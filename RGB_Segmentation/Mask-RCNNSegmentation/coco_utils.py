import copy
import os
import torch
import torch.utils.data
import torchvision
from torchvision import transforms as T
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO


class FilterAndRemapCocoCategories:
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, target):
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            target["annotations"] = anno
            return image, target
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        target["annotations"] = anno
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __call__(self, image, target):
        w, h = image.size
        convert_tensor = T.ToTensor()
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = [obj for obj in target["annotations"] if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = torch.tensor([obj["category_id"] for obj in anno], dtype=torch.int64)
        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            if keypoints.shape[0]:
                keypoints = keypoints.view(keypoints.shape[0], -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes, classes, masks = boxes[keep], classes[keep], masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {
            "boxes": boxes,
            "labels": classes,
            "masks": masks,
            "image_id": image_id,
            "area": torch.tensor([0 for _ in anno]),
            "iscrowd": torch.tensor([obj["iscrowd"] for obj in anno])
        }
        if keypoints is not None:
            target["keypoints"] = keypoints

        return convert_tensor(image), target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    def _has_valid_annotation(anno):
        if len(anno) == 0:
            return False
        if _has_only_empty_bbox(anno):
            return False
        if "keypoints" not in anno[0]:
            return True
        return _count_visible_keypoints(anno) >= 10

    if not isinstance(dataset, torchvision.datasets.CocoDetection):
        raise TypeError(f"Expected CocoDetection, got {type(dataset)}")

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    return torch.utils.data.Subset(dataset, ids)


def collate_fn(batch):
    return tuple(zip(*batch))


def convert_to_coco_api(ds):
    coco_ds = COCO()
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()

    for img_idx in range(len(ds)):
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {
            "id": image_id,
            "height": img.shape[-2],
            "width": img.shape[-1]
        }
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()

        if "masks" in targets:
            masks = targets["masks"].permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"].reshape(targets["keypoints"].shape[0], -1).tolist()

        for i in range(len(bboxes)):
            ann = {
                "image_id": image_id,
                "bbox": bboxes[i],
                "category_id": labels[i],
                "area": areas[i],
                "iscrowd": iscrowd[i],
                "id": ann_id
            }
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            categories.add(labels[i])
            ann_id += 1

    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            return dataset.coco
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    return convert_to_coco_api(dataset)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        int_img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            return int_img, *self._transforms(int_img, target)
        return int_img, int_img, target


def get_coco(img_folder, ann_file, mode="instances", train=True):
    dataset = CocoDetection(img_folder, ann_file, transforms=ConvertCocoPolysToMask())

    # ✅ Filter out missing image files
    valid_ids = []
    skipped = []
    for img_id in dataset.ids:
        file_name = dataset.coco.loadImgs(img_id)[0]["file_name"]
        file_path = os.path.join(img_folder, file_name)
        if os.path.exists(file_path):
            valid_ids.append(img_id)
        else:
            print(f"[Skipping] Missing file: {file_path}")
            skipped.append(file_path)

    dataset.ids = valid_ids

    if skipped:
        with open("skipped_images.log", "w") as f:
            for path in skipped:
                f.write(path + "\n")

    if train:
        dataset = _coco_remove_images_without_annotations(dataset)

    return dataset


def get_coco_kp(root, image_set, transforms):
    return get_coco(root, image_set, transforms, mode="person_keypoints")

