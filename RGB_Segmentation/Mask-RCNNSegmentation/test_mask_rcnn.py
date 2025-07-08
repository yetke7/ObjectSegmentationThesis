import argparse
import os
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

from PIL import Image, ImageOps
import PIL


from coco_utils import get_coco, collate_fn
from engine import train_one_epoch, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Test a detector")
    parser.add_argument(
        "--resume-from",
        help="The checkpoint file to load the model from",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--dataset_path",
        default=None,
        help="Path to folder with images. If not provided, a default will be used.",
    )

    args = parser.parse_args()
    return args

def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model

def save_result(img, int_img, prediction, output_path):
    img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

    output = prediction[0]
    masks, boxes = output["masks"], output["boxes"]

    detection_threshold = 0.8
    pred_scores = output["scores"].detach().cpu().numpy()
    pred_classes = [str(i) for i in output["labels"].cpu().numpy()]
    pred_bboxes = output["boxes"].detach().cpu().numpy()
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    pred_classes = pred_classes[: len(boxes)]

    int_img = np.array(int_img)
    int_img = np.transpose(int_img, [2, 0, 1])
    int_img = torch.tensor(int_img, dtype=torch.uint8)

    colors = np.random.randint(0, 255, size=(len(pred_bboxes), 3))
    colors = [tuple(color) for color in colors]
    result_with_boxes = draw_bounding_boxes(
        int_img,
        boxes=torch.tensor(boxes),
        width=4,
        colors=colors,
        labels=pred_classes,
    )

    final_masks = masks > 0.5
    final_masks = final_masks.squeeze(1)
    seg_result = draw_segmentation_masks(
        result_with_boxes, final_masks, colors=colors, alpha=0.8
    )

    seg_img = Image.fromarray(seg_result.mul(255).permute(1, 2, 0).byte().numpy())

    imgs = [img, seg_img]
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])

    imgs_comb = Image.fromarray(imgs_comb)

    imgs_comb.save(output_path)



def main():
    args = parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 3  # must match model training

    # Load model
    print("Loading model...")
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.resume_from}...")
    checkpoint = torch.load(args.resume_from, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Checkpoint loaded.")

    model.eval()

    # Determine image folder
    if args.dataset_path is not None:
        image_dir = args.dataset_path
    else:
        image_dir = "/home/yetke/Datasets/SileaneBrick/p_rgb/cycle_0000"
        print(f"No dataset path provided. Using default: {image_dir}")

    # Find images
    image_filenames = [
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if len(image_filenames) == 0:
        print(f"No images found in {image_dir}. Exiting.")
        return

    print(f"Found {len(image_filenames)} images in {image_dir}.")

    # Prepare output folder
    output_dir = "/home/yetke/results_brick_maskrcnn"
    os.makedirs(output_dir, exist_ok=True)

    # Inference loop
    for image_path in image_filenames:
        print(f"Processing {image_path}...")

        # Load and prepare image
        int_img = Image.open(image_path).convert("RGB")
        img = torchvision.transforms.ToTensor()(int_img)

        with torch.no_grad():
            prediction = model([img.to(device)])

        # Save result
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_result.jpg"
        output_path = os.path.join(output_dir, output_filename)

        save_result(img, int_img, prediction, output_path)
        print(f"Saved result to {output_path}.")

    print("All done!")


if __name__ == "__main__":
    main()
