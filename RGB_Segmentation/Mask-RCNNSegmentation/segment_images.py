import argparse
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image
from coco_utils import get_coco, collate_fn
from engine import evaluate



def parse_args():

    args = argparse.Namespace(
    	resume_from="/home/yetke/armbench/segmentation/latest.pt",
	dataset_path="/home/yetke/Datasets/IPARingScrew_part_1/p_rgb/cycle_0000"
    )
    return args
#    parser = argparse.ArgumentParser(description="Test a detector")
#    parser.add_argument("--resume-from", required=True, help="Checkpoint file to load")
#    parser.add_argument("-d", "--dataset_path", required=True, help="Path to dataset")
#    return parser.parse_args()

def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def save_segmented_images(images, predictions, file_names, output_dir="output-segment"):
    os.makedirs(output_dir, exist_ok=True)

    for img, pred, file_name in zip(images, predictions, file_names):
        masks = pred["masks"] > 0.5
        if masks.shape[0] == 0:
            print(f"[{file_name}] No masks detected.")
            continue

        img_uint8 = (img.cpu() * 255).to(torch.uint8)
        img_drawn = draw_segmentation_masks(img_uint8, masks.squeeze(1).cpu(), alpha=0.6)
        out_img = to_pil_image(img_drawn)

        base_name = os.path.splitext(os.path.basename(file_name))[0]
        out_path = os.path.join(output_dir, f"{base_name}_segmented.jpg")
        out_img.save(out_path)
        print(f"[{file_name}] Saved: {out_path}")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_data_path = os.path.join(args.dataset_path, "/home/yetke/Datasets/IPARingScrew_part_1/p_rgb/cycle_0000")
    #val_ann_data_path = os.path.join(args.dataset_path, "mix-object-tote/test.json")

    dataset = get_coco(img_data_path, mode="instances", train=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    num_classes = 3  # Adjust this based on your dataset
    model = get_instance_segmentation_model(num_classes)
    checkpoint = torch.load(args.resume_from, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    for i, (int_img, img, target) in enumerate(data_loader):
    	img_tensor = img[0].to(device)
    	image_id = dataset.ids[i]
    	file_name = dataset.coco.loadImgs(image_id)[0]["file_name"]
    	image_path = os.path.join(img_data_path, file_name)

    	if not os.path.exists(image_path):
        	print(f"[Warning] File does not exist: {image_path} — skipping.")
        	continue

    	with torch.no_grad():
        	predictions = model([img_tensor])
    	save_segmented_images([img_tensor.cpu()], predictions, [file_name], output_dir="output-segment")
    	del predictions
    	torch.cuda.empty_cache()
        
if __name__ == "__main__":
    main()

