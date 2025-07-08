import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ground_truth_annotations = "/home/yetke/subset_armbench_annotations"
segmentation_output_sam2 = "/home/yetke/subset_armbench_segmented_sam2"
segmentation_output_resnet50 = "/home/yetke/Mask_RCNN_segmented_subset"
eval_plots_output = "./eval_plots_segmentation/"
metrics_output = "./segmentation_evaluation.csv"

os.makedirs(eval_plots_output, exist_ok=True)


def load_ground_truth(ground_truth_path, image_size):
    try:
        with open(ground_truth_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR loading ground truth {ground_truth_path}: {e}")
        return None  # Return None to signal failure
    
    ground_truth_mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
    for shape in data['shapes']:
        if shape['label'] == 'Object':
            polygon = np.array(shape['points'], np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            cv2.fillPoly(ground_truth_mask, [polygon], color=1)
    
    return ground_truth_mask

def load_seg_mask(image_path):
    seg_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(seg_img, 127, 1, cv2.THRESH_BINARY)
    return binary_mask

def compute_metrics(ground_truth_mask, pred_mask):
    TP = np.sum((ground_truth_mask == 1) & (pred_mask == 1))
    FP = np.sum((ground_truth_mask == 0) & (pred_mask == 1))
    FN = np.sum((ground_truth_mask == 1) & (pred_mask == 0))
    TN = np.sum((ground_truth_mask == 0) & (pred_mask == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "IoU": iou,
        "Accuracy": accuracy
    }

image_folder = "/home/yetke/subset_armbench"

# List all .jpg files and extract base ID (filename without extension)
image_ids = [
    os.path.splitext(filename)[0]
    for filename in os.listdir(image_folder)
    if filename.lower().endswith(".jpg")
]

print(f"Found {len(image_ids)} image IDs.")
print(image_ids)


metrics_list = []

for image_id in image_ids:
    print(f"\n=== Processing: {image_id} ===")
    
    ground_truth_path = os.path.join(ground_truth_annotations, f"{image_id}.json")
    ground_truth_mask = load_ground_truth(ground_truth_path, (2448, 2048))
    
    if ground_truth_mask is None:
        print(f"Skipping image {image_id} due to bad ground truth.")
        continue  # Skip this image safely
    
    for model_name, model_folder in [("SAM2", segmentation_output_sam2), ("Mask-RCNN (ResNet50)", segmentation_output_resnet50)]:

        if model_name == "SAM2":
            seg_filename = f"segmented_{image_id}.jpg"
        else:  # Mask-RCNN (ResNet50)
            seg_filename = f"{image_id}_segmented.jpg"

        seg_path = os.path.join(model_folder, seg_filename)
        if not os.path.exists(seg_path):
            print(f"WARNING: Missing segmentation for {model_name}: {seg_path}")
            continue
        
        seg_mask = load_seg_mask(seg_path)
        
        if seg_mask.shape != ground_truth_mask.shape:
            seg_mask = cv2.resize(seg_mask, (ground_truth_mask.shape[1], ground_truth_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Compute metrics
        metrics = compute_metrics(ground_truth_mask, seg_mask)
        metrics_list.append({
            "image_id": image_id,
            "model": model_name,
            "precision": metrics["Precision"],
            "recall": metrics["Recall"],
            "f1_score": metrics["F1-score"],
            "iou": metrics["IoU"],
            "accuracy": metrics["Accuracy"]
        })
        
        # Save comparison plot
        output_plot_path = os.path.join(eval_plots_output, f"{image_id}_{model_name}_comparison.png")
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1,3,1)
        plt.title("Ground Truth Mask")
        plt.imshow(ground_truth_mask, cmap='gray')
        
        plt.subplot(1,3,2)
        plt.title(f"{model_name} Mask")
        plt.imshow(seg_mask, cmap='gray')
        
        plt.subplot(1,3,3)
        plt.title(f"Overlay ({model_name} vs GT)")
        overlay = np.stack([seg_mask * 255, ground_truth_mask * 255, np.zeros_like(ground_truth_mask)], axis=-1)
        plt.imshow(overlay)
        
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        
        print(f"Saved plot to {output_plot_path}")
        print(f"Metrics for {model_name}: {metrics}")

# === Save all metrics to CSV ===
df_metrics = pd.DataFrame(metrics_list)
df_metrics.to_csv(metrics_output, index=False)
print(f"\nSaved metrics table to: {metrics_output}")

# === Summary ===
print("\n=== Summary of all images ===")
print(df_metrics)
