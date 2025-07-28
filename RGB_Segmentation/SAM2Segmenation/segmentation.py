import sam2
from dounseen.core import UnseenClassifier
import dounseen.utils as dounseen_utils
import torch
import torchvision
import time
import dounseen
from PIL import Image
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import os
start_timer = time.time()


print("initializing unseen classifier")
unseen_classifier = dounseen.core.UnseenClassifier(
  gallery_images=None,  # Can be initialized later using update_gallery()
  gallery_buffered_path=None,
  augment_gallery=False,
  batch_size=80,
)
print("setting up cuda")
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
print("loading sam2")
# load SAM 2 from HuggingFace
sam2_mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
  'facebook/sam2-hiera-tiny',
   points_per_side=20,
   points_per_batch=20,
   pred_iou_thresh=0.7,
   stability_score_thresh=0.92,
   stability_score_offset=0.7,
   crop_n_layers=0,
   box_nms_thresh=0.7,
   multimask_output=False,
 )

# Input and output directories
input_dir = "/home/yetke/Datasets/pvcparts100/image"
output_dir = "/home/yetke/output_sam2_pvcparts"
os.makedirs(output_dir, exist_ok=True)
print("Generating masks and segments....")
# Outside the loop
background_filter = dounseen.core.BackgroundFilter(maskrcnn_model_path='DEFAULT')

# Inside the loop
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".png"):
        image_path = os.path.join(input_dir, filename)
        rgb_img = Image.open(image_path).convert("RGB")
        rgb_img_np = np.array(rgb_img)

        # Generate masks
        sam2_output = sam2_mask_generator.generate(rgb_img_np)
        sam2_masks, sam2_bboxes = dounseen.utils.reformat_sam2_output(sam2_output)

        # Segments (optional, if you need them later)
        segments = dounseen.utils.get_image_segments_from_binary_masks(rgb_img_np, sam2_masks, sam2_bboxes)

        # Background filtering (safe check should be inside dounseen.core.remove_background_masks)
        sam2_masks, sam2_bboxes = background_filter.filter_background_annotations(rgb_img_np, sam2_masks, sam2_bboxes)

        # Draw and save segmented image
        seg_image = dounseen.utils.draw_segmented_image(rgb_img_np, sam2_masks, sam2_bboxes)
        out_path = os.path.join(output_dir, f"segmented_{filename}")
        Image.fromarray(seg_image).save(out_path)

        print(f"Processed and saved: {out_path}")



end_time =time.time()
execution_time = end_time - start_timer
print(f"Execution time: {execution_time:.4f} seconds")


