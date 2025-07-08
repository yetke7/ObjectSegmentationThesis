import os
import numpy as np
import cv2
import argparse
from PIL import Image

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import dounseen.core, dounseen.utils

import torch

torch.manual_seed(0)

def save_image(image, name):
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", name)
    cv2.imwrite(path, image)
    print(f"Saved: {path}")

def main():
    parser = argparse.ArgumentParser()
    # input image and gallery
    parser.add_argument('--rgb-image-path', type=str, default='/home/yetke/sample_armbench_segmentation/IVW4RX878C.jpg')
    parser.add_argument('--gallery-images-path', type=str, default='/home/yetke/sample_armbench_segmentation')
    # model paths
    parser.add_argument('--classification-model-path', type=str, default='./models/dounseen/dounseen/vit_b_16_epoch_199_augment.pth')
    # Classifications parameters
    parser.add_argument('--classification-threshold', type=float, default=0.3)
    parser.add_argument('--multi-instance', action='store_true', default=False)
    parser.add_argument('--object-name', type=str, default='obj_000001')
    parser.add_argument('--classification-batch-size', type=int, default=80)
    # Background filter parameters
    parser.add_argument('--background-filter-model-path', type=str, default='./models/background_filter_Mask-RCNN.pth')

    args = parser.parse_args()

    args.rgb_image_path = os.path.abspath(args.rgb_image_path)
    args.gallery_images_path = os.path.abspath(args.gallery_images_path)

    rgb_img = Image.open(args.rgb_image_path)
    rgb_img = np.array(rgb_img.convert("RGB"))

    print("Create and run SAM 2")

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    sam2_output = sam2_mask_generator.generate(rgb_img)
    sam2_masks, sam2_bboxes = dounseen.utils.reformat_sam2_output(sam2_output)

    sam2_segmentation_image = dounseen.utils.draw_segmented_image(rgb_img, sam2_masks, sam2_bboxes)
    save_image(cv2.cvtColor(cv2.resize(sam2_segmentation_image, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_RGB2BGR), "unseen_segmentation.jpg")

    background_filter = dounseen.core.BackgroundFilter(maskrcnn_model_path='DEFAULT')
    sam2_masks, sam2_bboxes = background_filter.filter_background_annotations(rgb_img, sam2_masks, sam2_bboxes)

    sam2_filtered_background_segmentation_image = dounseen.utils.draw_segmented_image(rgb_img, sam2_masks, sam2_bboxes)
    save_image(cv2.cvtColor(cv2.resize(sam2_filtered_background_segmentation_image, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_RGB2BGR), "filtered_background.jpg")

    segments = dounseen.utils.get_image_segments_from_binary_masks(rgb_img, sam2_masks, sam2_bboxes)
    unseen_classifier = dounseen.core.UnseenClassifier(
        classification_model_path='DEFAULT',
        gallery_images=args.gallery_images_path,
        gallery_buffered_path=None,
        augment_gallery=False,
        batch_size=args.classification_batch_size,
    )

    matched_query, score = unseen_classifier.find_object(segments, obj_name=args.object_name, method="max")
    matched_query_ann_image = dounseen.utils.draw_segmented_image(
        rgb_img,
        [sam2_masks[matched_query]],
        [sam2_bboxes[matched_query]],
        classes_predictions=[0],
        classes_names=["obj_000001"]
    )
    save_image(cv2.cvtColor(cv2.resize(matched_query_ann_image, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_RGB2BGR), "matched_object.jpg")

    print("Classifying all objects")
    class_predictions, class_scores = unseen_classifier.classify_all_objects(
        segments,
        threshold=args.classification_threshold,
        multi_instance=args.multi_instance
    )
    filtered_class_predictions, filtered_masks, filtered_bboxes = dounseen.utils.remove_unmatched_query_segments(
        class_predictions, sam2_masks, sam2_bboxes)

    classified_image = dounseen.utils.draw_segmented_image(
        rgb_img,
        filtered_masks,
        filtered_bboxes,
        filtered_class_predictions,
        classes_names=os.listdir(args.gallery_images_path)
    )
    save_image(cv2.cvtColor(cv2.resize(classified_image, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_RGB2BGR), "classified_objects.jpg")


if __name__ == '__main__':
    main()

