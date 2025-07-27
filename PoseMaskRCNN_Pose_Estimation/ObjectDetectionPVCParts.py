import os
import numpy as np
import scipy.io
from PIL import Image
import open3d as o3d
import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt


def load_sim_annotation(filename, dataset_root):
    mat = scipy.io.loadmat(filename)
    gt = mat[next(k for k in mat if not k.startswith("__"))]

    # Read RGB image
    rgb_path = gt["RGBImagePath"][0, 0][0]
    rgb_path = os.path.join(dataset_root, *rgb_path.split("\\")[3:])
    im_rgb = np.array(Image.open(rgb_path).convert("RGB"))

    # Read depth
    depth_path = gt["DepthImagePath"][0, 0][0]
    depth_path = os.path.join(dataset_root, *depth_path.split("\\")[3:])
    depth = scipy.io.loadmat(depth_path)["depth"]

    # Sanitize labels
    labels = gt["instLabels"][0, 0]
    if labels.ndim == 2 and labels.shape[0] == 1:
        labels = labels.T
    if hasattr(labels[0], "__str__"):
        labels = np.array([str(x[0]) for x in labels])

    occ = gt["occPercentage"][0, 0].flatten()
    visible = occ > 0.5 if len(occ) == len(labels) else np.ones(len(labels), dtype=bool)

    bboxes = gt["instBBoxes"][0, 0][visible]
    masks = gt["instMasks"][0, 0][:, :, visible]

    # Rotation and translation -> 4x4 matrices
    rotations = gt["rotationMatrix"][0, 0]
    translations = gt["translation"][0, 0].T

    poses = []
    for R, t in zip(rotations.transpose(2, 0, 1), translations):
        pose = np.eye(4)
        pose[:3, :3] = R.T
        pose[:3, 3] = t
        poses.append(pose)
    poses = np.stack(poses)[visible]

    intr = gt["IntrinsicsMatrix"][0, 0]
    intrinsics = {
        "fx": intr[0, 0],
        "fy": intr[1, 1],
        "cx": intr[0, 2],
        "cy": intr[1, 2],
        "height": im_rgb.shape[0],
        "width": im_rgb.shape[1],
    }

    return im_rgb, depth, bboxes, labels[visible], masks, poses, intrinsics


def depth_to_pointcloud(depth, intr):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (i - intr['cx']) * z / intr['fx']
    y = (j - intr['cy']) * z / intr['fy']
    xyz = np.stack((x, y, z), axis=-1)
    return xyz.reshape(-1, 3)


def extract_object_point_clouds(depth, masks, intr):
    pc_all = depth_to_pointcloud(depth, intr)
    object_pcs = []
    for i in range(masks.shape[2]):
        mask = masks[:, :, i]
        indices = np.where(mask.flatten())[0]
        obj_pc = pc_all[indices]
        object_pcs.append(obj_pc)
    return object_pcs


def visualize_segmented_objects(object_pcs):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    vis = []
    for i, pc in enumerate(object_pcs):
        if pc.shape[0] == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.paint_uniform_color(colors[i % len(colors)])
        vis.append(pcd)
    o3d.visualization.draw_geometries(vis)


# === Mask R-CNN Prediction ===
def run_mask_rcnn(im_rgb):
    transform = transforms.Compose([transforms.ToTensor()])
    model = maskrcnn_resnet50_fpn(pretrained=True).eval()
    img_tensor = transform(im_rgb)
    with torch.no_grad():
        output = model([img_tensor])[0]
    return output


def visualize_predictions(im_rgb, output, score_thresh=0.5):
    masks = output['masks'] > 0.5
    scores = output['scores'].numpy()
    labels = output['labels'].numpy()
    boxes = output['boxes'].numpy()

    keep = scores >= score_thresh
    masks = masks[keep].squeeze(1).cpu().numpy()
    boxes = boxes[keep]

    overlay = im_rgb.copy()
    for i, mask in enumerate(masks):
        color = np.random.randint(0, 255, size=3)
        overlay[mask] = (0.4 * overlay[mask] + 0.6 * color).astype(np.uint8)

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Mask R-CNN Predictions")
    plt.show()

    return masks, boxes, labels[keep]


dataset_root = "/your/path/pvcparts100"
image_id = "00001"
gt_path = os.path.join(dataset_root, "GT", f"{image_id}.mat")

im_rgb, depth, bboxes, labels, masks, poses, intrinsics = load_sim_annotation(gt_path, dataset_root)

# 3D object segmentation from ground truth masks
object_pcs = extract_object_point_clouds(depth, masks, intrinsics)
visualize_segmented_objects(object_pcs)

# 2D instance segmentation with pretrained Mask R-CNN
output = run_mask_rcnn(im_rgb)
pred_masks, pred_boxes, pred_labels = visualize_predictions(im_rgb, output)

