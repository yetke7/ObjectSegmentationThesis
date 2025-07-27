# Novel Deep Learning with Point Clouds and Graph Convolutional Networks
This is the code for the paper "Deep Learning with Point Clouds and Graph
Convolutional Networks" by Yusuf Etke.

The segmenation pipelines for RGB images and the part object segmentation with a GCN require python and additional libararies that can be installed with.

	pip install -r requirements.txt

## Object segmentation with AmazonArmbench

To run the sam2 unseen segmenation execute the command:

	python3 segment_image.py

Ensure that the path to the dataset has been adjusted to your own path.

To run segmentation using Mask RCNN run the following code in the Mask-RCNNSegmentation folder

	python train_mask_rcnn.py --dataset_path $AB_SEG_DATA
	
Set dataset path to where you downloaded the dataset, view readme in Mask-RCNNSegmentation for clarification.

## Part object segmentation using the GCN
For part object segmentation using the GCN run the following code, using the pretrained ResGCN, category refers to categories from partnet, make sure to request access on https://shapenet.org/ and place the dataset in a directory named partnet.

	python3 -u eval.py --phase test --category 1 --pretrained_model --data_dir /home/data/partnet

To visualize the part object segments run this code:

	python3 -u visualize.py --dir_path /home/data/part_sem_seg/pretrained_partnet_ResGCN-28/result --category 20 --obj_no 1
	
## Object identification on the PVCparts
The Object identification on the PVCparts dataset using a Pose Mask R-CNN can be run with this command:

	python3 ObjectDetectionPVCParts.py

Make sure to adjust path to where you have the dataset stored.
