# Implementation of Deep Learning Methods on RGB Images and 3D Point Clouds for Object Segmentation
This is the code for the paper "Enlightening Object Segmentation with Point Clouds and Graph
Convolutional Networks" by Yusuf Etke. Original code can be found in the referred papers.

The segmenation pipelines for RGB images and the part object segmentation with a GCN require python and additional libararies that can be installed with.

	pip install -r requirements.txt

## Object segmentation on AmazonArmbench

To run the sam2 unseen segmenation execute the command:

	python3 segmentation.py
 
Ensure that the path to the dataset has been adjusted to your own path.

To run segmentation using Mask R-CNN run the following code in the Mask R-CNNSegmentation folder

	python test_mask_rcnn.py --dataset_path $AB_SEG_DATA
	
Set dataset path to where you downloaded the dataset, view readme in Mask-RCNNSegmentation for clarification.

## Part object segmentation using the GCN
For part object segmentation using the GCN run the following code, using the pretrained ResGCN, category refers to categories from partnet, make sure to request access on https://shapenet.org/ and place the dataset in a directory named partnet.

	python3 -u eval.py --phase test --category 1 --pretrained_model --data_dir /home/data/partnet

To visualize the part object segments run this code:

	python3 -u visualize.py --dir_path /home/data/part_sem_seg/pretrained_partnet_ResGCN-28/result --category 20 --obj_no 1
 
<img width="400" height="719" alt="Image" src="https://github.com/user-attachments/assets/0a7bb8b1-9851-4425-b2fb-6ef5f1f84c20" />

## Object Segmentation on the PVCparts
The Object identification on the PVCparts dataset using the experimental Mask R-CNN can be run with this command:

	run(ObjectSegmentationPVCParts)

Make sure to adjust path to where you have the dataset stored.

<img width="400" height="631" alt="Image" src="https://github.com/user-attachments/assets/dd6bb0c8-bd17-446d-b0fc-5bc67707dd0f" />
