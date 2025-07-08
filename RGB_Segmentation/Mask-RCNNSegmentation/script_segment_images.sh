#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=3
#SBATCH --time=10:00:00



source ~/DoSegment/bin/activate

python /home/yetke/armbench/segmentation/segment_images.py
