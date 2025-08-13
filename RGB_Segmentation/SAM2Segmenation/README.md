Installation
Please install SAM2 it as follows:

git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .

Install library for unseen:

git clone https://github.com/AnasIbrahim/image_agnostic_segmentation.git && cd image_agnostic_segmentation
pip install -e .

Download a pretrained models from HuggingFace using git LFS:

cd image_agnostic_segmentation
git lfs install  # if not installed
git clone https://huggingface.co/anas-gouda/dounseen models/
