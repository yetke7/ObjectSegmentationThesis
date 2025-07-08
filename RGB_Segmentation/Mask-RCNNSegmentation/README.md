## Requirements

Download the Segmentation dataset from [armbench.com](http://armbench.com)

Extract armbench-segmentation-0.1.tar.gz to filesystem 

```
tar -xf armbench-segmentation-0.1.tar.gz

cd armbench-segmentation-0.1
export AB_SEG_DATA=`pwd`
```

Download the pretrained mask-rcnn model

```
wget https://armbench-dataset.s3.amazonaws.com/segmentation/latest.pt
```
## Test a model
```
python test_mask_rcnn.py --dataset_path $AB_SEG_DATA --resume-from latest.pt 
```

