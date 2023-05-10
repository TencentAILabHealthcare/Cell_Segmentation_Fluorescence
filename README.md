# Cell Segmentation for Fluorescence Images

[![python >3.8.13](https://img.shields.io/badge/python-3.8.13-brightgreen)](https://www.python.org/) 

This repository provides a cell segmentation method for ssDNA images with ConA images assisted, which is implemented based on mmdetection. The method is mainly consisted of two parts, annotate cells on ConA images with an active learning procedure and train ssDNA image cell segmentation model with labels transferred from ConA images.

<img src="overview.png" width="600">

# Dependences

[![numpy-1.23.3](https://img.shields.io/badge/numpy-1.23.3-red)](https://github.com/numpy/numpy)
[![pycocotools-2.0.6](https://img.shields.io/badge/pycocotools-2.0.6-lightgrey)](https://github.com/cocodataset/cocoapi)
[![torch-1.12.1](https://img.shields.io/badge/torch-1.12.1-orange)](https://github.com/pytorch/pytorch)
[![mmcv-1.7.0](https://img.shields.io/badge/mmcv-1.7.0-green)](https://github.com/open-mmlab/mmcv/)


# Usage

## Data preparation
Run codes in dataset_generation/ to crop images into patches and generate COCO format json files.
```
python dataset_generation/conA/crop.py
python dataset_generation/conA/generate_coco_annotation.py.py
python dataset_generation/ssDNA/crop.py
python dataset_generation/ssDNA/generate_coco_annotation.py.py
```

## ConA model training and validation
The configuration file for training and validating a ConA model is in local_configs/conA_human_in_loop/, run the following code to train a ConA model.
```
bash tools/dist_train.sh local_configs/conA_human_in_loop/mask_rcnn_baseline.py 2
```

## ssDNA model training and validation
The configuration file for training and validating an ssDNA model is in local_configs/ssDNA_with_incomplete_annotations/, run the following code to train an ssDNA model.
```
bash tools/dist_train.sh local_configs/ssDNA_with_incomplete_annotations/mask_rcnn_baseline.py 2
```

# Disclaimer

This tool is for research purpose and not approved for clinical use.

This is not an official Tencent product.

# Coypright

This tool is developed in Tencent AI Lab.

The copyright holder for this project is Tencent AI Lab.

All rights reserved.
