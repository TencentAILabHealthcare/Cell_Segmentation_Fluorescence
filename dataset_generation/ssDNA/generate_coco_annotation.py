import json
import os
import cv2
from tqdm import tqdm
import numpy as np
from skimage import measure
from pycococreatortools import pycococreatortools
from pycocotools.coco import COCO


def image2coco(root, path_list):
    coco_output = {
        "categories": [{'id': 1, 'name': 'cell',}],
        "images": [],
        "annotations": []
    }
 
    image_id = 1
    segmentation_id = 1
    for image_path, mask_path in tqdm(path_list):
        image = cv2.imread(os.path.join(root, image_path), cv2.IMREAD_UNCHANGED)
        image_info = pycococreatortools.create_image_info(
            image_id, image_path, image.shape)
        coco_output["images"].append(image_info)
    

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask_pad = np.pad(mask, ((2,2),(2,2)), constant_values=0)
        labeled_mask_pad, cell_num = measure.label(mask_pad, background=0, return_num=True)
        labeled_mask = labeled_mask_pad[2:-2,2:-2]

        for label in range(1, cell_num+1):
            binary_mask = np.zeros_like(labeled_mask, dtype=np.uint8)
            binary_mask[labeled_mask == label] = 1
            category_info = {'id': 1, 'is_crowd': 1}
            annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.shape, tolerance=0)
            annotation_info["iscrowd"] = 0
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
                segmentation_id = segmentation_id + 1

        image_id = image_id + 1
    return coco_output


def process_train_1():
    root = "/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/ssDNA"
    target = "/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/ssDNA/train.json"
    path_list = []
    source = "/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/ssDNA/1_size256_overlap64"
    folder = os.path.basename(source)
    for f in os.listdir(source):
        name = "{}_{}_{}".format(*f.split("_")[:3])
        if name in ["T577"]:
            continue
        path_list.append([
            os.path.join(folder, f),
            os.path.join("/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/conA/1_size256_overlap64", f.replace("ssDNA", "mask"))
        ])     
    coco_output = image2coco(root, path_list)
    with open(target, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


def process_val():
    root = "/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/ssDNA"
    target = "/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/ssDNA/val.json"
    path_list = []
    source = "/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/ssDNA/val_size256_overlap64"
    folder = os.path.basename(source)
    for f in os.listdir(source):
        name = "{}_{}_{}".format(*f.split("_")[:3])
        if name not in ["T577"]:
            continue
        path_list.append([
            os.path.join(folder, f),
            os.path.join(source, f.replace("ssDNA", "mask"))
        ])     
    coco_output = image2coco(root, path_list)
    with open(target, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    process_train_1()
    process_val()
