# crop image to generate dataset
import numpy as np
import os
import cv2
from tqdm import tqdm


def crop(image, mask=None, patch_size=256, overlap=50):
    """
    crop whole image into patches
    args: 
        image: numpy array, (h, w, c)
        mask: numpy arrary, (h, w, c)
        patch_size: int
        overlap: int 
    """
    h, w = image.shape[:2]

    start_h_list = np.arange(h - (patch_size - overlap))[::(patch_size - overlap)]
    start_h_list[-1] = h - patch_size if start_h_list[-1] + patch_size > h else start_h_list[-1]
    start_w_list = np.arange(w - (patch_size - overlap))[::(patch_size - overlap)]
    start_w_list[-1] = w - patch_size if start_h_list[-1] + patch_size > w else start_h_list[-1]

    num = 0
    patches = []
    for h_t in start_h_list:
        for w_t in start_w_list:
            cropped_image = image[h_t:h_t+patch_size, w_t:w_t+patch_size]
            if cropped_image.sum() == 0:
                continue 
            cropped_mask = None
            if mask is not None:
                cropped_mask = mask[h_t:h_t+patch_size, w_t:w_t+patch_size]

            patches.append([(h_t, w_t), cropped_image, cropped_mask])
            num += 1
    return patches


def process(source, dist, size, overlap): 
    os.makedirs(dist, exist_ok=True)
    for f in tqdm(os.listdir(source)):
        if "mask" in f:
            continue
        path = os.path.join(source, f)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        msk = cv2.imread(os.path.join(source, f.replace("conA", "mask_denoised")), cv2.IMREAD_UNCHANGED)
        patches = crop(img, msk, patch_size=size, overlap=overlap)
        for (h, w), patch_image, patch_mask in patches:
            image_name = name = "{}_{}_{}_h{}_w{}_{}.png".format(*f.split("_")[:3], h, w, "conA")
            cv2.imwrite(os.path.join(dist, image_name), patch_image)
            mask_name = name = "{}_{}_{}_h{}_w{}_{}.png".format(*f.split("_")[:3], h, w, "mask")
            cv2.imwrite(os.path.join(dist, mask_name), patch_mask)


if __name__ == "__main__":
    process(
        "/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/conA/1",
        "/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/conA/1_size256_overlap64",
        256,
        64,
    )
    process(
        "/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/conA/2",
        "/mnt/zihanwu/wentaopan/cell_instance_segmentation/data/conA/2_size256_overlap64",
        256,
        64,
    )