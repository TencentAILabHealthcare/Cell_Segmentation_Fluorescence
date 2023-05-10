import cv2
import random
import numpy as np
import os.path as osp
from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class Load16BitImageFromFile:
    def __init__(self):
        pass

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}')
        return repr_str


@PIPELINES.register_module()
class CellBrightnessTransform:
    def __init__(self, brightness_range=(-32, 32), prob=0.5):
        self.brightness_range = brightness_range
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        img = results['img']
        delta = random.uniform(self.brightness_range[0], self.brightness_range[1])
        img += delta
        img = np.clip(img, 0, 65535)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(brightness_range={self.brightness_range}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class CellContrastTransform:
    def __init__(self, contrast_range=(0.5, 2), prob=0.5):
        self.contrast_range = contrast_range
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        img = results['img']
        alpha = random.uniform(self.contrast_range[0], self.contrast_range[1])
        img *= alpha
        img = np.clip(img, 0, 65535)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(contrast_range={self.brightness_range}, '
        repr_str += f'prob={self.prob})'
        return repr_str

@PIPELINES.register_module()
class CellRepeatTransform:
    def __init__(self,):
        pass

    def __call__(self, results):
        img = results['img']
        img = np.stack((img, img, img), axis=2)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


# @PIPELINES.register_module()
# class CellNormalizeTransform:
#     def __init__(self, window=(0, 15000)):
#         self.window = window

#     def __call__(self, results):
#         img = results['img']
#         # img = np.stack((img, img, img), axis=2)
#         img = np.clip(img, self.window[0], self.window[1]) / (self.window[1] - self.window[0])
#         results['img'] = img
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         return repr_str
@PIPELINES.register_module()
class CellNormalizeTransform:
    def __init__(self):
        pass

    def __call__(self, results):
        img = results['img']
        # img = np.stack((img, img, img), axis=2)
        img = (img - img.mean()) / img.std()
        results['img'] = img.astype(np.float32)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

import cv2

@PIPELINES.register_module()
class CellAdpativeBrightnessTransform:
    def __init__(self, target_brightness=50000):
        self.target_brightness = target_brightness

    def __call__(self, results):
        image = results['img']
        local_weight = cv2.blur((image > 0).astype(float), (51, 51))
        local_mean = cv2.blur(image.astype(float), (51, 51)) / (local_weight + 1e-10)
        image = (image * (self.target_brightness / local_mean).clip(0, 3)).clip(0, 65535)
        results['img'] = image
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str