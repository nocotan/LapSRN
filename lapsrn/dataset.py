# -*- coding: utf-8 -*-
import os
import six
import numpy as np
import cv2
import random
from PIL import Image
from chainer.dataset import dataset_mixin


class PILImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, resize=None, root='.'):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root = root
        self._resize = resize

    def __len__(self):
        return len(self._paths)

    def get_example(self, i) -> Image:
        path = os.path.join(self._root, self._paths[i])
        original_image = Image.open(path)
        if self._resize is not None:
            return original_image.resize(self._resize)
        else:
            return original_image


class ResizedImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, resize=None, root='.', dtype=np.float32):
        self.base = PILImageDataset(paths=paths, resize=resize, root=root)
        self._dtype = dtype

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image = self.base[i]
        image_ary = np.asarray(image, dtype=self._dtype)
        if len(image_ary.shape) == 2:
            image_ary = np.dstack((image_ary, image_ary, image_ary))
        image_data = image_ary.transpose(2, 0, 1)
        if image_data.shape[0] == 4:
            image_data = image_data[:3]
        return image_data


class ImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, cropsize, resize=None, root='.',
                 dtype=np.float32, gpu=-1, scale=4):
        self.base = ResizedImageDataset(paths=paths, resize=resize, root=root)
        self._dtype = dtype
        self.cropsize = cropsize
        self.scale = scale

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image = self.base[i]
        x = random.randint(0, image.shape[1] - self.cropsize)
        y = random.randint(0, image.shape[2] - self.cropsize)

        cropped_high_res = image[:, x:x + self.cropsize, y:y + self.cropsize]
        cropped_low_res = cv2.resize(
            cropped_high_res.transpose(1, 2, 0),
            dsize=(int(self.cropsize/self.scale),
                   int(self.cropsize/self.scale)),
            interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

        #cropped_low_res = cv2.cvtColor(cropped_low_res.reshape((24, 24, 3)), cv2.COLOR_RGB2GRAY)
        #cropped_high_res = cv2.cvtColor(cropped_high_res.reshape((96, 96, 3)), cv2.COLOR_RGB2GRAY)

        return cropped_low_res, cropped_high_res
