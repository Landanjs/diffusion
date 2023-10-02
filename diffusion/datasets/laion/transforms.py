# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Transforms for the laion dataset."""

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop, get_dimensions


def random_crop_params(img, output_size):
    """ Return the parameters for a random crop.

    Inputs:
        img: Tensor
        output_size: int
    Returns:
        crop_params: List[int]
    Based off of Algorithm 1 of SDXL paper: https://arxiv.org/pdf/2307.01952.pdf
    # TODO note: this only works for generating square images since output_size is scalar
    #  - need to adjust output_size for multi-aspect training!
    """
    _, image_height, image_width = get_dimensions(img)
    if image_height == image_width:
        c_left = 0
        c_top = 0
    elif image_height < image_width:
        c_left = np.random.randint(0, image_width - output_size)
        c_top = 0
    else:
        c_left = 0
        c_top = np.random.randint(0, image_height - output_size)
    cropped_im = crop(img, c_top, c_left, output_size, output_size)
    return cropped_im, c_top, c_left


class LargestCenterSquare:
    """Center crop to the largest square of a PIL image."""

    def __init__(self, size):
        self.size = size
        self.center_crop = transforms.CenterCrop(self.size)

    def __call__(self, img):
        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        img = transforms.functional.resize(img, self.size, antialias=True)
        # Then take a center crop to a square.
        img = self.center_crop(img)
        return img


class LargestCenterSquareReturnTransform:
    """Center crop to the largest square of a PIL image."""

    def __init__(self, size):
        self.size = size
        self.center_crop = transforms.CenterCrop(self.size)

    def __call__(self, img):
        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        orig_w, orig_h = img.size
        img = transforms.functional.resize(img, self.size, antialias=True)
        # Then take a center crop to a square.
        img = self.center_crop(img)
        return img, 0, 0, orig_h, orig_w


class RandomCropSquareReturnTransform:
    """Randomly crop square of a PIL image and return the crop parameters."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        orig_w, orig_h = img.size
        img = transforms.functional.resize(img, self.size, antialias=True)
        # Then take a center crop to a square & return crop params.
        img, c_top, c_left = random_crop_params(img, self.size)
        return img, c_top, c_left, orig_h, orig_w

class RandomCropAspectRatioTransorm:

    def __init__(self):
        self.height_buckets = torch.tensor([512, 512, 512, 512, 576, 576, 576, 640, 640, 704, 704, 704, 768, 768, 832, 832,
                                            896, 896, 960, 960, 1024, 1024, 1088, 1088, 1152, 1152, 1216, 1280, 1344,
                                            1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048])
        self.width_buckets = torch.tensor([2048, 1984, 1920, 1856, 1792, 1728, 1664, 1600, 1536, 1472, 1408, 1344,
                                           1344, 1280, 1216, 1152, 1152, 1088, 1088, 1024, 1024, 960, 960, 896, 896,
                                           832, 832, 768, 768, 704, 704, 640, 640, 576, 576, 576, 512, 512, 512, 512])
        self.aspect_ratio_buckets = torch.round(self.height_buckets / self.width_buckets, decimals=2)

    def __call__(self, img):
        orig_w, orig_h = img.size
        aspect_ratio = orig_h / orig_w
        bucket_ind = torch.abs(self.aspect_ratio_buckets - aspect_ratio).argmin()
        target_width, target_height = self.width_buckets[bucket_ind], self.height_buckets[bucket_ind]

        # Get the size to resize the shortest side
        resize_size = min(target_width, target_height)
        img = transforms.functional.resize(img, resize_size, antialias=True)

        # Crop based on aspect ratio
        top, left, height, width = transforms.RandomCrop.get_params(img, output_size=(target_height, target_width))
        img = crop(img, top, left, height, width)
        return img, top, left, orig_w, orig_h
