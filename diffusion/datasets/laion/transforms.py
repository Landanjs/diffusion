# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Transforms for the laion dataset."""

import numpy as np
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
