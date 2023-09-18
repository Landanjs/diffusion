# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Image-Caption dataset."""

from typing import Dict, Optional

import torch
from composer.utils import dist
from torch.utils.data import DataLoader, Dataset


class SyntheticImageCaptionDataset(Dataset):
    """Synthetic dataset imitating a dataset containing image-caption pairs.

    Args:
        image_size (int): Size of the synthetic images. Default: ``512``.
        caption_length (int): Length of the synthetic captions. Default: ``77``.
        num_samples (int): Number of samples in the synthetic dataset. Default: ``100_000``.
    """

    def __init__(self, image_size: int = 512, caption_length: int = 77, num_samples: int = 100_000, precompute_img_encoder: bool = False, precompute_txt_encoder: bool = False, txt_embed_dim: int = 1024):

        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.caption_length = caption_length
        self.img_channel_dim = 4 if precompute_img_encoder else 3
        self.precompute_txt_encoder = precompute_txt_encoder
        self.caption_length = caption_length
        self.txt_embed_dim = txt_embed_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.precompute_txt_encoder:
            captions = torch.randn(self.caption_length, self.txt_embed_dim)
        else:
            captions = torch.randint(0, 128, (self.caption_length,), dtype=torch.long)
        return {'image': torch.randn(self.img_channel_dim, self.image_size, self.image_size),
                'captions': captions,
                'captions_2': captions,
                'pooled_conditioning': torch.randn(1280),
                'cond_original_size': torch.tensor([256., 256.]),
                'cond_crops_coords_top_left': torch.tensor([0., 0.]),
                'cond_target_size': torch.tensor([256., 256.])}


def build_synthetic_image_caption_dataloader(
    batch_size: int,
    image_size: int = 512,
    caption_length: int = 77,
    num_samples: int = 100_000,
    precompute_img_encoder: bool = False,
    precompute_txt_encoder: bool = False,
    txt_embed_dim: int = 2048,
    dataloader_kwargs: Optional[Dict] = None,
):
    """Builds a dataloader for the synthetic image-caption dataset.

    Args:
        batch_size (int): Batch size for the dataloader.
        image_size (int): Size of the synthetic images. Default: ``512``.
        caption_length (int): Length of the synthetic captions. Default: ``77``.
        num_samples (int): Number of samples in the synthetic dataset. Default: ``100_000``.
        dataloader_kwargs (optional, dict): Additional arguments to pass to the dataloader. Default ``None``.
    """
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    dataset = SyntheticImageCaptionDataset(
        image_size=image_size,
        caption_length=caption_length,
        num_samples=num_samples,
        precompute_img_encoder=precompute_img_encoder,
        precompute_txt_encoder=precompute_txt_encoder,
        txt_embed_dim=txt_embed_dim,
    )

    dataloader = DataLoader(
        dataset=dataset,
        sampler=dist.get_sampler(dataset),
        batch_size=batch_size,
        **dataloader_kwargs,
    )

    return dataloader
