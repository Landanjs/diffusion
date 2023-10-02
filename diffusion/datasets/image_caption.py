# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming Image-Caption dataset."""

import random
from io import BytesIO
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from PIL import Image
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from diffusion.datasets.laion.transforms import RandomCropSquareReturnTransform, LargestCenterSquareReturnTransform, RandomCropAspectRatioTransorm

# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None


class StreamingImageCaptionDataset(StreamingDataset):
    """Streaming dataset for image-caption pairs.

    Args:
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from.
            ``StreamingImageCaptionDataset`` uses either ``streams`` or ``remote``/``local``. Default:``None``.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored. Default: ``None``.
        local (str, optional): Local filesystem directory where dataset is cached during operation. Default: ``None``.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        caption_selection (str): If there are multiple captions, specifies how to select a single caption.
            'first' selects the first caption in the list and 'random' selects a random caption in the list.
            If there is only one caption, this argument is ignored. Default: ``'first'``.
        transform (Optional[Callable]): The transforms to apply to the image. Default: ``None``.
        image_size (Optional[int]): The size to resize the image to. Default: ``None``.
        image_key (str): Key associated with the image in the streaming dataset. Default: ``'image'``.
        caption_key (str): Key associated with the caption in the streaming dataset. Default: ``'caption'``.
        **streaming_kwargs: Additional arguments to pass in the construction of the StreamingDataloader
    """

    def __init__(
        self,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
        tokenizer_name_or_path_2: Optional[str] = None,
        caption_drop_prob: float = 0.0,
        caption_selection: str = 'first',
        transform: Optional[Callable] = None,
        image_size: Optional[int] = None,
        image_key: str = 'image',
        caption_key: str = 'caption',
        sdxl: Optional[bool] = False,
        cond_drop_prob: float = 0.0,
        crop_type: Optional[str] = 'random',
        **streaming_kwargs,
    ) -> None:

        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            **streaming_kwargs,
        )
        caption_selection = caption_selection.lower()
        if caption_selection not in ['first', 'random']:
            raise ValueError(f'Invalid caption selection: {caption_selection}. Must be one of [random, first]')

        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, subfolder='tokenizer')
        if sdxl:
            if tokenizer_name_or_path_2:
                self.tokenizer_2 = AutoTokenizer.from_pretrained(tokenizer_name_or_path_2, subfolder='tokenizer_2')
            else:
                raise ValueError('Must provide value for tokenizer_name_or_path_2')
            
        self.caption_drop_prob = caption_drop_prob
        self.caption_selection = caption_selection
        self.image_size = image_size
        self.image_key = image_key
        self.caption_key = caption_key

        self.sdxl = sdxl
        if crop_type == 'static':
            self.crop = LargestCenterSquareReturnTransform(self.image_size)
        elif crop_type == 'random':
            self.crop = RandomCropSquareReturnTransform(self.image_size)
        elif crop_type == 'aspect_ratio':
            self.crop = RandomCropAspectRatioTransorm()
        elif crop_type is None:
            self.crop = None
        self.cond_drop_prob = cond_drop_prob # sdxl

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        # Image
        img = sample[self.image_key]
        if not isinstance(img, Image.Image):
            img = Image.open(BytesIO(sample[self.image_key]))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.crop is not None:
            img, crop_top, crop_left, image_height, image_width = self.crop(img)

        if self.transform is not None:
            img = self.transform(img)

        # Caption
        if torch.rand(1) < self.caption_drop_prob:
            caption = ''
        else:
            caption = sample[self.caption_key]
            if isinstance(caption, List) and self.caption_selection == 'first':
                caption = caption[0]
            if isinstance(caption, List) and self.caption_selection == 'random':
                caption = random.sample(caption, k=1)[0]
        tokenized_caption = self.tokenizer(caption,
                                           padding='max_length',
                                           max_length=self.tokenizer.model_max_length,
                                           truncation=True,
                                           return_tensors='pt')['input_ids'][0]

        out = {'image': img, 'captions': tokenized_caption}

        # optional SDXL tokenizer_2
        if self.sdxl:
            tokenized_caption_2 = self.tokenizer_2(
                caption,
                padding='max_length',
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors='pt'
            )['input_ids'][0]
            out['captions_2'] = tokenized_caption_2

            # microconditioning - add crop and img size params
            if torch.rand(1) < self.cond_drop_prob: # microconditioning drop-out
                out['cond_crops_coords_top_left'] = torch.tensor([crop_top, crop_left]) * 0
                out['cond_original_size'] = torch.tensor([image_width, image_height]) * 0
                out['cond_target_size'] = torch.tensor([self.image_size, self.image_size]) * 0
            else:
                out['cond_crops_coords_top_left'] = torch.tensor([crop_top, crop_left])
                out['cond_original_size'] = torch.tensor([image_width, image_height])
                out['cond_target_size'] = torch.tensor([self.image_size, self.image_size])
            
        return out


def build_streaming_image_caption_dataloader(
    remote: Union[str, List],
    local: Union[str, List],
    batch_size: int,
    tokenizer_name_or_path: str = 'stabilityai/stable-diffusion-2-base',
    tokenizer_name_or_path_2: Optional[str] = None,
    caption_drop_prob: float = 0.0,
    resize_size: int = 256,
    caption_selection: str = 'first',
    transform: Optional[List[Callable]] = None,
    image_key: str = 'image',
    caption_key: str = 'caption',
    sdxl: Optional[bool] = False,
    cond_drop_prob: float = 0.0,
    crop_type: Optional[str] = 'random',
    streaming_kwargs: Optional[Dict] = None,
    dataloader_kwargs: Optional[Dict] = None,
):
    """Builds a streaming LAION dataloader.

    Args:
        remote (str, Sequence[str]): One or more remote directories (S3 or local filesystem) where dataset is stored.
        local (str, Sequence[str]): One or more local filesystem directories where dataset is cached during operation.
        batch_size (int): The batch size to use for both the ``StreamingDataset`` and ``DataLoader``.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        caption_drop_prob (float): The probability of dropping a caption. Default: ``0.0``.
        resize_size (int): The size to resize the image to. Default: ``256``.
        caption_selection (str): If there are multiple captions, specifies how to select a single caption.
            'first' selects the first caption in the list and 'random' selects a random caption in the list.
            If there is only one caption, this argument is ignored. Default: ``'first'``.
        transform (Optional[Callable]): The transforms to apply to the image. Default: ``None``.
        image_key (str): Key associated with the image in the streaming dataset. Default: ``'image'``.
        caption_key (str): Key associated with the caption in the streaming dataset. Default: ``'caption'``.
        streaming_kwargs (dict, optional): Additional arguments to pass to the ``StreamingDataset``. Default: ``None``.
        dataloader_kwargs (dict, optional): Additional arguments to pass to the ``DataLoader``. Default: ``None``.
    """
    # Handle ``None`` kwargs
    if streaming_kwargs is None:
        streaming_kwargs = {}
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    # Check types for remote and local
    if isinstance(remote, str) and isinstance(local, str):
        # Hacky... make remote and local lists to simplify downstream code
        remote, local = [remote], [local]
    elif isinstance(remote, Sequence) and isinstance(local, Sequence):
        if len(remote) != len(local):
            ValueError(
                f'remote and local Sequences must be the same length, got lengths {len(remote)} and {len(local)}')
    else:
        ValueError(f'remote and local must be both Strings or Sequences, got types {type(remote)} and {type(local)}.')

    # Create a Stream for each (remote, local) pair
    streams = []
    for r, l in zip(remote, local):
        streams.append(Stream(remote=r, local=l))

    # Setup the transforms to apply
    if transform is None:
        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    transform = transforms.Compose(transform)
    assert isinstance(transform, Callable)

    dataset = StreamingImageCaptionDataset(
        streams=streams,
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_name_or_path_2=tokenizer_name_or_path_2,
        caption_drop_prob=caption_drop_prob,
        caption_selection=caption_selection,
        transform=transform,
        image_size=resize_size,
        image_key=image_key,
        caption_key=caption_key,
        batch_size=batch_size,
        sdxl=sdxl,
        cond_drop_prob=cond_drop_prob,
        crop_type=crop_type,
        **streaming_kwargs,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        **dataloader_kwargs,
    )

    return dataloader
