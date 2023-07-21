# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Tag LAION with latents."""

import os
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Sequence, Union

from streaming import MDSWriter, Stream, StreamingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def build_streaming_laion_dataloader(
    remote: Union[str, List],
    local: Union[str, List],
    batch_size: int,
    predownload: int = 100_000,
    download_retry: int = 2,
    download_timeout: float = 120,
    drop_last: bool = True,
    shuffle: bool = True,
    **dataloader_kwargs,
):
    """Builds a streaming LAION dataloader returning multiple image sizes.

    Args:
        remote (str, Sequence[str]): One or more remote directories (S3 or local filesystem) where dataset is stored.
        local (str, Sequence[str]): One or more local filesystem directories where dataset is cached during operation.
        batch_size (int): The batch size to use.
        tokenizer_name_or_path (str): The name or path of the tokenizer to use. Default: ``'stabilityai/stable-diffusion-2-base'``.
        caption_drop_prob (float): The probability of dropping a caption. Default: ``0.0``.
        resize_size (List[int]): The size or list of sizes to resize the image to. If None, defaults to ``[256, 512]``.
        num_samples (Optional[int]): The number of samples to use. Default: ``None`` uses all available samples.
        predownload (Optional[int]): The number of samples to prefetch. Default: ``100_000``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        drop_last (bool): Whether to drop the last batch if it is incomplete. Default: ``True``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``True``.
        **dataloader_kwargs: Additional arguments to pass to the dataloader.
    """
    if isinstance(remote, Sequence) or isinstance(local, Sequence):
        assert isinstance(remote, Sequence) and isinstance(
            local, Sequence), 'If either remote or local is a sequence, both must be sequences'
        assert len(remote) == len(
            local), f'remote and local must be lists of the same length, got lengths {len(remote)} and {len(local)}'
    else:
        # Hacky... make remote and local lists to simplify downstream code
        remote, local = [
            remote,
        ], [
            local,
        ]

    # Create a Stream for each (remote, local) pair
    streams = []
    for r, l in zip(remote, local):
        streams.append(Stream(remote=r, local=l, download_retry=download_retry, download_timeout=download_timeout))

    dataset = StreamingDataset(
        streams=streams,
        split=None,
        shuffle=shuffle,
        predownload=predownload,
        keep_zip=False,
        download_retry=download_retry,
        download_timeout=download_timeout,
        validate_hash=None,
        cache_limit=3_000_000_000_000,
        batch_size=batch_size,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--num_subsets', type=int, help='Number of subsets to create')
    args.add_argument('--num_samples_per_subset', type=int, help='Number of samples in each subsets')
    args.add_argument('--remote_upload', type=str, default='', help='Remote path to upload MDS-formatted shards to.')
    args.add_argument('--batch-size', type=int, default=64, help='Batch size to use for encoding.')
    return args.parse_args()


def main(args: Namespace) -> None:
    """Add latents to LAION dataset.

    Args:
        args (Namespace): Command-line arguments.
    """
    remote = [
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/256-512/1',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/256-512/2',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/256-512/3',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/256-512/4',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/512-768/1',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/512-768/2',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/512-768/3',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/512-768/4',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/768-1024/1',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/768-1024/2',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/768-1024/3',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/768-1024/4',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/1024-1048576/1',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/1024-1048576/2',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/1024-1048576/3',
        'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/1024-1048576/4',
    ]
    local = [
        '/tmp/mds-cache/laion/256-512/1',
        '/tmp/mds-cache/laion/256-512/2',
        '/tmp/mds-cache/laion/256-512/3',
        '/tmp/mds-cache/laion/256-512/4',
        '/tmp/mds-cache/laion/512-768/1',
        '/tmp/mds-cache/laion/512-768/2',
        '/tmp/mds-cache/laion/512-768/3',
        '/tmp/mds-cache/laion/512-768/4',
        '/tmp/mds-cache/laion/768-1024/1',
        '/tmp/mds-cache/laion/768-1024/2',
        '/tmp/mds-cache/laion/768-1024/3',
        '/tmp/mds-cache/laion/768-1024/4',
        '/tmp/mds-cache/laion/1024-1048576/1',
        '/tmp/mds-cache/laion/1024-1048576/2',
        '/tmp/mds-cache/laion/1024-1048576/3',
        '/tmp/mds-cache/laion/1024-1048576/4',
    ]

    dataloader = build_streaming_laion_dataloader(
        remote=remote,
        local=local,
        batch_size=args.batch_size,
        predownload=20_000,
        drop_last=False,
        shuffle=True,
        prefetch_factor=2,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        download_timeout=300,
    )

    columns = {
        'punsafe': 'float64',
        'pwatermark': 'float64',
        'similarity': 'float64',
        'caption': 'str',
        'url': 'str',
        'key': 'str',
        'status': 'str',
        'error_message': 'str',
        'width': 'int32',
        'height': 'int32',
        'original_width': 'int32',
        'original_height': 'int32',
        'exif': 'str',
        'jpg': 'bytes',
        'hash': 'int64',
        'aesthetic_score': 'float64',
    }

    resolutions = ['256-512', '512-768', '768-1024', '1024-1048576']
    writers = []
    for i in range(args.num_subsets):
        writers.append({})
        for r in resolutions:
            name = os.path.join(args.remote_upload, str(i), r)
            print(name)
            writers[i][r] = MDSWriter(out=name,
                                      columns=columns,
                                      compression=None,
                                      hash=[],
                                      size_limit=256 * (2**20),
                                      max_workers=64)

    count = 0
    curr_subset = 0
    for batch in tqdm(dataloader):
        sample = batch
        for i in range(len(sample['jpg'])):
            if count % 1_000 == 0:
                print(count)
            curr_subset = count // args.num_samples_per_subset
            if curr_subset == args.num_subsets:
                break
            mds_sample = {
                'punsafe': sample['punsafe'][i],
                'pwatermark': sample['pwatermark'][i],
                'similarity': sample['similarity'][i],
                'caption': sample['caption'][i],
                'url': sample['url'][i],
                'key': sample['key'][i],
                'status': sample['status'][i],
                'error_message': sample['error_message'][i],
                'width': sample['width'][i],
                'height': sample['height'][i],
                'original_width': sample['original_width'][i],
                'original_height': sample['original_height'][i],
                'exif': sample['exif'][i],
                'jpg': sample['jpg'][i],
                'hash': sample['hash'][i],
                'aesthetic_score': sample['aesthetic_score'][i],
            }
            if 256 <= min(mds_sample['width'], mds_sample['height']) < 512:
                res = '256-512'
            elif 512 <= min(mds_sample['width'], mds_sample['height']) < 768:
                res = '512-768'
            elif 768 <= min(mds_sample['width'], mds_sample['height']) < 1024:
                res = '768-1024'
            elif 1024 <= min(mds_sample['width'], mds_sample['height']) < 1048576:
                res = '1024-1048576'
            else:
                raise ValueError(f'This sample is too large! {count}')

            writers[curr_subset][res].write(mds_sample)
            count += 1
            if count % args.num_samples_per_subset == 0:
                for r in resolutions:
                    writers[curr_subset][r].finish()


if __name__ == '__main__':
    main(parse_args())
