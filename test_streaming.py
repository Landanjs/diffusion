import torch
import yaml
from tqdm import tqdm
from diffusion.datasets import build_streaming_image_caption_dataloader

torch.distributed.init_process_group()

with open('/mnt/config/parameters.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config)
dataloader = build_streaming_image_caption_dataloader(remote=config['remotes'],
                                                      batch_size=14338//8,
                                                      sdxl_conditioning=True,
                                                      microcond_drop_prob=0.1,
                                                      zero_dropped_captions=True,
                                                      caption_drop_prob=0.1,
                                                      crop_type='random',
                                                      image_key='IMAGE',
                                                      caption_key='DESCRIPTION',
                                                      resize_size=256,
                                                      streaming_kwargs={'download_timeout': 300,
                                                                        'num_canonical_nodes': 56,
                                                                        'shuffle': True},
                                                      dataloader_kwargs={'drop_last': True,
                                                                         'prefetch_factor': 2,
                                                                         'num_workers': 8,
                                                                         'persistent_workers': True,
                                                                         'pin_memory': True})

for batch in tqdm(dataloader):
    pass