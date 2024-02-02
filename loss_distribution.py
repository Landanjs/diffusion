from diffusion.datasets.image_caption import build_streaming_image_caption_dataloader
from diffusion.models.models import stable_diffusion_xl
from tqdm import tqdm
import torch

remotes = 'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/filter_v2/256-512/4.5-5.0/1'
locals = '/tmp/4.5v2/filter_v2/256-512/4.5-5.0/1.3'

print('Creating dataloader')
dataloader = build_streaming_image_caption_dataloader(
    remote=remotes,
    local=locals,
    batch_size=32,
    image_key='jpg',
    caption_key='caption',
    tokenizer_name_or_path='stabilityai/stable-diffusion-xl-base-1.0',
    streaming_kwargs={'shuffle': True, 'predownload': 32},
    dataloader_kwargs={'num_workers': 8},
)
print('Created Dataloader')

print('Creating model')
model = stable_diffusion_xl(fsdp=False, clip_qkv=None, loss_bins=[], val_guidance_scales=[], precomputed_latents=False, encode_latents_in_fp16=False)
print('Created model')
model = model.cuda()

print('Loading batch')
count = 0
losses = {k: [] for k in range(1000)}
with torch.no_grad():
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.cuda()
        out = model(batch)
        losses[out[-1].item()].append(model.loss(out, batch).item())
        if count == 100:
            break
        count += 1
losses = {k: (torch.tensor(v).mean().item(), torch.tensor(v).std().item()) for k, v in losses.items()}
print(losses)
exit()