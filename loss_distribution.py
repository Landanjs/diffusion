from diffusion.datasets.image_caption import build_streaming_image_caption_dataloader
from diffusion.models.models import stable_diffusion_xl
from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_batches', type=int)
args = parser.parse_args()

remotes = 'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/filter_v2/256-512/4.5-5.0/1'
locals = '/tmp/4.5v2/filter_v2/256-512/4.5-5.0/1.3'

print('Creating dataloader')
dataloader = build_streaming_image_caption_dataloader(
    remote=remotes,
    local=locals,
    batch_size=args.batch_size,
    image_key='jpg',
    caption_key='caption',
    tokenizer_name_or_path='stabilityai/stable-diffusion-xl-base-1.0',
    streaming_kwargs={'shuffle': True, 'predownload': args.batch_size},
    dataloader_kwargs={'num_workers': 8},
)
print('Created Dataloader')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Creating model')
model = stable_diffusion_xl(
    fsdp=False,
    clip_qkv=None,
    loss_bins=[],
    val_guidance_scales=[],
    precomputed_latents=False,
    encode_latents_in_fp16=False,
    pretrained=args.pretrained
)
print('Created model')
model = model.to(device)

print('Loading batch')
count = 0
losses = {k: [] for k in range(1000)}
with torch.no_grad():
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        out = model(batch)
        loss = F.mse_loss(out[0], out[1], reduction='none').mean(dim=(1, 2, 3))
        for t, l in zip(out[-1], loss):
            losses[t.cpu().item()].append(l.cpu().item())
        if count == args.num_batches:
            break
        count += 1
losses = {k: (torch.tensor(v).mean().item(), torch.tensor(v).std().item()) for k, v in losses.items()}
print(losses)
exit()