from composer.utils.file_helpers import get_file
from diffusion.datasets.image_caption import build_streaming_image_caption_dataloader
from diffusion.models.models import stable_diffusion_xl
from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse

LOCAL_CHECKPOINT_PATH = '/tmp/model.pt'

parser = argparse.ArgumentParser()
parser.add_argument('--remote_base', type=str)
parser.add_argument('--chkpt_path', type=str)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_batches', type=int)
args = parser.parse_args()

#remotes = 'oci://mosaicml-internal-dataset-laion2b-en/4.5v2/filter_v2/256-512/4.5-5.0/1'
#locals = '/tmp/4.5v2/filter_v2/256-512/4.5-5.0/1.3'
remotes = []
locals = []
remote_base = args.remote_base
local_base = '/tmp/4.5v2/filter_v2'
resolutions = [256, 512, 768, 1024, 1048576]
aesthetics = ['4.5', '5.0', '5.5', '6.0', '6.25', '6.5', '6.75', '7.0', '100']
for low_res, high_res in zip(resolutions[:-1], resolutions[1:]):
    for low_aes, high_aes in zip(aesthetics[:-1], aesthetics[1:]):
        for subdir in range(1, 5):
            remotes.append(f'{remote_base}/{low_res}-{high_res}/{low_aes}-{high_aes}/{subdir}')
            locals.append(f'{local_base}/{low_res}-{high_res}/{low_aes}-{high_aes}/{subdir}')


print('Creating dataloader')
dataloader = build_streaming_image_caption_dataloader(
    remote=remotes,
    local=locals,
    batch_size=args.batch_size,
    image_key='jpg',
    caption_key='caption',
    tokenizer_name_or_path='stabilityai/stable-diffusion-xl-base-1.0',
    streaming_kwargs={'shuffle': True, 'predownload': args.batch_size},
    dataloader_kwargs={'num_workers': 32},
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

# Load checkpoint
print('Loading Checkpoint')
if not args.pretrained:
    get_file(path=args.chkpt_path, destination=LOCAL_CHECKPOINT_PATH)
    state_dict = torch.load(LOCAL_CHECKPOINT_PATH)
    for key in list(state_dict['state']['model'].keys()):
        if 'val_metrics.' in key:
            del state_dict['state']['model'][key]
    model.load_state_dict(state_dict['state']['model'], strict=False)
print('Loaded Checkpoint')

print('Created model')
model.to(device)
model = model.eval()

print('Loading batch')
count = 0
losses = [[] for _ in range(1000)]
with torch.cuda.amp.autocast(True):
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
losses = [(torch.tensor(loss).mean().item(), torch.tensor(loss).std().item()) for loss in losses]
print(losses)
exit()
