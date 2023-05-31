import math
from typing import Tuple

import torch
import torch.nn as nn

# TODO: Always a 1x1 conv on the residual connection?
# TODO: Do the resnet blocks have attention? Depends on cond_dim
# TODO: Where did the scale shift method for incorporating time and text embeddings come from? Why is it applied at the second block?

class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim: int, max_period: int = 10_000):
        self.dim = dim
        self.max_period = max_period

    def forward(self, timestep: torch.Tensor):
        half_dim = self.dim // 2

        exponent = -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=timestep.dtype, device=timestep.device)
        exponent = exponent / (half_dim -1 )
        embed = torch.exp(exponent)

        embed = timestep[:, None] * embed[None, :]
        # Note: need to flip if we plan to load from SD weights
        embed = torch.cat([embed.sin(), embed.cos()], dim=-1)


class ResNetBlock(nn.Module):

    def __init__(self, num_channels: int, num_norm_groups: int = 32):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=num_norm_groups, num_channels=num_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=num_norm_groups, num_channels=num_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)

        # Convolution for shortcut
        self.conv_shortcut = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        shortcut = x

        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        shortcut = self.conv_shortcut(shortcut)

        return x + shortcut

class DownBlock(nn.Module):

    def __init__(self, is_downsample: bool, num_resnet_blocks: int)

class UNetTextConditional(nn.Module):

    def __init__(self,
        dims: Tuple[int] = (128, 256, 512, 1024),
        channels: int = 3, # Input and output channels
        pos_embed_dim: int = 16, # NOTE: This is a lot smaller than HF (I think they use the # of channels for the first output block)
        text_embed_dim: int = 768,
        cross_attention_dim: int = 1024,
        lowres_aug_cond: bool = False,
        cond_dim: int = 512,
    ):

        super().__init__()

        # When doing cascaded diffusion, concatenate the low resolution and noise, doubling the input channels
        self.in_channels = channels * (1 + int(lowres_aug_cond))
        self.out_channels = channels
        self.lowres_aug_cond = lowres_aug_cond

        # Setup time projection - NOTE: Lucidrains used Learned position embeddings based on crowsonkb
        time_embed_dim = cond_dim * 4 * (2 if lowres_aug_cond else 1)
        self.time_proj = nn.Sequential(
            SinusoidalPositionEmbeddings(dim=pos_embed_dim),
            nn.Linear(in_features=pos_embed_dim, out_features=time_embed_dim),
            nn.SiLU(),
            nn.Linear(in_features=time_embed_dim, out_features=time_embed_dim),
        )

        # TODO: what is to_time_tokens?

        # Embedding for the noise augumentation to the low resolutions image
        if lowres_aug_cond:
            self.lowres_time_proj = nn.Sequential(
                SinusoidalPositionEmbeddings(dim=pos_embed_dim),
                nn.Linear(in_features=pos_embed_dim, out_features=time_embed_dim),
                nn.SiLU(),
                nn.Linear(in_features=time_embed_dim, out_features=time_embed_dim),
            )

        # TODO: Uh, do we need this? -> applied to time_tokens
        # self.norm_cond = nn.LayerNorm(time_embed_dim)

        #TODO: Do we need this? -> Applied to text embedding for some reason?
        # self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)

        # TODO: do we need to_text_non_attn_cond? -> applied to non-attention pooled text? But it looks like attention pooled lol



        # Need some way to pool the text embeddings to add to the time embedding
        # HuggingFace uses attention pooling by adding an CLS token whereas lucidrains uses PerceiverResampler strategy
        # I don't think Imagen specifies this so unsure what to use
        # TODO: I'm just average pooling and projecting for now...
        self.proj_pool_text = nn.Sequential(
            nn.Linear(text_embed_dim, time_embed_dim),
            nn.Layer_norm(time_embed_dim),
        )

        # NOTE: Lucidrains did something fancy here
        self.init_conv = nn.Conv2d(channels, dims[0], kernel_size=3, padding=1)

    def forward(
        self,
        x,
        time,
        *,
        text_embeds = None
        lowres_cond_img = None,
        lowres_noise_time = None,
        text_mask = None,
        # TODO: what is self_cond?
        # TODO: what is cond_images?
        ):

        if lowres_cond_img is not None:
            x = torch.cat((x, lowres_cond_img), dim = 1)

        x = self.init_conv(x)

        # NOTE: ignored residual connection to final conv

        t_embed = self.time_proj(time)
        # TODO: to_time_tokens?

        if self.lowres_aug_cond:
            lowres_t_embed = self.lowres_time_proj(time)
            # TODO: lowres_time_tokens?

            t_embed = t_embed + lowres_t_embed

        if text_embeds is not None: # and self.cond_on_text
            # TODO: linear layer on text embeds? text_to_cond -> I guess needed if there is a difference in text embed dimension and the cross attention dim?
            text_tokens = text_embeds

            # Lucidrains does tuncation, padding, and dropping captions, but it is all handled in our dataloader
            # NOTE: He learns an sequence of embedding for null text?! null_text_embed
            # Replaces tokens specified by the mask with the learned embeddings using torch.where()

            # TODO: Pool tokens to another text_tokens? attn_pool -> what is this output dimension? How can he apply a mean afterwards?

            # Now a non-attention pooling with projection after the attention pooling (if it was present?!)
            mean_pooled_text_tokens = text_tokens.mean(dim=-2)
            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)
            t_embed = t_embed + text_hiddens

            # NOTE: Replace pooled embedding for the dropped captions with a learned embedding (null_text_hidden)






        time_embed = self.time_proj(time)
        text_pool = text_embeds.mean(dim=1) # TODO: should probably do something else...
        text_pool = self.proj_pool_text(text_pool)
