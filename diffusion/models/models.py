# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Constructors for diffusion models."""

from typing import List, Optional

import torch
from composer.devices import DeviceGPU
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel, EulerDiscreteScheduler
from torchmetrics import MeanSquaredError
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig

from diffusion.models.pixel_diffusion import PixelDiffusion
from diffusion.models.stable_diffusion import StableDiffusion
from diffusion.schedulers.schedulers import ContinuousTimeScheduler

try:
    import xformers  # type: ignore
    del xformers
    is_xformers_installed = True
except:
    is_xformers_installed = False

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class SDXLTextEncoder(torch.nn.Module):

    def __init__(self, model_name, encode_latents_in_fp16):
        super().__init__()
        torch_dtype = torch.float16 if encode_latents_in_fp16 else None
        self.text_encoder1 = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch_dtype)
        self.text_encoder2 = CLIPTextModelWithProjection.from_pretrained(model_name,
                                                                        subfolder='text_encoder_2',
                                                                        torch_dtype=torch_dtype)

    def forward(self, text):
        conditioning1 = self.text_encoder1(text[0], output_hidden_states=True).hidden_states[-2]
        text_encoder2_out = self.text_encoder2(text[1], output_hidden_states=True)
        pooled_conditioning = text_encoder2_out[0]
        conditioning2 = text_encoder2_out.hidden_states[-2]
        conditioning = torch.concat([conditioning1, conditioning2], dim=-1)
        # conditioning = torch.randn(text[0].shape[0], 77, 2048, device=text[0].device)
        # pooled_conditioning = torch.randn(text[0].shape[0], 1280, device=text[0].device)
        return conditioning, pooled_conditioning

def stable_diffusion_xl(
    model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    unet_model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    vae_model_name: str = 'madebyollin/sdxl-vae-fp16-fix',
    pretrained: bool = True,
    prediction_type: str = 'epsilon',
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    val_guidance_scales: Optional[List] = None,
    val_seed: int = 1138,
    loss_bins: Optional[List] = None,
    ignore_img_encoder: bool = False,
    ignore_txt_encoder: bool = False,
    encode_latents_in_fp16: bool = True,
    fsdp: bool = True,
):
    """Stable diffusion v2 training setup.

    Requires batches of matched images and text prompts to train. Generates images from text
    prompts.

    Args:
        model_name (str, optional): Name of the model to load. Determines the text encoder used.
            Defaults to 'stabilityai/stable-diffusion-xl-base-1.0'.
        unet_model_name (str, optional): Name of the UNet model to load. Defaults to
            'stabilityai/stable-diffusion-xl-base-1.0'.
        vae_model_name (str, optional): Name of the VAE model to load. Defaults to
            'madebyollin/sdxl-vae-fp16-fix' (SDXL VAE).
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        train_metrics (list, optional): List of metrics to compute during training. If None, defaults to
            [MeanSquaredError()].
        val_metrics (list, optional): List of metrics to compute during validation. If None, defaults to
            [MeanSquaredError(), FrechetInceptionDistance(normalize=True)].
        val_guidance_scales (list, optional): List of scales to use for validation guidance. If None, defaults to
            [1.0, 3.0, 7.0].
        val_seed (int, optional): Seed to use for generating evaluation images. Defaults to 1138.
        loss_bins (list, optional): List of tuples of (min, max) values to use for loss binning. If None, defaults to
            [(0, 1)].
        precomputed_latents (bool, optional): Whether to use precomputed latents. Defaults to False.
        encode_latents_in_fp16 (bool, optional): Whether to encode latents in fp16. Defaults to True.
        fsdp (bool, optional): Whether to use FSDP. Defaults to True.
    """
    if train_metrics is None:
        train_metrics = [MeanSquaredError()]
    if val_metrics is None:
        val_metrics = [MeanSquaredError(), FrechetInceptionDistance(normalize=True)]
    if val_guidance_scales is None:
        val_guidance_scales = [1.0, 3.0, 7.0]
    if loss_bins is None:
        loss_bins = [(0, 1)]
    # Fix a bug where CLIPScore requires grad
    for metric in val_metrics:
        if isinstance(metric, CLIPScore):
            metric.requires_grad_(False)

    if pretrained:
        unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder='unet')

    else:
        config = PretrainedConfig.get_config_dict(unet_model_name, subfolder='unet')

        # note: local only
        # config[0]['block_out_channels'] = [32, 32, 1280]  # make smaller and more manageable for local debug
        config[0]["cross_attention_dim"] = 1024
        unet = UNet2DConditionModel(**config[0])

        # zero out some params at init
        # see https://discuss.pytorch.org/t/why-would-someone-zero-out-the-parameters-of-a-module-in-the-constructor/157148

        print('doing FixUp init!')
        
        # zero out final conv output
        unet.conv_out = zero_module(unet.conv_out)

        for name, layer in unet.named_modules(): 
            # zero out final conv in resnet blocks
            if name.endswith('conv2'):
                layer = zero_module(layer)
            # zero out proj_out in attention blocks
            if name.endswith('to_out.0'):
                layer = zero_module(layer)
            # print(name, layer)

        # # smaller SDXL-style unet for debugging
        # unet = UNet2DConditionModel(
        #     act_fn='silu',
        #     addition_embed_type='text_time',
        #     addition_embed_type_num_heads=64,
        #     addition_time_embed_dim=256,
        #     attention_head_dim=[5, 10, 20],
        #     block_out_channels=[32, 64, 1280],  # make smaller and more manageable for local debug,
        #     # block_out_channels=[320, 640, 1280],
        #     center_input_sample=False,
        #     class_embed_type=None,
        #     class_embeddings_concat=False,
        #     conv_in_kernel=3,
        #     conv_out_kernel=3,
        #     cross_attention_dim=2048,
        #     cross_attention_norm=None,
        #     down_block_types=['DownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D'],
        #     downsample_padding=1,
        #     dual_cross_attention=False,
        #     encoder_hid_dim=None,
        #     encoder_hid_dim_type=None,
        #     flip_sin_to_cos=True,
        #     freq_shift=0,
        #     in_channels=4,
        #     layers_per_block=2,
        #     mid_block_only_cross_attention=None,
        #     mid_block_scale_factor=1,
        #     mid_block_type='UNetMidBlock2DCrossAttn',
        #     norm_eps=1e-05,
        #     norm_num_groups=32,
        #     num_attention_heads=None,
        #     num_class_embeds=None,
        #     only_cross_attention=False,
        #     out_channels=4,
        #     projection_class_embeddings_input_dim=2816,  # assuming use of text_encoder_2
        #     resnet_out_scale_factor=1.0,
        #     resnet_skip_time_act=False,
        #     resnet_time_scale_shift='default',
        #     sample_size=128,
        #     time_cond_proj_dim=None,
        #     time_embedding_act_fn=None,
        #     time_embedding_dim=None,
        #     time_embedding_type='positional',
        #     timestep_post_act=None,
        #     transformer_layers_per_block=[1, 2, 10],
        #     up_block_types=['CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'UpBlock2D'],
        #     upcast_attention=None,
        #     use_linear_projection=True)

    # if fsdp:  # SDXL
    #     # Can't fsdp wrap up_blocks or down_blocks because the forward pass calls length on these
    #     unet.up_blocks._fsdp_wrap = False
    #     unet.down_blocks._fsdp_wrap = False
    #     for block in unet.up_blocks:
    #         block._fsdp_wrap = True
    #     for block in unet.down_blocks:
    #         block._fsdp_wrap = True
    #     unet.mid_block._fsdp_wrap = True

    if encode_latents_in_fp16:
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_name, subfolder='vae', torch_dtype=torch.float16)
        except:  # for handling SDXL vae fp16 fixed checkpoint
            vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=torch.float16)
        # import pdb;pdb.set_trace()
    else:
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_name, subfolder='vae')
        except: #  for handling SDXL vae fp16 fixed checkpoint
            vae = AutoencoderKL.from_pretrained(vae_model_name)
    text_encoder = SDXLTextEncoder(model_name=model_name, encode_latents_in_fp16=encode_latents_in_fp16)

    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer_2')
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')
    inference_noise_scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder='scheduler')

    model = StableDiffusion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        tokenizer_2=tokenizer_2,
        prediction_type=prediction_type,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_guidance_scales=val_guidance_scales,
        val_seed=val_seed,
        loss_bins=loss_bins,
        ignore_img_encoder=ignore_img_encoder,
        ignore_txt_encoder=ignore_txt_encoder,
        encode_latents_in_fp16=encode_latents_in_fp16,
        fsdp=fsdp,
        sdxl=True,
    )
    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.unet.enable_xformers_memory_efficient_attention()
            model.vae.enable_xformers_memory_efficient_attention()
        # print('RE-ENABLE XFORMERS')
    return model


def stable_diffusion_2(
    model_name: str = 'stabilityai/stable-diffusion-2-base',
    unet_model_name: str = 'stabilityai/stable-diffusion-2-base',
    vae_model_name: str = 'stabilityai/stable-diffusion-2-base',
    pretrained: bool = True,
    prediction_type: str = 'epsilon',
    train_metrics: Optional[List] = None,
    val_metrics: Optional[List] = None,
    val_guidance_scales: Optional[List] = None,
    val_seed: int = 1138,
    loss_bins: Optional[List] = None,
    precomputed_latents: bool = False,
    encode_latents_in_fp16: bool = True,
    fsdp: bool = True,
):
    """Stable diffusion v2 training setup.

    Requires batches of matched images and text prompts to train. Generates images from text
    prompts.

    Args:
        model_name (str, optional): Name of the model to load. Determines the text encoder and autoencder.
            Defaults to 'stabilityai/stable-diffusion-2-base'.
        unet_model_name (str, optional): Name of the UNet model to load. Defaults to
            'stabilityai/stable-diffusion-2-base'.
        vae_model_name (str, optional): Name of the VAE model to load. Defaults to
            'stabilityai/stable-diffusion-2-base'.
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        train_metrics (list, optional): List of metrics to compute during training. If None, defaults to
            [MeanSquaredError()].
        val_metrics (list, optional): List of metrics to compute during validation. If None, defaults to
            [MeanSquaredError(), FrechetInceptionDistance(normalize=True)].
        val_guidance_scales (list, optional): List of scales to use for validation guidance. If None, defaults to
            [1.0, 3.0, 7.0].
        val_seed (int, optional): Seed to use for generating evaluation images. Defaults to 1138.
        loss_bins (list, optional): List of tuples of (min, max) values to use for loss binning. If None, defaults to
            [(0, 1)].
        precomputed_latents (bool, optional): Whether to use precomputed latents. Defaults to False.
        encode_latents_in_fp16 (bool, optional): Whether to encode latents in fp16. Defaults to True.
        fsdp (bool, optional): Whether to use FSDP. Defaults to True.
    """
    if train_metrics is None:
        train_metrics = [MeanSquaredError()]
    if val_metrics is None:
        val_metrics = [MeanSquaredError(), FrechetInceptionDistance(normalize=True)]
    if val_guidance_scales is None:
        val_guidance_scales = [1.0, 3.0, 7.0]
    if loss_bins is None:
        loss_bins = [(0, 1)]
    # Fix a bug where CLIPScore requires grad
    for metric in val_metrics:
        if isinstance(metric, CLIPScore):
            metric.requires_grad_(False)

    if pretrained:
        unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder='unet')
    else:
        config = PretrainedConfig.get_config_dict(unet_model_name, subfolder='unet')

        if unet_model_name == 'stabilityai/stable-diffusion-xl-refiner-1.0' or unet_model_name == 'stabilityai/stable-diffusion-xl-base-1.0':  # SDXL
            print('using SDXL unet!')
            config[0]['addition_embed_type'] = None
            config[0]['cross_attention_dim'] = 1024

        unet = UNet2DConditionModel(**config[0])

    if unet_model_name == 'stabilityai/stable-diffusion-xl-refiner-1.0' or unet_model_name == 'stabilityai/stable-diffusion-xl-base-1.0':  # SDXL
        # Can't fsdp wrap up_blocks or down_blocks because the forward pass calls length on these
        unet.up_blocks._fsdp_wrap = False
        unet.down_blocks._fsdp_wrap = False
        for block in unet.up_blocks:
            block._fsdp_wrap = True
        for block in unet.down_blocks:
            block._fsdp_wrap = True

    if encode_latents_in_fp16:
        try:
            vae = AutoencoderKL.from_pretrained(vae_model_name, subfolder='vae', torch_dtype=torch.float16)
        except:  # for handling SDXL vae fp16 fixed checkpoint
            vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=torch.float16)
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder', torch_dtype=torch.float16)
    else:
        vae = AutoencoderKL.from_pretrained(vae_model_name, subfolder='vae')
        text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder')

    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')
    inference_noise_scheduler = DDIMScheduler(num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                                              beta_start=noise_scheduler.config.beta_start,
                                              beta_end=noise_scheduler.config.beta_end,
                                              beta_schedule=noise_scheduler.config.beta_schedule,
                                              trained_betas=noise_scheduler.config.trained_betas,
                                              clip_sample=noise_scheduler.config.clip_sample,
                                              set_alpha_to_one=noise_scheduler.config.set_alpha_to_one,
                                              prediction_type=prediction_type)

    model = StableDiffusion(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        inference_noise_scheduler=inference_noise_scheduler,
        prediction_type=prediction_type,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_guidance_scales=val_guidance_scales,
        val_seed=val_seed,
        loss_bins=loss_bins,
        encode_latents_in_fp16=encode_latents_in_fp16,
        fsdp=fsdp,
    )
    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.unet.enable_xformers_memory_efficient_attention()
            model.vae.enable_xformers_memory_efficient_attention()
    return model


def discrete_pixel_diffusion(clip_model_name: str = 'openai/clip-vit-large-patch14', prediction_type='epsilon'):
    """Discrete pixel diffusion training setup.

    Args:
        clip_model_name (str, optional): Name of the clip model to load. Defaults to 'openai/clip-vit-large-patch14'.
        prediction_type (str, optional): Type of prediction to use. One of 'sample', 'epsilon', 'v_prediction'.
            Defaults to 'epsilon'.
    """
    # Create a pixel space unet
    unet = UNet2DConditionModel(in_channels=3,
                                out_channels=3,
                                attention_head_dim=[5, 10, 20, 20],
                                cross_attention_dim=768,
                                flip_sin_to_cos=True,
                                use_linear_projection=True)
    # Get the CLIP text encoder and tokenizer:
    text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    # Hard code the sheduler config
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,
                                    beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule='scaled_linear',
                                    trained_betas=None,
                                    variance_type='fixed_small',
                                    clip_sample=False,
                                    prediction_type=prediction_type,
                                    thresholding=False,
                                    dynamic_thresholding_ratio=0.995,
                                    clip_sample_range=1.0,
                                    sample_max_value=1.0)
    inference_scheduler = DDIMScheduler(num_train_timesteps=1000,
                                        beta_start=0.00085,
                                        beta_end=0.012,
                                        beta_schedule='scaled_linear',
                                        trained_betas=None,
                                        clip_sample=False,
                                        set_alpha_to_one=False,
                                        steps_offset=1,
                                        prediction_type=prediction_type,
                                        thresholding=False,
                                        dynamic_thresholding_ratio=0.995,
                                        clip_sample_range=1.0,
                                        sample_max_value=1.0)

    # Create the pixel space diffusion model
    model = PixelDiffusion(unet,
                           text_encoder,
                           tokenizer,
                           noise_scheduler,
                           inference_scheduler=inference_scheduler,
                           prediction_type=prediction_type,
                           train_metrics=[MeanSquaredError()],
                           val_metrics=[MeanSquaredError()])

    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.model.enable_xformers_memory_efficient_attention()
    return model


def continuous_pixel_diffusion(clip_model_name: str = 'openai/clip-vit-large-patch14',
                               prediction_type='epsilon',
                               use_ode=False,
                               train_t_max=1.570795,
                               inference_t_max=1.56):
    """Continuous pixel diffusion training setup.

    Uses the same clip and unet config as `discrete_pixel_diffusion`, but operates in continous time as in the VP
    process in https://arxiv.org/abs/2011.13456.

    Args:
        clip_model_name (str, optional): Name of the clip model to load. Defaults to 'openai/clip-vit-large-patch14'.
        prediction_type (str, optional): Type of prediction to use. One of 'sample', 'epsilon', 'v_prediction'.
            Defaults to 'epsilon'.
        use_ode (bool, optional): Whether to do generation using the probability flow ODE. If not used, uses the
            reverse diffusion process. Defaults to False.
        train_t_max (float, optional): Maximum timestep during training. Defaults to 1.570795 (pi/2).
        inference_t_max (float, optional): Maximum timestep during inference.
            Defaults to 1.56 (pi/2 - 0.01 for stability).
    """
    # Create a pixel space unet
    unet = UNet2DConditionModel(in_channels=3,
                                out_channels=3,
                                attention_head_dim=[5, 10, 20, 20],
                                cross_attention_dim=768,
                                flip_sin_to_cos=True,
                                use_linear_projection=True)
    # Get the CLIP text encoder and tokenizer:
    text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    # Need to use the continuous time schedulers for training and inference.
    noise_scheduler = ContinuousTimeScheduler(t_max=train_t_max, prediction_type=prediction_type)
    inference_scheduler = ContinuousTimeScheduler(t_max=inference_t_max,
                                                  prediction_type=prediction_type,
                                                  use_ode=use_ode)

    # Create the pixel space diffusion model
    model = PixelDiffusion(unet,
                           text_encoder,
                           tokenizer,
                           noise_scheduler,
                           inference_scheduler=inference_scheduler,
                           prediction_type=prediction_type,
                           continuous_time=True,
                           train_metrics=[MeanSquaredError()],
                           val_metrics=[MeanSquaredError()])

    if torch.cuda.is_available():
        model = DeviceGPU().module_to_device(model)
        if is_xformers_installed:
            model.model.enable_xformers_memory_efficient_attention()
    return model
