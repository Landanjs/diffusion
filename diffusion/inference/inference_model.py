# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Inference endpoint for Stable Diffusion."""

import base64
import io
from typing import Any, Dict, List, Optional

import torch
from composer.utils.file_helpers import get_file
from PIL import Image
from tqdm import tqdm

from diffusion.models import stable_diffusion_2, stable_diffusion_xl

# Local checkpoint params
LOCAL_CHECKPOINT_PATH = '/tmp/model.pt'
LOCAL_CHECKPOINT_PATH2 = 'tmp/model2.pt'


def download_checkpoint(chkpt_path: str, chkpt_path2: str):
    """Downloads the Stable Diffusion checkpoint to the local filesystem.

    Args:
        chkpt_path (str): The path to the local folder, URL or object score that contains the checkpoint.
    """
    get_file(path=chkpt_path, destination=LOCAL_CHECKPOINT_PATH)
    get_file(path=chkpt_path2, destination=LOCAL_CHECKPOINT_PATH2)


class StableDiffusionInference():
    """Inference endpoint class for Stable Diffusion 2.

    Args:
        model_name (str, optional): Name of the model to load. Default: 'stabilityai/stable-diffusion-2-base'.
        pretrained (bool): Whether to load pretrained weights. Default: True.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        local_checkpoint_path (str): Path to the local checkpoint. Default: '/tmp/model.pt'.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    def __init__(self,
                 model_name: str = 'stabilityai/stable-diffusion-2-base',
                 pretrained: bool = False,
                 prediction_type: str = 'epsilon',
                 local_checkpoint_path: str = LOCAL_CHECKPOINT_PATH,
                 **kwargs):
        self.device = torch.cuda.current_device()

        model = stable_diffusion_2(
            model_name=model_name,
            pretrained=pretrained,
            prediction_type=prediction_type,
            encode_latents_in_fp16=True,
            fsdp=False,
            **kwargs,
        )

        if not pretrained:
            state_dict = torch.load(local_checkpoint_path)
            for key in list(state_dict['state']['model'].keys()):
                if 'val_metrics.' in key:
                    del state_dict['state']['model'][key]
            model.load_state_dict(state_dict['state']['model'], strict=False)
        model.to(self.device)
        self.model = model.eval()

    def predict(self, model_requests: List[Dict[str, Any]]):
        prompts = []
        negative_prompts = []
        generate_kwargs = {}

        # assumes the same generate_kwargs across all samples
        for req in model_requests:
            if 'input' not in req:
                raise RuntimeError('"input" must be provided to generate call')
            inputs = req['input']

            # Prompts and negative prompts if available
            if isinstance(inputs, str):
                prompts.append(inputs)
            elif isinstance(inputs, Dict):
                if 'prompt' not in inputs:
                    raise RuntimeError('"prompt" must be provided to generate call if using a dict as input')
                prompts.append(inputs['prompt'])
                if 'negative_prompt' in inputs:
                    negative_prompts.append(inputs['negative_prompt'])
            else:
                raise RuntimeError(f'Input must be of type string or dict, but it is type: {type(inputs)}')

            generate_kwargs = req['parameters']

        # Check for prompts
        if len(prompts) == 0:
            raise RuntimeError('No prompts provided, must be either a string or dictionary with "prompt"')

        # Check negative prompt length
        if len(negative_prompts) == 0:
            negative_prompts = None
        elif len(prompts) != len(negative_prompts):
            raise RuntimeError('There must be the same number of negative prompts as prompts.')

        # Generate images
        with torch.cuda.amp.autocast(True):
            imgs = self.model.generate(prompt=prompts, negative_prompt=negative_prompts, **generate_kwargs).cpu()

        # Send as bytes
        png_images = []
        for i in range(imgs.shape[0]):
            img = (imgs[i].permute(1, 2, 0).numpy() * 255).round().astype('uint8')
            pil_image = Image.fromarray(img, 'RGB')
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            base64_encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            png_images.append(base64_encoded_image)
        return png_images


class StableDiffusionXLInference():
    """Inference endpoint class for Stable Diffusion XL.

    Args:
        model_name (str): Name of the model to load. Default: 'stabilityai/stable-diffusion-xl-base-1.0'.
        unet_model_name (str): Name of the UNet model to load. Default: 'stabilityai/stable-diffusion-xl-base-1.0'.
        vae_model_name (str): Name of the VAE model to load. Defaults to
            'madebyollin/sdxl-vae-fp16-fix' as the official VAE checkpoint (from
            'stabilityai/stable-diffusion-xl-base-1.0') is not compatible with fp16.
        clip_qkv (float, optional): If not None, clip the qkv values to this value. Defaults to 6.0. Improves stability
            of training.
        pretrained (bool): Whether to load pretrained weights. Default: True.
        prediction_type (str): The type of prediction to use. Must be one of 'sample',
            'epsilon', or 'v_prediction'. Default: `epsilon`.
        **kwargs: Additional keyword arguments to pass to the model.
    """

    def __init__(self,
                 model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
                 unet_model_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
                 vae_model_name: str = 'madebyollin/sdxl-vae-fp16-fix',
                 clip_qkv: Optional[float] = None,
                 pretrained: bool = False,
                 prediction_type: str = 'epsilon',
                 local_checkpoint_path: str = LOCAL_CHECKPOINT_PATH,
                 local_checkpoint_path2: str = LOCAL_CHECKPOINT_PATH2,
                 **kwargs):
        self.device = torch.cuda.current_device()


        # Model 1
        model = stable_diffusion_xl(
            model_name=model_name,
            unet_model_name=unet_model_name,
            vae_model_name=vae_model_name,
            clip_qkv=clip_qkv,
            pretrained=pretrained,
            prediction_type=prediction_type,
            encode_latents_in_fp16=True,
            fsdp=False,
            **kwargs,
        )

        if not pretrained:
            state_dict = torch.load(local_checkpoint_path)
            for key in list(state_dict['state']['model'].keys()):
                if 'val_metrics.' in key:
                    del state_dict['state']['model'][key]
            model.load_state_dict(state_dict['state']['model'], strict=False)
        model.to(self.device)
        self.model = model.eval()

        # Model 2
        model2 = stable_diffusion_xl(
            model_name=model_name,
            unet_model_name=unet_model_name,
            vae_model_name=vae_model_name,
            clip_qkv=clip_qkv,
            pretrained=pretrained,
            prediction_type=prediction_type,
            encode_latents_in_fp16=True,
            fsdp=False,
            **kwargs,
        )

        if not pretrained:
            state_dict = torch.load(local_checkpoint_path2)
            for key in list(state_dict['state']['model'].keys()):
                if 'val_metrics.' in key:
                    del state_dict['state']['model'][key]
        model2.to(self.device)
        self.model2 = model2.eval()



    def predict(self, model_requests: List[Dict[str, Any]]):
        prompts = []
        negative_prompts = []
        generate_kwargs = {}

        # assumes the same generate_kwargs across all samples
        for req in model_requests:
            if 'input' not in req:
                raise RuntimeError('"input" must be provided to generate call')
            inputs = req['input']

            # Prompts and negative prompts if available
            if isinstance(inputs, str):
                prompts.append(inputs)
            elif isinstance(inputs, Dict):
                if 'prompt' not in inputs:
                    raise RuntimeError('"prompt" must be provided to generate call if using a dict as input')
                prompts.append(inputs['prompt'])
                if 'negative_prompt' in inputs:
                    negative_prompts.append(inputs['negative_prompt'])
            else:
                raise RuntimeError(f'Input must be of type string or dict, but it is type: {type(inputs)}')

            generate_kwargs = req['parameters']

        # Check for prompts
        if len(prompts) == 0:
            raise RuntimeError('No prompts provided, must be either a string or dictionary with "prompt"')

        # Check negative prompt length
        if len(negative_prompts) == 0:
            negative_prompts = None
        elif len(prompts) != len(negative_prompts):
            raise RuntimeError('There must be the same number of negative prompts as prompts.')

        # Generate images
        with torch.cuda.amp.autocast(True):
            imgs = generate(self.model, self.model2, prompt=prompts, negative_prompt=negative_prompts, **generate_kwargs).cpu()

        # Send as bytes
        png_images = []
        for i in range(imgs.shape[0]):
            img = (imgs[i].permute(1, 2, 0).numpy() * 255).round().astype('uint8')
            pil_image = Image.fromarray(img, 'RGB')
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            base64_encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            png_images.append(base64_encoded_image)
        return png_images

def generate(
    model,
    model2,
    prompt: Optional[list] = None,
    negative_prompt: Optional[list] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 3.0,
    rescaled_guidance: Optional[float] = None,
    num_images_per_prompt: Optional[int] = 1,
    seed: Optional[int] = None,
    progress_bar: Optional[bool] = True,
    zero_out_negative_prompt: bool = True,
    crop_params: Optional[torch.Tensor] = None,
    input_size_params: Optional[torch.Tensor] = None,
):
    """Generates image from noise.

    Performs the backward diffusion process, each inference step takes
    one forward pass through the unet.

    Args:
        prompt (str or List[str]): The prompt or prompts to guide the image generation.
        negative_prompt (str or List[str]): The prompt or prompts to guide the
            image generation away from. Ignored when not using guidance
            (i.e., ignored if guidance_scale is less than 1).
            Must be the same length as list of prompts. Default: `None`.
        height (int, optional): The height in pixels of the generated image.
            Default: `self.unet.config.sample_size * 8)`.
        width (int, optional): The width in pixels of the generated image.
            Default: `self.unet.config.sample_size * 8)`.
        num_inference_steps (int): The number of denoising steps.
            More denoising steps usually lead to a higher quality image at the expense
            of slower inference. Default: `50`.
        guidance_scale (float): Guidance scale as defined in
            Classifier-Free Diffusion Guidance. guidance_scale is defined as w of equation
            2. of Imagen Paper. Guidance scale is enabled by setting guidance_scale > 1.
            Higher guidance scale encourages to generate images that are closely linked
            to the text prompt, usually at the expense of lower image quality.
            Default: `3.0`.
        rescaled_guidance (float, optional): Rescaled guidance scale. If not specified, rescaled guidance will
            not be used. Default: `None`.
        num_images_per_prompt (int): The number of images to generate per prompt.
                Default: `1`.
        progress_bar (bool): Whether to use the tqdm progress bar during generation.
            Default: `True`.
        seed (int): Random seed to use for generation. Set a seed for reproducible generation.
            Default: `None`.
        zero_out_negative_prompt (bool): Whether or not to zero out negative prompt if it is
            an empty string. Default: `True`.
        crop_params (torch.FloatTensor of size [Bx2], optional): Crop parameters to use
            when generating images with SDXL. Default: `None`.
        input_size_params (torch.FloatTensor of size [Bx2], optional): Size parameters
            (representing original size of input image) to use when generating images with SDXL.
            Default: `None`.
    """
    _check_prompt_lengths(prompt, negative_prompt)

    # Create rng for the generation
    device = model.vae.device
    rng_generator = torch.Generator(device=device)
    if seed:
        rng_generator = rng_generator.manual_seed(seed)  # type: ignore

    height = height or model.unet.config.sample_size * model.downsample_factor
    width = width or model.unet.config.sample_size * model.downsample_factor
    assert height is not None  # for type checking
    assert width is not None  # for type checking

    do_classifier_free_guidance = guidance_scale > 1.0  # type: ignore

    text_embeddings, pooled_text_embeddings, pad_attn_mask = model._prepare_text_embeddings(
        prompt, None, None, None, num_images_per_prompt)
    batch_size = len(text_embeddings)  # len prompts * num_images_per_prompt
    # classifier free guidance + negative prompts
    # negative prompt is given in place of the unconditional input in classifier free guidance
    pooled_embeddings, encoder_attn_mask = pooled_text_embeddings, pad_attn_mask
    if do_classifier_free_guidance:
        if not negative_prompt and zero_out_negative_prompt:
            # Negative prompt is empty and we want to zero it out
            unconditional_embeddings = torch.zeros_like(text_embeddings)
            pooled_unconditional_embeddings = torch.zeros_like(pooled_text_embeddings) if model.sdxl else None
            uncond_pad_attn_mask = torch.zeros_like(pad_attn_mask) if pad_attn_mask is not None else None
        else:
            if not negative_prompt:
                negative_prompt = [''] * (batch_size // num_images_per_prompt)  # type: ignore
            unconditional_embeddings, pooled_unconditional_embeddings, uncond_pad_attn_mask = _prepare_text_embeddings(
                negative_prompt, None, None, None, num_images_per_prompt)

        # concat uncond + prompt
        text_embeddings = torch.cat([unconditional_embeddings, text_embeddings])
        if model.sdxl:
            pooled_embeddings = torch.cat([pooled_unconditional_embeddings, pooled_text_embeddings])  # type: ignore
        if pad_attn_mask is not None:
            encoder_attn_mask = torch.cat([uncond_pad_attn_mask, pad_attn_mask])  # type: ignore
    else:
        if model.sdxl:
            pooled_embeddings = pooled_text_embeddings

    # prepare for diffusion generation process
    latents = torch.randn(
        (batch_size, model.unet.config.in_channels, height // model.downsample_factor,
            width // model.downsample_factor),
        device=device,
        generator=rng_generator,
    )

    model.inference_scheduler.set_timesteps(num_inference_steps)
    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * model.inference_scheduler.init_noise_sigma

    added_cond_kwargs = {}
    # if using SDXL, prepare added time ids & embeddings
    if model.sdxl and pooled_embeddings is not None:
        if crop_params is None:
            crop_params = torch.zeros((batch_size, 2), dtype=text_embeddings.dtype)
        if input_size_params is None:
            input_size_params = torch.tensor([width, height], dtype=text_embeddings.dtype).repeat(batch_size, 1)
        output_size_params = torch.tensor([width, height], dtype=text_embeddings.dtype).repeat(batch_size, 1)

        if do_classifier_free_guidance:
            crop_params = torch.cat([crop_params, crop_params])
            input_size_params = torch.cat([input_size_params, input_size_params])
            output_size_params = torch.cat([output_size_params, output_size_params])

        add_time_ids = torch.cat([input_size_params, crop_params, output_size_params], dim=1).to(device)
        added_cond_kwargs = {'text_embeds': pooled_embeddings, 'time_ids': add_time_ids}

    # backward diffusion process
    for t in tqdm(model.inference_scheduler.timesteps, disable=not progress_bar):
        if do_classifier_free_guidance:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents

        latent_model_input = model.inference_scheduler.scale_model_input(latent_model_input, t)
        # Model prediction
        if t > (0.2 * len(model.inference_scheduler.timesteps)):
            pred = model.unet(latent_model_input,
                              t,
                              encoder_hidden_states=text_embeddings,
                              encoder_attention_mask=encoder_attn_mask,
                              added_cond_kwargs=added_cond_kwargs).sample
        else:
            pred = model2.unet(latent_model_input,
                               t,
                               encoder_hidden_states=text_embeddings,
                               encoder_attention_mask=encoder_attn_mask,
                               added_cond_kwargs=added_cond_kwargs).sample

        if do_classifier_free_guidance:
            # perform guidance. Note this is only techincally correct for prediction_type 'epsilon'
            pred_uncond, pred_text = pred.chunk(2)
            pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
            # Optionally rescale the classifer free guidance
            if rescaled_guidance is not None:
                std_pos = torch.std(pred_text, dim=(1, 2, 3), keepdim=True)
                std_cfg = torch.std(pred, dim=(1, 2, 3), keepdim=True)
                pred_rescaled = pred * (std_pos / std_cfg)
                pred = pred_rescaled * rescaled_guidance + pred * (1 - rescaled_guidance)
        # compute the previous noisy sample x_t -> x_t-1
        latents = model.inference_scheduler.step(pred, t, latents, generator=rng_generator).prev_sample

    # We now use the vae to decode the generated latents back into the image.
    # scale and decode the image latents with vae
    latents = 1 / model.latent_scale * latents
    image = model.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image.detach()  # (batch*num_images_per_prompt, channel, h, w)

def _prepare_text_embeddings(self, prompt, tokenized_prompts, tokenized_pad_mask, prompt_embeds,
                                num_images_per_prompt):
    """Tokenizes and embeds prompts if needed, then duplicates embeddings to support multiple generations per prompt."""
    device = self.text_encoder.device
    pooled_text_embeddings = None
    if prompt_embeds is None:
        max_length = None if self.sdxl else self.tokenizer.model_max_length
        if tokenized_prompts is None:
            tokenized_out = self.tokenizer(prompt,
                                            padding='max_length',
                                            max_length=max_length,
                                            truncation=True,
                                            return_tensors='pt')
            tokenized_prompts = tokenized_out.input_ids
            if self.mask_pad_tokens:
                tokenized_pad_mask = tokenized_out.attention_mask
            if self.sdxl:
                tokenized_prompts = torch.stack([tokenized_prompts[0], tokenized_prompts[1]], dim=1)
                if self.mask_pad_tokens:
                    # For cross attention mask, take union of masks (want [B, 77])
                    tokenized_pad_mask = torch.logical_or(tokenized_pad_mask[0], tokenized_pad_mask[1]).to(
                        tokenized_pad_mask[0].dtype).to(device)
        if self.sdxl:
            text_embeddings, pooled_text_embeddings = self.text_encoder(
                [tokenized_prompts[:, 0, :].to(device), tokenized_prompts[:, 1, :].to(device)])  # type: ignore
        else:
            text_embeddings = self.text_encoder(tokenized_prompts.to(device))[0]  # type: ignore
    else:
        if self.sdxl:
            raise NotImplementedError('SDXL not yet supported with precomputed embeddings')
        text_embeddings = prompt_embeds

    # duplicate text embeddings for each generation per prompt
    bs_embed, seq_len, _ = text_embeddings.shape
    text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)  # type: ignore
    text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if tokenized_pad_mask is not None:
        tokenized_pad_mask = tokenized_pad_mask.repeat(1, num_images_per_prompt, 1)
        tokenized_pad_mask = tokenized_pad_mask.view(bs_embed * num_images_per_prompt, seq_len)  # [B, 77]

    if self.sdxl and pooled_text_embeddings is not None:
        pooled_text_embeddings = pooled_text_embeddings.repeat(1, num_images_per_prompt)
        pooled_text_embeddings = pooled_text_embeddings.view(bs_embed * num_images_per_prompt, -1)
    return text_embeddings, pooled_text_embeddings, tokenized_pad_mask


def _check_prompt_lengths(prompt, negative_prompt):
    if prompt is None and negative_prompt is None:
        return
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    if negative_prompt:
        negative_prompt_bs = 1 if isinstance(negative_prompt, str) else len(negative_prompt)
        if negative_prompt_bs != batch_size:
            raise ValueError('len(prompts) and len(negative_prompts) must be the same. \
                    A negative prompt must be provided for each given prompt.')
