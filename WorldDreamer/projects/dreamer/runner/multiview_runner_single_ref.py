import logging
import os
import contextlib
from omegaconf import OmegaConf
from functools import partial
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers import (
    ModelMixin,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler

from .base_runner import BaseRunner
from .utils import smart_param_count
from dataset.utils import collate_fn_singleframe
from projects.dreamer.utils.common import load_module, convert_outputs_to_fp16, move_to
from projects.dreamer.runner.single_ref_validator import SingleRefBaseValidator
from projects.dreamer.networks.clip_embedder import FrozenOpenCLIPImageEmbedderV2
from diffusers.models.attention_processor import Attention, AttnProcessor, XFormersAttnProcessor # ssk
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAXFormersAttnProcessor

import numpy as np


class ReferenceOnlyAttnProc(torch.nn.Module):
    def __init__(self, chained_proc, enabled=False, name=None):
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,
        mode=None, ref_dict: dict = None, is_cfg_guidance=False
    ) -> torch.Tensor:
        if not self.enabled or mode not in {"w", "r"} or ref_dict is None:
            return self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)

        context_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        if mode == 'w':
            if is_cfg_guidance:
                _, context_cond = context_hidden_states.chunk(2)
                ref_dict[self.name] = context_cond
            else:
                ref_dict[self.name] = context_hidden_states
            return self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)

        # mode == 'r'
        if is_cfg_guidance:
            hidden_uncond, hidden_cond = hidden_states.chunk(2)
            context_uncond, context_cond = context_hidden_states.chunk(2)

            res_uncond = self.chained_proc(attn, hidden_uncond, context_uncond, attention_mask)

            ref_context_state = ref_dict.get(self.name, None)
            if ref_context_state is not None:
                context_cond = torch.cat([context_cond, ref_context_state], dim=1)

            res_cond = self.chained_proc(attn, hidden_cond, context_cond, attention_mask)
            return torch.cat([res_uncond, res_cond], dim=0)
        else:
            ref_context_state = ref_dict.get(self.name, None)
            if ref_context_state is not None:
                context_hidden_states = torch.cat([context_hidden_states, ref_context_state], dim=1)
            return self.chained_proc(attn, hidden_states, context_hidden_states, attention_mask)


class ControlnetUnetWrapper(ModelMixin):
    """As stated in https://github.com/huggingface/accelerate/issues/668, we
    should not use accumulate provided by accelerator, but create a wrapper to
    two modules.
    """

    def __init__(
        self, controlnet, unet, weight_dtype=torch.float32, unet_in_fp16=True
    ) -> None:
        super().__init__()
        self.controlnet = controlnet
        self.unet = unet
        self.weight_dtype = weight_dtype
        self.unet_in_fp16 = unet_in_fp16
        self._reference_injection_setup = False

    def forward(
        self,
        noisy_latents,
        noisy_latents_ref_img,
        timesteps,
        camera_param,
        encoder_hidden_states,
        encoder_hidden_states_uncond,
        bev_hdmap,
        rel_pose,
        ref_rel_pose,
        ref_images,
        layout_canvas,
        occupancy,
        ref_layout_canvas,
        ref_occupancy,
        flatten_ref_attn_alt=False,
        camera_params_raw=None,
        ref_attn_drop_ratio=0.5,
        **kwargs,
    ):
        N_cam = noisy_latents.shape[1]

        additional_param = {}
        if hasattr(self.unet, "crossview_attn_type"):
            additional_param['camera_params_raw'] = camera_params_raw

        drop_ref_attn = random.random() < ref_attn_drop_ratio

        if flatten_ref_attn_alt:
            self.setup_reference_injection()
            ref_dict = {}
            if not drop_ref_attn:
                down_block_res_samples_ref, mid_block_res_sample_ref, encoder_hidden_states_with_cam_ref = self.controlnet(
                    sample=noisy_latents_ref_img,
                    timestep=timesteps,
                    camera_param=camera_param,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_uncond=encoder_hidden_states_uncond,
                    bev_hdmap=bev_hdmap,
                    rel_pose=ref_rel_pose,
                    ref_images=ref_images,
                    layout_canvas=ref_layout_canvas,
                    occupancy=ref_occupancy,
                    return_dict=False,
                    **kwargs,
                )
                noisy_latents_ref_img = rearrange(noisy_latents_ref_img, "b n ... -> (b n) ...")
        else:
            ref_dict = {}

        down_block_res_samples, mid_block_res_sample, encoder_hidden_states_with_cam = self.controlnet(
            sample=noisy_latents,
            timestep=timesteps,
            camera_param=camera_param,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_uncond=encoder_hidden_states_uncond,
            bev_hdmap=bev_hdmap,
            rel_pose=rel_pose,
            ref_images=ref_images,
            layout_canvas=layout_canvas,
            occupancy=occupancy,
            return_dict=False,
            # camera_params_raw=camera_params_raw,
            **kwargs,
        )

        # starting from here, we use (B n) as batch_size
        noisy_latents = rearrange(noisy_latents, "b n ... -> (b n) ...")
        if timesteps.ndim == 1:
            timesteps = repeat(timesteps, "b -> (b n)", n=N_cam)

        # Predict the noise residual
        # NOTE: Since we fix most of the model, we cast the model to fp16 and
        # disable autocast to prevent it from falling back to fp32. Please
        # enable autocast on your customized/trainable modules.
        context = contextlib.nullcontext
        context_kwargs = {}
        if self.unet_in_fp16:
            context = torch.cuda.amp.autocast
            context_kwargs = {"enabled": False}
        with context(**context_kwargs):
            if flatten_ref_attn_alt and not drop_ref_attn:
                self.unet(
                    noisy_latents_ref_img,
                    timesteps.reshape(-1),
                    encoder_hidden_states=encoder_hidden_states_with_cam_ref.to(dtype=self.weight_dtype),
                    down_block_additional_residuals=[s.to(dtype=self.weight_dtype) for s in down_block_res_samples_ref],
                    mid_block_additional_residual=mid_block_res_sample_ref.to(dtype=self.weight_dtype),
                    cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
                    **additional_param,
                )

            if flatten_ref_attn_alt:
                model_pred = self.unet(
                    noisy_latents,
                    timesteps.reshape(-1),
                    encoder_hidden_states=encoder_hidden_states_with_cam.to(dtype=self.weight_dtype),
                    down_block_additional_residuals=[s.to(dtype=self.weight_dtype) for s in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
                    cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict),
                    **additional_param
                ).sample
            else:
                model_pred = self.unet(
                    noisy_latents,
                    timesteps.reshape(-1),
                    encoder_hidden_states=encoder_hidden_states_with_cam.to(dtype=self.weight_dtype),
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    **additional_param
                ).sample

        model_pred = rearrange(model_pred, "(b n) ... -> b n ...", n=N_cam)
        return model_pred
    
    def setup_reference_injection(self):
        if hasattr(self, '_reference_injection_setup') and self._reference_injection_setup:
            return

        attn_procs = {}
        named_modules = dict(self.unet.named_modules())
        for name in self.unet.attn_processors.keys():
            module_path = name.replace(".attn1.processor", "").replace(".attn2.processor", "").replace(".transformer_blocks.0", "")
        
            parent_module = named_modules.get(module_path)
            
            is_target = getattr(parent_module, 'is_reference_injection_target', False)

            should_be_enabled = (
                'up_blocks' in name 
                and name.endswith("attn1.processor") 
                and is_target
            )
            
            existing_proc = self.unet.attn_processors[name]
            if isinstance(existing_proc, ReferenceOnlyAttnProc):
                # already wrapped (re-entry) — keep as is
                attn_procs[name] = existing_proc
                continue
            # preserve existing processor (e.g. LoRA) by chaining instead of replacing
            attn_procs[name] = ReferenceOnlyAttnProc(
                chained_proc=existing_proc,
                enabled=should_be_enabled,
                name=name,
            )
        self.unet.set_attn_processor(attn_procs)
        self._reference_injection_setup = True

def _cast_attn_inputs_fp16(hidden_states, encoder_hidden_states):
    # LoRALinearLayer derives its output dtype from its input. If the caller passed
    # fp32 (e.g. validator's outer autocast upcasts LayerNorm output), LoRA returns
    # fp32 while base attn returns fp16, breaking xformers (Q/K/V dtype mismatch).
    # Force inputs to fp16 so LoRA stays consistent with the base attn dtype.
    if hidden_states is not None:
        hidden_states = hidden_states.to(torch.float16)
    if encoder_hidden_states is not None:
        encoder_hidden_states = encoder_hidden_states.to(torch.float16)
    return hidden_states, encoder_hidden_states


class _AutocastLoRAXFormersAttnProcessor(LoRAXFormersAttnProcessor):
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, scale=1.0, temb=None):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            hidden_states, encoder_hidden_states = _cast_attn_inputs_fp16(
                hidden_states, encoder_hidden_states
            )
            out = super().__call__(attn, hidden_states, encoder_hidden_states,
                                   attention_mask, scale, temb)
        return out.to(torch.float16) if isinstance(out, torch.Tensor) else out


class _AutocastLoRAAttnProcessor(LoRAAttnProcessor):
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, scale=1.0, temb=None):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            hidden_states, encoder_hidden_states = _cast_attn_inputs_fp16(
                hidden_states, encoder_hidden_states
            )
            out = super().__call__(attn, hidden_states, encoder_hidden_states,
                                   attention_mask, scale, temb)
        return out.to(torch.float16) if isinstance(out, torch.Tensor) else out


class MultiviewSingleRefRunner(BaseRunner):
    def __init__(self, cfg, accelerator, train_set, val_set) -> None:
        self._strategy = cfg.runner.get("trainable_strategy", "baseline")
        self._global_step_counter = 0
        self._toggle_done = False
        super().__init__(cfg, accelerator, train_set, val_set)
        self._apply_trainable_strategy()
        pipe_cls = load_module(cfg.model.pipe_module)
        self.validator = SingleRefBaseValidator(
            self.cfg,
            self.val_dataset,
            pipe_cls,
            pipe_param={
                "vae": self.vae,
                "text_encoder": self.text_encoder,
                "tokenizer": self.tokenizer,
                "embedder": self.embedder,
            },
        )

    def _apply_trainable_strategy(self):
        if self._strategy == "baseline":
            return
        else:
            raise ValueError(f"Unknown trainable_strategy: {self._strategy}")

    def _init_fixed_models(self, cfg):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.model.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.model.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(
            cfg.model.pretrained_model_name_or_path, subfolder="vae"
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            cfg.model.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.embedder = FrozenOpenCLIPImageEmbedderV2(
            arch="ViT-B-32",
            model_path="./pretrained/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin",
        )

    def _init_trainable_models(self, cfg):
        unet = UNet2DConditionModel.from_pretrained(
            cfg.model.pretrained_model_name_or_path, subfolder="unet"
        )

        model_cls = load_module(cfg.model.unet_module)
        unet_param = OmegaConf.to_container(self.cfg.model.unet, resolve=True)
        self.unet = model_cls.from_unet_2d_condition(unet, **unet_param)

        model_cls = load_module(cfg.model.model_module)
        controlnet_param = OmegaConf.to_container(
            self.cfg.model.controlnet, resolve=True
        )
        self.controlnet = model_cls.from_unet(unet, **controlnet_param)

        self.image_proj_model = nn.Linear(
            cfg.model.image_proj_model.input_dim, cfg.model.image_proj_model.output_dim
        )

    def _set_model_trainable_state(self, train=True):
        # set trainable status
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet.train(train)
        self.unet.requires_grad_(False)
        self.image_proj_model.requires_grad_(True)

        self.embedder.train = False
        for param in self.embedder.parameters():
            param.requires_grad = False

        for name, mod in self.unet.trainable_module.items():
            logging.debug(f"[MultiviewRunner] set {name} to requires_grad = True")
            mod.requires_grad_(train)

    def set_optimizer_scheduler(self):
        # optimizer and lr_schedulers
        if self.cfg.runner.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        base_lr = self.cfg.runner.learning_rate
        controlnet_params = list(self.controlnet.parameters())
        proj_params = list(self.image_proj_model.parameters())
        unet_params = list(self.unet.trainable_parameters)

        params_to_optimize = controlnet_params + unet_params + proj_params
        param_groups = params_to_optimize
        logging.info(
            f"[MultiviewRunner] single group: "
            f"{smart_param_count(params_to_optimize)} params @ lr={base_lr:.2e}"
        )

        self.optimizer = optimizer_class(
            param_groups,
            lr=base_lr,
            betas=(self.cfg.runner.adam_beta1, self.cfg.runner.adam_beta2),
            weight_decay=self.cfg.runner.adam_weight_decay,
            eps=self.cfg.runner.adam_epsilon,
        )

        # lr scheduler
        self._calculate_steps()
        self.lr_scheduler = get_scheduler(
            self.cfg.runner.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.runner.lr_warmup_steps
            * self.cfg.runner.gradient_accumulation_steps,
            num_training_steps=self.cfg.runner.max_train_steps
            * self.cfg.runner.gradient_accumulation_steps,
            num_cycles=self.cfg.runner.lr_num_cycles,
            power=self.cfg.runner.lr_power,
        )

    def _set_dataset_loader(self):
        collate_fn_param = {
            "tokenizer": self.tokenizer,
            "template": self.cfg.dataset.template,
            "bbox_mode": self.cfg.model.bbox_mode,
            "bbox_view_shared": self.cfg.model.bbox_view_shared,
            "bbox_drop_ratio": self.cfg.runner.bbox_drop_ratio,
            "bbox_add_ratio": self.cfg.runner.bbox_add_ratio,
            "bbox_add_num": self.cfg.runner.bbox_add_num,
            "with_ref_bboxes": self.cfg.runner.with_ref_bboxes,
        }
        if self.train_dataset is not None:
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                shuffle=self.cfg.dataset.shuffle_train if hasattr(self.cfg.dataset, "shuffle_train") else True, # True
                collate_fn=partial(
                    collate_fn_singleframe, is_train=True, **collate_fn_param
                ),
                batch_size=self.cfg.runner.train_batch_size,
                num_workers=self.cfg.runner.num_workers,
                pin_memory=True,
                prefetch_factor=self.cfg.runner.prefetch_factor,
                persistent_workers=self.cfg.runner.persistent_workers,
            )
        if self.val_dataset is not None:
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                shuffle=False,
                collate_fn=partial(
                    collate_fn_singleframe, is_train=False, **collate_fn_param
                ),
                batch_size=self.cfg.runner.validation_batch_size,
                num_workers=self.cfg.runner.num_workers,
                prefetch_factor=self.cfg.runner.prefetch_factor,
            )

    def prepare_device(self):
        self.controlnet_unet = ControlnetUnetWrapper(self.controlnet, self.unet)
        # accelerator
        ddp_modules = (
            self.controlnet_unet,
            self.image_proj_model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        ddp_modules = self.accelerator.prepare(*ddp_modules)
        (
            self.controlnet_unet,
            self.image_proj_model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = ddp_modules

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.embedder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.image_proj_model.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.cfg.runner.unet_in_fp16 and self.weight_dtype == torch.float16:
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
            # move optimized params to fp32. TODO: is this necessary?
            if self.cfg.model.use_fp32_for_unet_trainable:
                for name, mod in self.unet.trainable_module.items():
                    logging.debug(f"[MultiviewRunner] set {name} to fp32")
                    mod.to(dtype=torch.float32)
                    if isinstance(mod, (LoRAXFormersAttnProcessor, LoRAAttnProcessor)):
                        # LoRA processors override __call__ (bypassing forward).
                        # autocast is added via _AutocastLoRA* subclass, so skip the forward-wrap here.
                        continue
                    mod._original_forward = mod.forward
                    # autocast intermediate is necessary since others are fp16
                    mod.forward = torch.cuda.amp.autocast(dtype=torch.float16)(
                        mod.forward
                    )
                    # we ensure output is always fp16
                    mod.forward = convert_outputs_to_fp16(mod.forward)
            else:
                raise TypeError(
                    "There is an error/bug in accumulation wrapper, please "
                    "make all trainable param in fp32."
                )
        controlnet_unet = self.accelerator.unwrap_model(self.controlnet_unet)
        controlnet_unet.weight_dtype = self.weight_dtype
        controlnet_unet.unet_in_fp16 = self.cfg.runner.unet_in_fp16
        self.accelerator.unwrap_model(self.embedder)
        self.accelerator.unwrap_model(self.image_proj_model)

        with torch.no_grad():
            self.accelerator.unwrap_model(self.controlnet).prepare(
                self.cfg, tokenizer=self.tokenizer, text_encoder=self.text_encoder
            )

        # We need to recalculate our total training steps as the size of the
        # training dataloader may have changed.
        self._calculate_steps()

    def _save_model(self, root=None):
        if root is None:
            root = self.cfg.log_root
        # if self.accelerator.is_main_process:
        controlnet = self.accelerator.unwrap_model(self.controlnet)
        controlnet.save_pretrained(os.path.join(root, self.cfg.model.controlnet_dir))
        unet = self.accelerator.unwrap_model(self.unet)
        unet.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))
        image_proj_model = self.accelerator.unwrap_model(self.image_proj_model)
        os.makedirs(os.path.join(root, self.cfg.model.image_proj_model_dir), exist_ok=True)
        torch.save(
            image_proj_model.state_dict(),
            os.path.join(
                root, self.cfg.model.image_proj_model_dir, "image_proj_model.bin"
            ),
        )
        logging.info(f"Save your model to: {root}")


    def _train_one_step(self, batch):
        self.controlnet_unet.train()
        with self.accelerator.accumulate(self.controlnet_unet):
            N_cam = batch["pixel_values"].shape[1]

            # Convert ref_images to latent space
            latents = self.vae.encode(
                rearrange(batch["pixel_values"], "b n c h w -> (b n) c h w").to(
                    dtype=self.weight_dtype
                )
            ).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            latents = rearrange(latents, "(b n) c h w -> b n c h w", n=N_cam)
            
            latents_ref_img = self.vae.encode(
                rearrange(batch["ref_images"], "b n c h w -> (b n) c h w").to(
                    dtype=self.weight_dtype
                )
            ).latent_dist.sample()
            latents_ref_img = latents_ref_img * self.vae.config.scaling_factor
            latents_ref_img = rearrange(latents_ref_img, "(b n) c h w -> b n c h w", n=N_cam)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            noise_ref_img = torch.randn_like(latents_ref_img)
            # make sure we use same noise for different views, only take the first
            if self.cfg.model.train_with_same_noise:
                noise = repeat(noise[:, 0], "b ... -> b r ...", r=N_cam)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            if self.cfg.model.train_with_same_t:
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
            else:
                timesteps = torch.stack(
                    [
                        torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=latents.device,
                        )
                        for _ in range(N_cam)
                    ],
                    dim=1,
                )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self._add_noise(latents, noise, timesteps)
            noisy_latents_ref_img = self._add_noise(latents_ref_img, noise_ref_img, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
            encoder_hidden_states_uncond = self.text_encoder(batch["uncond_ids"])[0]

            bev_hdmap = batch["bev_hdmap"].to(dtype=self.weight_dtype)
            camera_param = batch["camera_param"].to(self.weight_dtype)
            rel_pose = batch["relative_pose"].to(self.weight_dtype)
            ref_rel_pose = torch.eye(rel_pose.size(-1), dtype=rel_pose.dtype, device=rel_pose.device).expand_as(rel_pose)

            layout_canvas = batch["layout_canvas"].to(dtype=self.weight_dtype)
            layout_bbox = batch["layout_bbox"].to(dtype=self.weight_dtype)
            occupancy = batch["occupancy"].to(dtype=self.weight_dtype)
            
            ref_layout_canvas = batch["ref_layout_canvas"].to(dtype=self.weight_dtype)
            ref_occupancy = batch["ref_occupancy"].to(dtype=self.weight_dtype)
            
            if hasattr(self, "embedder"):
                image_tensor = rearrange(
                    batch["ref_images"], "b n c h w -> (b n) c h w"
                ).to(
                    dtype=self.weight_dtype
                )  # torch.Size([6, 3, 224, 400])
                # img: b c h w >> b l c
                ref_images = self.embedder(image_tensor)  # torch.Size([6, 50, 768])
                ref_images = self.image_proj_model(ref_images)
                ref_images = rearrange(ref_images, "(b n) c l -> b n c l", n=N_cam)

            kwargs = {
                **batch["kwargs"],
                "captions": batch["captions"],
            }
            model_pred = self.controlnet_unet(
                noisy_latents,
                noisy_latents_ref_img,
                timesteps,
                camera_param,
                encoder_hidden_states,
                encoder_hidden_states_uncond,
                bev_hdmap,
                rel_pose,
                ref_rel_pose,
                ref_images,
                layout_canvas,
                occupancy,
                ref_layout_canvas,
                ref_occupancy,
                self.cfg.model.unet.flatten_ref_attn_alt,
                self.cfg.runner.ref_attn_drop_ratio,
                **kwargs,
            )

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            
            instances = self.cfg.model.instance
            if instances is not None:
                instance_total = instances['total']
                instance_loss_weight = [instances['car'], instances['truck'], instances['construction_vehicle'], instances['bus'], instances['trailer'], instances['barrier'], instances['motorcycle'], instances['bicycle'], instances['pedestrian'], instances['traffic_cone']]
                instance_loss_weight = np.array(instance_loss_weight) / instance_total
                instance_loss_weight = 1.0 / torch.tensor(instance_loss_weight).unsqueeze(1).unsqueeze(1)
                instance_loss_weight = instance_loss_weight.to(layout_bbox.device) * layout_bbox / 255.0
                instance_loss_weight, _ = torch.max(instance_loss_weight, dim=2)
                weighted_loss = loss * instance_loss_weight.unsqueeze(2)
                loss = loss + weighted_loss * 0.02
            
            loss = loss.mean()

            self.accelerator.backward(loss)
            if (
                self.accelerator.sync_gradients
                and self.cfg.runner.max_grad_norm is not None
            ):
                params_to_clip = self.controlnet_unet.parameters()
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.runner.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=self.cfg.runner.set_grads_to_none)

        return loss

    def _validation(self, step):
        controlnet = self.accelerator.unwrap_model(self.controlnet)
        unet = self.accelerator.unwrap_model(self.unet)
        image_proj_model = self.accelerator.unwrap_model(self.image_proj_model)
        with torch.inference_mode():
            image_logs = self.validator.validate(
                controlnet,
                unet,
                self.accelerator.trackers,
                step,
                self.weight_dtype,
                self.accelerator.device,
                image_proj_model,
            )
