import logging
import os
from functools import partial
from typing import List, Tuple, Union

import copy
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

from accelerate.utils import set_seed
from diffusers import UniPCMultistepScheduler
from hydra.core.hydra_config import HydraConfig
from mmdet3d.datasets import build_dataset
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from dataset import ApiSetWrapper, ListSetWrapper, collate_fn_singleframe
from projects.dreamer.utils.common import load_module, move_to
from projects.dreamer.pipeline.pipeline_controlnet_single_ref import (
    StableDiffusionSingleRefControlNetPipeline
)
# from projects.dreamer.pipeline.pipeline_controlnet_single_ref_attn import (
#     StableDiffusionSingleRefControlNetPipeline,
# )
from projects.dreamer.networks.clip_embedder import FrozenOpenCLIPImageEmbedderV2
from projects.dreamer.runner.utils import img_m11_to_01, show_box_on_views, visualize_map
from safetensors.torch import load_file

import accelerate.utils

if not hasattr(accelerate.utils, "is_timm_available"):
    accelerate.utils.is_timm_available = lambda: False


def load_lora_weights(pipeline, checkpoint_path):
    # load base model
    pipeline.to("cuda")
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    alpha = 0.75
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device="cuda")
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline


def insert_pipeline_item(cfg: DictConfig, search_type, item=None) -> None:
    if item is None:
        return
    assert OmegaConf.is_list(cfg)
    ori_cfg: List = OmegaConf.to_container(cfg)
    for index, _it in enumerate(cfg):
        if _it["type"] == search_type:
            break
    else:
        raise RuntimeError(f"cannot find type: {search_type}")
    ori_cfg.insert(index + 1, item)
    cfg.clear()
    cfg.merge_with(ori_cfg)


def draw_box_on_imgs(cfg, idx, val_input, ori_imgs, transparent_bg=False) -> Tuple[Image.Image, ...]:
    if transparent_bg:
        in_imgs = [Image.new("RGB", img.size) for img in ori_imgs]
    else:
        in_imgs = ori_imgs

    if "filename" in val_input["meta_data"]["metas"][0].data:
        filename = val_input["meta_data"]["metas"][0].data["filename"][0]
    else:
        # Handle the case where 'filename' is missing
        print("Filename not found in the data.")
        filename = ''

    if "nuplan" in filename:
        nuplan = True
    else:
        nuplan = False
        
    out_imgs = show_box_on_views(
        OmegaConf.to_container(cfg.dataset.object_classes, resolve=True),
        in_imgs,
        val_input["meta_data"]["gt_bboxes_3d"][idx].data,
        val_input["meta_data"]["gt_labels_3d"][idx].data.numpy(),
        val_input["meta_data"]["lidar2image"][idx].data.numpy(),  # 644
        val_input["meta_data"]["img_aug_matrix"][idx].data.numpy(),  # 644
        nuplan=nuplan
    )
    if transparent_bg:
        for i in range(len(out_imgs)):
            out_imgs[i].putalpha(
                Image.fromarray(
                    (np.any(np.asarray(out_imgs[i]) > 0, axis=2) * 255).astype(np.uint8)
                )
            )
    return out_imgs


def update_progress_bar_config(pipe, **kwargs):
    if hasattr(pipe, "_progress_bar_config"):
        config = pipe._progress_bar_config
        config.update(kwargs)
    else:
        config = kwargs
    pipe.set_progress_bar_config(**config)


def setup_logger_seed(cfg):
    #### setup logger ####
    # only log debug info to log file
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
    # handle log from some packages
    logging.getLogger("shapely.geos").setLevel(logging.WARN)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("accelerate.tracking").setLevel(logging.INFO)
    logging.getLogger("numba").setLevel(logging.WARN)
    logging.getLogger("PIL").setLevel(logging.WARN)
    logging.getLogger("matplotlib").setLevel(logging.WARN)
    setattr(cfg, "log_root", HydraConfig.get().runtime.output_dir)
    if cfg.seed is not None:
        set_seed(cfg.seed)


def build_pipe(cfg, device):
    weight_dtype = torch.float16
    if cfg.resume_from_checkpoint.endswith("/"):
        cfg.resume_from_checkpoint = cfg.resume_from_checkpoint[:-1]
    pipe_param = {}
    model_cls = load_module(cfg.model.model_module)
    controlnet_path = os.path.join(cfg.resume_from_checkpoint, cfg.model.controlnet_dir)
    logging.info(f"Loading controlnet from {controlnet_path} with {model_cls}")
    controlnet = model_cls.from_pretrained(controlnet_path, torch_dtype=weight_dtype)
    controlnet.eval()  # from_pretrained will set to eval mode by default
    pipe_param["controlnet"] = controlnet

    if hasattr(cfg.model, "unet_module"):
        unet_cls = load_module(cfg.model.unet_module)
        unet_path = os.path.join(cfg.resume_from_checkpoint, cfg.model.unet_dir)
        logging.info(f"Loading unet from {unet_path} with {unet_cls}")
        if cfg.runner.get("trainable_strategy", "baseline") == "lora":
            from projects.dreamer.runner.multiview_runner_single_ref import _inject_lora_into_unet
            unet = unet_cls.from_config(unet_cls.load_config(unet_path))
            unet.to(dtype=weight_dtype)
            _inject_lora_into_unet(
                unet,
                rank=cfg.runner.lora_rank,
                use_xformers=cfg.runner.enable_xformers_memory_efficient_attention,
            )
            sd_path = os.path.join(unet_path, "diffusion_pytorch_model.bin")
            sd = torch.load(sd_path, map_location="cpu")
            sd = {k.replace(".chained_proc.", "."): v for k, v in sd.items()}
            missing, unexpected = unet.load_state_dict(sd, strict=False)
            logging.info(f"LoRA unet load: {len(missing)} missing, {len(unexpected)} unexpected")
        else:
            unet = unet_cls.from_pretrained(unet_path, torch_dtype=weight_dtype)
        logging.warn(f"We reset sc_attn_index from config.")
        for mod in unet.modules():
            if hasattr(mod, "_sc_attn_index"):
                mod._sc_attn_index = OmegaConf.to_container(
                    cfg.model.sc_attn_index, resolve=True
                )
        unet.eval()
        pipe_param["unet"] = unet


    if hasattr(cfg.model, "image_proj_model"):
        image_proj_model = nn.Linear(
            cfg.model.image_proj_model.input_dim, cfg.model.image_proj_model.output_dim
        )
        image_proj_model_path = os.path.join(
            cfg.resume_from_checkpoint,
            cfg.model.image_proj_model_dir,
            "image_proj_model.bin",
        )
        state_dict = torch.load(image_proj_model_path, map_location="cpu")
        image_proj_model.load_state_dict(state_dict)
        image_proj_model = image_proj_model.to(device)
        image_proj_model.type(torch.cuda.HalfTensor)
        image_proj_model.eval()
        pipe_param["image_proj_model"] = image_proj_model

    embedder = FrozenOpenCLIPImageEmbedderV2(
        arch="ViT-B-32",
        model_path="./pretrained/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin",
    )
    embedder = embedder.to(device)
    embedder.type(torch.cuda.HalfTensor)
    embedder.eval()
    pipe_param["embedder"] = embedder

    pipe_cls = load_module(cfg.model.pipe_module)
    logging.info(f"Build pipeline with {pipe_cls}")
    pipe = pipe_cls.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        **pipe_param,
        safety_checker=None,
        feature_extractor=None,  # since v1.5 has default, we need to override
        torch_dtype=weight_dtype,
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    if cfg.runner.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    pipe = pipe.to(device)

    # when inference, memory is not the issue. we do not need this.
    # pipe.enable_model_cpu_offload()
    return pipe, weight_dtype


def prepare_all(cfg, device="cuda", need_loader=True):
    assert cfg.resume_from_checkpoint is not None, "Please set model to load"
    setup_logger_seed(cfg)

    #### model ####
    pipe, weight_dtype = build_pipe(cfg, device)
    update_progress_bar_config(pipe, leave=False)

    if not need_loader:
        return pipe, weight_dtype

    #### datasets ####
    if cfg.runner.validation_index == "demo":
        val_dataset = ApiSetWrapper("data/demo_data/demo_data.pth")
    else:
        val_dataset = build_dataset(
            OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
        )
        if cfg.runner.validation_index != "all":
            val_dataset = ListSetWrapper(val_dataset, cfg.runner.validation_index)

    #### dataloader ####
    assert cfg.runner.validation_batch_size == 1, "Do not support more."
    collate_fn_param = {
        "tokenizer": pipe.tokenizer,
        "template": cfg.dataset.template,
        "bbox_mode": cfg.model.bbox_mode,
        "bbox_view_shared": cfg.model.bbox_view_shared,
        "bbox_drop_ratio": cfg.runner.bbox_drop_ratio,
        "bbox_add_ratio": cfg.runner.bbox_add_ratio,
        "bbox_add_num": cfg.runner.bbox_add_num,
    }

    def _collate_fn(examples, *args, **kwargs):
        return collate_fn_singleframe(examples, *args, **kwargs)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=partial(_collate_fn, is_train=False, **collate_fn_param),
        batch_size=cfg.runner.validation_batch_size,
        num_workers=cfg.runner.num_workers,
    )
    return pipe, val_dataloader, weight_dtype


def new_local_seed(global_generator):
    local_seed = torch.randint(
        0x7FFFFFFFFFFFFFF0, [1], generator=global_generator
    ).item()
    logging.debug(f"Using seed: {local_seed}")
    return local_seed


def run_one_batch_pipe(
    cfg,
    pipe: StableDiffusionSingleRefControlNetPipeline,
    pixel_values: torch.FloatTensor,  # useless
    captions: Union[str, List[str]],
    bev_hdmap: torch.FloatTensor,
    camera_param: Union[torch.Tensor, None],
    rel_pose,
    ref_rel_pose,
    ref_images,
    layout_canvas,
    occupancy,
    ref_layout_canvas,
    ref_occupancy,
    meta_data,
    bev_controlnet_kwargs: dict,
    global_generator=None,
    is_first_frame=False,
    cal_flops=False,
):
    """call pipe several times to generate images

    Args:
        cfg (_type_): _description_
        pipe (StableDiffusionBEVControlNetPipeline): _description_
        captions (Union[str, List[str]]): _description_
        bev_map_with_aux (torch.FloatTensor): (B=1, C=26, 200, 200), float32
        camera_param (Union[torch.Tensor, None]): (B=1, N=6, 3, 7), if None,
            use learned embedding for uncond_cam

    Returns:
        List[List[List[Image.Image]]]: 3-dim list of PIL Image: B, Times, views
    """
    # for each input param, we generate several times to check variance.
    if isinstance(captions, str):
        batch_size = 1
    else:
        batch_size = len(captions)

    # let different prompts have the same random seed
    device = pipe.device
    
    if cfg.seed is None:
        generator = None
    else:
        if global_generator is not None:
            if cfg.fix_seed_within_batch:
                generator = []
                for _ in range(batch_size):
                    local_seed = new_local_seed(global_generator)
                    generator.append(
                        torch.Generator(device=device).manual_seed(local_seed)
                    )
            else:
                local_seed = new_local_seed(global_generator)
                generator = torch.Generator(device=device).manual_seed(local_seed)
        else:
            if cfg.fix_seed_within_batch:
                generator = [
                    torch.Generator(device=device).manual_seed(cfg.seed)
                    for _ in range(batch_size)
                ]
            else:
                generator = torch.Generator(device=device).manual_seed(cfg.seed)

    pipeline_param = {k: v for k, v in cfg.runner.pipeline_param.items()}
    gen_imgs_list = [[] for _ in range(batch_size)]
    
    for ti in range(cfg.runner.validation_times):
        is_first_frame = meta_data[0].data['is_first_frame']
        if is_first_frame:
            image: StableDiffusionSingleRefControlNetPipeline = pipe(
                prompt=captions,
                bev_hdmap=bev_hdmap,
                camera_param=camera_param,
                rel_pose=rel_pose,
                ref_rel_pose=ref_rel_pose,
                ref_images=ref_images,
                layout_canvas=layout_canvas,
                occupancy=occupancy,
                ref_layout_canvas=ref_layout_canvas,
                ref_occupancy=ref_occupancy,
                height=cfg.dataset.image_size[0],
                width=cfg.dataset.image_size[1],
                generator=generator,
                bev_controlnet_kwargs=bev_controlnet_kwargs,
                **pipeline_param,
                is_first_frame=is_first_frame,
                use_first_frame=cfg.get('use_first_frame', False),
            )
        else:
            image: StableDiffusionSingleRefControlNetPipeline = pipe(
                prompt=captions,
                bev_hdmap=bev_hdmap,
                camera_param=camera_param,
                rel_pose=rel_pose,
                ref_rel_pose=ref_rel_pose,
                ref_images=ref_images,
                layout_canvas=layout_canvas,
                occupancy=occupancy,
                ref_layout_canvas=ref_layout_canvas,
                ref_occupancy=ref_occupancy,
                height=cfg.dataset.image_size[0],
                width=cfg.dataset.image_size[1],
                generator=generator,
                bev_controlnet_kwargs=bev_controlnet_kwargs,
                use_first_frame=cfg.get('use_first_frame', False),
                **pipeline_param,
            )
        image: List[List[Image.Image]] = image.images
        for bi, imgs in enumerate(image):
            gen_imgs_list[bi].append(imgs)
    return gen_imgs_list


def run_one_batch(
    cfg,
    pipe,
    val_input,
    weight_dtype,
    global_generator=None,
    run_one_batch_pipe_func=run_one_batch_pipe,
    transparent_bg=False,
    map_size=400,
):
    """Run one batch of data according to your configuration

    Returns:
        List[Image.Image]: map image
        List[List[Image.Image]]: ori images
        List[List[Image.Image]]: ori images with bbox, can be []
        List[List[Tuple[Image.Image]]]: generated images list
        List[List[Tuple[Image.Image]]]: generated images list, can be []
        if 2-dim: B, views; if 3-dim: B, Times, views
    """
    bs = len(val_input["meta_data"]["metas"])
    camera_param = val_input["camera_param"].to(weight_dtype)
    rel_pose = val_input["relative_pose"].to(weight_dtype)
    ref_rel_pose = torch.eye(rel_pose.size(-1), dtype=rel_pose.dtype, device=rel_pose.device).expand_as(rel_pose)
    ref_images = val_input["ref_images"].to(weight_dtype)
    layout_canvas = val_input["layout_canvas"].to(weight_dtype)
    occupancy = val_input["occupancy"].to(weight_dtype)
    ref_layout_canvas = val_input["ref_layout_canvas"].to(weight_dtype)
    ref_occupancy = val_input["ref_occupancy"].to(weight_dtype)
    is_first_frame = val_input['meta_data']['metas'][0].data.get('is_first_frame', False)

    # 3-dim list: B, Times, views
    gen_imgs_list = run_one_batch_pipe_func(
        cfg,
        pipe,
        val_input["pixel_values"],
        val_input["captions"],
        val_input["bev_hdmap"],
        camera_param,
        rel_pose,
        ref_rel_pose,
        ref_images,
        layout_canvas,
        occupancy,
        ref_layout_canvas,
        ref_occupancy,
        val_input["meta_data"]["metas"],
        val_input["kwargs"],
        global_generator=global_generator,
        is_first_frame=is_first_frame,
    )
    
    # map
    map_imgs = []
    for bev_map in val_input["bev_hdmap"]:
        if 'map_classes' in cfg.dataset:
            map_layers = len(cfg.dataset.map_classes)
        else:
            map_layers = len(cfg.dataset.map_classes_nuscenes)

        vis_map = torch.zeros(map_layers, *bev_map.shape[1:])
        # Change order for visualization
        for i, j in enumerate([6,1,5,0]):
            vis_map[j] = bev_map[i]
        map_img_np = visualize_map(cfg, vis_map, target_size=map_size)
        map_imgs.append(Image.fromarray(map_img_np))
    # ori
    if val_input["pixel_values"] is not None:
        ori_imgs = [
            [
                to_pil_image(img_m11_to_01(val_input["pixel_values"][bi][i]))
                for i in range(6)
            ]
            for bi in range(bs)
        ]
        if cfg.show_box_on_img:
            ori_imgs_with_box = [
                draw_box_on_imgs(cfg, bi, val_input, ori_imgs[bi],
                                 transparent_bg=transparent_bg)
                for bi in range(bs)
            ]
        else:
            ori_imgs_with_box = copy.deepcopy(ori_imgs)

        if cfg.show_map_on_img:
            for bi, images in enumerate(ori_imgs_with_box):
                for vi in range(len(images)):
                    tmp = copy.deepcopy(layout_canvas[bi][vi][:3, ...])
                    tmp = tmp.permute(1,2,0).numpy()
                    ori_imgs_with_box[bi][vi] = Image.fromarray(np.where(tmp, 255, ori_imgs_with_box[bi][vi]))
    else:
        ori_imgs = [None for bi in range(bs)]
        ori_imgs_with_box = [None for bi in range(bs)]

    # save gen with box
    gen_imgs_wb_list = []
    if cfg.show_box_on_img:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wb_list.append(
                [
                    draw_box_on_imgs(
                        cfg, bi, val_input, images[ti], transparent_bg=transparent_bg
                    )
                    for ti in range(len(images))
                ]
            )
    else:
        gen_imgs_wb_list = copy.deepcopy(gen_imgs_list)

    if cfg.show_map_on_img:
        for bi, images in enumerate(gen_imgs_wb_list):
            for ti in range(len(images)):
                for vi in range(len(images[ti])):
                    tmp = copy.deepcopy(layout_canvas[bi][vi][:3, ...])
                    tmp = tmp.permute(1, 2, 0).numpy()
                    gen_imgs_wb_list[bi][ti][vi] = Image.fromarray(
                        np.where(tmp, 255, gen_imgs_wb_list[bi][ti][vi])
                    )


    return (
        map_imgs,
        ori_imgs,
        ori_imgs_with_box,
        gen_imgs_list,
        gen_imgs_wb_list,
    )

class PipelineFlopsWrapper(torch.nn.Module):
    def __init__(self, pipe: StableDiffusionSingleRefControlNetPipeline):
        super().__init__()
        self.pipe = pipe

        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.controlnet = pipe.controlnet
        self.embedder = pipe.embedder
        self.image_proj_model = pipe.image_proj_model

        self.safety_checker = getattr(pipe, "safety_checker", None)

    def forward(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs["output_type"] = "latent"
        kwargs["return_dict"] = False

        latents, _ = self.pipe(**kwargs)

        return latents