import logging
import os
import sys
# fmt: off
# bypass annoying warning
import warnings

import cv2
import hydra
import numpy as np
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from moviepy.editor import *
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from shapely.errors import ShapelyDeprecationWarning
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

sys.path.append(".")  # noqa
from data.demo_data.img_style import style_dict
from projects.dreamer.runner.utils import (concat_6_views, img_concat_h,
                                           img_concat_v)
from projects.dreamer.utils.test_utils import prepare_all, run_one_batch

target_map_size = 400

def output_func(x): return concat_6_views(x, oneline=True)

def make_video_with_filenames(filenames, outname, fps=2):
    clips = [ImageClip(m).set_duration(1 / fps) for m in filenames]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(outname, fps=fps)

transform1 = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(
                                     mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                 ])


# For visualizing semantic occupancy
classname16_to_color = {  # Custom defined colors for 16 classes.
        "noise": (0, 0, 0),                  # Black
        "barrier": (128, 128, 128),          # Gray
        "bicycle": (255, 0, 0),              # Red
        "bus": (255, 165, 0),                # Orange
        "car": (255, 255, 0),                # Yellow
        "construction_vehicle": (0, 128, 0), # Green
        "motorcycle": (0, 255, 0),           # Lime
        "pedestrian": (0, 0, 255),           # Blue
        "traffic_cone": (255, 105, 180),     # Hot pink
        "trailer": (138, 43, 226),           # Blue violet
        "truck": (210, 105, 30),             # Chocolate
        "driveable_surface": (0, 255, 255),  # Cyan
        "other_flat": (255, 20, 147),        # Deep pink
        "sidewalk": (128, 0, 128),           # Purple
        "terrain": (189, 183, 107),          # Dark khaki
        "manmade": (205, 133, 63),           # Peru
        "vegetation": (34, 139, 34),         # Forest green
}
with open("./configs/dataset/Nuscenes_labels.yaml", 'r') as stream:
    nuscenes_yaml = yaml.safe_load(stream)
labels_16_to_classname = nuscenes_yaml["labels_16"]
labels16_to_color = {k:classname16_to_color[v] for k, v in labels_16_to_classname.items()}
max_dist = 63.0


@hydra.main(version_base=None, config_path="../configs", config_name="test_config_nus_448_depth_infl_ref_attn")
def main(cfg: DictConfig):
    if cfg.debug:
        import debugpy

        debugpy.listen(cfg.get('debug_port', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Attached, continue...")

    output_dir = to_absolute_path(cfg.resume_from_checkpoint)
    original_overrides = OmegaConf.load(
        os.path.join(output_dir, "../hydra/overrides.yaml")
    )
    current_overrides = HydraConfig.get().overrides.task

    # getting the config name of this job.
    config_name = HydraConfig.get().job.config_name
    # concatenating the original overrides with the current overrides
    overrides = original_overrides + current_overrides
    # compose a new config from scratch
    cfg = hydra.compose(config_name, overrides=overrides)

    logging.info(f"Your validation index: {cfg.runner.validation_index}")

    #### setup everything ####
    pipe, val_dataloader, weight_dtype = prepare_all(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(cfg.log_root, "run_config.yaml"))

    #### start ####
    batch_index = -1
    progress_bar = tqdm(
        range(len(val_dataloader) * cfg.runner.validation_times),
        desc="Steps",
    )
    os.makedirs(os.path.join(cfg.log_root, "frames"), exist_ok=True)
    gen_ref = None
    
    save_only_gen = "e2e_eval_mode" in cfg and cfg.e2e_eval_mode
    print(f"Save only generated images for evaluation: {save_only_gen}")
    
    for val_input in val_dataloader:
        batch_index += 1
        batch_img_index = 0
        if cfg.runner.validation_index in ['demo', 'all']:
            curr_index = batch_index
        else:
            curr_index = cfg.runner.validation_index[batch_index]
        
        if "num_gen_frames" in cfg and batch_index >= cfg.num_gen_frames:
            print(f"Completed generating {cfg.num_gen_frames} frames, exiting.")
            break

        ori_img_paths = []
        gen_img_paths = {}
        if cfg.runner.validation_index == 'demo' and batch_index == 0:
            ref_image_key = 'boston_rain'
            val_input["ref_images"][0, ...] = style_dict(ref_image_key, cfg.dataset.dataset_root_nuscenes)
            val_input['relative_pose'][0] = torch.eye(4)
        elif val_input['meta_data']['metas'][0].data.get('is_first_frame', False):
            print(curr_index)
            pass
        elif gen_ref is None:
            pass
        else:
            val_input["ref_images"][0, ...] = gen_ref
        
        # You can change the description with the folowing commands.
        """
        # For adverse weather condition
        python tools/test.py \
            resume_from_checkpoint=./dreamer-log/double_controlnet/weight-E2-S60001 \
            +text_prompt="Night. Photo-realistic." \

        # For normal weather condition
        python tools/test.py \
            resume_from_checkpoint=./dreamer-log/double_controlnet/weight-E2-S60001 \
        """
        if "text_prompt" in cfg:
            val_input['captions'] = [cfg.text_prompt]
        
        # Example prompts:
        # ['A driving scene image. Snow. Winter. Snow covered road. Snowfall. Snow blankets. Snow are everywhere. Snow covers all grasses and trees.']
        # ['A driving scene image at boston-seaport. night, clear, downtown, straight road, white buildings, construction zone.']
        # ['delicious cake']
        # ['Snowy weather. Photo-realistic.']
        # ['Heavy snow. Snowy weather. Photo-realistic.']
        # ['Snow storm. Blizzard. Snowy weather. Photo-realistic.']
        # ['Snowy weather at night. Photo-realistic.']
        # ['Foggy weather. Photo-realistic.']
        # ['Dusty road. Photo-realistic.']
        # ['Dusty weather. Photo-realistic.']
        # ['Desert. Photo-realistic.']
        # ['Ocean. Photo-realistic.']
        # ['Amazon forest. Photo-realistic.']
        # ['night']
        
        return_tuples = run_one_batch(cfg, pipe, val_input, weight_dtype,
                                      transparent_bg=cfg.transparent_bg,
                                      map_size=target_map_size)
        
        # for saving caption
        with open(os.path.join(cfg.log_root, "captions.txt"), "a") as txt_file:
            if isinstance(val_input['captions'][0], str) and len(val_input['captions'][0]) > 0:
                txt_file.write(val_input['captions'][0] + "\n")
            else:
                print(val_input['captions'][0])

        for map_img, ori_imgs, ori_imgs_wb, gen_imgs_list, gen_imgs_wb_list in zip(*return_tuples):
            # save map
            if "save_map" in cfg and cfg.save_map:
                map_img.save(
                    os.path.join(
                        cfg.log_root,
                        "frames",
                        f"{curr_index}_{batch_img_index}_map.png",
                    )
                )

            #####################################################################
            # save occ
            # val_input['occupancy'][:, :, 0] : depth
            # val_input['occupancy'][:, :, 1] : occ
            
            if "occ_only_loader" in cfg and cfg.occ_only_loader:
                for idx in range(16):
                    val_input['occupancy'][:,:,idx] = val_input['occupancy'][:,:,idx] * (idx+1)
                occupancy, _ = val_input['occupancy'].max(dim=2)
            else:
                occupancy = val_input['occupancy'][:,:,1:17].clone()
                for idx in range(16):
                    occupancy[:,:,idx] = occupancy[:,:,idx] * (idx+1)
                occupancy, _ = occupancy.max(dim=2)
            
            n_views = occupancy.shape[1]


            if not save_only_gen:
                label_imgs = []
                label_blended_ori_imgs = []
                label_blended_gen_imgs = []
                for i_view in range(n_views):
                    label_img = occupancy[batch_img_index, i_view]
                    ori_img = np.array(ori_imgs[i_view])
                    assert len(gen_imgs_list) == 1, "only support single frame currently."
                    gen_img = np.array(gen_imgs_list[0][i_view])
                    H, W = ori_img.shape[:2]
                    
                    # Create rgb label image
                    label_img_rgb = np.zeros((H, W, 3), dtype=np.uint8)
                    for label, color in labels16_to_color.items():
                        label_img_rgb[label_img == label] = (color[0], color[1], color[2]) # Already RGB format
                    
                    label_imgs.append(Image.fromarray(label_img_rgb))
                    
                    # Blend label image
                    label_blended_ori_img = cv2.addWeighted(ori_img, 0.7, label_img_rgb, 0.3, 0)
                    label_blended_ori_imgs.append(Image.fromarray(label_blended_ori_img))
                    label_blended_gen_img = cv2.addWeighted(gen_img, 0.7, label_img_rgb, 0.3, 0)
                    label_blended_gen_imgs.append(Image.fromarray(label_blended_gen_img))
                    
                # save label images
                label_img = output_func(label_imgs)
                save_path = os.path.join(
                    cfg.log_root,
                    "frames",
                    f"{curr_index}_{batch_img_index}_occ.png",
                )
                label_img.save(save_path)
                
                label_blended_ori_img = output_func(label_blended_ori_imgs)
                save_path = os.path.join(
                    cfg.log_root,
                    "frames",
                    f"{curr_index}_{batch_img_index}_ori_occ.png",
                )
                label_blended_ori_img.save(save_path)

                label_blended_gen_img = output_func(label_blended_gen_imgs)
                save_path = os.path.join(
                    cfg.log_root,
                    "frames",
                    f"{curr_index}_{batch_img_index}_gen0_occ.png",
                )
                label_blended_gen_img.save(save_path)

                #####################################################################
                # save distance
                distance = val_input['occupancy'][:,:,0]
                dist_rgb_imgs = []
                dist_blended_ori_imgs = []
                dist_blended_gen_imgs = []
                for i_view in range(n_views):
                    dist_img = distance[batch_img_index, i_view]
                    ori_img = np.array(ori_imgs[i_view])
                    assert len(gen_imgs_list) == 1, "only support single frame currently."
                    gen_img = np.array(gen_imgs_list[0][i_view])
                    
                    # Create rgb dist image
                    dist_img_processed = dist_img.cpu().numpy().copy()
                    dist_img_processed[~(dist_img > 0)] = max_dist
                    dist_img_processed = np.clip(dist_img_processed, 0, max_dist)
                    dist_img_processed = (dist_img_processed / max_dist * 255).astype(np.uint8)
                    dist_img_rgb = cv2.applyColorMap(dist_img_processed, cv2.COLORMAP_JET)
                    
                    dist_rgb_imgs.append(Image.fromarray(dist_img_rgb))
                    
                    # Blend dist image
                    dist_blended_ori_img = cv2.addWeighted(ori_img, 0.7, dist_img_rgb, 0.3, 0)
                    dist_blended_ori_imgs.append(Image.fromarray(dist_blended_ori_img))
                    dist_blended_gen_img = cv2.addWeighted(gen_img, 0.7, dist_img_rgb, 0.3, 0)
                    dist_blended_gen_imgs.append(Image.fromarray(dist_blended_gen_img))
                    
                dist_only_img = output_func(dist_rgb_imgs)
                save_path = os.path.join(
                    cfg.log_root,
                    "frames",
                    f"{curr_index}_{batch_img_index}_dist.png",
                )
                dist_only_img.save(save_path)    
                
                dist_blended_ori_img = output_func(dist_blended_ori_imgs)
                save_path = os.path.join(
                    cfg.log_root,
                    "frames",
                    f"{curr_index}_{batch_img_index}_ori_dist.png",
                )
                dist_blended_ori_img.save(save_path)
                
                dist_blended_gen_img = output_func(dist_blended_gen_imgs)
                save_path = os.path.join(
                    cfg.log_root,
                    "frames",
                    f"{curr_index}_{batch_img_index}_gen0_dist.png",
                )
                dist_blended_gen_img.save(save_path)
            
                #####################################################################

                # save ori
                if ori_imgs is not None:
                    ori_img = output_func(ori_imgs)
                    save_path = os.path.join(
                        cfg.log_root,
                        "frames",
                        f"{curr_index}_{batch_img_index}_ori.png",
                    )
                    ori_img.save(save_path)
                    ori_img_paths.append(save_path)

                if cfg.show_box_on_img or cfg.show_map_on_img:
                    # save ori with box
                    if ori_imgs_wb is not None:
                        ori_img_with_box = output_func(ori_imgs_wb)
                        ori_img_with_box.save(
                            os.path.join(
                                cfg.log_root,
                                "frames",
                                f"{curr_index}_{batch_img_index}_ori_box.png",
                            )
                        )
                    # save gen with box
                    for ti, gen_imgs_wb in enumerate(gen_imgs_wb_list):
                        gen_img_with_box = output_func(gen_imgs_wb)
                        gen_img_with_box.save(
                            os.path.join(
                                cfg.log_root,
                                "frames",
                                f"{curr_index}_{batch_img_index}_gen{ti}_box.png",
                            )
                        )
            
            # save gen
            for ti, gen_imgs in enumerate(gen_imgs_list):
                gen_img = output_func(gen_imgs)
                save_path = os.path.join(
                    cfg.log_root,
                    "frames",
                    f"{curr_index}_{batch_img_index}_gen{ti}.png",
                )
                gen_img.save(save_path)
                if ti in gen_img_paths:
                    gen_img_paths[ti].append(save_path)
                else:
                    gen_img_paths[ti] = [save_path]
            
            # process ref img (indentation is important)
            gen_imgs = gen_imgs_list[0]
            ref_image_list = []
            for cam_i in range(6):
                img_i = gen_imgs[cam_i]
                img_i = np.array(img_i)
                img_i = transform1(img_i)
                ref_image_list.append(img_i)
            gen_ref = torch.stack(ref_image_list)
            gen_ref = gen_ref.to(
                memory_format=torch.contiguous_format).float()

            batch_img_index += 1
        # update bar
        progress_bar.update(cfg.runner.validation_times)


if __name__ == "__main__":
    main()
