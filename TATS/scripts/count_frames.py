####################################################################################################

# Total number of frames in train set: 28130
# Total number of frames in val set: 6019

####################################################################################################
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import re
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# List of camera names in NuScenes
CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]


class MinimalNuScenesDataset(Dataset):
    def __init__(self, dataroot, splits):
        self.dataroot = dataroot
        self.splits = splits
        
        if "train" in self.splits or "val" in self.splits:
            self.nusc_trainval = NuScenes(version="v1.0-trainval", dataroot=dataroot, verbose=True)
            self.scene_name_to_token_trainval = {
                scene["name"]: scene["token"] for scene in self.nusc_trainval.scene
            }
            
        if "test" in self.splits:
            self.nusc_test = NuScenes(version="v1.0-test", dataroot=dataroot, verbose=True)
            self.scene_name_to_token_test = {
                scene["name"]: scene["token"] for scene in self.nusc_test.scene
            }
        
        scene_splits = create_splits_scenes()
        self.data = []
        for split in self.splits:
            if split in ["train", "val"]:
                scene_names = scene_splits[split]
                scene_tokens = [self.scene_name_to_token_trainval[name] for name in scene_names]
                self.data.extend([{"token": x, "split": split} for x in scene_tokens])
                
            elif split == "test":
                scene_names = scene_splits[split]
                scene_tokens = [self.scene_name_to_token_test[name] for name in scene_names]
                self.data.extend([{"token": x, "split": split} for x in scene_tokens])
                
            else:
                raise ValueError(f"Unknown split: {split}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def main():
    # 1. Init
    device = torch.device("cuda")
    
    
    # 2. Data
    nusc_ds = MinimalNuScenesDataset(Path("PATH_TO_NUSCENES"), ["train", "val"])
    nusc_dl = DataLoader(nusc_ds, batch_size=1, shuffle=False, num_workers=0)
    
    
    # 3. Count
    train_frames = 0
    val_frames = 0
    for i, batch in enumerate(nusc_dl):
        print(f"Processing batch {i+1}/{len(nusc_dl)}", end="\r")
        scene_token = batch["token"][0]
        scene_split = batch["split"][0]
        
        nusc = nusc_ds.nusc_trainval if scene_split in ["train", "val"] else nusc_ds.nusc_test
        scene = nusc.get("scene", scene_token)
        
        if scene_split == "train":
            train_frames += scene["nbr_samples"]
        elif scene_split == "val":
            val_frames += scene["nbr_samples"]
        else:
            raise ValueError(f"Unknown split: {scene_split}")

        breakpoint()
        
    print(f"Total number of frames in train set: {train_frames}")
    print(f"Total number of frames in val set: {val_frames}")

    

if __name__ == "__main__":
    main()
    