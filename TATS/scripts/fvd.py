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

from tats.fvd.fvd import (frechet_distance, get_fvd_logits, load_fvd_model,
                          polynomial_mmd)

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


class ArenaGeneratedDataset(Dataset):
    """
    Dataset class for loading organized images from the Arena dataset.
    
    The dataset structure is expected to be:
    dataroot/
        clip_000000/
            CAM_FRONT_LEFT/
                000000.jpg
                ...
            CAM_FRONT/
                000000.jpg
                ...
            ...
        clip_000001/
            ...
    
    Returns a dictionary with camera views as keys and stacked image arrays as values.
    """
    
    def __init__(self, dataroot):
        """
        Initialize the dataset.
        
        Args:
            dataroot (str or Path): Path to the root directory containing clip folders
        """
        self.dataroot = Path(dataroot)
        
        # Find all clip directories
        self.clip_dirs = sorted([
            d for d in self.dataroot.iterdir() 
            if d.is_dir() and d.name.startswith("clip_")
        ], key=lambda x: natural_sort_key(x.name))
        
        print(f"Found {len(self.clip_dirs)} clips in {dataroot}")
    
    def __len__(self):
        """Return the number of clips in the dataset."""
        return len(self.clip_dirs)
    
    def __getitem__(self, idx):
        """
        Return images from all cameras for the specified clip index.
        
        Args:
            idx (int): Clip index
        
        Returns:
            dict: Dictionary with camera names as keys and image arrays (T, H, W, C) as values
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} clips")
        
        clip_dir = self.clip_dirs[idx]
        result = {}
        
        # Process each camera view
        for camera in CAMERAS:
            camera_dir = clip_dir / camera
            
            if not camera_dir.exists():
                raise ValueError(f"Camera directory {camera} not found in clip {clip_dir.name}")
            
            # Get all images for this camera and clip
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(camera_dir.glob(ext))
            
            # Sort images by filename
            images = sorted(images, key=lambda x: natural_sort_key(x.name))
            
            if not images:
                raise ValueError(f"No images found in {camera_dir}")
            
            # Load images into numpy array
            frames = []
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError(f"Failed to load image at {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frames.append(img)
            
            # Stack frames into (T, H, W, C) format
            frames = np.stack(frames, axis=0)
            result[camera] = frames
        
        return result


def crop_image(image, crop_size):
    """Crop the image to the center square of size crop_size."""
    h, w, _ = image.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return image[start_h:start_h + crop_size, start_w:start_w + crop_size]


def pad_image(image):
    """Pad the image to make it square with target size."""
    h, w, _ = image.shape
    pad_size = abs(h - w) // 2
    if h > w:
        return np.pad(image, ((0, 0), (pad_size, pad_size), (0, 0)), mode='constant')
    else:
        return np.pad(image, ((pad_size, pad_size), (0, 0), (0, 0)), mode='constant')


def crop_video(video, crop_size):
    """Crop each frame of the video to the center square of size crop_size.
    
    Args:
        video: Video tensor of shape (B, T, H, W, C)
        crop_size: Size of the square crop
        
    Returns:
        Cropped video of shape (B, T, crop_size, crop_size, C)
    """
    b, t, h, w, c = video.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return video[:, :, start_h:start_h + crop_size, start_w:start_w + crop_size, :]


def pad_video(video):
    """Pad each frame of the video to make it square.
    
    Args:
        video: Video tensor of shape (B, T, H, W, C)
        
    Returns:
        Padded video with square frames
    """
    b, t, h, w, c = video.shape
    pad_size = abs(h - w) // 2
    
    if h > w:
        # Pad width dimension
        padded = np.zeros((b, t, h, h, c), dtype=video.dtype)
        padded[:, :, :, pad_size:pad_size+w, :] = video
    else:
        # Pad height dimension
        padded = np.zeros((b, t, w, w, c), dtype=video.dtype)
        padded[:, :, pad_size:pad_size+h, :, :] = video
    
    return padded


def natural_sort_key(s):
    """Sort strings with numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]


def main(args):
    # 1. Init
    device = torch.device("cuda")
    if args.batch_size != 1:
        raise ValueError("Batch size must be 1 for FVD computation.")
    if args.crop and args.pad:
        raise ValueError("Cannot crop and pad at the same time.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    
    # 2. Data
    nusc_ds = MinimalNuScenesDataset(args.nusc_dataroot, args.splits)
    nusc_dl = DataLoader(nusc_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    
    # 3. Model
    i3d = load_fvd_model(device)
    
    
    # 4. Real video embeddings
    if args.load_real_embeddings_from is not None:  # Load precomputed embeddings
        print(f"Loading real embeddings from {args.load_real_embeddings_from}")
        real_embeddings = np.load(args.load_real_embeddings_from)
        real_embeddings = {camera: real_embeddings[camera] for camera in CAMERAS}
        print("Real embeddings loaded successfully.")
        
    else:  # Compute real video embeddings
        print("Computing fvd embeddings for real videos")
        
        num_samples = []
        real_embeddings = {camera: [] for camera in CAMERAS}
        for batch in tqdm(nusc_dl):
            scene_tokens = batch["token"]
            scene_splits = batch["split"]
            
            batch_videos = {camera: [] for camera in CAMERAS}
            for scene_token, scene_split in zip(scene_tokens, scene_splits):
                nusc = nusc_ds.nusc_trainval if scene_split in ["train", "val"] else nusc_ds.nusc_test
                scene = nusc.get("scene", scene_token)

                for i in range(args.num_videos_per_scene):
                    # initialize frame
                    start_frame = i * args.shift_stride
                    if start_frame + args.max_frames > scene["nbr_samples"]:
                        break
                    cur_frame = start_frame
                    
                    # initialize sample token
                    sample_token = scene["first_sample_token"]
                    for j in range(start_frame):
                        sample = nusc.get("sample", sample_token)
                        sample_token = sample["next"]
                        
                    scene_videos = {camera: [] for camera in CAMERAS}
                    while cur_frame < start_frame + args.max_frames and sample_token != "":
                        sample = nusc.get("sample", sample_token)
                        
                        for camera in CAMERAS:
                            # Get the sample data for the camera
                            sample_data = nusc.get("sample_data", sample["data"][camera])
                            image_path = nusc.get_sample_data_path(sample_data["token"])
                            image = iio.imread(image_path)
                            if args.crop:
                                image = crop_image(image, min(image.shape[-3], image.shape[-2]))
                            
                            elif args.pad:
                                image = pad_image(image)
                                
                            image = torch.from_numpy(image).to(device)
                            
                            # Store the image in the sample_videos dictionary
                            scene_videos[camera].append(image)  # Append (H, W, C) to scene_videos[camera]
                        
                        # Get next sample token
                        sample_token = sample["next"]
                        
                        cur_frame += 1
                    
                    scene_videos = {camera: torch.stack(scene_videos[camera], dim=0) for camera in CAMERAS}
                    for camera in CAMERAS:
                        batch_videos[camera].append(scene_videos[camera])  # Append (T, H, W, C) to batch_videos[camera].
                
                num_samples.append(scene["nbr_samples"])

            for camera in CAMERAS:
                video_batch = torch.stack(batch_videos[camera], dim=0)  # (B, T, H, W, C)
                real_embeddings[camera].append(get_fvd_logits(video_batch.cpu().numpy(), i3d=i3d, device=device))
            
        print(f"Total number of samples in real video set: {sum(num_samples)}")

        print("Concat fvd embeddings for real videos")
        real_embeddings = {camera: torch.cat(real_embeddings[camera], dim=0).cpu().numpy() for camera in CAMERAS}
        
        output_file = Path(args.output_dir) / f"{'_'.join(args.splits)}_real_embeddings_max{args.max_frames}.npz"
        np.savez_compressed(output_file, **real_embeddings)
        
    
    # 5. Fake video embeddings
    if args.load_fake_embeddings_from is not None:  # Load precomputed embeddings
        print(f"Loading fake embeddings from {args.load_fake_embeddings_from}")
        fake_embeddings = np.load(args.load_fake_embeddings_from)
        fake_embeddings = {camera: fake_embeddings[camera] for camera in CAMERAS}
        print("Fake embeddings loaded successfully.")
        
    else:
        print("Computing fvd embeddings for fake videos")
        
        # Initialize Arena dataset
        arena_ds = ArenaGeneratedDataset(args.arena_dataroot)
        arena_dl = DataLoader(arena_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        fake_embeddings = {camera: [] for camera in CAMERAS}
        # Process fake videos
        for batch in tqdm(arena_dl):
            for camera in CAMERAS:
                video_batch = batch[camera]  # Shape (B, T, H, W, C)

                for i in range(args.num_videos_per_scene):
                    start_frame = i * args.shift_stride
                    if start_frame + args.max_frames > video_batch.shape[1]:
                        break
                    video = video_batch[:, start_frame:start_frame + args.max_frames]
                    # Apply crop or pad if specified
                    if args.crop:
                        video = crop_video(video.numpy(), min(video.shape[-3], video.shape[-2]))
                    elif args.pad:
                        video = pad_video(video.numpy())
                
                    # Get embeddings for this camera view
                    embeddings = get_fvd_logits(video, i3d=i3d, device=device)
                    fake_embeddings[camera].append(embeddings)
            
        print("Concat fvd embeddings for fake videos")
        fake_embeddings = {camera: torch.cat(fake_embeddings[camera], dim=0).cpu().numpy() for camera in CAMERAS}
        
        output_file = Path(args.output_dir) / f"fake_embeddings_max{args.max_frames}.npz"
        np.savez_compressed(output_file, **fake_embeddings)
        
        
    # 6. Compute FVD and KVD
    fake_embeddings_all = torch.cat([torch.from_numpy(fake_embeddings[camera]) for camera in args.cameras_to_evaluate], dim=0)
    real_embeddings_all = torch.cat([torch.from_numpy(real_embeddings[camera]) for camera in args.cameras_to_evaluate], dim=0)
    print("FVD = %.2f"%(frechet_distance(fake_embeddings_all, real_embeddings_all)))
    print("KVD = %.2f"%(polynomial_mmd(fake_embeddings_all.cpu(), real_embeddings_all.cpu())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["val"],
        choices=["train", "val", "test"],
        help="List of splits to process.",
    )
    parser.add_argument(
        "--nusc_dataroot",
        type=str,
        default="data/nuscenes",
        help="Root directory of the NuScenes dataset.",
    )
    parser.add_argument(
        "--arena_dataroot",
        type=str,
        default="data/arena",
        help="Root directory of the Arena dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the output.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing videos.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=2048,
        help="Maximum number of frames to process per video.",
    )
    parser.add_argument(
        "--num_videos_per_scene",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--shift_stride",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--cameras_to_evaluate",
        type=str,
        nargs="+",
        default=CAMERAS,
        choices=CAMERAS,
        help="List of cameras to evaluate.",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        default=False,
        help="Whether to crop the images to match square aspect ratio.",
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        default=False,
        help="Whether to pad the images to match square aspect ratio.",
    )
    parser.add_argument(
        "--load_real_embeddings_from",
        type=str,
        default=None,
        help="Path to load precomputed real embeddings from.",
    )
    parser.add_argument(
        "--load_fake_embeddings_from",
        type=str,
        default=None,
        help="Path to load precomputed fake embeddings from.",
    )
    args = parser.parse_args()
    main(args)
