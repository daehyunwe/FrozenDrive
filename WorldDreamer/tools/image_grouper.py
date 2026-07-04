import cv2
from pathlib import Path
import re
import shutil


PART_MATCHING = {
    "part1": "CAM_FRONT_LEFT",
    "part2": "CAM_FRONT",
    "part3": "CAM_FRONT_RIGHT",
    "part4": "CAM_BACK_RIGHT",
    "part5": "CAM_BACK",
    "part6": "CAM_BACK_LEFT",
}


def group_images(src_path, target_path, frame_count_path):
    """
    Group images from src_path into target_path based on frame counts.
    
    Args:
        src_path (Path): Source directory containing 6 subdirectories with images
        target_path (Path): Target directory to save the grouped images
        frame_count_path (Path): Path to file with frame counts for each clip
    """
    # Read the frame counts file
    clip_frame_counts = []
    with open(frame_count_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                frame_count = int(parts[1].strip())
                clip_frame_counts.append(frame_count)
    
    # Get images from each camera directory
    camera_images = {}
    for part_key, camera_dir in PART_MATCHING.items():
        src_dir = src_path / camera_dir
        if not src_dir.exists():
            print(f"Warning: Source directory {src_dir} not found, skipping")
            continue
        
        # Get all images in this directory
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images.extend(src_dir.glob(ext))
        images = sorted(images, key=natural_sort_key)
        
        if not images:
            print(f"Warning: No images found in {src_dir}, skipping")
            continue
        
        print(f"Found {len(images)} images in {camera_dir}")
        camera_images[camera_dir] = images
    
    # Process clips first, then cameras
    process_clips(camera_images, target_path, clip_frame_counts)


def process_clips(camera_images, target_path, clip_frame_counts):
    """
    Process clips with images for each camera.
    
    Args:
        camera_images (dict): Dictionary mapping camera names to lists of images
        target_path (Path): Target directory to save the grouped images
        clip_frame_counts (list): List of frame counts for each clip
    """
    # Track image index for each camera
    camera_indices = {camera: 0 for camera in camera_images}
    
    # Create clip directories and copy images from each camera
    for clip_index, frame_count in enumerate(clip_frame_counts):
        clip_dir_name = f"clip_{clip_index:06d}"
        clip_dir = target_path / clip_dir_name
        clip_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Processing {clip_dir_name} ({frame_count} frames)")
        
        # Process each camera for this clip
        for camera, images in camera_images.items():
            camera_dir = clip_dir / camera
            camera_dir.mkdir(exist_ok=True)
            
            img_index = camera_indices[camera]
            total_images = len(images)
            images_copied = 0
            
            # Copy the next frame_count images for this camera
            for i in range(frame_count):
                if img_index < total_images:
                    src_img = images[img_index]
                    dst_img = camera_dir / src_img.name
                    shutil.copy2(src_img, dst_img)
                    
                    img_index += 1
                    images_copied += 1
                else:
                    print(f"  - Warning: Not enough images for {camera}")
                    break
            
            print(f"  - Copied {images_copied} images to {camera}")
            camera_indices[camera] = img_index
    
    # Check for unused images
    for camera, images in camera_images.items():
        img_index = camera_indices[camera]
        if img_index < len(images):
            print(f"Warning: {len(images) - img_index} unused images for {camera}")


# Configuration paths
SRC_PATH = Path("./dreamer-log/test/SDv1.5_mv_single_ref_.../frames_split_interp")
TARGET_PATH = Path("./dreamer-log/test/SDv1.5_mv_single_ref_.../frames_split_interp_grouped")
FRAME_COUNT_PATH = Path("frame_counts.txt")


def natural_sort_key(s):
    """Sort strings with numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]


if __name__ == "__main__":
    group_images(SRC_PATH, TARGET_PATH, FRAME_COUNT_PATH)
