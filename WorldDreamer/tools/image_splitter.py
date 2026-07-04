import cv2
from pathlib import Path
import re


PART_MATCHING = {
    "part1": "CAM_FRONT_LEFT",
    "part2": "CAM_FRONT",
    "part3": "CAM_FRONT_RIGHT",
    "part4": "CAM_BACK_RIGHT",
    "part5": "CAM_BACK",
    "part6": "CAM_BACK_LEFT",
}


def split_image(image_path, output_dir, num_splits=4):
    """
    Split an image horizontally into N equal parts.
    
    Args:
        image_path (str or Path): Path to the input image
        output_dir (str or Path): Directory to save the split images
        num_splits (int): Number of horizontal splits to make
    
    Returns:
        list: Paths to the split images
    """
    # Convert paths to Path objects if they aren't already
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    
    # Get image name without extension
    img_name = image_path.stem
    # Extract frame number if it exists in the filename
    frame_number = None
    match = re.search(r'(\d+)', img_name)
    if match:
        frame_number = match.group(1)
    
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Calculate width of each split
    split_width = width // num_splits
    
    # Split and save each part
    output_paths = []
    for i in range(num_splits):
        start_x = i * split_width
        # Handle the last piece (might be slightly larger due to integer division)
        end_x = start_x + split_width if i < num_splits - 1 else width
        
        # Extract the slice
        img_slice = img[:, start_x:end_x]
        
        # Create part directory
        part_dir = output_dir / PART_MATCHING.get(f"part{i+1}", f"part{i+1}")
        part_dir.mkdir(exist_ok=True, parents=True)
        
        # Create output filename
        if frame_number:
            output_name = f"{int(frame_number):06d}.jpg"
        else:
            output_name = f"{img_name}_part{i+1}{image_path.suffix}"
        
        output_path = part_dir / output_name
        
        # Save the slice
        cv2.imwrite(str(output_path), img_slice)
        output_paths.append(output_path)
    
    return output_paths


src_path = Path("./dreamer-log/test/SDv1.5_mv_single_ref_.../frames")
target_path = Path("./dreamer-log/test/SDv1.5_mv_single_ref_.../frames_split")
def natural_sort_key(s):
    """Sort strings with numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]

img_paths = sorted(src_path.glob("*_gen0.png"), key=natural_sort_key)
for index, i_path in enumerate(img_paths):
    print(f"Splitting image {index+1}/{len(img_paths)}: {i_path}", end="\r")
    split_image(i_path, target_path, num_splits=6)
