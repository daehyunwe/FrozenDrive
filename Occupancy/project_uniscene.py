import argparse
import concurrent.futures
import re
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import yaml
# from mayavi import mlab
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from tqdm import tqdm

from utils.voxel_traversal import batch_range_filtered_raycast_first_hit

###############################################################################################################################

placeholder = None
import cv2  # If cv2 is imported before mayavi, it will cause an error.

################################################################################################################################

# Set these paths as needed
NUSCENES_ROOT = 'PATH/TO/NUSCENES'  # <-- CHANGE THIS
NUSCENES_VERSION = 'advanced_12Hz_trainval'
OCCUPANCY_DIR = './uniscene/val_occupancy_800_down_0.1/dense_voxels_with_semantic'
LABEL_MAPPING = './project/config/label_mapping/nuscenes.yaml'

# More parameters
num_workers = 10
save_images = True
cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
point_cloud_range = [-50.0, -50.0, -5, 50.0, 50.0, 3]
voxel_size = 0.125
img_size = np.array([448, 800])
ignore_config = {
    "label_type": "labels_16",  # "labels_16" or "labels"
    "labels": [0]
}
filter_config = {
    "filter_type": "min_dist",  # "min_dist" or "xyz_mask"
    "min_dist": 0.5
}
max_dist = 63.0
classname_to_color = {  # RGB.
        "noise": (0, 0, 0),  # Black.
        "animal": (70, 130, 180),  # Steelblue
        "human.pedestrian.adult": (0, 0, 230),  # Blue
        "human.pedestrian.child":(0, 0, 230),  # Skyblue,
        "human.pedestrian.construction_worker":(0, 0, 230),  # Cornflowerblue
        "human.pedestrian.personal_mobility": (0, 0, 230),  # Palevioletred
        "human.pedestrian.police_officer":(0, 0, 230),  # Navy,
        "human.pedestrian.stroller": (0, 0, 230),  # Lightcoral
        "human.pedestrian.wheelchair": (0, 0, 230),  # Blueviolet
        "movable_object.barrier": (112, 128, 144),  # Slategrey
        "movable_object.debris": (112, 128, 144),  # Chocolate
        "movable_object.pushable_pullable":(112, 128, 144),  # Dimgrey
        "movable_object.trafficcone":(112, 128, 144),  # Darkslategrey
        "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
        "vehicle.bicycle": (220, 20, 60),  # Crimson
        "vehicle.bus.bendy":(255, 158, 0),  # Coral
        "vehicle.bus.rigid": (255, 158, 0),  # Orangered
        "vehicle.car": (255, 158, 0),  # Orange
        "vehicle.construction":(255, 158, 0),  # Darksalmon
        "vehicle.emergency.ambulance":(255, 158, 0),
        "vehicle.emergency.police": (255, 158, 0),  # Gold
        "vehicle.motorcycle": (255, 158, 0),  # Red
        "vehicle.trailer":(255, 158, 0),  # Darkorange
        "vehicle.truck": (255, 158, 0),  # Tomato
        "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
        "flat.other":(0, 207, 191),
        "flat.sidewalk": (75, 0, 75),
        "flat.terrain": (0, 207, 191),
        "static.manmade": (222, 184, 135),  # Burlywood
        "static.other": (0, 207, 191),  # Bisque
        "static.vegetation": (0, 175, 0),  # Green
        "vehicle.ego": (255, 240, 245)
    }
classname16_to_color = {  # Custom defined colors for 16 classes.
        "noise": (0, 0, 0),  # Black.
        "barrier": (112, 128, 144),  # Slategrey
        "bicycle": (220, 20, 60),  # Crimson
        "bus": (255, 158, 0),  # Coral
        "car": (255, 158, 0),  # Orange
        "construction_vehicle": (255, 158, 0),  # Darksalmon
        "motorcycle": (255, 158, 0),  # Red
        "pedestrian": (0, 0, 230),  # Blue
        "traffic_cone": (112, 128, 144),  # Slategrey
        "trailer": (255, 158, 0),  # Darkorange
        "truck": (255, 158, 0),  # Tomato
        "driveable_surface": (0, 207, 191),  # nuTonomy green
        "other_flat": (0, 207, 191),  # Bisque
        "sidewalk": (75, 0, 75),  # Purple
        "terrain": (0, 207, 191),  # nuTonomy green
        "manmade": (222, 184, 135),  # Burlywood
        "vegetation": (0, 175, 0),  # Green
}

################################################################################################################################

# Loaded/Derived parameters
OUTPUT_IMG_DIR = f'./output/proj_occ_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_ROOT, verbose=True)
with open(LABEL_MAPPING, 'r') as stream:
    nuscenes_yaml = yaml.safe_load(stream)
labels_to_classname = nuscenes_yaml["labels"]
labels_16_to_classname = nuscenes_yaml["labels_16"]
learning_map = nuscenes_yaml["learning_map"]
labels_to_color = {k:classname_to_color[v] for k, v in labels_to_classname.items()}
labels16_to_color = {k:classname16_to_color[v] for k, v in labels_16_to_classname.items()}
voxel_shape=(int((point_cloud_range[3]-point_cloud_range[0])/voxel_size),
             int((point_cloud_range[4]-point_cloud_range[1])/voxel_size),
             int((point_cloud_range[5]-point_cloud_range[2])/voxel_size))
ignored_labels = ignore_config["labels"]
learning_map_np = -np.ones(len(learning_map) + 1, dtype=np.int8)
for k, v in learning_map.items():
    learning_map_np[k] = v

# Save parameters
Path(OUTPUT_IMG_DIR).mkdir(parents=True, exist_ok=True)
with open(OUTPUT_IMG_DIR + '/params.yaml', 'w') as f:
    yaml.dump({
        "nuscenes_root": NUSCENES_ROOT,
        "nuscenes_version": NUSCENES_VERSION,
        "occupancy_dir": OCCUPANCY_DIR,
        "label_mapping": LABEL_MAPPING,
        "num_workers": num_workers,
        "save_images": save_images,
        "cameras": cameras,
        "point_cloud_range": point_cloud_range,
        "voxel_size": voxel_size,
        "img_size": img_size.tolist(),
        "classname_to_color": classname_to_color,
        "ignore_config": ignore_config,
        "filter_config": filter_config,
        "max_dist": max_dist,
    }, f, default_flow_style=False)

################################################################################################################################
# External functions

def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: "lidar".

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    # if os.getcwd() in data_path:  # path from lyftdataset is absolute path
    #     data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    ).squeeze(0)
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: np.ndarray = np.eye(3),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation matrix (3x3).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

################################################################################################################################
# Backup old implementations

def remove_far(points, point_cloud_range):
    mask = (points[:, 0]>point_cloud_range[0]) & (points[:, 0]<point_cloud_range[3]) & (points[:, 1]>point_cloud_range[1]) & (points[:, 1]<point_cloud_range[4]) \
            & (points[:, 2]>point_cloud_range[2]) & (points[:, 2]<point_cloud_range[5])
    return points[mask, :]


def voxelize_original(voxel: np.array, label_count: np.array):
    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                if label_count[x, y, z] == 0:
                    continue
                labels = voxel[x, y, z]
                try:
                    label_count[x, y, z] = np.argmax(np.bincount(labels[labels!=0]))
                except:
                    label_count[x, y, z] = 0
    return label_count


def points2voxel_original(points, voxel_shape, voxel_size, max_points=5, specific_category=None):
    t0 = time()
    
    voxel = np.zeros((*voxel_shape, max_points), dtype=np.int64)
    label_count = np.zeros((voxel_shape), dtype=np.int64)
    index = points[:, 4].argsort()
    points = points[index]
    
    for point in points:
        x, y, z = point[0], point[1], point[2]
        x = round((x - point_cloud_range[0]) / voxel_size)
        y = round((y - point_cloud_range[1]) / voxel_size)
        z = round((z - point_cloud_range[2]) / voxel_size)

        try:
            voxel[x, y, z, label_count[x, y, z]] = int(point[4])  # map_label[int(point[4])]
            label_count[x, y, z] += 1
        except:
            continue
        
    t1 = time()
    
    voxel = voxelize(voxel, label_count)
    voxel = voxel.astype(np.float64)

    t2 = time()
    print(f"{t1 - t0:.4f}s, {t2 - t1:.4f}s, total: {t2 - t0:.4f}s")

    return voxel


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """
    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def draw_mayavi(
    voxel,
    voxel_size=0.2,
    output_path="temp.png",
    colormap="viridis",
    opacity=1.0,
    flip_z_axis=False,
):
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxel.shape[0], voxel.shape[1], voxel.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxel.reshape(-1)]).T

    grid_voxels = grid_coords[
        (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)
    ]

    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))
    
    plt_plot = mlab.points3d(
        grid_voxels[:, 0],
        grid_voxels[:, 1],
        grid_voxels[:, 2],
        grid_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )
    
    colors = np.array(list(classname_to_color.values())).astype(np.uint8)
    alpha = np.ones((colors.shape[0], 1), dtype=np.uint8) * 255
    colors = np.hstack([colors, alpha])

    plt_plot.glyph.scale_mode = "scale_by_vector"

    plt_plot.module_manager.scalar_lut_manager.lut.table = colors
    plt_plot.module_manager.scalar_lut_manager.data_range = [0, 31]
    # mlab.savefig("temp.png")


################################################################################################################################
# New implementations

def voxelize(voxel: np.array, label_count: np.array):
    output = np.zeros_like(label_count)
    nonzero_idx = np.argwhere(label_count > 0)
    
    for x, y, z in nonzero_idx:
        labels = voxel[x, y, z][:label_count[x, y, z]]
        labels = labels[labels != 0]
        if labels.size > 0:
            output[x, y, z] = np.argmax(np.bincount(labels))
    
    return output


def points2voxel(points, voxel_shape, voxel_size, max_points=5, specific_category=None, verbose=False):
    t0 = time()
    if isinstance(voxel_shape, tuple):
        voxel_shape = np.array(voxel_shape)
    
    # Sort points by label
    index = points[:, 4].argsort()
    points = points[index]
    
    # Filter valid points
    indices = np.round((points[:, :3] - point_cloud_range[:3]) / voxel_size).astype(np.int64)  # (N, 3)
    valid_mask = np.all((indices >= 0) & (indices < voxel_shape[None, ...]), axis=1)  # (N, )
    points = points[valid_mask]  # (M, 5)
    indices = indices[valid_mask]  # (M, 3)
    
    # Construct voxel grid
    voxel = np.zeros((*voxel_shape, max_points), dtype=np.int64)  # (X, Y, Z, max_points)
    label_count = np.zeros((voxel_shape), dtype=np.int64)  # (X, Y, Z)

    for idx, point in zip(indices, points):
        x, y, z = idx
        if label_count[x, y, z] < max_points:
            voxel[x, y, z, label_count[x, y, z]] = int(point[4])
            label_count[x, y, z] += 1
    
    t1 = time()
    voxel = voxelize(voxel, label_count)
    voxel = voxel.astype(np.float64)
    
    t2 = time()
    
    if verbose:
        print(f"{t1 - t0:.4f}s, {t2 - t1:.4f}s, total: {t2 - t0:.4f}s")
    
    return voxel


def draw(
    voxel,
    R,
    t,
    intrinsic,
    img_size,
    voxel_size=0.2,
    output_path="temp.png",
):
    H, W = img_size
    scale_x, scale_y = img_size[0] / 900, img_size[1] / 1600
    K = intrinsic.copy()
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y
    c2w = transform_matrix(t, R, inverse=False)
    
    rays_o, rays_d = get_rays_np(H, W, K, c2w[:3,:4])
    
    start = rays_o[0, 0].copy()
    start[0] = start[0] - point_cloud_range[0]
    start[1] = start[1] - point_cloud_range[1]
    start[2] = start[2] - point_cloud_range[2]

    occupancy_grid = np.logical_and.reduce(
        [voxel != c for c in ignored_labels]
    )
    
    starts = np.tile(start, (H*W, 1))
    directions = rays_d.reshape(H*W, 3)
    if filter_config["filter_type"] == "min_dist":
        first_hits, distances = batch_range_filtered_raycast_first_hit(
            starts,
            directions,
            occupancy_grid,
            voxel_size,
            min_dist=filter_config["min_dist"],
        )
    else:
        raise ValueError("Invalid filter type")
    first_hits = first_hits.reshape(H, W, 3)
    distances = distances.reshape(H, W)
    
    mask = np.all(first_hits != -1, axis=-1)
    semantic_map = voxel[first_hits[..., 0], first_hits[..., 1], first_hits[..., 2]]
    semantic_map[~mask] = -1  # Void as -1
    semantic_map = semantic_map.astype(np.int8)
    
    processed_distances = distances.copy()
    processed_distances[~mask] = max_dist
    processed_distances = np.clip(processed_distances, 0, max_dist)
    processed_distances = (processed_distances / max_dist * 255).astype(np.uint8)
    distance_image = cv2.applyColorMap(processed_distances, cv2.COLORMAP_JET)
    
    semantic_image = np.zeros((H, W, 3), dtype=np.uint8)
    for label, color in labels16_to_color.items():
        mask = semantic_map == label
        semantic_image[mask] = (color[2], color[1], color[0])  # OpenCV uses BGR format
    
    np.savez_compressed(
        output_path.parent / output_path.name.replace("occ_", "proj_").replace(".png", ".npz"),
        labels=semantic_map,
        distances=distances,
    )
    if save_images:
        cv2.imwrite(str(output_path), semantic_image)
        cv2.imwrite(str(output_path.parent / output_path.name.replace("occ_", "dist_")), distance_image)


def draw_with_info(
    nusc,
    sample_info,
):
    scene_token = sample_info["scene_token"]
    sample_token = sample_info["sample_token"]
    output_path = sample_info["output_path"]
    output_path.mkdir(parents=True, exist_ok=True)
    
    sample = nusc.get("sample", sample_token)
    
    # Points to voxel
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    lidar_path = nusc.get_sample_data_path(lidar_token)
    occ_path = Path(OCCUPANCY_DIR) / sample_token / f"{lidar_token}.npy"
    
    points = np.load(occ_path, allow_pickle=True)  # (N, 4)
    voxel = np.zeros(voxel_shape, dtype=np.int64)  # (X, Y, Z)
    indices = points[:, :3].astype(np.int64)
    values = points[:, 3].astype(np.int64)
    assert np.all((indices >= 0) & (indices < np.array(voxel.shape))), "Indices out of bounds"
    voxel[indices[:, 0], indices[:, 1], indices[:, 2]] = values
    
    # Project to images
    for cam in cameras:
        cam_token = sample["data"][cam]
        cam_data = nusc.get("sample_data", cam_token)
        cam_sensor = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        lidar_sensor = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
        
        l2e_r = lidar_sensor["rotation"]
        l2e_t = (lidar_sensor["translation"],)
        e2g_r = ego_pose["rotation"]
        e2g_t = ego_pose["translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
    
        cam_info = obtain_sensor2top(
            nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
        )
        
        rot = cam_info["sensor2lidar_rotation"]
        trans = cam_info["sensor2lidar_translation"]
        intrinsic = np.array(cam_sensor["camera_intrinsic"], dtype=np.float32).reshape(3, 3)
        
        draw(
            voxel, rot, trans, intrinsic, img_size=img_size, voxel_size=voxel_size,
            output_path=output_path / f"occ_{cam}.png"
        )
        
        # Save camera image
        if save_images:
            cam_path = nusc.get_sample_data_path(cam_token)
            cam_img = cv2.imread(cam_path)
            cv2.imwrite(str(output_path / f"img_{cam}.png"), cam_img)
    
    
################################################################################################################################

def run(start=0):
    all_sample_infos = []
    for scene in nusc.scene[start::num_workers]:
        scene_token = scene["token"]
        
        sample_token = scene["first_sample_token"]
        while sample_token != "":
            output_path = (Path(OUTPUT_IMG_DIR) / f"sample_{sample_token}")
            sample = nusc.get("sample", sample_token)
            
            sample_info = {
                "scene_token": scene_token,
                "sample_token": sample_token,
                "output_path": output_path,
            }
            all_sample_infos.append(sample_info)
            sample_token = sample["next"]

    pbar = tqdm(total=len(all_sample_infos), desc=f"Worker{start}", position=start % num_workers, dynamic_ncols=True)
    for sample_info in all_sample_infos:
        draw_with_info(nusc, sample_info)
        pbar.update(1)


if __name__ == "__main__":
    if num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for start in range(num_workers):
                executor.submit(run, start=start)
    else:
        run(start=0)
        