import logging
import os
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import PIL.ImageDraw as ImageDraw
from pyquaternion import Quaternion

from mmdet.datasets.builder import PIPELINES
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS

from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.database.maps_db.map_api import NuPlanMapWrapper
from nuplan.database.maps_db.map_explorer import NuPlanMapExplorer
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS
from PIL import Image

from mmdet3d.datasets.pipelines.loading_utils import (
    one_hot_decode,
)


def distance_to_depth(distance_map, intrinsic):
    H, W = distance_map.shape
    
    scale_x, scale_y = H / 900, W / 1600
    K = intrinsic.copy()
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y
    
    depth_map = distance_map.copy()
    mask = depth_map > 0
    
    H, W = distance_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    norm = np.sqrt(dx**2 + dy**2 + 1)

    depth_map = distance_map / norm
    depth_map[~mask] = -1
    
    return depth_map


@PIPELINES.register_module()
class LoadBEVSegmentationS:
    """This only loads map annotations
    In this map, the origin is at lower-left corner, with x-y transposed.
                          FRONT                             RIGHT
         Nuscenes                       transposed
        --------->  LEFT   EGO   RIGHT  ----------->  BACK   EGO   FRONT
           map                            output
                    (0,0)  BACK                       (0,0)  LEFT
    Guess reason, in cv2 / PIL coord, this is a BEV as follow:
        (0,0)  LEFT

        BACK   EGO   FRONT

              RIGHT
    All masks in np channel first format.
    """

    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
        cache_file: str = None,
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes
        self.lidar2canvas = np.array(
            [
                [canvas_h / patch_h, 0, canvas_h / 2],
                [0, canvas_w / patch_w, canvas_w / 2],
                [0, 0, 1],
            ]
        )
        
        self.maps = {}
        if "nuplan" in dataset_root.lower():
            pass
        else:
            for location in LOCATIONS:
                self.maps[location] = NuScenesMap(dataset_root, location)

        if cache_file and os.path.isfile(cache_file):
            logging.info(f"using data cache from: {cache_file}")
            # load to memory and ignore all possible changes.
            self.cache = cache_file
        else:
            self.cache = None
        # this should be set through main process afterwards
        self.shared_mem_cache = None

    def _load_from_cache(self, data: Dict[str, Any], cache_dict) -> Dict[str, Any]:
        token = data["token"]
        labels = one_hot_decode(
            cache_dict["gt_masks_bev"][token][:],
            len(self.classes),
        )
        data["gt_masks_bev"] = labels

        if not np.any(data["gt_masks_bev"]):
            print('gt_masks_bev are all 0')
        return data

    def _get_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        
        location = data["location"]
        if location in MAP_LOCATIONS:
            rotation_z_neg90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            lidar2global = ego2global @ lidar2ego
            lidar2global = rotation_z_neg90 @ lidar2global
        else:
            lidar2global = ego2global @ lidar2ego

        if "lidar_aug_matrix" in data:  # it is I if no lidar aux or no train
            lidar2point = data["lidar_aug_matrix"]
            point2lidar = np.linalg.inv(lidar2point)
            lidar2global = lidar2global @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])  # angle between v and x-axis
        patch_angle = yaw / np.pi * 180

        mappings = {}
        # cut semantics from nuscenesMap
        location = data["location"]
        if location in LOCATIONS:
            for name in self.classes:
                if name == "drivable_area*":
                    mappings[name] = ["road_segment", "lane"]
                elif name == "divider":
                    mappings[name] = ["road_divider", "lane_divider"]
                else:
                    mappings[name] = [name]
            layer_names = []
            for name in mappings:
                layer_names.extend(mappings[name])
            layer_names = list(set(layer_names))
            masks = self.maps[location].get_map_mask(
                patch_box=patch_box,
                patch_angle=patch_angle,
                layer_names=layer_names,
                canvas_size=self.canvas_size,
            )
        else:
            if location not in self.maps:
                mapdb = GPKGMapsDB("nuplan-maps-v1.0", f"{self.dataset_root}/maps")
                map_api = NuPlanMapWrapper(mapdb, map_name=location)
                map_explorer = NuPlanMapExplorer(map_api=map_api)
                self.maps[location] = map_explorer

            for name in self.classes:
                mappings[name] = [name]

            layer_names = []
            for name in mappings:
                layer_names.extend(mappings[name])
            layer_names = list(layer_names)
            masks = self.maps[location].get_map_mask(
                patch_box=patch_box,
                patch_angle=patch_angle,
                layer_names=self.classes,
                output_size=self.canvas_size,
            )

        masks = masks.transpose(0, 2, 1)  # TODO why need transpose here?
        masks = masks.astype(np.bool_)
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.int64)  # long)

        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1
        data["gt_masks_bev"] = labels
        return data

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # if set cache, use it.
        if self.cache is not None:
            try:
                with h5py.File(self.cache, "r") as cache_file:
                    return self._load_from_cache(data, cache_file)
            except:
                pass
        if self.shared_mem_cache is not None:
            try:
                return self._load_from_cache(data, self.shared_mem_cache)
            except:
                pass

        # cache miss, load normally
        data = self._get_data(data)

        # if set, add this item into it.
        if self.shared_mem_cache is not None:
            token = data["token"]
            for key in self.shared_mem_cache.keys():
                if key in data:
                    self.shared_mem_cache[key][token] = data[key]
        return data

@PIPELINES.register_module()
class LoadOccupancy:
    """This only loads map annotations
    In this map, the origin is at lower-left corner, with x-y transposed.
                          FRONT                             RIGHT
         Nuscenes                       transposed
        --------->  LEFT   EGO   RIGHT  ----------->  BACK   EGO   FRONT
           map                            output
                    (0,0)  BACK                       (0,0)  LEFT
    Guess reason, in cv2 / PIL coord, this is a BEV as follow:
        (0,0)  LEFT

        BACK   EGO   FRONT

              RIGHT
    All masks in np channel first format.
    
    NOTE: occupancy(not one-hot) + distance
    """

    def __init__(
        self,
        dataset_root: str,
        dataset_occ: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
        cache_file: str = None,
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes
        self.lidar2canvas = np.array(
            [
                [canvas_h / patch_h, 0, canvas_h / 2],
                [0, canvas_w / patch_w, canvas_w / 2],
                [0, 0, 1],
            ]
        )
        
        self.dataset_occ = dataset_occ
        
        self.maps = {}
        if "nuplan" in dataset_root.lower():
            pass
        else:
            for location in LOCATIONS:
                self.maps[location] = NuScenesMap(dataset_root, location)

        # if cache_file and os.path.isfile(cache_file):
        #     logging.info(f"using data cache from: {cache_file}")
        #     # load to memory and ignore all possible changes.
        #     self.cache = cache_file
        # else:
        self.cache = None
        # this should be set through main process afterwards
        self.shared_mem_cache = None

    def _load_from_cache(self, data: Dict[str, Any], cache_dict) -> Dict[str, Any]:
        token = data["token"]
        labels = one_hot_decode(
            cache_dict["gt_masks_bev"][token][:],
            len(self.classes),
        )
        data["gt_masks_bev"] = labels

        if not np.any(data["gt_masks_bev"]):
            print('gt_masks_bev are all 0')
        return data

    def _get_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        token = data["token"]
        sample_token = f"{self.dataset_occ}/sample_{token}/"
        views = ['proj_CAM_FRONT_LEFT.npz', 'proj_CAM_FRONT.npz', 'proj_CAM_FRONT_RIGHT.npz', 'proj_CAM_BACK_RIGHT.npz', 'proj_CAM_BACK.npz','proj_CAM_BACK_LEFT.npz', ]
        
        occupancies = []
        for view in views:
            view_sample_token = sample_token + view
            view_file = np.load(view_sample_token)
            labels = view_file['labels']
            distances = view_file['distances']
            occupancies.append(np.stack([distances, labels]))
        
        data["occupancy"] = occupancies
        return data

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # if set cache, use it.
        if self.cache is not None:
            try:
                with h5py.File(self.cache, "r") as cache_file:
                    return self._load_from_cache(data, cache_file)
            except:
                pass
        if self.shared_mem_cache is not None:
            try:
                return self._load_from_cache(data, self.shared_mem_cache)
            except:
                pass

        # cache miss, load normally
        data = self._get_data(data)

        # if set, add this item into it.
        if self.shared_mem_cache is not None:
            token = data["token"]
            for key in self.shared_mem_cache.keys():
                if key in data:
                    self.shared_mem_cache[key][token] = data[key]
        return data


@PIPELINES.register_module()
class LoadOccupancyOnlyOcc:
    """This only loads map annotations
    In this map, the origin is at lower-left corner, with x-y transposed.
                          FRONT                             RIGHT
         Nuscenes                       transposed
        --------->  LEFT   EGO   RIGHT  ----------->  BACK   EGO   FRONT
           map                            output
                    (0,0)  BACK                       (0,0)  LEFT
    Guess reason, in cv2 / PIL coord, this is a BEV as follow:
        (0,0)  LEFT

        BACK   EGO   FRONT

              RIGHT
    All masks in np channel first format.
    
    NOTE: occupancy(one-hot)
    """

    def __init__(
        self,
        dataset_root: str,
        dataset_occ: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
        cache_file: str = None,
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes
        self.lidar2canvas = np.array(
            [
                [canvas_h / patch_h, 0, canvas_h / 2],
                [0, canvas_w / patch_w, canvas_w / 2],
                [0, 0, 1],
            ]
        )
        
        self.dataset_occ = dataset_occ
        
        self.maps = {}
        if "nuplan" in dataset_root.lower():
            pass
        else:
            for location in LOCATIONS:
                self.maps[location] = NuScenesMap(dataset_root, location)

        # if cache_file and os.path.isfile(cache_file):
        #     logging.info(f"using data cache from: {cache_file}")
        #     # load to memory and ignore all possible changes.
        #     self.cache = cache_file
        # else:
        self.cache = None
        # this should be set through main process afterwards
        self.shared_mem_cache = None

    def _load_from_cache(self, data: Dict[str, Any], cache_dict) -> Dict[str, Any]:
        token = data["token"]
        labels = one_hot_decode(
            cache_dict["gt_masks_bev"][token][:],
            len(self.classes),
        )
        data["gt_masks_bev"] = labels

        if not np.any(data["gt_masks_bev"]):
            print('gt_masks_bev are all 0')
        return data
    
    def mask_to_onehot(self, mask, num_classes=17):
        h, w = mask.shape
        onehot = np.zeros((num_classes, h, w), dtype=np.uint8)
        for cls in range(num_classes):
            onehot[cls] = (mask == cls).astype(np.uint8)
        return onehot

    def _get_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        token = data["token"]
        sample_token = f"{self.dataset_occ}/sample_{token}/"
        views = ['proj_CAM_FRONT_LEFT.npz', 'proj_CAM_FRONT.npz', 'proj_CAM_FRONT_RIGHT.npz', 'proj_CAM_BACK_RIGHT.npz', 'proj_CAM_BACK.npz','proj_CAM_BACK_LEFT.npz', ]
        
        occupancies = []
        for view in views:
            view_sample_token = sample_token + view
            view_file = np.load(view_sample_token)
            labels = view_file['labels']
            # distances = view_file['distances']
            labels_seperate = self.mask_to_onehot(labels)[1:]
            occupancies.append(labels_seperate)
        
        data["occupancy"] = occupancies
        return data

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # if set cache, use it.
        if self.cache is not None:
            try:
                with h5py.File(self.cache, "r") as cache_file:
                    return self._load_from_cache(data, cache_file)
            except:
                pass
        if self.shared_mem_cache is not None:
            try:
                return self._load_from_cache(data, self.shared_mem_cache)
            except:
                pass

        # cache miss, load normally
        data = self._get_data(data)

        # if set, add this item into it.
        if self.shared_mem_cache is not None:
            token = data["token"]
            for key in self.shared_mem_cache.keys():
                if key in data:
                    self.shared_mem_cache[key][token] = data[key]
        return data


@PIPELINES.register_module()
class LoadOccupancyDepth:
    """This only loads map annotations
    In this map, the origin is at lower-left corner, with x-y transposed.
                          FRONT                             RIGHT
         Nuscenes                       transposed
        --------->  LEFT   EGO   RIGHT  ----------->  BACK   EGO   FRONT
           map                            output
                    (0,0)  BACK                       (0,0)  LEFT
    Guess reason, in cv2 / PIL coord, this is a BEV as follow:
        (0,0)  LEFT

        BACK   EGO   FRONT

              RIGHT
    All masks in np channel first format.
    
    NOTE: occupancy(one-hot) + depth
    """

    def __init__(
        self,
        dataset_root: str,
        dataset_occ: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
        cache_file: str = None,
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes
        self.lidar2canvas = np.array(
            [
                [canvas_h / patch_h, 0, canvas_h / 2],
                [0, canvas_w / patch_w, canvas_w / 2],
                [0, 0, 1],
            ]
        )
        
        self.dataset_occ = dataset_occ
        
        self.maps = {}
        if "nuplan" in dataset_root.lower():
            pass
        else:
            for location in LOCATIONS:
                self.maps[location] = NuScenesMap(dataset_root, location)

        # if cache_file and os.path.isfile(cache_file):
        #     logging.info(f"using data cache from: {cache_file}")
        #     # load to memory and ignore all possible changes.
        #     self.cache = cache_file
        # else:
        self.cache = None
        # this should be set through main process afterwards
        self.shared_mem_cache = None

    def _load_from_cache(self, data: Dict[str, Any], cache_dict) -> Dict[str, Any]:
        token = data["token"]
        labels = one_hot_decode(
            cache_dict["gt_masks_bev"][token][:],
            len(self.classes),
        )
        data["gt_masks_bev"] = labels

        if not np.any(data["gt_masks_bev"]):
            print('gt_masks_bev are all 0')
        return data
    
    def mask_to_onehot(self, mask, num_classes=17):
        h, w = mask.shape
        onehot = np.zeros((num_classes, h, w), dtype=np.uint8)
        for cls in range(num_classes):
            onehot[cls] = (mask == cls).astype(np.uint8)
        return onehot

    def _get_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        token = data["token"]
        sample_token = f"{self.dataset_occ}/sample_{token}/"
        views = ['proj_CAM_FRONT_LEFT.npz', 'proj_CAM_FRONT.npz', 'proj_CAM_FRONT_RIGHT.npz', 'proj_CAM_BACK_RIGHT.npz', 'proj_CAM_BACK.npz','proj_CAM_BACK_LEFT.npz', ]
        
        occupancies = []
        for view in views:
            cam_type = view.removesuffix('.npz').removeprefix('proj_')
            cam_index = None
            for i, img_path in enumerate(data["image_paths"]):
                if f"/{cam_type}/" in img_path:
                    cam_index = i
                    break
            if cam_index is None:
                raise ValueError(f"Camera type {cam_type} not found in image paths.")
            view_sample_token = sample_token + view
            view_file = np.load(view_sample_token)
            labels = view_file['labels']
            distances = view_file['distances']
            depths = distance_to_depth(distances, data["camera_intrinsics"][cam_index])
            
            labels_seperate = self.mask_to_onehot(labels)[1:]
            occupancies.append(np.concatenate([depths[np.newaxis,:], labels_seperate]))
        
        data["occupancy"] = occupancies
        return data

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # if set cache, use it.
        if self.cache is not None:
            try:
                with h5py.File(self.cache, "r") as cache_file:
                    return self._load_from_cache(data, cache_file)
            except:
                pass
        if self.shared_mem_cache is not None:
            try:
                return self._load_from_cache(data, self.shared_mem_cache)
            except:
                pass

        # cache miss, load normally
        data = self._get_data(data)

        # if set, add this item into it.
        if self.shared_mem_cache is not None:
            token = data["token"]
            for key in self.shared_mem_cache.keys():
                if key in data:
                    self.shared_mem_cache[key][token] = data[key]
        return data
