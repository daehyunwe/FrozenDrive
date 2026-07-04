import copy
import logging
import random
from time import time

import cv2
import mmcv
import numpy as np
from PIL import Image
import torch
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from scipy.ndimage import zoom

from .map_utils import (
    LiDARInstanceLines,
    VectorizedLocalMap,
    project_box_to_image,
    project_map_to_image,
    visualize_bev_hdmap,
)


@DATASETS.register_module()
class NuScenesMapDataset_occ(NuScenesDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        map_bound=None,
        video_length=None,
        start_on_keyframe=True,
        start_on_firstframe=False,
        fixed_ptsnum_per_line=-1,
        padding_value=-10000,
        fps=12,
        image_size=None,
        equalize_file=None,
        balancing=False,
        add_firstframe_pair=True,
        ref_random_index=True,
    ) -> None:
        self.video_length = video_length
        self.start_on_keyframe = start_on_keyframe
        self.start_on_firstframe = start_on_firstframe
        self.fps = fps
        self.equalize_file = equalize_file
        self.balancing = balancing
        self.add_firstframe_pair = add_firstframe_pair
        self.ref_random_index = ref_random_index
        super().__init__(
            ann_file, pipeline, dataset_root, object_classes, map_classes,
            load_interval, with_velocity, modality, box_type_3d,
            filter_empty_gt, test_mode, eval_version, use_valid_flag,
            force_all_boxes)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.object_classes = object_classes
        self.map_classes = map_classes
        self.map_bound = map_bound
        xbound = map_bound['x']
        ybound = map_bound['y']
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        if image_size[0] == 224:
            self.scale = 1.0
        elif image_size[0] == 448:
            self.scale = 2.0
        elif image_size[0] == 704: 
            self.scale = 3.14
        elif image_size[0] == 576:
            self.scale = 2.57
        
        self.vector_map = VectorizedLocalMap(dataset_root, 
                            patch_size=self.patch_size, map_classes=map_classes, 
                            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
                            padding_value=padding_value)
        self.clip_infos = sorted(self.clip_infos, key=lambda x: x[-1])
        
        # the curr scan is the last index

        # weather info
        if 'train' in self.ann_file and self.balancing:
            filename = 'train_rain_night_indices.txt'
            f = open(filename, 'r')
            lines = f.readlines()
            assert len(lines) == len(self.clip_infos)
            
            self.clip_infos_by_weather = {
                "normal": [],
                "rain": [],
                "night": [],
            }
            for i in range(len(self.clip_infos)):
                weather = lines[i].strip().split(', ')[-1]
                assert weather in self.clip_infos_by_weather
                
                self.clip_infos_by_weather[weather].append(self.clip_infos[i])

    def __len__(self):
        return len(self.clip_infos)

    def build_clips(self, data_infos, scene_tokens):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene 

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)} # item['scene_token']
        all_clips = []
        for scene in scene_tokens:
            keyframes_list = []
            for idx in range(len(scene)):
                if len(scene[idx]) == 32:
                    keyframes_list.append(scene[idx])
            data_infos[self.token_data_dict[keyframes_list[0]]]['is_first_frame'] = True  # track first frames
            if self.add_firstframe_pair:
                assert self.video_length == 2
                all_clips.append([self.token_data_dict[keyframes_list[0]], self.token_data_dict[keyframes_list[0]]])  # prepend duplicated first frame pair
            for start in range(len(scene) - self.video_length + 1):
                if self.start_on_keyframe and ";" in scene[start]:
                    continue  # this is not a keyframe
                if self.start_on_keyframe and len(scene[start]) >= 33:
                    continue  # this is not a keyframe
                clip = [self.token_data_dict[token]
                        for token in scene[start: start + self.video_length]] # token setting
                all_clips.append(clip)
                if self.start_on_firstframe:
                    break
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into clips of length {self.video_length}, "
                     f"which has {len(all_clips)} in total.")
        return all_clips

    def build_clips_2hz(self, data_infos, scene_tokens):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene 

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        all_clips = []
        
        for scene in scene_tokens:
            keyframes_list = []
            for idx in range(len(scene)):
                if len(scene[idx]) == 32:
                    keyframes_list.append(scene[idx])
            data_infos[self.token_data_dict[keyframes_list[0]]]['is_first_frame']=True
            all_clips.append([self.token_data_dict[keyframes_list[0]], self.token_data_dict[keyframes_list[0]]])
            for idx in range(len(keyframes_list)-1):
                clip = [self.token_data_dict[keyframes_list[idx]], self.token_data_dict[keyframes_list[idx+1]]]
                all_clips.append(clip)
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into clips of length 2, only keyframes, "
                     f"which has {len(all_clips)} in total.")
        return all_clips

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        if self.fps == 2 and self.test_mode:
            self.clip_infos = self.build_clips_2hz(data_infos, data['scene_tokens'])
        else:
            self.clip_infos = self.build_clips(data_infos, data['scene_tokens'])
        return data_infos

    def vectormap_pipeline(self, example, input_dict):
        '''
        `example` type: <class 'dict'>
            keys: 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img';
                  all keys type is 'DataContainer';
                  'img_metas' cpu_only=True, type is dict, others are false;
                  'gt_labels_3d' shape torch.size([num_samples]), stack=False,
                                padding_value=0, cpu_only=False
                  'gt_bboxes_3d': stack=False, cpu_only=True
        '''
        
        lidar2ego = input_dict['lidar2ego']
        ego2global = input_dict['ego2global']
        lidar2global = ego2global @ lidar2ego

        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
        location = input_dict['location']
        anns_results = self.vector_map.gen_vectorized_samples(location, lidar2global_translation, lidar2global_rotation)
        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        '''
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
            gt_lines_instance = gt_vecs_pts_loc.instance_list
            gt_map_pts = []
            for i in range(len(gt_lines_instance)):
                pts = np.array(list(gt_lines_instance[i].coords))
                gt_map_pts.append(pts)
        example['gt_vecs_label'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_vecs_pts_loc'] = DC(gt_map_pts, cpu_only=True)

        drivable_mask = example['gt_masks_bev'][0, ...] + example['gt_masks_bev'][-1, ...]    # 200*200
        drivable_mask = drivable_mask.astype(bool)
        bev_map = visualize_bev_hdmap(example['gt_vecs_pts_loc'].data, example['gt_vecs_label'].data, self.canvas_size, num_classes=len(self.map_classes), bound=self.map_bound['x'], drivable_mask=drivable_mask)

        bev_map = bev_map.transpose(2, 0, 1)    # C, H, W

        example['bev_hdmap'] = DC(to_tensor(bev_map), cpu_only=False)
        
        return example

    def project_bev2img(self, example):
        lidar2image = example['lidar2image'].data
        camera2ego = example['camera2ego'].data
        camera_intrinsics = example['camera_intrinsics'].data

        drivable_mask = example['gt_masks_bev'][0, ...] + example['gt_masks_bev'][-1, ...]    # 200*200
        drivable_mask = drivable_mask.astype(bool)
        layout_canvas = []
        layout_bbox = []
        
        for i in range(len(lidar2image)):
            map_canvas = project_map_to_image(example['gt_vecs_pts_loc'].data, example['gt_vecs_label'].data, camera_intrinsics[i], camera2ego[i], num_classes=len(self.map_classes), drivable_mask=drivable_mask, scale=self.scale)

            box_canvas, bbox = project_box_to_image(example['gt_bboxes_3d'].data, example['gt_labels_3d'].data, lidar2image[i], object_classes=self.object_classes, scale=self.scale)

            layout_canvas.append(np.concatenate([map_canvas, box_canvas], axis=-1))
            layout_bbox.append(bbox)
        layout_canvas = np.stack(layout_canvas, axis=0)
        layout_canvas = np.transpose(layout_canvas, (0, 3, 1, 2))    # 6, C, H, W
        example['layout_canvas'] = DC(to_tensor(layout_canvas), cpu_only=False)
        
        layout_bbox = np.stack(layout_bbox, axis=0)
        layout_bbox = np.transpose(layout_bbox, (0, 3, 1, 2))    # 6, C, H, W
        example['layout_bbox'] = DC(to_tensor(layout_bbox), cpu_only=False)
        
        return example
    
    def project_ref_bev2img(self, example, ref_example):
        if ref_example is None:
            _, _, H, W = example['layout_canvas'].data.shape
            C = example['layout_canvas'].data.shape[1]
            ref_layout_canvas = np.zeros((1, C, H, W), dtype=np.float32)
            example['ref_layout_canvas'] = DC(to_tensor(ref_layout_canvas), cpu_only=False)
            return example

        lidar2image = ref_example['lidar2image'].data
        camera2ego = ref_example['camera2ego'].data
        camera_intrinsics = ref_example['camera_intrinsics'].data

        drivable_mask = ref_example['gt_masks_bev'][0, ...] + ref_example['gt_masks_bev'][-1, ...]
        drivable_mask = drivable_mask.astype(bool)
        layout_canvas_ref = []
        
        for i in range(len(lidar2image)):
            map_canvas = project_map_to_image(ref_example['gt_vecs_pts_loc'].data, ref_example['gt_vecs_label'].data, camera_intrinsics[i], camera2ego[i], num_classes=len(self.map_classes), drivable_mask=drivable_mask, scale=self.scale)
            
            box_canvas, bbox = project_box_to_image(ref_example['gt_bboxes_3d'].data, ref_example['gt_labels_3d'].data, lidar2image[i], object_classes=self.object_classes, scale=self.scale)
            layout_canvas_ref.append(np.concatenate([map_canvas, box_canvas], axis=-1))

            
        layout_canvas_ref = np.stack(layout_canvas_ref, axis=0)
        layout_canvas_ref = np.transpose(layout_canvas_ref, (0, 3, 1, 2))  # 6, C, H, W
        
        example['ref_layout_canvas'] = DC(to_tensor(layout_canvas_ref), cpu_only=False)
        return example
    
    def process_ref_image(self, example, ref_example):
        example["ref_images"] = ref_example["img"]
        relative_pose = torch.matmul(torch.inverse(example["ego2global"].data), ref_example["ego2global"].data)

        example["relative_pose"] = DC(relative_pose, cpu_only=False)
        example["ref_bboxes_3d"] = ref_example["gt_bboxes_3d"]
        example["ref_labels_3d"] = ref_example["gt_labels_3d"]
        example["obj_ids"] = example["metas"].data["obj_ids"]
        example["ref_obj_ids"] = ref_example["metas"].data["obj_ids"]
        return example
    
    def process_ref_occupancy(self, example, ref_example):
        example["ref_occupancy"] = ref_example["occupancy"]
        return example
    
    def process_prev_ref_pose(self, example, ref_index, prev_ref_index):
        ref_info = self.get_data_info(ref_index)
        prev_ref_info = self.get_data_info(prev_ref_index)
        
        if ref_info is None or prev_ref_info is None:
            identity_pose = torch.eye(4, dtype=torch.float32)
            example['ref_relative_pose'] = DC(identity_pose, cpu_only=False)
            return example
        
        self.pre_pipeline(ref_info)
        ref_info = self.pipeline(ref_info)
        self.pre_pipeline(prev_ref_info)
        prev_ref_info = self.pipeline(prev_ref_info)
        
        prev_ref_relative_pose = torch.matmul(torch.inverse(ref_info["ego2global"].data), prev_ref_info["ego2global"].data)

        example['ref_relative_pose'] = DC(prev_ref_relative_pose.float(), cpu_only=False)
        return example
    
    def prepare_train_data(self, index):
        """This is called by `__getitem__`."""
        if self.balancing:
            choice = np.random.choice(np.arange(3))
            if choice == 0:
                new_index = index % len(self.clip_infos_by_weather["normal"])
                clips = self.clip_infos_by_weather["normal"][new_index]
            elif choice == 1:
                new_index = index % len(self.clip_infos_by_weather["rain"])
                clips = self.clip_infos_by_weather["rain"][new_index]
            elif choice == 2:
                new_index = index % len(self.clip_infos_by_weather["night"])
                clips = self.clip_infos_by_weather["night"][new_index]
        else:
            clips = self.clip_infos[index]
        curr_index = clips[-1]
        if self.ref_random_index:
            random_index = random.choice(clips)
        else:
            random_index = clips[-2]

        curr_input_dict = self.get_data_info(curr_index)
        ref_input_dict = self.get_data_info(random_index)
        if curr_input_dict is None or ref_input_dict is None: return None

        self.pre_pipeline(curr_input_dict)
        example = self.pipeline(curr_input_dict)

        self.pre_pipeline(ref_input_dict)
        ref_example = self.pipeline(ref_input_dict)

        if example is None or ref_example is None:
            return None

        example = self.vectormap_pipeline(example, curr_input_dict)
        ref_example = self.vectormap_pipeline(ref_example, ref_input_dict)
        example = self.project_bev2img(example)
        example = self.project_ref_bev2img(example, ref_example)
        example = self.process_ref_image(example, ref_example)
        example = self.process_ref_occupancy(example, ref_example)
        
        if self.scale == 1.0 and "occupancy" in example and "ref_occupancy" in example:
            sliced_occ = example["occupancy"].data[:,:,::2,::2]
            sliced_ref_occ = example["ref_occupancy"].data[:,:,::2,::2]

            example["occupancy"] = DC(sliced_occ, cpu_only=False)
            example["ref_occupancy"] = DC(sliced_ref_occ, cpu_only=False)
        
        if self.filter_empty_gt and (example is None or ~(example["gt_labels_3d"]._data != -1).any()):
            return None

        return example

    def prepare_test_data(self, index):
        clips = self.clip_infos[index]
        curr_index = clips[-1]
        ref_index = clips[-2]

        curr_input_dict = self.get_data_info(curr_index)
        ref_input_dict = self.get_data_info(ref_index)
        if curr_input_dict is None or ref_input_dict is None: return None
        self.pre_pipeline(curr_input_dict)
        example = self.pipeline(curr_input_dict)
        self.pre_pipeline(ref_input_dict)
        ref_example = self.pipeline(ref_input_dict)

        if example is None or ref_example is None:
            return None

        example = self.vectormap_pipeline(example, curr_input_dict)
        ref_example = self.vectormap_pipeline(ref_example, ref_input_dict)
        example = self.project_bev2img(example)
        example = self.project_ref_bev2img(example, ref_example)
        example = self.process_ref_image(example, ref_example)
        example = self.process_ref_occupancy(example, ref_example)
        
        if self.scale == 1.0 and "occupancy" in example and "ref_occupancy" in example:
            sliced_occ = example["occupancy"].data[:,:,::2,::2]
            sliced_ref_occ = example["ref_occupancy"].data[:,:,::2,::2]

            example["occupancy"] = DC(sliced_occ, cpu_only=False)
            example["ref_occupancy"] = DC(sliced_ref_occ, cpu_only=False)
        elif self.scale == 3.14 and "occupancy" in example and "ref_occupancy" in example: 
            import torch.nn.functional as F
            target_size = (704, 1280)

            sliced_occ = F.interpolate(example["occupancy"].data, size=target_size, mode='bilinear', align_corners=False)
            sliced_ref_occ = F.interpolate(example["ref_occupancy"].data, size=target_size, mode='bilinear', align_corners=False)

            example["occupancy"] = DC(sliced_occ, cpu_only=False)
            example["ref_occupancy"] = DC(sliced_ref_occ, cpu_only=False)
        elif self.scale == 2.57 and "occupancy" in example and "ref_occupancy" in example: 
            import torch.nn.functional as F
            target_size = (576, 1024)

            sliced_occ = F.interpolate(example["occupancy"].data, size=target_size, mode='bilinear', align_corners=False)
            sliced_ref_occ = F.interpolate(example["ref_occupancy"].data, size=target_size, mode='bilinear', align_corners=False)

            example["occupancy"] = DC(sliced_occ, cpu_only=False)
            example["ref_occupancy"] = DC(sliced_ref_occ, cpu_only=False)
        
        if self.filter_empty_gt and (example is None or ~(example["gt_labels_3d"]._data != -1).any()):
            return None
        
        return example
