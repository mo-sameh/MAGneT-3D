import copy
import json
import os
import os.path as osp
from collections import defaultdict
from typing import Tuple, Union

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox.structures import Box3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines.compose import Compose
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from multi_dataset_nuscenes_metric.detection.evaluate import \
    multi_dataset_evaluation

# camera frame (x right, y down, z forward) -> pseudo-LiDAR frame
# (x right, y forward, z up)
CAM_TO_PSEUDO_LIDAR = np.asarray([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])


def get_4x4_matrix(transform: list) -> np.ndarray:
    transform = np.asarray(transform)
    transform4x4 = np.eye(4)
    transform4x4[:transform.shape[0], :transform.shape[1]] = transform
    return transform4x4


def get_frame_idx(index, flags):
    """Position of frame ``index`` within its sequence."""
    frame_idx = 0
    while index - frame_idx - 1 >= 0 and flags[index - frame_idx - 1] == flags[index]:
        frame_idx += 1
    return frame_idx


def project_to_image(pts_3d: np.ndarray, calibs: np.ndarray,
                     return_depth=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Projects (B, 3) camera-frame points with (B or 1, 3, 4) calibrations."""
    calibs = np.asarray(calibs).reshape(-1, 3, 4)
    assert pts_3d.dtype in [np.float32, np.float64]
    assert calibs.dtype in [np.float32, np.float64]
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    if len(calibs) == len(pts_3d):
        pts_2d = np.einsum('bij,bj->bi', calibs, pts_3d_homo)
    else:
        pts_2d = np.einsum('ij,bj->bi', calibs[0], pts_3d_homo)
    pts_2d, depth = pts_2d[:, :2] / pts_2d[:, 2:], pts_2d[:, 2]
    if return_depth:
        return pts_2d, depth
    return pts_2d


@DATASETS.register_module()
class MonoVideoDataset(Dataset):
    """Monocular video dataset in the cross-dataset benchmark format.

    The annotation file is COCO-style with ``images`` (one front-camera frame
    each, with calibration and poses), ``videos``, ``annotations`` (3D boxes
    in the global frame) and ``categories`` (the shared taxonomy). Boxes are
    returned in a pseudo-LiDAR frame attached to the camera.

    Args:
        data_root (str): Root that ``annotation_file`` and image file names
            are relative to.
        annotation_file (str): Annotation file.
        pipeline (list[dict]): Data pipeline.
        test_mode (bool): Test mode.
        sequence_length (int, optional): Videos are split into sequences of
            at most this many frames (each sequence is one sampler group).
        limit_sequence_length (bool): Drop the last frames of every video so
            that its length is a multiple of ``sequence_length``.
        branched_gt_prefix (str, optional): Prefix of the per-camera gt copy
            created by ``BranchOutGTBoxes`` (used by DRAG).
        tp_error_averaging (str): 'recall' or 'all_tp', averaging of the
            TP errors in the benchmark metric.
        overfit (bool): Only use the first ``overfit_samples`` frames.
    """

    def __init__(self,
                 data_root: str,
                 annotation_file: str,
                 pipeline: list,
                 test_mode: bool,
                 sequence_length: int = None,
                 limit_sequence_length: bool = False,
                 branched_gt_prefix: str = None,
                 img_path_prefix: str = '',
                 classes=None,
                 class_weighted_eval: bool = False,
                 tp_error_averaging: str = 'recall',
                 overfit: bool = False,
                 overfit_samples: int = 400):
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.class_weighted_eval = class_weighted_eval
        self.tp_error_averaging = tp_error_averaging
        self.overfit = overfit
        self.overfit_samples = overfit_samples
        self.img_path_prefix = img_path_prefix
        self.branched_gt_prefix = branched_gt_prefix
        self.cam_to_pseudo_lidar = CAM_TO_PSEUDO_LIDAR

        self.ann_path = os.path.join(data_root, annotation_file)
        with open(self.ann_path, 'r') as file:
            self.coco_data = json.load(file)
        if limit_sequence_length:
            self.fix_sequence_length(self.sequence_length)
        self.img_id_to_anns = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.img_id_to_anns[ann['image_id']].append(ann)
        self.video_id_to_video_info = {
            video_info['id']: video_info['name']
            for video_info in self.coco_data['videos']
        }
        self.CLASSES = classes if classes is not None else [
            cat_info['name'] for cat_info in self.coco_data['categories']
        ]
        self.cat_id_to_name = {c['id']: c['name'] for c in self.coco_data['categories']}
        self.name_to_cat_id = {c['name']: c['id'] for c in self.coco_data['categories']}
        self.assign_group_ids()

    @property
    def images(self):
        return self.coco_data['images']

    def __len__(self):
        if self.overfit:
            return self.overfit_samples
        return len(self.images)

    def fix_sequence_length(self, sequence_length: int):
        """Drops frames so that each video has a multiple of
        ``sequence_length`` frames."""
        if sequence_length is None:
            return
        video_id_to_frames = defaultdict(list)
        for img_info in self.images:
            video_id_to_frames[img_info['video_id']].append(img_info)
        filtered_images = []
        for frames in video_id_to_frames.values():
            num_full_sequences = len(frames) // sequence_length
            filtered_images.extend(frames[:num_full_sequences * sequence_length])
        self.coco_data['images'] = filtered_images

    def assign_group_ids(self):
        """``self.flag[i]`` is the sequence id of frame ``i``."""
        group_id = -1
        current_video_id = None
        count = 0
        group_ids = []
        for img_info in self.images:
            if img_info['video_id'] != current_video_id or (
                    self.sequence_length is not None and count >= self.sequence_length):
                group_id += 1
                count = 0
            group_ids.append(group_id)
            current_video_id = img_info['video_id']
            count += 1
        self.flag = np.asarray(group_ids).astype(int)

    def compute_transformations(self, img_info: dict) -> dict:
        camera_to_global = get_4x4_matrix(img_info['trans_matrix'])
        return {
            'lidar_to_global': camera_to_global @ np.linalg.inv(self.cam_to_pseudo_lidar),
            'lidar_to_img': get_4x4_matrix(img_info['calib']) @ np.linalg.inv(self.cam_to_pseudo_lidar),
            'global_to_lidar': self.cam_to_pseudo_lidar @ np.linalg.inv(camera_to_global),
            'lidar_to_cam': np.linalg.inv(self.cam_to_pseudo_lidar)
        }

    def __getitem__(self, index: int):
        input_dict = self.get_input_dict(index)
        self.pre_pipeline(input_dict)
        data = self.pipeline(input_dict)
        if not self.test_mode:
            data = self.union2one([data])
        return data

    def get_input_dict(self, index: int) -> dict:
        img_info = self.images[index]
        transformations = self.compute_transformations(img_info)
        return {
            'sample_idx': None,
            'pts_filename': None,
            'sweeps': None,
            'ego_pose': transformations['lidar_to_global'].astype(np.float32),
            'ego_pose_inv': np.linalg.inv(transformations['lidar_to_global']).astype(np.float32),
            'prev_idx': None,
            'next_idx': None,
            'scene_token': self.video_id_to_video_info[img_info['video_id']],
            # used by MultiResizeCropFlipRotImage to cycle input resolutions
            'frame_idx': get_frame_idx(index, self.flag),
            'timestamp': img_info['timestamp'] / 1e6,
            'img_timestamp': img_info['timestamp'] / 1e6,
            'img_filename': [os.path.join(self.data_root, self.img_path_prefix, img_info['file_name'])],
            'lidar2img': [transformations['lidar_to_img']],
            'intrinsics': [get_4x4_matrix(img_info['calib'])],
            'extrinsics': [np.linalg.inv(self.cam_to_pseudo_lidar).astype(np.float32)],
            'prev_exists': index == 0 or self.flag[index] == self.flag[index - 1],
            'supported_classes': self.CLASSES,
            'ann_info': self.get_ann_info(img_info, transformations)
        }

    def get_ann_info(self, img_info: dict, transformations: dict):
        anns = copy.deepcopy(self.img_id_to_anns[img_info['id']])
        gt_names, gt_labels_3d, gt_bboxes_3d, centers_2d, depths, bbox_xywh = [], [], [], [], [], []
        for ann in anns:
            cat_name = self.cat_id_to_name[ann['category_id']]
            lidar_translation = (transformations['global_to_lidar'] @ np.asarray([*ann['translation'], 1]))[:3]
            lidar_rotation_matrix = transformations['global_to_lidar'][:3, :3] @ \
                Rotation.from_rotvec(ann['rotation']).as_matrix()
            lidar_angle = Rotation.from_matrix(lidar_rotation_matrix).as_euler('xyz')[2]
            lwh = np.asarray(ann['dim'])[::-1]  # hwl => lwh
            lidar_velocity_2d = (transformations['global_to_lidar'][:3, :3] @
                                 np.asarray([*ann['velocity'][:2], 0.0]))[:2]
            gt_bbox_3d = np.concatenate((lidar_translation, lwh, [lidar_angle], lidar_velocity_2d))

            # project the gravity center to the image
            center_of_object = lidar_translation.copy()
            center_of_object[2] += lwh[2] / 2
            cam_translation = (transformations['lidar_to_cam'] @ np.asarray([*center_of_object, 1]))[:3]
            center_2d, depth = project_to_image(
                cam_translation[None], np.asarray([img_info['calib']]), return_depth=True)
            if depth[0] <= 0:
                continue

            gt_bboxes_3d.append(gt_bbox_3d)
            gt_names.append(cat_name)
            gt_labels_3d.append(self.CLASSES.index(cat_name))
            centers_2d.append(center_2d[0])
            depths.append(depth[0])
            bbox_xywh.append(ann['bbox'])

        return {
            'gt_names': np.asarray(gt_names),
            'gt_labels_3d': np.asarray(gt_labels_3d),
            'gt_bboxes_3d': LiDARInstance3DBoxes(
                torch.from_numpy(np.asarray(gt_bboxes_3d).astype(np.float32)), box_dim=9),
            'bboxes': [np.asarray(bbox_xywh).astype(np.float32)],
            'labels': [np.asarray(gt_labels_3d)],
            'bboxes_ignore': [np.asarray([False for _ in anns])],
            'centers2d': [np.asarray(centers_2d).astype(np.float32)],
            'depths': [np.asarray(depths).astype(np.float32)],
        }

    def pre_pipeline(self, results):
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = LiDARInstance3DBoxes
        results['box_mode_3d'] = Box3DMode.LIDAR

    def union2one(self, queue):
        collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                        'img_timestamp', 'ego_pose', 'ego_pose_inv', 'img',
                        'prev_exists', 'img_metas']
        if 'lidar2img_inv' in queue[0]:
            collect_keys.append('lidar2img_inv')
        for key in collect_keys:
            if key != 'img_metas':
                queue[-1][key] = DC(torch.stack([each[key].data for each in queue]),
                                    cpu_only=False, stack=True, pad_dims=None)
            else:
                queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
        keys = ['gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'gt_bboxes_3d', 'gt_labels_3d']
        if self.branched_gt_prefix is not None:
            keys.append(f'{self.branched_gt_prefix}_gt_bboxes_3d')
            keys.append(f'{self.branched_gt_prefix}_gt_labels_3d')
        for key in keys:
            queue[-1][key] = DC([each[key].data for each in queue],
                                cpu_only=key.endswith('gt_bboxes_3d'))
        return queue[-1]

    def format_results(self, predictions_list, train_classes=None):
        """Converts the predictions to the benchmark (global frame) format.

        Args:
            train_classes (list[str], optional): Class names in the label order
                of the model (``class_names`` of its config). Labels are
                mapped to the categories of this dataset by name. If None, the
                labels are assumed to follow this dataset's category order.
        """
        images = self.images[:self.overfit_samples] if self.overfit else self.images
        if train_classes is not None:
            category_map = {i: self.name_to_cat_id[name] for i, name in enumerate(train_classes)}
        else:
            category_map = {cat_id: cat_id for cat_id in self.cat_id_to_name.keys()}

        converted_results = []
        for pred, img_info in zip(predictions_list, images):
            transformations = self.compute_transformations(img_info)
            boxes_3d = pred['pts_bbox']['boxes_3d']
            if boxes_3d.bottom_center.shape[0] == 0:
                continue
            translations_lidar = boxes_3d.bottom_center.numpy()
            translations_lidar_homo = np.hstack((translations_lidar, np.ones((len(translations_lidar), 1))))
            translations_global = (transformations['lidar_to_global'] @ translations_lidar_homo.T)[:3].T

            lidar_rotation_matrices = Rotation.from_euler('z', boxes_3d.yaw.cpu()).as_matrix()
            global_rotation_matrices = (transformations['lidar_to_global'][:3, :3] @ lidar_rotation_matrices.T).T
            global_rotations = Rotation.from_matrix(global_rotation_matrices).as_rotvec()

            velocity_lidar = boxes_3d.tensor[:, 7:9].numpy()
            velocity_lidar_homo = np.hstack((velocity_lidar, np.zeros((len(velocity_lidar), 1))))
            velocity_global = (transformations['lidar_to_global'][:3, :3] @ velocity_lidar_homo.T)[:2].T

            dims = boxes_3d.dims.cpu()[:, [2, 1, 0]]  # lwh -> hwl
            scores = pred['pts_bbox']['scores_3d'].numpy().tolist()
            labels = [category_map[label] for label in pred['pts_bbox']['labels_3d'].numpy().tolist()]
            translation_in_camera_frame = (transformations['lidar_to_cam'] @ translations_lidar_homo.T)[:3].T
            for i in range(len(dims)):
                if scores[i] < 0.05:
                    continue
                converted_results.append({
                    'translation_in_camera_frame': translation_in_camera_frame[i].tolist(),
                    'translation': translations_global[i].tolist(),
                    'rotation': global_rotations[i].tolist(),
                    'dim': dims[i].tolist(),
                    'velocity': velocity_global[i].tolist(),
                    'score': scores[i],
                    'category_id': labels[i],
                    'image_id': img_info['id'],
                    'id': len(converted_results),
                    'file_name': img_info['file_name'],
                    'global_to_lidar': transformations['global_to_lidar'].tolist(),
                    'lidar_to_img': transformations['lidar_to_img'].tolist(),
                })
        return converted_results

    def evaluate(self, predictions_list, jsonfile_prefix, *args,
                 train_classes=None, return_detailed=False, **kwargs):
        """Saves the predictions to ``<jsonfile_prefix>/results.json`` and
        evaluates them with the cross-dataset benchmark metric."""
        converted_results = self.format_results(predictions_list, train_classes)
        os.makedirs(jsonfile_prefix, exist_ok=True)
        with open(osp.join(jsonfile_prefix, 'results.json'), 'w') as f:
            json.dump(converted_results, f)
        print(f'Results have been saved to {osp.join(jsonfile_prefix, "results.json")}')
        eval_results = multi_dataset_evaluation(
            self.ann_path, converted_results, jsonfile_prefix,
            self.class_weighted_eval, only_predicted_imgs=self.overfit,
            tp_error_averaging=self.tp_error_averaging)
        return eval_results if return_detailed else eval_results[0]
