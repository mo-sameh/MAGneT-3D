# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------

import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.bbox import LiDARInstance3DBoxes,Box3DMode, CameraInstance3DBoxes,DepthInstance3DBoxes
from scipy.spatial.distance import cdist
from mmdet.datasets.pipelines import PhotoMetricDistortion

from numpy import random
import torch
from PIL import Image
import math

import copy
from copy import deepcopy

@PIPELINES.register_module()
class PhotoMetricDistortion3D(PhotoMetricDistortion):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    PhotoMetricDistortion3D further support using predefined randomness
    variable to do the augmentation.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Required Keys:

    - img (np.uint8)

    Modified Keys:

    - img (np.float32)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (sequence): range of contrast.
        saturation_range (sequence): range of saturation.
        hue_delta (int): delta of hue.
    """
    def __call__(self, results):
        results['img_fields'] = ['img']
        new_imgs = []
        for img in results['img']:
            img_result = results.copy()
            img_result['img'] = img
            img_result = super().__call__(img_result)
            new_imgs.append(img_result['img'])
        results['img'] = new_imgs
        return results

@PIPELINES.register_module()
class MultiCamRandomGamma:
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, results):
        new_imgs = []
        for img in results['img']:
            img = img.astype(np.float32) / 255.0
            gamma = random.uniform(*self.gamma_range)
            img = np.power(img, gamma)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            new_imgs.append(img)
        results['img'] = new_imgs
        return results
@PIPELINES.register_module()     
class MultiCamGaussianNoise:
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std

    def __call__(self, results):
        new_imgs = []
        for img in results['img']:
            img = img.astype(np.float32) / 255.0
            img += np.random.normal(0, self.noise_std, img.shape)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            new_imgs.append(img)
        results['img'] = new_imgs
        return results
@PIPELINES.register_module()
class PadMultiViewImage():
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size_divisor is None or size is None
    
    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(img,
                                shape = self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(img,
                                self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fix_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
    
    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class InverseLidarToImage():
    def __call__(self, results):
        """Call function to compute inverse lidar to image transformation.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict with 'img2lidar' key.
        """
        results['lidar2img_inv'] = [np.linalg.inv(results['lidar2img'][i]) for i in range(len(results['lidar2img']))]
        return results

@PIPELINES.register_module()
class ResizeCropFlipRotImage():
    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 0.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

    def __call__(self, results):

        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"
        img_shape = imgs[0].shape[:2]
        # reuse the stored seed so that repeated calls apply the same augmentation
        if 'mixup_seed' in results:
            np.random.seed(results['mixup_seed'])
        else: 
            mixup_seed = np.random.randint(0, 10000)
            np.random.seed(mixup_seed)
            results['mixup_seed'] = mixup_seed
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation(orig_shape=img_shape)

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if self.training and self.with_2d:
                gt_bboxes = results['gt_bboxes'][i]
                centers2d = results['centers2d'][i]
                gt_labels = results['gt_labels'][i]
                depths = results['depths'][i]
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers2d, gt_labels, depths,keep = self._bboxes_transform(
                        gt_bboxes, 
                        centers2d,
                        gt_labels,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                    )
                    if '3d_to_2d_indices' in results and '2d_to_3d_indices' in results:
                        results['3d_to_2d_indices'], results['2d_to_3d_indices'] = update_correspondance_indices(
                            results['3d_to_2d_indices'], 
                            results['2d_to_3d_indices'], 
                            mask_2d =keep
                        )

                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers2d, gt_labels, depths, keep =  self._filter_invisible(gt_bboxes, centers2d, gt_labels, depths)
                    if '3d_to_2d_indices' in results and '2d_to_3d_indices' in results:
                        results['3d_to_2d_indices'], results['2d_to_3d_indices'] = update_correspondance_indices(
                            results['3d_to_2d_indices'], 
                            results['2d_to_3d_indices'], 
                            mask_2d = keep
                        )
                    results['percentage_visible'] = np.sum(keep) / len(keep)
                else:
                    results['percentage_visible'] = 1.0
                new_gt_bboxes.append(gt_bboxes)
                new_centers2d.append(centers2d)
                new_gt_labels.append(gt_labels)
                new_depths.append(depths)

            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
        results['gt_bboxes'] = new_gt_bboxes
        results['centers2d'] = new_centers2d
        results['gt_labels'] = new_gt_labels
        results['depths'] = new_depths
        results['img'] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]
        return results

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths,resize, crop, flip):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim_res"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH) 
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) > self.min_size)
        
        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d  = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH) 
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths,keep


 
    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths):
        """
        Filters invisible bboxes due to occlusion while preserving the original order.
        """
        original_bboxes = copy.deepcopy(bboxes)
        original_centers2d = copy.deepcopy(centers2d)
        original_gt_labels = copy.deepcopy(gt_labels)
        original_depths = copy.deepcopy(depths)

        N = len(gt_labels)

        if N == 0:
            return bboxes, centers2d, gt_labels, depths, np.zeros(0, dtype=bool)

        # paint boxes far-to-near; boxes with no visible pixel are dropped
        sort_idx = np.argsort(-original_depths, kind='stable')

        tmp_bboxes = np.zeros_like(original_bboxes)
        tmp_bboxes[:, :2] = np.ceil(original_bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(original_bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes[sort_idx].astype(np.int64)

        fH, fW = self.data_aug_conf["final_dim_res"]
        indices_map = np.zeros((fH, fW))
        for i in range(tmp_bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_map[v1:v2, u1:u2] = i + 1 

        visible_sorted_indices = np.unique(indices_map).astype(np.int64)
        visible_sorted_indices = visible_sorted_indices[visible_sorted_indices != 0] - 1

        visible_original_indices = sort_idx[visible_sorted_indices]

        keep_mask = np.zeros(N, dtype=bool)
        keep_mask[visible_original_indices] = True

        final_bboxes = original_bboxes[keep_mask]
        final_centers2d = original_centers2d[keep_mask]
        final_gt_labels = original_gt_labels[keep_mask]
        final_depths = original_depths[keep_mask]

        return final_bboxes, final_centers2d, final_gt_labels, final_depths, keep_mask



    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self, orig_shape=None):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        if H == -1 or W == -1:
            assert orig_shape is not None, "orig_shape should not be None if H or W is -1"
            H, W = orig_shape 
        final_dim = self.data_aug_conf["final_dim"]
        if isinstance(final_dim, float) or isinstance(final_dim, int):
            fH, fW = int(H * final_dim), int(W * final_dim)
            # round to a multiple of 32
            fH = int(np.round(fH / 32) * 32)
            fW = int(np.round(fW / 32) * 32)
            self.data_aug_conf["final_dim_res"] = (fH, fW)
        else:
            fH, fW = final_dim
            self.data_aug_conf["final_dim_res"] = (fH, fW)
        fH, fW = self.data_aug_conf["final_dim_res"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate

@PIPELINES.register_module()
class MultiResizeCropFlipRotImage(ResizeCropFlipRotImage):
    def __init__(self, data_aug_configs=None, with_2d=True, filter_invisible=True, training=True):
        assert isinstance(data_aug_configs, list) and len(data_aug_configs) > 0, "data_aug_configs when using MultiResizeCropFlipRotImage should be a list"
        super().__init__(data_aug_conf=data_aug_configs[0], with_2d=with_2d, filter_invisible=filter_invisible, training=training)

        self.data_aug_configs = data_aug_configs
    def __call__(self, results):
        # cycle configs over frames of a sequence
        frame_idx = results['frame_idx']
        number_of_configs = len(self.data_aug_configs)
        config_idx = frame_idx % number_of_configs
        self.data_aug_conf = self.data_aug_configs[config_idx]
        return super().__call__(results)
@PIPELINES.register_module()
class BranchOutGTBoxes():
    """
    Duplicate the ground truth boxes in the results. Since the monocular head and the tracking head will have different 3d ground truth boxes,
    """
    def __init__(self, prefix = 'monocular', training=True):
        self.prefix = prefix
        self.training = training

    def __call__(self, results):
        """
        Duplicate the ground truth boxes in the results.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict with duplicated ground truth boxes.
        """
        if self.training:
            results[f'{self.prefix}_gt_bboxes_3d'] = results['gt_bboxes_3d'].clone()
            results[f'{self.prefix}_gt_labels_3d'] = copy.deepcopy(results['gt_labels_3d'])

        # transform from the branched gt frame to the original gt frame
        results['GlobalRotScaleTrans'] = np.eye(4, dtype=np.float32)

        return results
@PIPELINES.register_module()
class GlobalRotScaleTransImage():
    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
        gt_prefix=None,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.gt_prefix = gt_prefix
        self.reverse_angle = reverse_angle
        self.training = training


    def __call__(self, results):
        translation_std = np.array(self.translation_std, dtype=np.float32)

        rot_angle = np.random.uniform(*self.rot_range)
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        trans = np.random.normal(scale=translation_std, size=3).T
        
        gt_bbboxes_key = f'{self.gt_prefix}_gt_bboxes_3d' if self.gt_prefix else 'gt_bboxes_3d'
     

        self._rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle = rot_angle * -1

        results[gt_bbboxes_key].rotate(
            np.array(rot_angle)
        )
    
        self._scale_xyz(results, scale_ratio)
        results[gt_bbboxes_key].scale(scale_ratio)

        self._trans_xyz(results, trans)
        results[gt_bbboxes_key].translate(trans)

        

        return results

    def _trans_xyz(self, results, trans):
        trans_mat = torch.eye(4, 4)
        trans_mat[:3, -1] = torch.from_numpy(trans).reshape(1, 3)
        trans_mat_inv = torch.inverse(trans_mat)
        num_view = len(results["lidar2img"])
        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ trans_mat_inv).numpy()
        results['ego_pose_inv'] = (trans_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()

        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ trans_mat_inv).numpy()
        if self.gt_prefix is None:
            results['GlobalRotScaleTrans'] = (trans_mat.float() @ torch.tensor(results['GlobalRotScaleTrans']).float()).numpy()
        else:
            results['GlobalRotScaleTrans'] = (trans_mat_inv @ torch.tensor(results['GlobalRotScaleTrans']).float()).numpy()

    def _rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, rot_sin, 0, 0], [-rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ rot_mat_inv).numpy()
        results['ego_pose_inv'] = (rot_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
        if self.gt_prefix is None:
            results['GlobalRotScaleTrans'] = (rot_mat.float() @ torch.tensor(results['GlobalRotScaleTrans']).float()).numpy()
        else:
            results['GlobalRotScaleTrans'] = (rot_mat_inv @ torch.tensor(results['GlobalRotScaleTrans']).float()).numpy()

    def _scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = torch.inverse(scale_mat)

        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ scale_mat_inv).numpy()
        results['ego_pose_inv'] = (scale_mat @ torch.tensor(results["ego_pose_inv"]).float()).numpy()

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ scale_mat_inv).numpy()
        if self.gt_prefix is None:
            results['GlobalRotScaleTrans'] = (scale_mat.float() @ torch.tensor(results['GlobalRotScaleTrans']).float()).numpy()
        else:
            results['GlobalRotScaleTrans'] = (scale_mat_inv @ torch.tensor(results['GlobalRotScaleTrans']).float()).numpy()

@PIPELINES.register_module()
class ObjectCameraFrustumFilter():
    """
    Filter the 3D bounding boxes that are not in any camera frustum. By defualt this runs on the original branch. and uses the same mask for the rest of the branches.
    Since the lidar2img by defintion is for the original branch. 

    Args:
        use_center (bool): if set a boundig box is removed if its center is not in the camera frustum.
            If False, the bounding box is removed if its 3D corners are not in the camera frustum.
            Default: True.
    """
    def __init__(self, use_center=True):
        self.use_center = use_center

    def __call__(self, results):

        keys= results.keys()
        branches_keys = [key for key in keys if key.endswith('_gt_bboxes_3d')]
        for key in branches_keys:
            assert results["gt_bboxes_3d"].tensor.shape == results[key].tensor.shape, \
                f"All branches gt_bboxes_3d should have the same shape, but got {results['gt_bboxes_3d'].tensor.shape} and {results[key].tensor.shape} for original branch and {key}."
        if self.use_center:
            centers3d = results["gt_bboxes_3d"].center
            centers3d = np.concatenate([centers3d, np.ones((centers3d.shape[0], 1))], axis=1)
            objects_masks = []
            for view in range(len(results['lidar2img'])):
                lidar2img = results['lidar2img'][view]
                img_shape = results['img_shape'][view]
                in_frustum = np.ones(centers3d.shape[0], dtype=bool)
                centers_in_img = np.array(lidar2img @ centers3d.T,  dtype=np.float32)
                in_frustum &= (centers_in_img[2, :] > 0)
                centers_in_img = centers_in_img[:2, :] / centers_in_img[2, :]
                in_frustum &= (centers_in_img[0, :] > 0) & (centers_in_img[1, :] > 0)
                in_frustum &= (centers_in_img[0, :] < img_shape[1]) & (centers_in_img[1, :] < img_shape[0])

                objects_masks.append(in_frustum)
            objects_masks = np.logical_or.reduce(objects_masks)
            results["gt_bboxes_3d"].tensor = results["gt_bboxes_3d"].tensor[objects_masks]
            results["gt_labels_3d"] = results["gt_labels_3d"][objects_masks]
            if '3d_to_2d_indices' in results and '2d_to_3d_indices' in results:
                results['3d_to_2d_indices'], results['2d_to_3d_indices'] = update_correspondance_indices(
                            results['3d_to_2d_indices'], 
                            results['2d_to_3d_indices'], 
                            mask_3d =objects_masks
                        )
        else:
            # keep boxes with at least one corner in the frustum
            corners3d = results["gt_bboxes_3d"].corners
            ones = np.ones((corners3d.shape[0], corners3d.shape[1], 1), dtype=np.float32)  # [N, 8, 1]
            corners3d = np.concatenate([corners3d, ones], axis=-1)  # [N, 8, 4]
            objects_masks = []
            for view in range(len(results['lidar2img'])):
                lidar2img = results['lidar2img'][view]
                img_shape = results['img_shape'][view]

                corners_in_img = corners3d @ lidar2img.T  # [N, 8, 4]

                in_frustum = np.ones(corners_in_img.shape[:2], dtype=bool)  # [N, 8]
                in_frustum &= (corners_in_img[..., 2] > 0)
                z = np.clip(corners_in_img[..., 2], a_min=1e-5, a_max=None)

                corners_in_img = corners_in_img[..., :2] / z[..., None]  # [N, 8, 2]

                in_frustum &= (corners_in_img[..., 0] > 0) & (corners_in_img[..., 0] < img_shape[1])
                in_frustum &= (corners_in_img[..., 1] > 0) & (corners_in_img[..., 1] < img_shape[0])

                objects_masks.append(in_frustum.any(axis=1))
            objects_masks = np.logical_or.reduce(objects_masks)
            results["gt_bboxes_3d"].tensor = results["gt_bboxes_3d"].tensor[objects_masks]
            results["gt_labels_3d"] = results["gt_labels_3d"][objects_masks]
            if '3d_to_2d_indices' in results and '2d_to_3d_indices' in results:
                results['3d_to_2d_indices'], results['2d_to_3d_indices'] = update_correspondance_indices(
                            results['3d_to_2d_indices'], 
                            results['2d_to_3d_indices'], 
                            mask_3d =objects_masks
                        )

        for branch in branches_keys:
            results[branch] = results[branch][objects_masks]
            branch_labels = branch.replace('bboxes', 'labels')
            results[branch_labels] = results[branch_labels][objects_masks]

        return results


@PIPELINES.register_module()
class ParameterFreeObjectNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n < len(input_dict['supported_classes']) for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]
        if '3d_to_2d_indices' in input_dict and '2d_to_3d_indices' in input_dict:
            input_dict['3d_to_2d_indices'], input_dict['2d_to_3d_indices'] = update_correspondance_indices(
                input_dict['3d_to_2d_indices'], 
                input_dict['2d_to_3d_indices'], 
                mask_3d =gt_bboxes_mask
            )

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str  

@PIPELINES.register_module()
class RangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        if '3d_to_2d_indices' in input_dict and '2d_to_3d_indices' in input_dict:
            input_dict['3d_to_2d_indices'], input_dict['2d_to_3d_indices'] = update_correspondance_indices(
                        input_dict['3d_to_2d_indices'], 
                        input_dict['2d_to_3d_indices'], 
                        mask_3d =mask.numpy().astype(np.bool)
                    )
                    
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str

@PIPELINES.register_module()
class AlignBoxes():
    """
    Align the 2D bounding boxes with the 3D bounding boxes
    by projecting 3D box centers into the camera and matching
    them to the 2D centers. Add "2d_to_3d_indices" and "3d_to_2d_indices" to results.
    """
    def __init__(self, threshold=10.0):
        self.threshold = threshold

    def __call__(self, results):
        assert len(results['lidar2img']) == 1, "Only one camera is supported to align 2D and 3D boxes."
        
        lidar2img = results['lidar2img'][0]
        
        # [N, 3]
        centers3d = results['gt_bboxes_3d'].gravity_center.numpy()

        # [M, 2]
        centers2d = results['centers2d'][0]
    
        centers3d_hom = np.concatenate([centers3d, np.ones((centers3d.shape[0], 1))], axis=1)  # [N, 4]
        projected = (lidar2img @ centers3d_hom.T).T  # [N_3D, 3]

        valid_depth = projected[:, 2] > 0
        projected = projected[valid_depth]
        centers3d_valid_indices = np.where(valid_depth)[0]  # map to original indices

        projected_centers = projected[:, :2] / projected[:, 2:3]  # [N_valid_3D, 2]

        n_2d = len(centers2d)
        n_3d = len(centers3d)

        if projected_centers.shape[0] == 0 or n_2d == 0:
            results['2d_to_3d_indices'] = -np.ones(n_2d, dtype=np.int64)
            results['3d_to_2d_indices'] = -np.ones(n_3d, dtype=np.int64)
            return results

        # [N_valid_3D, N_2D]
        distances = cdist(projected_centers, centers2d)

        map_2d_to_3d = -np.ones(n_2d, dtype=np.int64)
        map_3d_to_2d = -np.ones(n_3d, dtype=np.int64)

        for i_proj3d, dists in enumerate(distances):
            i_3d = centers3d_valid_indices[i_proj3d]
            i_2d = np.argmin(dists)
            dist = dists[i_2d]
            if dist < self.threshold:
                if map_2d_to_3d[i_2d] == -1:
                    map_2d_to_3d[i_2d] = i_3d
                    map_3d_to_2d[i_3d] = i_2d
                else:
                    # 2D box already taken: keep the closer 3D box
                    existing_3d = map_2d_to_3d[i_2d]
                    prev_dist = np.linalg.norm(projected_centers[centers3d_valid_indices == existing_3d] - centers2d[i_2d])
                    if dist < prev_dist:
                        map_3d_to_2d[existing_3d] = -1
                        map_2d_to_3d[i_2d] = i_3d
                        map_3d_to_2d[i_3d] = i_2d
        results['2d_to_3d_indices'] = map_2d_to_3d
        results['3d_to_2d_indices'] = map_3d_to_2d
        return results


@PIPELINES.register_module()
class FilterUnmatchedBboxes():
    """
    Filter 3D bounding boxes that do not have a corresponding 2D bounding box and vice versa.
    """
    def __init__(self, gt_prefix=None):
        """
        Args:
            prefix_3d (list[str]): List of prefixes for 3D bounding boxes.
                Default: ['monocular'].
        """
        self.prefix = gt_prefix
    def __call__(self, results):
        assert '2d_to_3d_indices' in results and '3d_to_2d_indices' in results, "AlignBoxes should be called before FilterUnmatchedBbox."
        gt_bboxes_key = f'{self.prefix}_gt_bboxes_3d' if self.prefix else 'gt_bboxes_3d'
        gt_labels_key = f'{self.prefix}_gt_labels_3d' if self.prefix else 'gt_labels_3d'
        if results[gt_bboxes_key].tensor.shape[0] > 0 and len(results['3d_to_2d_indices']) > 0:
            mask = results['3d_to_2d_indices'] != -1
            results[gt_bboxes_key].tensor = results[gt_bboxes_key].tensor[mask]
            results[gt_labels_key] = results[gt_labels_key][mask]

            results['3d_to_2d_indices'], results['2d_to_3d_indices'] = update_correspondance_indices(
                            results['3d_to_2d_indices'], 
                            results['2d_to_3d_indices'], 
                            mask_3d =mask
             )

        if results['centers2d'][0].shape[0] > 0 and len(results['2d_to_3d_indices']) > 0:            
            mask = results['2d_to_3d_indices'] != -1
            results['centers2d'][0] = results['centers2d'][0][mask]
            results['gt_bboxes'][0] = results['gt_bboxes'][0][mask]
            results['gt_labels'][0] = results['gt_labels'][0][mask]
            results['depths'][0] = results['depths'][0][mask]
            results['3d_to_2d_indices'], results['2d_to_3d_indices'] = update_correspondance_indices(
                            results['3d_to_2d_indices'], 
                            results['2d_to_3d_indices'], 
                            mask_2d =mask
             )
        return results


def update_correspondance_indices(idx_3d_to_2d, idx_2d_to_3d, mask_3d=None, mask_2d=None):
    """
    Update the correspondence indices between 3D and 2D bounding boxes based on the provided masks.
    If a box is removed (mask = False), the relevant index is dropped or marked -1.
    Index references to later elements are shifted appropriately.
    
    Args:
        idx_3d_to_2d (np.ndarray): Mapping from 3D box index to 2D box index.
        idx_2d_to_3d (np.ndarray): Mapping from 2D box index to 3D box index.
        mask_3d (np.ndarray): Boolean mask indicating which 3D boxes to keep.
        mask_2d (np.ndarray): Boolean mask indicating which 2D boxes to keep.
    
    Returns:
        (np.ndarray, np.ndarray): Updated idx_3d_to_2d, idx_2d_to_3d
    """
    assert mask_3d is not None or mask_2d is not None, "At least one of the masks must be provided"
    idx_3d_to_2d = np.array(idx_3d_to_2d, dtype=np.int64)
    idx_2d_to_3d = np.array(idx_2d_to_3d, dtype=np.int64)

    idx_3d_to_2d = idx_3d_to_2d.copy()
    idx_2d_to_3d = idx_2d_to_3d.copy()

    if mask_3d is not None:
        assert len(mask_3d) == len(idx_3d_to_2d), f"mask_3d must match length of idx_3d_to_2d got {len(mask_3d)} vs {len(idx_3d_to_2d)}"
        # old -> new 3D index (-1 if removed)
        map_old_to_new_3d = -np.ones(len(mask_3d), dtype=np.int64)
        kept_indices_3d = np.flatnonzero(mask_3d)
        map_old_to_new_3d[kept_indices_3d] = np.arange(len(kept_indices_3d))

        for i, ref_3d in enumerate(idx_2d_to_3d):
            if ref_3d == -1:
                continue
            if not mask_3d[ref_3d]:
                idx_2d_to_3d[i] = -1
            else:
                idx_2d_to_3d[i] = map_old_to_new_3d[ref_3d]

        idx_3d_to_2d = idx_3d_to_2d[mask_3d]

    if mask_2d is not None:
        assert len(mask_2d) == len(idx_2d_to_3d), "mask_2d must match length of idx_2d_to_3d"
        # old -> new 2D index (-1 if removed)
        map_old_to_new_2d = -np.ones(len(mask_2d), dtype=np.int64)
        kept_indices_2d = np.flatnonzero(mask_2d)
        map_old_to_new_2d[kept_indices_2d] = np.arange(len(kept_indices_2d))

        for i, ref_2d in enumerate(idx_3d_to_2d):
            if ref_2d == -1:
                continue
            if not mask_2d[ref_2d]:
                idx_3d_to_2d[i] = -1
            else:
                idx_3d_to_2d[i] = map_old_to_new_2d[ref_2d]

        idx_2d_to_3d = idx_2d_to_3d[mask_2d]
    return idx_3d_to_2d, idx_2d_to_3d