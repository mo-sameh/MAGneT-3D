# ------------------------------------------------------------------------
# MAGneT-3D
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.runner import force_fp32
from torch import nn as nn

from mmdet3d.core import box3d_multiclass_nms, xywhr2xyxyr
from mmdet.core import multi_apply
from mmdet3d.models.builder import HEADS
from mmdet3d.models.dense_heads import FCOSMono3DHead

from projects.mmdet3d_plugin.core.bbox.util import points_img2cam

INF = 1e8


@HEADS.register_module()
class DRAGHead(FCOSMono3DHead):
    """Domain-Robust Anchor Generator (DRAG).

    An FCOS3D head that predicts per-frame 3D anchors for the query head.
    Compared to FCOS3D it adds:

    * **Virtual depth** (``use_virtual_depth``): the network regresses
      ``z / f`` and the metric depth is recovered with the focal length ``f``
      of the (augmented) image, which makes the backbone features independent
      of the camera intrinsics.
    * **Scale-invariant feature assignment** (``normalized_regress_ranges``):
      ``regress_ranges`` and ``center_sample_radius`` are given in units of
      the focal length (i.e. divided by a reference focal length) and are
      rescaled with the focal length of each image, so an object is always
      assigned to the same FPN level regardless of the intrinsics.
    * **Ray map** (``append_ray_map``): the normalised viewing ray of every
      feature location is embedded and concatenated to the features.

    Boxes are predicted in the (pseudo-)LiDAR frame, i.e. ``lidar2img`` is
    used instead of ``cam2img``. The focal length is read from
    ``lidar2img[0, 0]``, which is valid because the pseudo-LiDAR frame is an
    axis permutation of the camera frame.
    """

    def __init__(self,
                 append_ray_map=True,
                 normalized_regress_ranges=True,
                 use_virtual_depth=True,
                 bbox_coder=dict(type='DRAGBBoxCoder', code_size=9),
                 **kwargs):
        self.append_ray_map = append_ray_map
        self.normalized_regress_ranges = normalized_regress_ranges
        self.use_virtual_depth = use_virtual_depth
        if append_ray_map:
            # the ray map is embedded to 32 channels and concatenated
            kwargs['in_channels'] += 32
        super().__init__(bbox_coder=bbox_coder, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        if self.append_ray_map:
            self.ray_embed = nn.Conv2d(3, 32, kernel_size=1)

    def init_weights(self):
        super().init_weights()
        if self.append_ray_map:
            normal_init(self.ray_embed, std=0.01)

    def forward(self, feats, lidar2img_inv):
        """
        Args:
            feats (list[Tensor]): FPN features.
            lidar2img_inv (list[Tensor]): (4, 4) inverse projection per image.
        """
        focal_lengths = [1 / m[0, 0] for m in lidar2img_inv]
        num_levels = len(feats)
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides, [lidar2img_inv] * num_levels,
                           [focal_lengths] * num_levels)[:5]

    def ray_map(self, x, stride, lidar2img_inv):
        """Unit viewing rays of every feature location, (B, 3, H, W)."""
        featmap_size = x.size()[-2:]
        batch_size = x.size(0)
        points = self._get_points_single(featmap_size, stride, x.dtype, x.device)
        points = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
        points = points.t().unsqueeze(0).expand(batch_size, -1, -1)
        img2lidar = torch.stack([m[:3, :3] for m in lidar2img_inv], dim=0)
        rays = F.normalize(img2lidar @ points, p=2, dim=1)
        return rays.reshape(batch_size, 3, featmap_size[0], featmap_size[1])

    def forward_single(self, x, scale, stride, lidar2img_inv, focal_lengths):
        if self.append_ray_map:
            x = torch.cat([x, self.ray_embed(self.ray_map(x, stride, lidar2img_inv))], dim=1)
        outs = super().forward_single(x, scale, stride)
        bbox_pred = outs[1]
        if self.use_virtual_depth:
            # virtual depth -> metric depth
            focal_lengths = torch.stack(focal_lengths).to(bbox_pred.device)
            bbox_pred[:, 2, :, :] = bbox_pred[:, 2, :, :] * focal_lengths.view(-1, 1, 1)
        return outs

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds', 'attr_preds',
                  'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             attr_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             centers2d,
             depths,
             attr_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """FCOS3D loss with focal-length aware target assignment."""
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(attr_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        focal_lengths = [
            torch.tensor(img_meta['lidar2img'][0, 0], dtype=bbox_preds[0].dtype,
                         device=bbox_preds[0].device) for img_meta in img_metas
        ]
        labels_3d, bbox_targets_3d, centerness_targets, attr_targets = \
            self.get_targets(all_level_points, gt_bboxes, gt_labels,
                             gt_bboxes_3d, gt_labels_3d, centers2d, depths,
                             attr_labels, focal_lengths)
        num_points_per_lvl = [t.shape[0] for t in centerness_targets]

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ])
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, sum(self.group_reg_dims))
            for bbox_pred in bbox_preds
        ])
        flatten_dir_cls_preds = torch.cat([
            dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            for dir_cls_pred in dir_cls_preds
        ])
        flatten_centerness = torch.cat([
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ])
        flatten_labels_3d = torch.cat(labels_3d).long()
        flatten_bbox_targets_3d = torch.cat(bbox_targets_3d)
        flatten_centerness_targets = torch.cat(centerness_targets)
        flatten_level = torch.cat([
            torch.full((n, ), i, dtype=torch.int64, device=flatten_cls_scores.device)
            for i, n in enumerate(num_points_per_lvl)
        ])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels_3d >= 0)
                    & (flatten_labels_3d < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        num_pos_per_level = [
            int((flatten_level[pos_inds] == i).sum()) for i in range(len(num_points_per_lvl))
        ]

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels_3d,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if self.pred_attrs:
            flatten_attr_preds = torch.cat([
                attr_pred.permute(0, 2, 3, 1).reshape(-1, self.num_attrs)
                for attr_pred in attr_preds
            ])
            flatten_attr_targets = torch.cat(attr_targets)
            pos_attr_preds = flatten_attr_preds[pos_inds]

        loss_velo = loss_dir = loss_attr = None
        if num_pos > 0:
            pos_bbox_targets_3d = flatten_bbox_targets_3d[pos_inds]
            pos_centerness_targets = flatten_centerness_targets[pos_inds]
            if self.pred_attrs:
                pos_attr_targets = flatten_attr_targets[pos_inds]
            bbox_weights = pos_centerness_targets.new_ones(
                len(pos_centerness_targets), sum(self.group_reg_dims))
            equal_weights = pos_centerness_targets.new_ones(pos_centerness_targets.shape)
            avg_factor = equal_weights.sum()

            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                assert len(code_weight) == sum(self.group_reg_dims)
                bbox_weights = bbox_weights * bbox_weights.new_tensor(code_weight)

            if self.use_direction_classifier:
                pos_dir_cls_targets = self.get_direction_target(
                    pos_bbox_targets_3d, self.dir_offset, self.dir_limit_offset,
                    one_hot=False)
            if self.diff_rad_by_sin:
                pos_bbox_preds, pos_bbox_targets_3d = self.add_sin_difference(
                    pos_bbox_preds, pos_bbox_targets_3d)

            loss_offset = self.loss_bbox(
                pos_bbox_preds[:, :2], pos_bbox_targets_3d[:, :2],
                weight=bbox_weights[:, :2], avg_factor=avg_factor)
            loss_depth = self.loss_bbox(
                pos_bbox_preds[:, 2], pos_bbox_targets_3d[:, 2],
                weight=bbox_weights[:, 2], avg_factor=avg_factor)
            loss_size = self.loss_bbox(
                pos_bbox_preds[:, 3:6], pos_bbox_targets_3d[:, 3:6],
                weight=bbox_weights[:, 3:6], avg_factor=avg_factor)
            loss_rotsin = self.loss_bbox(
                pos_bbox_preds[:, 6], pos_bbox_targets_3d[:, 6],
                weight=bbox_weights[:, 6], avg_factor=avg_factor)
            if self.pred_velo:
                loss_velo = self.loss_bbox(
                    pos_bbox_preds[:, 7:9], pos_bbox_targets_3d[:, 7:9],
                    weight=bbox_weights[:, 7:9], avg_factor=avg_factor)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets,
                weight=equal_weights, avg_factor=avg_factor)
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_preds, pos_dir_cls_targets, equal_weights,
                    avg_factor=avg_factor)
            if self.pred_attrs:
                loss_attr = self.loss_attr(
                    pos_attr_preds, pos_attr_targets, pos_centerness_targets,
                    avg_factor=pos_centerness_targets.sum())
        else:
            loss_offset = pos_bbox_preds[:, :2].sum()
            loss_depth = pos_bbox_preds[:, 2].sum()
            loss_size = pos_bbox_preds[:, 3:6].sum()
            loss_rotsin = pos_bbox_preds[:, 6].sum()
            if self.pred_velo:
                loss_velo = pos_bbox_preds[:, 7:9].sum()
            loss_centerness = pos_centerness.sum()
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_preds.sum()
            if self.pred_attrs:
                loss_attr = pos_attr_preds.sum()

        loss_dict = dict(
            loss_cls=loss_cls,
            loss_offset=loss_offset,
            loss_depth=loss_depth,
            loss_size=loss_size,
            loss_rotsin=loss_rotsin,
            loss_centerness=loss_centerness)
        if loss_velo is not None:
            loss_dict['loss_velo'] = loss_velo
        if loss_dir is not None:
            loss_dict['loss_dir'] = loss_dir
        if loss_attr is not None:
            loss_dict['loss_attr'] = loss_attr
        # number of positives per FPN level (logged only)
        for i, num in enumerate(num_pos_per_level):
            loss_dict[f'num_pos_lvl_{i}'] = torch.tensor(
                num, dtype=torch.float32, device=loss_cls.device)
        return loss_dict

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           dir_cls_preds,
                           attr_preds,
                           centernesses,
                           mlvl_points,
                           input_meta,
                           cfg,
                           rescale=False):
        """Decodes the boxes of one image (in the pseudo-LiDAR frame)."""
        view = np.array(input_meta['lidar2img'])
        intrinsic = np.array(input_meta['intrinsics'])
        scale_factor = input_meta['scale_factor']
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        mlvl_attr_scores = []
        mlvl_centerness = []

        for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
                points in zip(cls_scores, bbox_preds, dir_cls_preds,
                              attr_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, sum(self.group_reg_dims))
            bbox_pred = bbox_pred[:, :self.bbox_code_size]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                attr_score = attr_score[topk_inds]
            # change the offset to actual center predictions
            bbox_pred[:, :2] = points - bbox_pred[:, :2]
            if rescale:
                bbox_pred[:, :2] /= bbox_pred[:, :2].new_tensor(scale_factor)
            pred_center2d = bbox_pred[:, :3].clone()
            # back-project (u, v, z) with the full lidar2img projection
            bbox_pred[:, :3] = points_img2cam(bbox_pred[:, :3], view)
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)

        mlvl_centers2d = torch.cat(mlvl_centers2d)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        # change local yaw to global yaw for 3D nms
        cam2img = mlvl_centers2d.new_zeros((4, 4))
        cam2img[:intrinsic.shape[0], :intrinsic.shape[1]] = mlvl_centers2d.new_tensor(intrinsic)
        mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d,
                                                 mlvl_dir_scores,
                                                 self.dir_offset, cam2img)

        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5)).bev)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # FG labels are [0, num_class-1], BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_attr_scores = torch.cat(mlvl_attr_scores)
        mlvl_centerness = torch.cat(mlvl_centerness)
        mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_nms_scores, cfg.score_thr,
                                       cfg.max_per_img, cfg, mlvl_dir_scores,
                                       mlvl_attr_scores)
        bboxes, scores, labels, dir_scores, attrs = results
        attrs = attrs.to(labels.dtype)
        # gt centers2d are gravity centers, hence origin (0.5, 0.5, 0.5)
        bboxes = input_meta['box_type_3d'](
            bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
        if not self.pred_attrs:
            attrs = None
        return bboxes, scores, labels, attrs

    def get_targets(self, points, gt_bboxes_list, gt_labels_list,
                    gt_bboxes_3d_list, gt_labels_3d_list, centers2d_list,
                    depths_list, attr_labels_list, focal_lengths):
        """FCOS3D targets; ``focal_lengths`` rescales the focal-normalized
        regress ranges and center-sampling radius per image."""
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i])
            for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]

        if attr_labels_list is None:
            attr_labels_list = [
                gt_labels.new_full(gt_labels.shape, self.attr_background_label)
                for gt_labels in gt_labels_list
            ]

        _, _, labels_3d_list, bbox_targets_3d_list, centerness_targets_list, \
            attr_targets_list = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                gt_bboxes_3d_list,
                gt_labels_3d_list,
                centers2d_list,
                depths_list,
                attr_labels_list,
                focal_lengths,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        # split to per img, per level
        labels_3d_list = [labels_3d.split(num_points, 0) for labels_3d in labels_3d_list]
        bbox_targets_3d_list = [t.split(num_points, 0) for t in bbox_targets_3d_list]
        centerness_targets_list = [t.split(num_points, 0) for t in centerness_targets_list]
        attr_targets_list = [t.split(num_points, 0) for t in attr_targets_list]

        # concat per level image
        concat_lvl_labels_3d = []
        concat_lvl_bbox_targets_3d = []
        concat_lvl_centerness_targets = []
        concat_lvl_attr_targets = []
        for i in range(num_levels):
            concat_lvl_labels_3d.append(torch.cat([labels[i] for labels in labels_3d_list]))
            concat_lvl_centerness_targets.append(
                torch.cat([t[i] for t in centerness_targets_list]))
            bbox_targets_3d = torch.cat([t[i] for t in bbox_targets_3d_list])
            concat_lvl_attr_targets.append(torch.cat([t[i] for t in attr_targets_list]))
            if self.norm_on_bbox:
                bbox_targets_3d[:, :2] = bbox_targets_3d[:, :2] / self.strides[i]
            concat_lvl_bbox_targets_3d.append(bbox_targets_3d)
        return concat_lvl_labels_3d, concat_lvl_bbox_targets_3d, \
            concat_lvl_centerness_targets, concat_lvl_attr_targets

    def _get_target_single(self, gt_bboxes, gt_labels, gt_bboxes_3d,
                           gt_labels_3d, centers2d, depths, attr_labels,
                           focal_length, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if not isinstance(gt_bboxes_3d, torch.Tensor):
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label, dtype=torch.long), \
                gt_bboxes.new_zeros((num_points, 4)), \
                gt_labels_3d.new_full((num_points,), self.background_label, dtype=torch.long), \
                gt_bboxes_3d.new_zeros((num_points, self.bbox_code_size)), \
                gt_bboxes_3d.new_zeros((num_points,)), \
                attr_labels.new_full((num_points,), self.attr_background_label)

        # change orientation to local yaw (pseudo-LiDAR frame)
        gt_bboxes_3d[..., 6] = gt_bboxes_3d[..., 6] - torch.atan2(
            gt_bboxes_3d[..., 1], gt_bboxes_3d[..., 0])

        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        centers2d = centers2d[None].expand(num_points, num_gts, 2)
        gt_bboxes_3d = gt_bboxes_3d[None].expand(num_points, num_gts, self.bbox_code_size)
        depths = depths[None, :, None].expand(num_points, num_gts, 1)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        delta_xs = (xs - centers2d[..., 0])[..., None]
        delta_ys = (ys - centers2d[..., 1])[..., None]
        bbox_targets_3d = torch.cat((delta_xs, delta_ys, depths, gt_bboxes_3d[..., 3:]), dim=-1)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        assert self.center_sampling is True, \
            'Setting center_sampling to False has not been implemented for FCOS3D.'
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        if self.normalized_regress_ranges:
            radius = radius * focal_length
        center_xs = centers2d[..., 0]
        center_ys = centers2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride
        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack((cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        if self.normalized_regress_ranges:
            inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0] * focal_length)
                & (max_regress_distance <= regress_ranges[..., 1] * focal_length))
        else:
            inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # center-based criterion to deal with ambiguity
        dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2]**2, dim=-1))
        dists[inside_gt_bbox_mask == 0] = INF
        dists[inside_regress_range == 0] = INF
        min_dist, min_dist_inds = dists.min(dim=1)

        labels = gt_labels[min_dist_inds]
        labels_3d = gt_labels_3d[min_dist_inds]
        attr_labels = attr_labels[min_dist_inds]
        labels[min_dist == INF] = self.background_label  # set as BG
        labels_3d[min_dist == INF] = self.background_label  # set as BG
        attr_labels[min_dist == INF] = self.attr_background_label

        bbox_targets = bbox_targets[range(num_points), min_dist_inds]
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_dist_inds]
        relative_dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2]**2, dim=-1)) / (
            1.414 * stride[:, 0])
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists)
        return labels, bbox_targets, labels_3d, bbox_targets_3d, \
            centerness_targets, attr_labels
