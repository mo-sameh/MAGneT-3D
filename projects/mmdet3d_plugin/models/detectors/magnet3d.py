# ------------------------------------------------------------------------
# MAGneT-3D
# ------------------------------------------------------------------------
# Modified from StreamPETR (https://github.com/exiawsh/StreamPETR)
# Copyright (c) Shihao Wang
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations


@DETECTORS.register_module()
class MAGneT3D(MVXTwoStageDetector):
    """MAGneT-3D: monocular temporal 3D detector.

    Instead of static learnable queries, the query head (``pts_bbox_head``,
    TRIM) is initialised every frame from 3D anchors predicted by the
    Domain-Robust Anchor Generator (``drag_head``, DRAG).

    Without ``pts_bbox_head`` only DRAG is trained (DRAG warm-up).
    """

    def __init__(self,
                 img_backbone=None,
                 img_neck=None,
                 img_roi_head=None,
                 drag_head=None,
                 pts_bbox_head=None,
                 use_grid_mask=False,
                 num_frame_head_grads=1,
                 num_frame_backbone_grads=1,
                 num_frame_losses=1,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(MAGneT3D, self).__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.drag_head = builder.build_head(drag_head)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.test_flag = False

    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Returns the features of ``position_level`` reshaped to
        (B, [T,] N, C, H, W), and all FPN levels (for DRAG)."""
        B = img.size(0)
        if img.dim() == 6:
            img = img.flatten(1, 2)
        if img.dim() == 5:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(
                B, len_queue, int(BN / B / len_queue), C, H, W)
        else:
            img_feats_reshaped = img_feats[self.position_level].view(
                B, int(BN / B / len_queue), C, H, W)
        return img_feats_reshaped, img_feats

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        return self.extract_img_feat(img, T, training_mode)

    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        return locations(x, self.stride, pad_h, pad_w)[None].repeat(
            bs * n, 1, 1, 1)

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or \
                not self.with_img_roi_head:
            return {'topk_indexes': None}
        return self.img_roi_head(location, **data)

    def forward_drag(self, full_feats, img_metas, **data):
        """Runs DRAG.

        Returns the raw DRAG outputs and the per-sample metas (with
        ``intrinsics`` and ``lidar2img``) used to decode them.
        """
        if 'lidar2img_inv' in data:
            lidar2img_inv = [
                data['lidar2img_inv'][i].clone().squeeze(0)
                for i in range(len(img_metas))
            ]
        else:  # pipelines without InverseLidarToImage
            lidar2img_inv = [
                torch.inverse(data['lidar2img'][i].clone().squeeze(0))
                for i in range(len(img_metas))
            ]
        drag_out = self.drag_head(full_feats, lidar2img_inv)
        metas = list(img_metas)
        for i, meta in enumerate(metas):
            meta['intrinsics'] = data['intrinsics'][i].clone().cpu().squeeze(0)
            meta['lidar2img'] = data['lidar2img'][i].clone().cpu().squeeze(0)
        return drag_out, metas

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          gt_bboxes_ignore=None,
                          full_feats=None,
                          return_losses=False,
                          **data):
        location = self.prepare_location(img_metas, **data)

        drag_out, metas = self.forward_drag(full_feats, img_metas, **data)
        drag_out_detached = [[
            val.detach().clone() if isinstance(val, torch.Tensor) else val
            for val in item
        ] for item in drag_out]
        anchors = self.drag_head.get_bboxes(*drag_out_detached, metas)
        centers = [item[0].center for item in anchors]

        outs_roi = self.forward_roi_head(location, **data)
        outs = self.pts_bbox_head(location, img_metas, outs_roi['topk_indexes'],
                                  centers, **data)
        if not return_losses:
            return None

        losses = dict()
        drag_losses = self.drag_head.loss(
            *drag_out,
            img_metas=metas,
            gt_bboxes=[val[0] for val in gt_bboxes],
            gt_labels=[val[0] for val in gt_labels],
            gt_bboxes_3d=list(data['monocular_gt_bboxes_3d']),
            gt_labels_3d=list(data['monocular_gt_labels_3d']),
            centers2d=[val[0] for val in centers2d],
            depths=[val[0] for val in depths],
            attr_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore[-1]
            if gt_bboxes_ignore is not None else None)
        for key, value in drag_losses.items():
            losses['anchor_head_' + key] = value

        losses.update(
            self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs,
                                    list(img_metas)))
        if self.with_img_roi_head:
            losses.update(
                self.img_roi_head.loss(gt_bboxes, gt_labels, centers2d,
                                       depths, outs_roi, img_metas))
        return losses

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        if return_loss:
            target_keys = [
                'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels',
                'centers2d', 'depths', 'img_metas',
                'monocular_gt_bboxes_3d', 'monocular_gt_labels_3d'
            ]
            for key in target_keys:
                data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      monocular_gt_bboxes_3d=None,
                      monocular_gt_labels_3d=None,
                      **data):
        if self.test_flag and self.with_pts_bbox:  # after an interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False

        T = data['img'].size(1)
        assert T - self.num_frame_head_grads <= 0, \
            'MAGneT-3D is trained in streaming mode (one frame per sample).'
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        rec_img_feats, full_feats = self.extract_feat(
            rec_img, self.num_frame_backbone_grads)
        if T - self.num_frame_backbone_grads > 0:
            prev_img = data['img'][:, :-self.num_frame_backbone_grads]
            self.eval()
            with torch.no_grad():
                prev_img_feats, _ = self.extract_feat(
                    prev_img, T - self.num_frame_backbone_grads, True)
            self.train()
            data['img_feats'] = torch.cat([prev_img_feats, rec_img_feats],
                                          dim=1)
        else:
            data['img_feats'] = rec_img_feats

        if not self.with_pts_bbox:
            return self.forward_drag_train(
                img_metas[-1], gt_bboxes[-1], gt_labels[-1], centers2d[-1],
                depths[-1], monocular_gt_bboxes_3d[-1],
                monocular_gt_labels_3d[-1], full_feats,
                {key: data[key][:, -1] for key in data})

        losses = dict()
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            data_t = {key: data[key][:, i] for key in data}
            data_t['monocular_gt_bboxes_3d'] = monocular_gt_bboxes_3d[i]
            data_t['monocular_gt_labels_3d'] = monocular_gt_labels_3d[i]
            gt_bboxes_ignore_i = gt_bboxes_ignore[i] \
                if gt_bboxes_ignore is not None else None
            loss = self.forward_pts_train(
                gt_bboxes_3d[i], gt_labels_3d[i], gt_bboxes[i], gt_labels[i],
                img_metas[i], centers2d[i], depths[i], gt_bboxes_ignore_i,
                full_feats=full_feats,
                return_losses=i >= num_grad_losses,
                **data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses[f'frame_{i}_{key}'] = value
        return losses

    def forward_drag_train(self, img_metas, gt_bboxes, gt_labels, centers2d,
                           depths, gt_bboxes_3d, gt_labels_3d, full_feats,
                           data):
        """DRAG warm-up: only the FCOS3D losses of DRAG."""
        drag_out, metas = self.forward_drag(full_feats, img_metas, **data)
        drag_losses = self.drag_head.loss(
            *drag_out,
            img_metas=metas,
            gt_bboxes=[val[0] for val in gt_bboxes],
            gt_labels=[val[0] for val in gt_labels],
            gt_bboxes_3d=list(gt_bboxes_3d),
            gt_labels_3d=list(gt_labels_3d),
            centers2d=[val[0] for val in centers2d],
            depths=[val[0] for val in depths],
            attr_labels=None)
        return {f'frame_0_anchor_head_{k}': v for k, v in drag_losses.items()}

    def forward_test(self, img_metas, rescale=None, **data):
        self.test_flag = True
        if not isinstance(img_metas, list):
            raise TypeError(
                f'img_metas must be a list, but got {type(img_metas)}')
        for key in data:
            if key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, full_feats, **data):
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)

        # reset the temporal memory at the start of every sequence
        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        drag_out, metas = self.forward_drag(full_feats, img_metas, **data)
        anchors = self.drag_head.get_bboxes(*drag_out, metas)
        centers = [item[0].center for item in anchors]

        outs = self.pts_bbox_head(location, img_metas, outs_roi['topk_indexes'],
                                  centers, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
        return [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

    def simple_test_drag(self, img_metas, full_feats, **data):
        """DRAG-only model (anchor warm-up): its anchors are the detections."""
        drag_out, metas = self.forward_drag(full_feats, img_metas, **data)
        anchors = self.drag_head.get_bboxes(*drag_out, metas)
        return [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels, _ in anchors
        ]

    def simple_test(self, img_metas, **data):
        data['img_feats'], full_feats = self.extract_img_feat(data['img'], 1)
        test = self.simple_test_pts if self.with_pts_bbox else self.simple_test_drag
        return [dict(pts_bbox=pts_bbox) for pts_bbox in test(img_metas, full_feats, **data)]
