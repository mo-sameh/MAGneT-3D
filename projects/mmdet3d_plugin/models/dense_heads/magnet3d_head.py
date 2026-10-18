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
import torch.nn as nn
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.models.utils.misc import (MLN, SELayer_Linear,
                                                       memory_refresh,
                                                       topk_gather,
                                                       transform_reference_points)
from projects.mmdet3d_plugin.models.utils.positional_encoding import (
    nerf_positional_encoding, pos2posemb1d, pos2posemb3d)


@HEADS.register_module()
class MAGneT3DHead(AnchorFreeHead):
    """Temporal prediction refiner of MAGneT-3D (TRIM).

    A StreamPETR-style query head whose queries are the 3D anchors predicted by
    DRAG (plus queries propagated from the memory bank). Compared to
    StreamPETR it

    * builds its queries from ``anchor_centers`` instead of learnable
      reference points (``num_query=0``),
    * is supervised with a dense soft assignment (``SoftTopKAssigner3D``),
    * predicts an identity embedding per query trained with a supervised
      contrastive loss (``loss_reid``), used at inference by the bbox coder to
      merge duplicate predictions.

    Args:
        num_query (int): Number of additional learnable reference points.
        memory_len (int): Number of queries kept in the memory bank.
        topk_proposals (int): Number of queries pushed into memory per frame.
        num_propagated (int): Number of memory queries used as extra queries.
        reid_dim (int): Dimension of the identity embedding.
        with_dn (bool): Use query denoising during training.
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 stride=16,
                 embed_dims=256,
                 num_query=0,
                 num_reg_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_propagated=256,
                 reid_dim=128,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=0.25),
                 loss_reid=dict(
                     type='SupConLoss', temperature=0.15, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=dict(max_per_img=100),
                 depth_num=64,
                 LID=False,
                 depth_start=1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 scalar=5,
                 noise_scale=0.4,
                 noise_trans=0.0,
                 dn_weight=1.0,
                 split=0.5,
                 init_cfg=None,
                 **kwargs):
        self.code_size = kwargs.get('code_size', 10)
        if code_weights is None:
            code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = code_weights[:self.code_size]
        self.match_costs = match_costs if match_costs is not None \
            else self.code_weights
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        if train_cfg:
            assert 'assigner' in train_cfg, \
                'assigner should be provided when train_cfg is set.'
            self.assigner = build_assigner(train_cfg['assigner'])
            self.sampler = build_sampler(dict(type='PseudoSampler'), context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.LID = LID
        self.depth_start = depth_start
        self.stride = stride
        self.reid_dim = reid_dim
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split
        self.act_cfg = transformer.get('act_cfg', dict(type='ReLU', inplace=True))
        self.num_pred = 6
        super(MAGneT3DHead, self).__init__(num_classes, in_channels, init_cfg=init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_reid = build_loss(loss_reid)
        self.cls_out_channels = num_classes if self.loss_cls.use_sigmoid \
            else num_classes + 1

        self.transformer = build_transformer(transformer)
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights), requires_grad=False)
        self.match_costs = nn.Parameter(
            torch.tensor(self.match_costs), requires_grad=False)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = nn.Parameter(
            torch.tensor(self.bbox_coder.pc_range), requires_grad=False)
        self.position_range = nn.Parameter(
            torch.tensor(position_range), requires_grad=False)

        index = torch.arange(start=0, end=self.depth_num, step=1).float()
        if self.LID:
            bin_size = (self.position_range[3] - self.depth_start) / (
                self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * (index + 1)
        else:
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index
        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        reid_branch = []
        for _ in range(self.num_reg_fcs):
            reid_branch.append(Linear(self.embed_dims, self.embed_dims))
            reid_branch.append(nn.ReLU())
        reid_branch.append(Linear(self.embed_dims, self.reid_dim))
        reid_branch = nn.Sequential(*reid_branch)

        # the prediction branches are shared across decoder layers
        self.cls_branches = nn.ModuleList([fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList([reg_branch for _ in range(self.num_pred)])
        self.reid_branches = nn.ModuleList([reid_branch for _ in range(self.num_pred)])

        self.position_encoder = nn.Sequential(
            nn.Linear(self.position_dim, self.embed_dims * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        self.memory_embed = nn.Sequential(
            nn.Linear(self.in_channels, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.featurized_pe = SELayer_Linear(self.embed_dims)
        self.reference_points = nn.Embedding(self.num_query, 3) \
            if self.num_query > 0 else None
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)

        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.spatial_alignment = MLN(8)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims))
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

    def init_weights(self):
        if self.reference_points is not None:
            nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # bypass AnchorFreeHead._load_from_state_dict
        super(AnchorFreeHead, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None

    def pre_update_memory(self, data):
        x = data['prev_exists']
        B = x.size(0)
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 3)
            self.memory_timestamp = x.new_zeros(B, self.memory_len, 1)
            self.memory_egopose = x.new_zeros(B, self.memory_len, 4, 4)
            self.memory_velo = x.new_zeros(B, self.memory_len, 2)
        else:
            # move the memory into the current ego frame and reset it when a
            # new sequence starts (prev_exists == 0)
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points(
                self.memory_reference_point, data['ego_pose_inv'], reverse=False)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)

        # for the first frame, pad with (non-learnable) pseudo reference points
        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (
                self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.memory_reference_point[:, :self.num_propagated] = \
                self.memory_reference_point[:, :self.num_propagated] + \
                (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated] = \
                self.memory_egopose[:, :self.num_propagated] + \
                (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores,
                           all_bbox_preds, outs_dec, mask_dict):
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            pad = mask_dict['pad_size']
            rec_reference_points = all_bbox_preds[:, :, pad:, :3][-1]
            rec_velo = all_bbox_preds[:, :, pad:, -2:][-1]
            rec_memory = outs_dec[:, :, pad:, :][-1]
            rec_score = all_cls_scores[:, :, pad:, :][-1].sigmoid().topk(
                1, dim=-1).values[..., 0:1]
        else:
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            rec_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)

        # push the top-k queries into the memory bank
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose = torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat(
            [rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        self.memory_reference_point = transform_reference_points(
            self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose

    def position_embeding(self, data, memory_centers, topk_indexes, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)
        # focal lengths (fx, fy) in units of 1000 px for the spatial alignment
        intrinsic = torch.stack(
            [data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H * W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)
        num_sample_tokens = topk_indexes.size(1) if topk_indexes is not None else LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]
        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = topk_gather(memory_centers, topk_indexes).repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1, 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)
        coords = coords.unsqueeze(-1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H * W, D, 1, 1).view(B, LEN, D, 4, 4)
        img2lidars = topk_gather(img2lidars, topk_indexes)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (
            self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D * 3)

        pos_embed = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)
        intrinsic = topk_gather(intrinsic, topk_indexes)

        # spatial alignment as in Focal-PETR
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)
        return coords_position_embeding, cone

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (
            self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point))
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(
            B, query_pos.size(1), 1, 1)

        if self.with_ego_pos:
            # ego-motion conditioning; velocity and time stamps are not used
            rec_ego_motion = torch.cat([
                torch.zeros_like(reference_points[..., :3]),
                rec_ego_pose[..., :3, :].flatten(-2)
            ], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            memory_ego_motion = torch.cat([
                torch.zeros_like(self.memory_reference_point[..., :3]),
                self.memory_egopose[..., :3, :].flatten(-2)
            ], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos += self.time_embedding(
            pos2posemb1d(torch.zeros_like(reference_points[..., :1])))
        temp_pos += self.time_embedding(
            pos2posemb1d(torch.zeros_like(self.memory_timestamp).float()))

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat(
                [reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(
                B, query_pos.shape[1] + self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]

        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        num_query = reference_points.size(1)
        if not (self.training and self.with_dn):
            return reference_points, None, None

        targets = [
            torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center,
                       img_meta['gt_bboxes_3d']._data.tensor[:, 3:]), dim=1)
            for img_meta in img_metas
        ]
        labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas]
        known = [(torch.ones_like(t)).cuda() for t in labels]
        know_idx = known
        unmask_bbox = unmask_label = torch.cat(known)
        known_num = [t.size(0) for t in targets]

        labels = torch.cat([t for t in labels])
        boxes = torch.cat([t for t in targets])
        batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox).view(-1)
        known_indice = known_indice.repeat(self.scalar, 1).view(-1)
        known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
        known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
        known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
        known_bbox_center = known_bboxs[:, :3].clone()
        known_bbox_scale = known_bboxs[:, 3:6].clone()

        if self.bbox_noise_scale > 0:
            diff = known_bbox_scale / 2 + self.bbox_noise_trans
            rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
            known_bbox_center += torch.mul(rand_prob, diff) * self.bbox_noise_scale
            known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (
                self.pc_range[3:6] - self.pc_range[0:3])
            known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
            mask = torch.norm(rand_prob, 2, 1) > self.split
            known_labels[mask] = self.num_classes

        single_pad = int(max(known_num))
        pad_size = int(single_pad * self.scalar)
        padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
        padded_reference_points = torch.cat(
            [padding_bbox.unsqueeze(0).expand(batch_size, -1, -1), reference_points], dim=1)

        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat(
                [map_known_indice + single_pad * i for i in range(self.scalar)]).long()
        if len(known_bid):
            padded_reference_points[(known_bid.long(), map_known_indice)] = \
                known_bbox_center.to(reference_points.device)

        tgt_size = pad_size + num_query
        attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
        # matching queries cannot see the denoising queries
        attn_mask[pad_size:, :pad_size] = True
        # denoising groups cannot see each other
        for i in range(self.scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == self.scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

        # extend the mask to the propagated queries and the memory
        query_size = pad_size + num_query + self.num_propagated
        tgt_size = pad_size + num_query + self.memory_len
        temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
        temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask
        temporal_attn_mask[pad_size:, :pad_size] = True
        attn_mask = temporal_attn_mask

        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'know_idx': know_idx,
            'known_pid': known_bid,
            'pad_size': pad_size
        }
        return padded_reference_points, attn_mask, mask_dict

    def anchors_to_reference_points(self, anchor_centers, device):
        """Normalises the DRAG anchors (one (N_i, 3) tensor per sample) and pads
        them to a (B, N, 3) tensor of reference points in [0, 1]."""
        anchor_centers = [
            anchor[(anchor[:, 0] >= self.pc_range[0]) & (anchor[:, 0] <= self.pc_range[3]) &
                   (anchor[:, 1] >= self.pc_range[1]) & (anchor[:, 1] <= self.pc_range[4]) &
                   (anchor[:, 2] >= self.pc_range[2]) & (anchor[:, 2] <= self.pc_range[5])]
            for anchor in anchor_centers
        ]
        anchor_centers = [
            (anchor - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[:3])
            for anchor in anchor_centers
        ]
        learned = self.reference_points.weight if self.reference_points is not None \
            else torch.empty(0, 3, device=device)
        reference_points = []
        for anchor in anchor_centers:
            if anchor.shape[0] == 0:
                reference_points.append(
                    learned if learned.shape[0] > 0 else torch.zeros(1, 3, device=device))
            else:
                reference_points.append(torch.cat((anchor.to(learned.device), learned), dim=0))
        max_len = max([ref.shape[0] for ref in reference_points])
        reference_points = [
            torch.cat((ref, torch.zeros(max_len - ref.shape[0], 3, device=ref.device)), dim=0)
            for ref in reference_points
        ]
        return torch.stack(reference_points, dim=0)

    def forward(self, memory_center, img_metas, topk_indexes=None,
                anchor_centers=None, **data):
        """
        Args:
            memory_center (Tensor): Normalised pixel centers of the image
                tokens, shape (B*N, H, W, 2).
            anchor_centers (list[Tensor]): DRAG anchors per sample, (N_i, 3)
                in the (pseudo-)LiDAR frame.
        Returns:
            dict: per decoder layer class scores, boxes and identity embeddings.
        """
        reference_points = self.anchors_to_reference_points(
            anchor_centers, memory_center.device)
        self.pre_update_memory(data)

        x = data['img_feats']
        B, N, C, H, W = x.shape
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, N * H * W, C)
        memory = topk_gather(memory, topk_indexes)
        pos_embed, cone = self.position_embeding(data, memory_center, topk_indexes, img_metas)

        memory = self.memory_embed(memory)
        memory = self.spatial_alignment(memory, cone)
        pos_embed = self.featurized_pe(pos_embed, memory)

        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas)
        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        tgt = torch.zeros_like(query_pos)
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = \
            self.temporal_alignment(query_pos, tgt, reference_points)

        outs_dec, _ = self.transformer(memory, tgt, query_pos, pos_embed,
                                       attn_mask, temp_memory, temp_pos)
        outs_dec = torch.nan_to_num(outs_dec)

        outputs_classes, outputs_coords, outputs_reid = [], [], []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()
            outputs_reid.append(self.reid_branches[lvl](outs_dec[lvl]))
            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_reid_feats = torch.stack(outputs_reid)
        all_bbox_preds[..., 0:3] = all_bbox_preds[..., 0:3] * (
            self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]

        self.post_update_memory(data, rec_ego_pose, all_cls_scores,
                                all_bbox_preds, outs_dec, mask_dict)

        if mask_dict and mask_dict['pad_size'] > 0:
            pad = mask_dict['pad_size']
            mask_dict['output_known_lbs_bboxes'] = (all_cls_scores[:, :, :pad, :],
                                                    all_bbox_preds[:, :, :pad, :])
            mask_dict['output_known_reid'] = all_reid_feats[:, :, :pad, :]
            return {
                'all_cls_scores': all_cls_scores[:, :, pad:, :],
                'all_bbox_preds': all_bbox_preds[:, :, pad:, :],
                'all_reid_feats': all_reid_feats[:, :, pad:, :],
                'dn_mask_dict': mask_dict,
            }
        return {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'all_reid_feats': all_reid_feats,
            'dn_mask_dict': None,
        }

    def prepare_for_loss(self, mask_dict):
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        output_known_reid = mask_dict['output_known_reid']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long().cpu()
        batch_idx = mask_dict['batch_idx'].long()
        known_bid = mask_dict['known_pid'].long()
        # denoising queries of the same gt share an identity
        reid_labels = known_indice + (known_bid * 1000)

        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_reid = output_known_reid.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, reid_labels, output_known_class, \
            output_known_coord, output_known_reid, num_tgt

    def _get_target_single(self, cls_score, bbox_pred, gt_labels, gt_bboxes,
                           layer_idx):
        num_bboxes = bbox_pred.size(0)
        gt_labels = gt_labels.to(torch.int64)
        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, None,
            self.match_costs, self.match_with_velo, layer=layer_idx)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        reg_pos_inds = sampling_result.pos_inds

        labels = gt_bboxes.new_full((num_bboxes, ), self.num_classes, dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes, dtype=torch.float)
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        if sampling_result.num_gts > 0:
            bbox_targets[reg_pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[reg_pos_inds] = 1.0

        # classification positives; the other matched queries only regress
        pos_inds = (assign_result.labels >= 0).nonzero(as_tuple=True)[0]
        labels[pos_inds] = assign_result.labels[pos_inds]
        neg_inds = (assign_result.labels == self.num_classes).nonzero(as_tuple=True)[0]
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, assign_result.gt_inds)

    def get_targets(self, cls_scores_list, bbox_preds_list, gt_bboxes_list,
                    gt_labels_list, layer_idx):
        """Targets of one decoder layer for all samples."""
        num_imgs = len(cls_scores_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list, assign_indices) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list,
             [layer_idx for _ in range(num_imgs)])
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, assign_indices)

    def loss_single(self, cls_scores, bbox_preds, reid_feats, gt_bboxes_list,
                    gt_labels_list, layer_idx):
        """Loss of one decoder layer."""
        num_imgs = cls_scores.size(0)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, assign_indices) = self.get_targets(
             [cls_scores[i] for i in range(num_imgs)],
             [bbox_preds[i] for i in range(num_imgs)],
             gt_bboxes_list, gt_labels_list, layer_idx)
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # identity (re-id) loss: queries assigned to the same gt are positives.
        # gt indices are offset per sample so identities are unique in a batch.
        reid_labels = torch.stack(assign_indices, dim=0)
        batch_offset = torch.arange(num_imgs, device=bbox_preds.device) * 1000
        reid_labels = reid_labels + batch_offset.view(-1, 1)
        labels_flat = reid_labels.reshape(-1)
        fg_labels = labels_flat[labels_flat % 1000 != 0]
        if fg_labels.numel() == 0:
            reid_avg_factor = 1.0
        else:
            # normalise by the number of queries with at least one positive
            _, inverse, counts = torch.unique(
                fg_labels, return_inverse=True, return_counts=True)
            num_valid_anchors = (counts[inverse] > 1).sum()
            reid_avg_factor = torch.clamp(
                reduce_mean(num_valid_anchors.float()), min=1).item()
        loss_reid = self.loss_reid(
            reid_feats.reshape(-1, reid_feats.size(-1)), labels_flat,
            avg_factor=reid_avg_factor)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # regression loss
        bbox_avg_factor = torch.clamp(reduce_mean(bbox_weights[:, 0].sum()), min=1).item()
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10], avg_factor=bbox_avg_factor)

        return torch.nan_to_num(loss_cls), torch.nan_to_num(loss_bbox), \
            torch.nan_to_num(loss_reid)

    def dn_loss_single(self, cls_scores, bbox_preds, reid_embeds, known_bboxs,
                       known_labels, reid_labels, num_total_pos=None):
        """Denoising loss of one decoder layer."""
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 3.14159 / 6 * self.split * self.split * self.split
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, known_labels.long(), label_weights,
                                 avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_reid = self.loss_reid(
            reid_embeds.reshape(-1, reid_embeds.size(-1)), reid_labels.reshape(-1),
            avg_factor=num_total_pos)

        return self.dn_weight * torch.nan_to_num(loss_cls), \
            self.dn_weight * torch.nan_to_num(loss_bbox), \
            self.dn_weight * torch.nan_to_num(loss_reid)

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, img_metas,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports gt_bboxes_ignore=None.'
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_reid_feats = preds_dicts['all_reid_feats']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(device)
            for gt_bboxes in gt_bboxes_list
        ]
        losses_cls, losses_bbox, losses_reid = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_reid_feats,
            [gt_bboxes_list for _ in range(num_dec_layers)],
            [gt_labels_list for _ in range(num_dec_layers)],
            list(range(num_dec_layers)))

        loss_dict = dict()
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_reid'] = losses_reid[-1]
        for i, (loss_cls_i, loss_bbox_i, loss_reid_i) in enumerate(
                zip(losses_cls[:-1], losses_bbox[:-1], losses_reid[:-1])):
            loss_dict[f'd{i}.loss_cls'] = loss_cls_i
            loss_dict[f'd{i}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{i}.loss_reid'] = loss_reid_i

        if preds_dicts['dn_mask_dict'] is not None:
            (known_labels, known_bboxs, known_reid_labels, output_known_class,
             output_known_coord, output_known_reid, num_tgt) = \
                self.prepare_for_loss(preds_dicts['dn_mask_dict'])
            dn_losses_cls, dn_losses_bbox, dn_losses_reid = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                output_known_reid,
                [known_bboxs for _ in range(num_dec_layers)],
                [known_labels for _ in range(num_dec_layers)],
                [known_reid_labels for _ in range(num_dec_layers)],
                [num_tgt for _ in range(num_dec_layers)])
        elif self.with_dn:
            # no gt in the batch: log the matching losses as dn losses (no grad)
            dn_losses_cls = [l.detach() for l in losses_cls]
            dn_losses_bbox = [l.detach() for l in losses_bbox]
            dn_losses_reid = [l.detach() for l in losses_reid]
        else:
            return loss_dict

        loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
        loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
        loss_dict['dn_loss_reid'] = dn_losses_reid[-1]
        for i, (loss_cls_i, loss_bbox_i, loss_reid_i) in enumerate(
                zip(dn_losses_cls[:-1], dn_losses_bbox[:-1], dn_losses_reid[:-1])):
            loss_dict[f'd{i}.dn_loss_cls'] = loss_cls_i
            loss_dict[f'd{i}.dn_loss_bbox'] = loss_bbox_i
            loss_dict[f'd{i}.dn_loss_reid'] = loss_reid_i
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Decodes the predictions; duplicates are merged with the identity
        embeddings inside ``bbox_coder`` (TRIM)."""
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        return [
            preds_to_boxes(preds, img_metas[i]['box_type_3d'])
            for i, preds in enumerate(preds_dicts)
        ]


def preds_to_boxes(preds, box_type_3d):
    """Decoded predictions (gravity centers) -> [boxes, scores, labels]."""
    bboxes = preds['bboxes']
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
    return [box_type_3d(bboxes, bboxes.size(-1)), preds['scores'], preds['labels']]
