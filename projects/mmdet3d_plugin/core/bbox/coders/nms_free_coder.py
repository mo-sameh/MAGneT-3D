import numpy as np
import torch
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


def identity_dedup(bboxes, scores, embeddings, mode='greedy',
                   sim_threshold=0.8, dist_threshold=3.0, dist_weight=0.5):
    """TRIM inference-time deduplication.

    ``mode='greedy'`` (default): the highest scoring prediction becomes a
    cluster leader and absorbs every remaining prediction that is both similar
    in identity space (cosine similarity above ``sim_threshold``) and close in
    3D (center distance below ``dist_threshold`` metres); only the leader of
    each cluster is kept.

    ``mode='connected'``: predictions ``i, j`` are linked if
    ``cos_sim - dist_weight * dist > sim_threshold`` and
    ``dist < dist_threshold``; the best prediction of each connected component
    is kept.

    Args:
        bboxes (Tensor): (N, >=3) boxes, centers first.
        scores (Tensor): (N, ) confidence scores.
        embeddings (Tensor): (N, D) identity embeddings.

    Returns:
        list[int]: indices of the kept predictions.
    """
    if mode == 'connected':
        return _connected_dedup(bboxes, scores, embeddings, sim_threshold,
                                dist_threshold, dist_weight)
    assert mode == 'greedy', f'Unknown dedup mode {mode}'
    order = scores.argsort(descending=True)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    keep = []
    while order.numel() > 0:
        leader = order[0]
        keep.append(leader.item())
        others = order[1:]
        sim = torch.mm(embeddings[leader:leader + 1], embeddings[others].T).squeeze(0)
        dist = torch.norm(bboxes[leader, :3] - bboxes[others, :3], dim=-1)
        order = others[~((sim > sim_threshold) & (dist < dist_threshold))]
    return keep


def _connected_dedup(bboxes, scores, embeddings, sim_threshold,
                     dist_threshold, dist_weight):
    if bboxes.size(0) == 0:
        return []
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    dist = torch.cdist(bboxes[:, :3], bboxes[:, :3])
    adj = ((embeddings @ embeddings.T - dist_weight * dist) > sim_threshold) & \
        (dist < dist_threshold)
    adj = adj.cpu().numpy()

    # union-find over the adjacency matrix
    num = adj.shape[0]
    parent = list(range(num))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.nonzero(np.triu(adj, k=1))):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    clusters = {}
    for i in range(num):
        clusters.setdefault(find(i), []).append(i)

    keep = []
    for cluster in clusters.values():
        cluster = torch.tensor(cluster, device=scores.device)
        keep.append(cluster[scores[cluster].argmax()].item())
    return keep


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detectors (StreamPETR) with optional TRIM
    deduplication when identity embeddings are predicted.

    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
        max_num (int): Max number of (query, class) pairs kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
        num_classes (int): Number of classes.
        dedup_cfg (dict): Arguments of :func:`identity_dedup`.
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10,
                 dedup_cfg=dict(sim_threshold=0.8, dist_threshold=3.0)):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.dedup_cfg = dedup_cfg

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, reid=None):
        max_num = min(self.max_num, cls_scores.shape[0])
        scores, indexs = cls_scores.sigmoid().view(-1).topk(max_num)
        bbox_index = torch.div(indexs, self.num_classes, rounding_mode='floor')
        labels = indexs % self.num_classes
        bboxes = denormalize_bbox(bbox_preds[bbox_index], self.pc_range)
        if reid is not None and self.dedup_cfg is not None:
            keep = identity_dedup(bboxes, scores, reid[bbox_index], **self.dedup_cfg)
            bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]

        post_center_range = torch.tensor(self.post_center_range, device=scores.device)
        mask = (bboxes[..., :3] >= post_center_range[:3]).all(1)
        mask &= (bboxes[..., :3] <= post_center_range[3:]).all(1)
        if self.score_threshold:
            mask &= scores >= self.score_threshold
        return {'bboxes': bboxes[mask], 'scores': scores[mask], 'labels': labels[mask]}

    def decode(self, preds_dicts):
        """Decode the boxes of the last decoder layer for every sample."""
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        reid = preds_dicts['all_reid_feats'][-1] if 'all_reid_feats' in preds_dicts else None
        return [
            self.decode_single(all_cls_scores[i], all_bbox_preds[i],
                               reid[i] if reid is not None else None)
            for i in range(all_cls_scores.size(0))
        ]
