import numpy as np
import torch
from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from scipy.optimize import linear_sum_assignment

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox


@BBOX_ASSIGNERS.register_module()
class SoftTopKAssigner3D(BaseAssigner):
    """One-to-many assignment: each gt is matched to up to ``top_k``
    predictions by repeated Hungarian matching.

    The first round uses the full cost and gives the classification positives;
    the later rounds use the regression cost only and give regression-only
    matches. Matches further than ``radius`` from the gt are discarded.

    Args:
        cls_cost (dict): Classification cost.
        reg_cost (dict): Regression cost.
        pc_range (list[float]): Point cloud range used to normalize boxes.
        top_k (int): Max number of predictions per gt.
        radius (float): Max center distance of a match in metres.
        topk_decay (bool): Use ``max(top_k - layer, 1)`` at decoder layer ``layer``.
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 pc_range=None,
                 top_k=5,
                 radius=5.0,
                 topk_decay=True):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.pc_range = pc_range
        self.top_k = top_k
        self.radius = radius
        self.topk_decay = topk_decay

    def assign(self,
               bbox_pred,
               cls_pred=None,
               gt_bboxes=None,
               gt_labels=None,
               gt_bboxes_ignore=None,
               code_weights=None,
               with_velo=False,
               layer=0):
        """Assigns predictions to gts.

        The matching runs on the CPU (``scipy``); costs are copied from the
        GPU once and the result is copied back once.
        """
        assert gt_bboxes_ignore is None, 'Only gt_bboxes_ignore=None supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        device = bbox_pred.device
        top_k = max(self.top_k - layer, 1) if self.topk_decay else self.top_k

        if num_gts == 0 or num_bboxes == 0:
            assigned_gt_inds = bbox_pred.new_full((num_bboxes, ), -1, dtype=torch.long)
            assigned_labels = bbox_pred.new_full((num_bboxes, ), -1, dtype=torch.long)
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # costs: rows = predictions, cols = gts
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        if code_weights is not None:
            bbox_pred_weighted = bbox_pred * code_weights
            normalized_gt_bboxes_weighted = normalized_gt_bboxes * code_weights
        else:
            bbox_pred_weighted = bbox_pred
            normalized_gt_bboxes_weighted = normalized_gt_bboxes
        if with_velo:
            reg_cost = self.reg_cost(bbox_pred_weighted, normalized_gt_bboxes_weighted)
        else:
            reg_cost = self.reg_cost(bbox_pred_weighted[:, :8], normalized_gt_bboxes_weighted[:, :8])
        cost = cls_cost + reg_cost

        # move all cost matrices to the CPU in one copy
        mats = [cost.detach(), reg_cost.detach()]
        if self.radius > 0:
            mats.append(torch.cdist(bbox_pred[:, :3], gt_bboxes[:, :3], p=2).detach())
        if all(m.dtype == mats[0].dtype for m in mats):
            mats = torch.stack(mats).cpu().numpy()
        else:
            mats = [m.cpu().numpy() for m in mats]
        cost_raw = mats[0]
        cost_np = np.nan_to_num(cost_raw, nan=100.0, posinf=100.0, neginf=-100.0)
        reg_cost_np = np.nan_to_num(mats[1], nan=100.0, posinf=100.0, neginf=-100.0)
        if self.radius > 0:
            cost_np[mats[2] > self.radius] = 1e9

        gt_inds, labels = self._match(cost_np, reg_cost_np, cost_raw,
                                      gt_labels.cpu().numpy(), num_gts, num_bboxes, top_k)
        return AssignResult(num_gts, torch.from_numpy(gt_inds).to(device), None,
                            labels=torch.from_numpy(labels).to(device))

    @staticmethod
    def _match(cost_np, reg_cost_np, cost_raw, gt_labels_np, num_gts, num_bboxes, top_k):
        assigned_gt_inds = np.zeros(num_bboxes, dtype=np.int64)
        assigned_labels = np.full(num_bboxes, -1, dtype=np.int64)
        K = min(top_k, num_bboxes // num_gts)
        if K == 0:
            # fewer predictions than gts: plain Hungarian matching
            rows, cols = linear_sum_assignment(cost_np)
            assigned_gt_inds[rows] = cols + 1
            assigned_labels[rows] = gt_labels_np[cols]
            return assigned_gt_inds, assigned_labels

        available_mask = np.ones(num_bboxes, dtype=bool)
        for i in range(K):
            unassigned_rows = np.nonzero(available_mask)[0]
            if unassigned_rows.size == 0:
                break
            round_cost = cost_np if i == 0 else reg_cost_np
            matched_row_sub, matched_col = linear_sum_assignment(round_cost[unassigned_rows, :])
            matched_rows = unassigned_rows[matched_row_sub]
            valid = cost_np[matched_rows, matched_col] < 1e8  # within the radius
            matched_rows, matched_col = matched_rows[valid], matched_col[valid]
            assigned_gt_inds[matched_rows] = matched_col + 1
            if i == 0:
                assigned_labels[matched_rows] = gt_labels_np[matched_col]
            available_mask[matched_rows] = False

        # a matched gt without a round-0 match gets its lowest-cost match as positive
        for gt_idx in range(num_gts):
            pred_indices = np.nonzero(assigned_gt_inds == gt_idx + 1)[0]
            if len(pred_indices) and (assigned_labels[pred_indices] == -1).all():
                best = pred_indices[np.argmin(cost_raw[pred_indices, gt_idx])]
                assigned_labels[best] = gt_labels_np[gt_idx]
        return assigned_gt_inds, assigned_labels
