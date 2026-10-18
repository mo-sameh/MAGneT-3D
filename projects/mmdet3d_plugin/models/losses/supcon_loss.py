import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    """Supervised contrastive loss per embedding.

    Args:
        embeddings (Tensor): (N, D) embeddings (not necessarily normalized).
        labels (Tensor): (N, ) identity labels; equal labels are positives.
        temperature (float): Scaling factor of the logits.
    """
    device = embeddings.device
    if embeddings.size(0) <= 1:
        return embeddings.new_zeros(1)

    embeddings = F.normalize(embeddings, p=2, dim=1)

    # cosine similarity [N, N]
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # positive pairs: same label
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # exclude self-pairs (diagonal)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(mask.shape[0]).view(-1, 1).to(device),
        0)
    mask = mask * logits_mask

    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

    # mean log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
    return -mean_log_prob_pos


@LOSSES.register_module()
class SupConLoss(nn.Module):
    """Supervised contrastive loss on the identity embeddings.

    Labels are the assigned gt indices offset by ``1000 * sample index``;
    unassigned queries (``label % 1000 == 0``) are ignored.

    Args:
        temperature (float): Scaling factor of the logits.
        loss_weight (float): Weight of the loss.
        reduction (str): 'none', 'mean' or 'sum'.
    """

    def __init__(self, temperature=0.07, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,
                embeddings,
                labels,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        pos_inds = labels % 1000 != 0
        if not pos_inds.any():
            return embeddings.sum() * 0.0

        pos_embeddings = embeddings[pos_inds]
        pos_labels = labels[pos_inds]
        if weight is not None:
            weight = weight[pos_inds]

        return self.loss_weight * supervised_contrastive_loss(
            pos_embeddings,
            pos_labels,
            temperature=self.temperature,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
