# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import math
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from .sampler import SAMPLER


@SAMPLER.register_module()
class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset=None,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        if self.shuffle:
            assert False
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # pad to make it evenly divisible
        indices = (indices *
                   math.ceil(self.total_size / len(indices)))[:self.total_size]
        assert len(indices) == self.total_size

        # contiguous chunk per rank
        per_replicas = self.total_size//self.num_replicas
        indices = indices[self.rank*per_replicas:(self.rank+1)*per_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
