# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, Sampler


class PairedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    """

    def __init__(self, sampler, group_ids, batch_size):
        """
        Args:
            sampler (Sampler): Base sampler.
            group_ids (list[int]): If the sampler produces indices in range [0, N),
                `group_ids` must be a list of `N` ints which contains the group id of each sample.
                The group ids must be a set of integers in the range [0, num_groups).
            batch_size (int): Size of mini-batch.
        """
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        # assert self.group_ids.ndim == 1
        self.batch_size = batch_size
        merge_group_ids = [j for i in self.group_ids for j in i]
        groups = np.unique(merge_group_ids).tolist()

        # buffer the indices of each group until batch size is reached
        self.buffer_per_group = {k: [] for k in groups}
        self.idx_members = {}

    def __iter__(self):
        for idx in self.sampler:
            has_repeated = False
            if idx.item() not in self.idx_members:
                self.idx_members[idx.item()] = 1
            else:
                has_repeated = True
            group_id = self.group_ids[idx]
            for gd in group_id:
                group_buffer = self.buffer_per_group[gd]
                if idx.item() not in group_buffer or has_repeated:
                    group_buffer.append(idx.item())
                if len(group_buffer) == self.batch_size:
                    yield [torch.tensor(id) for id in group_buffer] # yield a copy of the list
                    for k in group_buffer:
                        if k in self.idx_members:
                            del self.idx_members[k]
                    del group_buffer[:]

    def __len__(self):
        raise NotImplementedError("len() of GroupedBatchSampler is not well-defined.")