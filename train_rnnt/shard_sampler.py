"""Shard-local sampler: keeps batch samples from the same shard to avoid I/O thrashing.

The original ShardBucketingSampler sorts globally by duration, breaking shard locality.
This sampler groups by shard first, then sorts within each shard by duration.
"""

import numpy as np
import torch
from torch.utils.data import Sampler


class ShardLocalSampler(Sampler):
    """Shard-aware batching that minimizes cross-shard I/O.

    Strategy: shuffle shards, then within each shard form batches by duration.
    This ensures each batch's samples come from at most 1-2 shards.
    """

    def __init__(
        self,
        dataset,
        max_duration_per_batch: float = 800.0,
        shuffle: bool = True,
        drop_last: bool = True,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.max_duration = max_duration_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self._build_shard_groups()

    def _build_shard_groups(self):
        """Group indices by shard_idx."""
        self.shard_groups = {}
        for global_idx, (shard_idx, _within) in enumerate(self.dataset._index):
            if shard_idx not in self.shard_groups:
                self.shard_groups[shard_idx] = []
            self.shard_groups[shard_idx].append(global_idx)
        self.shard_ids = sorted(self.shard_groups.keys())

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        shard_order = self.shard_ids[:]
        if self.shuffle:
            perm = torch.randperm(len(shard_order), generator=g).tolist()
            shard_order = [shard_order[i] for i in perm]

        all_batches = []
        durations = np.array(self.dataset._durations, dtype=np.float32)

        for shard_id in shard_order:
            indices = np.array(self.shard_groups[shard_id])
            shard_durs = durations[indices]
            sort_order = np.argsort(shard_durs)
            sorted_indices = indices[sort_order]
            sorted_durs = shard_durs[sort_order]

            pos = 0
            while pos < len(sorted_durs):
                cum = 0.0
                end = pos
                while end < len(sorted_durs) and cum + sorted_durs[end] <= self.max_duration:
                    cum += sorted_durs[end]
                    end += 1
                if end == pos:
                    end = pos + 1
                all_batches.append(sorted_indices[pos:end].tolist())
                pos = end

        if self.drop_last and all_batches:
            all_batches.pop()

        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        if self.world_size > 1:
            per_rank = len(all_batches) // self.world_size
            start = self.rank * per_rank
            all_batches = all_batches[start: start + per_rank]

        for batch in all_batches:
            yield batch

    def __len__(self):
        total_dur = float(np.sum(self.dataset._durations))
        return max(1, int(total_dur / self.max_duration))
