"""Severity-aware curriculum learning for dysarthric ASR.

Implements a progressive training curriculum that starts with
easier (mild) dysarthric speech and gradually introduces more
difficult (severe) samples.

This helps the model:
1. Build stable representations on clearer speech first
2. Avoid being overwhelmed by highly variable severe speech early on
3. Gradually adapt to the full spectrum of dysarthric patterns
"""

import logging
from typing import Dict, List, Optional, Set

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class SeverityAwareSampler(Sampler):
    """Batch sampler that filters data by severity level based on curriculum schedule.

    Args:
        dataset: dataset with severity labels accessible via get_severity(idx)
        schedule: list of curriculum stages, each with epochs and severity_filter
        max_duration_per_batch: maximum total audio duration per batch
        current_epoch: starting epoch
        rank: distributed rank
        world_size: distributed world size
    """

    def __init__(
        self,
        dataset,
        schedule: List[Dict],
        max_duration_per_batch: float = 500.0,
        current_epoch: int = 1,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.schedule = schedule
        self.max_duration_per_batch = max_duration_per_batch
        self.current_epoch = current_epoch
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed

        # Index samples by severity
        self.severity_indices: Dict[str, List[int]] = {}
        self._build_severity_index()

    def _build_severity_index(self):
        """Build mapping from severity level to sample indices."""
        for idx in range(len(self.dataset)):
            severity = self.dataset.get_severity(idx)
            if severity not in self.severity_indices:
                self.severity_indices[severity] = []
            self.severity_indices[severity].append(idx)

        for sev, indices in self.severity_indices.items():
            logger.info(f"Severity '{sev}': {len(indices)} samples")

    def _get_active_severities(self) -> Set[str]:
        """Get active severity levels for current epoch based on schedule."""
        active = set()
        for stage in self.schedule:
            epoch_range = stage["epochs"]
            if epoch_range[0] <= self.current_epoch <= epoch_range[1]:
                active.update(stage["severity_filter"])
                break

        # Fallback: use all severities if no matching stage
        if not active:
            active = set(self.severity_indices.keys())

        return active

    def _get_lr_scale(self) -> float:
        """Get learning rate scale factor for current epoch."""
        for stage in self.schedule:
            epoch_range = stage["epochs"]
            if epoch_range[0] <= self.current_epoch <= epoch_range[1]:
                return stage.get("lr_scale", 1.0)
        return 1.0

    def set_epoch(self, epoch: int):
        """Update current epoch (call before each epoch)."""
        self.current_epoch = epoch
        active = self._get_active_severities()
        logger.info(f"Epoch {epoch}: active severities = {active}")

    def __iter__(self):
        """Generate batches respecting severity curriculum."""
        active_severities = self._get_active_severities()

        # Gather eligible indices
        eligible_indices = []
        for sev in active_severities:
            if sev in self.severity_indices:
                eligible_indices.extend(self.severity_indices[sev])

        if not eligible_indices:
            eligible_indices = list(range(len(self.dataset)))

        # Shuffle with epoch-dependent seed for reproducibility
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.current_epoch)
            perm = torch.randperm(len(eligible_indices), generator=g).tolist()
            eligible_indices = [eligible_indices[i] for i in perm]

        # Shard for distributed training
        if self.world_size > 1:
            indices_per_rank = len(eligible_indices) // self.world_size
            start = self.rank * indices_per_rank
            end = start + indices_per_rank
            eligible_indices = eligible_indices[start:end]

        # Create batches based on duration
        batches = []
        current_batch = []
        current_duration = 0.0

        for idx in eligible_indices:
            duration = self.dataset.get_duration(idx)
            if current_duration + duration > self.max_duration_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_duration = 0.0

            current_batch.append(idx)
            current_duration += duration

        if current_batch:
            batches.append(current_batch)

        # Shuffle batches
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.current_epoch + 1000)
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        active_severities = self._get_active_severities()
        total = sum(
            len(self.severity_indices.get(sev, []))
            for sev in active_severities
        )
        return max(1, total // 10)  # Approximate batch count


class SeverityBalancedSampler(Sampler):
    """Sampler that balances samples across severity levels.

    Used during Stage 2 (distillation) to ensure all severity levels
    are represented in each epoch regardless of their natural frequency.
    """

    def __init__(
        self,
        dataset,
        weights: Dict[str, float],
        max_duration_per_batch: float = 400.0,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.weights = weights  # e.g., {"mild": 0.3, "moderate": 0.4, "severe": 0.3}
        self.max_duration_per_batch = max_duration_per_batch
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.severity_indices: Dict[str, List[int]] = {}
        self._build_severity_index()

    def _build_severity_index(self):
        for idx in range(len(self.dataset)):
            severity = self.dataset.get_severity(idx)
            if severity not in self.severity_indices:
                self.severity_indices[severity] = []
            self.severity_indices[severity].append(idx)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        """Generate balanced batches sampling from each severity proportionally."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Determine how many samples to draw from each severity
        total_samples = len(self.dataset)
        severity_quotas = {}
        for sev, weight in self.weights.items():
            if sev in self.severity_indices:
                quota = int(total_samples * weight)
                severity_quotas[sev] = quota

        # Sample with replacement if needed
        all_indices = []
        for sev, quota in severity_quotas.items():
            indices = self.severity_indices[sev]
            if len(indices) >= quota:
                perm = torch.randperm(len(indices), generator=g)[:quota].tolist()
                all_indices.extend([indices[i] for i in perm])
            else:
                # Oversample
                repeats = quota // len(indices) + 1
                expanded = (indices * repeats)[:quota]
                all_indices.extend(expanded)

        # Include unknown severity samples
        for sev, indices in self.severity_indices.items():
            if sev not in self.weights:
                all_indices.extend(indices)

        # Shuffle all
        perm = torch.randperm(len(all_indices), generator=g).tolist()
        all_indices = [all_indices[i] for i in perm]

        # Shard for distributed
        if self.world_size > 1:
            per_rank = len(all_indices) // self.world_size
            start = self.rank * per_rank
            all_indices = all_indices[start : start + per_rank]

        # Batch by duration
        batches = []
        current_batch = []
        current_duration = 0.0

        for idx in all_indices:
            duration = self.dataset.get_duration(idx)
            if current_duration + duration > self.max_duration_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_duration = 0.0
            current_batch.append(idx)
            current_duration += duration

        if current_batch:
            batches.append(current_batch)

        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]

        for batch in batches:
            yield batch

    def __len__(self):
        return max(1, len(self.dataset) // 10)
