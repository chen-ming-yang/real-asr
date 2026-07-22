"""ASR Dataset and DataLoader utilities.

Provides Lhotse-compatible dataset and sampler classes for training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class AsrDataset(Dataset):
    """ASR dataset that loads from Lhotse JSONL manifests.

    Reads recordings and supervisions, computes Fbank features on-the-fly.
    """

    def __init__(
        self,
        manifest_dir: Path,
        max_duration: float = 30.0,
        min_duration: float = 0.3,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
    ):
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.max_duration = max_duration
        self.min_duration = min_duration

        # Load manifests
        self.recordings = self._load_jsonl(manifest_dir / "recordings.jsonl")
        self.supervisions = self._load_jsonl(manifest_dir / "supervisions.jsonl")

        # Build index
        self.rec_map = {r["id"]: r for r in self.recordings}

        # Filter by duration
        self.supervisions = [
            s for s in self.supervisions
            if min_duration <= s["duration"] <= max_duration
        ]

        # Fbank extractor
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=int(0.025 * sample_rate),
            hop_length=int(0.010 * sample_rate),
            n_mels=num_mel_bins,
        )

        logger.info(f"Loaded {len(self.supervisions)} utterances from {manifest_dir}")

    def _load_jsonl(self, path: Path) -> List[dict]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def __len__(self) -> int:
        return len(self.supervisions)

    def get_duration(self, idx: int) -> float:
        """Get duration of utterance at index."""
        return self.supervisions[idx]["duration"]

    def get_severity(self, idx: int) -> str:
        """Get severity level of utterance at index."""
        custom = self.supervisions[idx].get("custom", {})
        return custom.get("severity", "unknown")

    def __getitem__(self, idx: int) -> Dict:
        sup = self.supervisions[idx]
        rec = self.rec_map[sup["recording_id"]]

        # Load audio
        audio_path = rec["sources"][0]["source"]
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Extract segment
        start_sample = int(sup["start"] * self.sample_rate)
        num_samples = int(sup["duration"] * self.sample_rate)
        waveform = waveform[:, start_sample : start_sample + num_samples]

        # Mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Compute Fbank features
        features = self.fbank(waveform)  # (1, n_mels, time)
        features = features.squeeze(0).transpose(0, 1)  # (time, n_mels)

        # Log scale
        features = torch.log(features.clamp(min=1e-10))

        # Encode text to token IDs (placeholder: character-level)
        text = sup["text"]
        token_ids = self._text_to_tokens(text)

        return {
            "features": features,
            "feature_lens": features.size(0),
            "targets": torch.tensor(token_ids, dtype=torch.long),
            "target_lens": len(token_ids),
            "text": text,
            "severity": self.get_severity(idx),
            "utt_id": sup["id"],
        }

    def _text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token IDs.

        Placeholder: uses character-level encoding.
        In production, use SentencePiece tokenizer.
        """
        tokens = []
        for char in text:
            if char.strip():
                tokens.append(ord(char) % 5999 + 1)  # Map to [1, 5999]
        return tokens

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Collate function for DataLoader."""
        # Pad features to max length
        max_feat_len = max(item["feature_lens"] for item in batch)
        max_target_len = max(item["target_lens"] for item in batch)
        num_mel_bins = batch[0]["features"].size(-1)

        features = torch.zeros(len(batch), max_feat_len, num_mel_bins)
        feature_lens = torch.zeros(len(batch), dtype=torch.long)
        targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)
        target_lens = torch.zeros(len(batch), dtype=torch.long)
        texts = []

        for i, item in enumerate(batch):
            feat_len = item["feature_lens"]
            tgt_len = item["target_lens"]
            features[i, :feat_len] = item["features"]
            feature_lens[i] = feat_len
            targets[i, :tgt_len] = item["targets"]
            target_lens[i] = tgt_len
            texts.append(item["text"])

        return {
            "features": features,
            "feature_lens": feature_lens,
            "targets": targets,
            "target_lens": target_lens,
            "texts": texts,
        }


class AsrBucketingSampler(Sampler):
    """Dynamic bucketing sampler based on audio duration.

    Groups utterances by similar duration to minimize padding waste.
    """

    def __init__(
        self,
        dataset: AsrDataset,
        max_duration_per_batch: float = 500.0,
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

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        # Sort by duration for bucketing
        indices = list(range(len(self.dataset)))
        durations = [self.dataset.get_duration(i) for i in indices]
        sorted_indices = sorted(indices, key=lambda i: durations[i])

        # Create batches
        batches = []
        current_batch = []
        current_duration = 0.0

        for idx in sorted_indices:
            dur = durations[idx]
            if current_duration + dur > self.max_duration and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_duration = 0.0
            current_batch.append(idx)
            current_duration += dur

        if current_batch and not self.drop_last:
            batches.append(current_batch)

        # Shuffle batches (not within batch, for padding efficiency)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]

        # Shard for distributed
        if self.world_size > 1:
            per_rank = len(batches) // self.world_size
            start = self.rank * per_rank
            batches = batches[start : start + per_rank]

        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.dataset) // 10  # Approximate
