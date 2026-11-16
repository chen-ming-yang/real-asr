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

    Uses lazy loading: only stores file offsets in memory, reads full
    JSON records on demand to avoid OOM with large datasets.
    """

    def __init__(
        self,
        manifest_dir: Path,
        tokenizer_path: str = None,
        max_duration: float = 30.0,
        min_duration: float = 0.3,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
    ):
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.manifest_dir = Path(manifest_dir)

        # Load SentencePiece tokenizer
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        if tokenizer_path and Path(tokenizer_path).exists():
            self.sp.Load(tokenizer_path)
            logger.info(f"Loaded tokenizer from {tokenizer_path}, vocab_size={self.sp.GetPieceSize()}")
        else:
            raise ValueError(f"Tokenizer not found: {tokenizer_path}")

        # Build recording index (much smaller than supervisions)
        self.rec_map = {}
        rec_path = self.manifest_dir / "recordings.jsonl"
        logger.info(f"Loading recordings from {rec_path}")
        with open(rec_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    self.rec_map[rec["id"]] = rec
        logger.info(f"Loaded {len(self.rec_map)} recordings")

        # Build supervision offset index with caching
        self.sup_path = str(self.manifest_dir / "supervisions.jsonl")
        cache_path = self.manifest_dir / f".sup_index_{min_duration}_{max_duration}.npz"
        self.sup_offsets, self.sup_durations = self._load_or_build_index(cache_path)

        # Fbank extractor
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=int(0.025 * sample_rate),
            hop_length=int(0.010 * sample_rate),
            n_mels=num_mel_bins,
        )

        logger.info(f"Loaded {len(self.sup_offsets)} utterances from {manifest_dir}")

    def _load_or_build_index(self, cache_path: Path):
        """Load cached index or build from scratch."""
        import os
        sup_mtime = os.path.getmtime(self.sup_path)

        if cache_path.exists():
            cache_mtime = os.path.getmtime(str(cache_path))
            if cache_mtime > sup_mtime:
                logger.info(f"Loading cached index from {cache_path}")
                data = np.load(str(cache_path))
                return data["offsets"].tolist(), data["durations"].tolist()
            else:
                logger.info("Cache outdated, rebuilding index...")

        logger.info(f"Building index for {self.sup_path} (first time, may take a few minutes)...")
        offsets = []
        durations = []
        count = 0
        with open(self.sup_path, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                dur_start = line.find('"duration":')
                if dur_start == -1:
                    continue
                dur_start += len('"duration":')
                dur_end = line.find(',', dur_start)
                if dur_end == -1:
                    dur_end = line.find('}', dur_start)
                try:
                    dur = float(line[dur_start:dur_end].strip())
                except ValueError:
                    continue
                if self.min_duration <= dur <= self.max_duration:
                    offsets.append(offset)
                    durations.append(dur)
                count += 1
                if count % 2000000 == 0:
                    logger.info(f"  Indexed {count} lines, {len(offsets)} valid...")

        # Save cache
        np.savez(
            str(cache_path),
            offsets=np.array(offsets, dtype=np.int64),
            durations=np.array(durations, dtype=np.float32),
        )
        logger.info(f"Index cached to {cache_path}")
        return offsets, durations

    def _read_supervision(self, idx: int) -> dict:
        """Read a single supervision by file offset."""
        with open(self.sup_path, "r", encoding="utf-8") as f:
            f.seek(self.sup_offsets[idx])
            line = f.readline()
            return json.loads(line)

    def __len__(self) -> int:
        return len(self.sup_offsets)

    def get_duration(self, idx: int) -> float:
        """Get duration of utterance at index."""
        return self.sup_durations[idx]

    def get_severity(self, idx: int) -> str:
        """Get severity level of utterance at index."""
        sup = self._read_supervision(idx)
        custom = sup.get("custom", {})
        return custom.get("severity", "unknown")

    def __getitem__(self, idx: int) -> Dict:
        sup = self._read_supervision(idx)
        rec = self.rec_map[sup["recording_id"]]

        # Load only the required audio segment (avoid reading entire file)
        audio_path = rec["sources"][0]["source"]
        import soundfile as sf
        start_sample = int(sup["start"] * self.sample_rate)
        num_samples = int(sup["duration"] * self.sample_rate)

        file_sr = rec.get("sampling_rate", self.sample_rate)
        file_start = int(sup["start"] * file_sr)
        file_frames = int(sup["duration"] * file_sr)

        with sf.SoundFile(audio_path, "r") as f:
            f.seek(file_start)
            data = f.read(frames=file_frames, dtype="float32")
            sr = f.samplerate

        if len(data.shape) == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

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
            "severity": sup.get("custom", {}).get("severity", "unknown"),
            "utt_id": sup["id"],
        }

    def _text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token IDs using SentencePiece."""
        return self.sp.EncodeAsIds(text)

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


class PrecomputedShardDataset(Dataset):
    """ASR dataset that loads precomputed fbank features from .pt shard files.

    Skips audio decoding and fbank computation entirely — reads directly from
    shards produced by scripts/precompute_fbank_mp.py.
    """

    def __init__(self, shard_dir: str):
        self.shard_dir = Path(shard_dir)
        meta_path = self.shard_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {shard_dir}")

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.num_shards = self.meta["num_shards"]
        self.shard_size = self.meta["shard_size"]

        # Build flat index with caching to avoid re-scanning all shards each run
        # Cache next to shard directory for persistence across restarts
        # If shard_dir is read-only, fall back to parent directory
        import os
        if os.access(str(self.shard_dir), os.W_OK):
            index_cache = self.shard_dir / ".shard_index.npz"
        else:
            index_cache = self.shard_dir.parent / f".{self.shard_dir.name}_shard_index.npz"
        self._index, self._durations = self._load_or_build_shard_index(index_cache)

        logger.info(f"PrecomputedShardDataset: {len(self._index)} utterances "
                    f"across {self.num_shards} shards")

        # Cache loaded shards to avoid re-loading on consecutive accesses
        self._cached_shard_idx = -1
        self._cached_shard = None

    def _load_or_build_shard_index(self, cache_path: Path):
        """Build or load cached (shard_idx, within_shard_idx) index + durations."""
        if cache_path.exists():
            logger.info(f"Loading shard index cache from {cache_path}")
            data = np.load(str(cache_path))
            shard_idxs = data["shard_idxs"].tolist()
            within_idxs = data["within_idxs"].tolist()
            durations = data["durations"].tolist()
            return list(zip(shard_idxs, within_idxs)), durations

        logger.info(f"Building shard index for {self.num_shards} shards "
                    f"(will be cached for future runs)...")
        index = []
        durations = []
        for shard_idx in range(self.num_shards):
            shard_path = self.shard_dir / f"shard_{shard_idx:05d}.pt"
            if not shard_path.exists():
                logger.warning(f"Missing shard: {shard_path}, skipping")
                continue
            data = torch.load(shard_path, map_location="cpu", weights_only=False)
            n = len(data["features"])
            for i in range(n):
                index.append((shard_idx, i))
                durations.append(float(data["durations"][i]))
            if shard_idx % 100 == 0:
                logger.info(f"  Indexed shard {shard_idx}/{self.num_shards}...")

        shard_idxs_arr = np.array([x[0] for x in index], dtype=np.int32)
        within_idxs_arr = np.array([x[1] for x in index], dtype=np.int32)
        try:
            np.savez(str(cache_path),
                     shard_idxs=shard_idxs_arr,
                     within_idxs=within_idxs_arr,
                     durations=np.array(durations, dtype=np.float32))
            logger.info(f"Shard index cached to {cache_path}")
        except OSError as e:
            logger.warning(f"Could not cache shard index to {cache_path}: {e}")
        return index, durations

    def __len__(self) -> int:
        return len(self._index)

    def get_duration(self, idx: int) -> float:
        return self._durations[idx]

    def __getitem__(self, idx: int) -> Dict:
        shard_idx, within_idx = self._index[idx]

        if shard_idx != self._cached_shard_idx:
            shard_path = self.shard_dir / f"shard_{shard_idx:05d}.pt"
            self._cached_shard = torch.load(shard_path, map_location="cpu",
                                            weights_only=False)
            self._cached_shard_idx = shard_idx

        shard = self._cached_shard
        features = shard["features"][within_idx].to(torch.float32)
        targets_raw = shard["targets"][within_idx]

        if isinstance(targets_raw, torch.Tensor):
            token_ids = targets_raw.to(torch.long)
        else:
            token_ids = torch.tensor(targets_raw, dtype=torch.long)

        return {
            "features": features,
            "feature_lens": features.size(0),
            "targets": token_ids,
            "target_lens": token_ids.size(0),
            "utt_id": shard["utt_ids"][within_idx],
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Collate function for DataLoader (same interface as AsrDataset)."""
        max_feat_len = max(item["feature_lens"] for item in batch)
        max_target_len = max(item["target_lens"] for item in batch)
        num_mel_bins = batch[0]["features"].size(-1)

        features = torch.zeros(len(batch), max_feat_len, num_mel_bins)
        feature_lens = torch.zeros(len(batch), dtype=torch.long)
        targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)
        target_lens = torch.zeros(len(batch), dtype=torch.long)

        for i, item in enumerate(batch):
            feat_len = item["feature_lens"]
            tgt_len = item["target_lens"]
            features[i, :feat_len] = item["features"]
            feature_lens[i] = feat_len
            targets[i, :tgt_len] = item["targets"]
            target_lens[i] = tgt_len

        return {
            "features": features,
            "feature_lens": feature_lens,
            "targets": targets,
            "target_lens": target_lens,
        }


class ShardBucketingSampler(Sampler):
    """Bucketing sampler for PrecomputedShardDataset.

    Same interface as AsrBucketingSampler but works with shard-based dataset.
    """

    def __init__(
        self,
        dataset: "PrecomputedShardDataset",
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
        durations = np.array(self.dataset._durations, dtype=np.float32)
        sorted_indices = np.argsort(durations)
        sorted_durations = durations[sorted_indices]

        cumsum = np.cumsum(sorted_durations)
        batches = []
        pos = 0
        while pos < len(sorted_durations):
            target = (cumsum[pos - 1] if pos > 0 else 0.0) + self.max_duration
            next_pos = int(np.searchsorted(cumsum, target, side="right"))
            if next_pos <= pos:
                next_pos = pos + 1
            batches.append(sorted_indices[pos:next_pos].tolist())
            pos = next_pos

        if self.drop_last and batches:
            batches.pop()

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]

        if self.world_size > 1:
            per_rank = len(batches) // self.world_size
            start = self.rank * per_rank
            batches = batches[start: start + per_rank]

        for batch in batches:
            yield batch

    def __len__(self):
        total_dur = float(np.sum(self.dataset._durations))
        return max(1, int(total_dur / self.max_duration))


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
        # Sort by duration using numpy (fast for 14M+ items)
        durations = np.array(self.dataset.sup_durations, dtype=np.float32)
        sorted_indices = np.argsort(durations)
        sorted_durations = durations[sorted_indices]

        # Vectorized batch creation using cumsum
        cumsum = np.cumsum(sorted_durations)
        batch_boundaries = [0]
        pos = 0
        while pos < len(sorted_durations):
            # Find where cumulative sum exceeds next batch boundary
            target = cumsum[pos - 1] + self.max_duration if pos > 0 else self.max_duration
            next_pos = np.searchsorted(cumsum, target, side='right')
            if next_pos <= pos:
                next_pos = pos + 1
            batch_boundaries.append(int(next_pos))
            pos = int(next_pos)

        # Build batch list from boundaries
        batches = []
        for i in range(len(batch_boundaries) - 1):
            start = batch_boundaries[i]
            end = batch_boundaries[i + 1]
            batches.append(sorted_indices[start:end].tolist())

        if self.drop_last and batches:
            batches.pop()  # drop incomplete last batch

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
        # Cache actual batch count from first iteration
        if hasattr(self, '_cached_len'):
            return self._cached_len
        # Estimate: total_duration / max_duration_per_batch
        import numpy as np
        total_dur = np.sum(self.dataset.sup_durations)
        return max(1, int(total_dur / self.max_duration))
