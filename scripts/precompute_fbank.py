"""Precompute log-mel fbank features for all utterances.

Reads from Lhotse JSONL manifests, extracts features in parallel,
and stores them as sharded .pt files (float16) for fast training.

Usage:
    python scripts/precompute_fbank.py \
        --manifest-dir data/manifests/wenetspeech/train_ms \
        --output-dir data/fbank_shards/wenetspeech_ms \
        --tokenizer data/tokenizer/sp_6000.model \
        --num-workers 32 \
        --shard-size 1000
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 512
WIN_LENGTH = int(0.025 * SAMPLE_RATE)
HOP_LENGTH = int(0.010 * SAMPLE_RATE)


_worker_sp = None
_worker_fbank = None


def _init_worker(tokenizer_path):
    """Initialize per-worker state (called once per process)."""
    global _worker_sp, _worker_fbank
    import sentencepiece as spm
    import torchaudio
    _worker_sp = spm.SentencePieceProcessor()
    _worker_sp.Load(tokenizer_path)
    _worker_fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )


def extract_one(args_tuple):
    """Extract fbank for a single utterance."""
    audio_path, start, duration, file_sr, text, utt_id = args_tuple
    import soundfile as sf
    global _worker_sp, _worker_fbank

    try:
        file_start = int(start * file_sr)
        file_frames = int(duration * file_sr)

        with sf.SoundFile(audio_path, "r") as f:
            f.seek(file_start)
            data = f.read(frames=file_frames, dtype="float32")
            sr = f.samplerate

        if len(data) == 0:
            return None

        if len(data.shape) == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)

        if sr != SAMPLE_RATE:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        features = _worker_fbank(waveform).squeeze(0).transpose(0, 1)
        features = torch.log(features.clamp(min=1e-10))
        features = features.to(torch.float16)

        token_ids = _worker_sp.EncodeAsIds(text)

        return {
            "features": features,
            "targets": token_ids,
            "duration": duration,
            "utt_id": utt_id,
        }
    except Exception:
        return None




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=str,
                        default="data/manifests/wenetspeech/train_ms")
    parser.add_argument("--output-dir", type=str,
                        default="data/fbank_shards/wenetspeech_ms")
    parser.add_argument("--tokenizer", type=str,
                        default="data/tokenizer/sp_6000.model")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--start-shard", type=int, default=0,
                        help="Resume from this shard index")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load recordings
    logger.info("Loading recordings...")
    rec_map = {}
    with open(manifest_dir / "recordings.jsonl", "r") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rec_map[rec["id"]] = rec
    logger.info(f"Loaded {len(rec_map)} recordings")

    # Load supervisions
    logger.info("Loading supervisions...")
    all_items = []
    with open(manifest_dir / "supervisions.jsonl", "r") as f:
        for line in f:
            if not line.strip():
                continue
            sup = json.loads(line)
            dur = sup["duration"]
            if dur < args.min_duration or dur > args.max_duration:
                continue
            rec = rec_map.get(sup["recording_id"])
            if rec is None:
                continue
            audio_path = rec["sources"][0]["source"]
            file_sr = rec.get("sampling_rate", SAMPLE_RATE)
            all_items.append((
                audio_path, sup["start"], dur, file_sr,
                sup["text"], sup["id"],
            ))
    logger.info(f"Total valid utterances: {len(all_items)}")

    # Split into shards
    num_shards = (len(all_items) + args.shard_size - 1) // args.shard_size
    logger.info(f"Will create {num_shards} shards of size {args.shard_size}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Workers: {args.num_workers}")

    total_processed = 0
    total_errors = 0
    start_time = time.time()

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=_init_worker,
        initargs=(args.tokenizer,),
    ) as executor:
        for shard_idx in range(args.start_shard, num_shards):
            shard_start = shard_idx * args.shard_size
            shard_end = min(shard_start + args.shard_size, len(all_items))
            shard_items = all_items[shard_start:shard_end]

            # Skip if shard already exists
            shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
            if shard_path.exists():
                total_processed += len(shard_items)
                if shard_idx % 100 == 0:
                    logger.info(
                        f"Shard {shard_idx}/{num_shards} exists, skipping"
                    )
                continue

            # Submit all items in this shard
            futures = [executor.submit(extract_one, item) for item in shard_items]
            results = []
            errors = 0
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    errors += 1

            if results:
                shard_data = {
                    "features": [r["features"] for r in results],
                    "targets": [
                        torch.tensor(r["targets"], dtype=torch.int16)
                        for r in results
                    ],
                    "durations": [r["duration"] for r in results],
                    "utt_ids": [r["utt_id"] for r in results],
                }
                torch.save(shard_data, shard_path)

            total_processed += len(results)
            total_errors += errors

            if shard_idx % 10 == 0:
                elapsed = time.time() - start_time
                speed = total_processed / elapsed if elapsed > 0 else 0
                eta = (len(all_items) - total_processed) / speed if speed > 0 else 0
                logger.info(
                    f"Shard {shard_idx}/{num_shards} done | "
                    f"processed={total_processed} errors={total_errors} | "
                    f"{speed:.0f} utt/s | ETA {eta/60:.1f}min"
                )

    elapsed = time.time() - start_time
    logger.info(f"Done! {total_processed} utterances in {elapsed/60:.1f} min")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Output: {output_dir} ({num_shards} shards)")

    # Save metadata
    meta = {
        "num_utterances": total_processed,
        "num_shards": num_shards,
        "shard_size": args.shard_size,
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "dtype": "float16",
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
