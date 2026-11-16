"""Precompute fbank features - reads each opus file ONCE, slices segments in memory.

Avoids the catastrophic seek overhead of opus by loading each recording fully,
then extracting all segments from it in one pass.

Usage:
    python scripts/precompute_fbank_fast.py \
        --manifest-dir data/manifests/wenetspeech/train_ms \
        --output-dir data/fbank_shards/wenetspeech_ms \
        --tokenizer data/tokenizer/sp_6000.model \
        --shard-size 1000
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torchaudio.transforms as T

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 512
WIN_LENGTH = int(0.025 * SAMPLE_RATE)
HOP_LENGTH = int(0.010 * SAMPLE_RATE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=str,
                        default="data/manifests/wenetspeech/train_ms")
    parser.add_argument("--output-dir", type=str,
                        default="data/fbank_shards/wenetspeech_ms")
    parser.add_argument("--tokenizer", type=str,
                        default="data/tokenizer/sp_6000.model")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=20.0)
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)

    fbank_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    logger.info("Loading recordings...")
    rec_map = {}
    with open(manifest_dir / "recordings.jsonl") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rec_map[rec["id"]] = rec
    logger.info(f"Loaded {len(rec_map)} recordings")

    logger.info("Loading supervisions and grouping by recording...")
    rec_segments = defaultdict(list)
    total_filtered = 0
    with open(manifest_dir / "supervisions.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            sup = json.loads(line)
            dur = sup["duration"]
            if dur < args.min_duration or dur > args.max_duration:
                total_filtered += 1
                continue
            if sup["recording_id"] not in rec_map:
                continue
            rec_segments[sup["recording_id"]].append(sup)

    total_utts = sum(len(v) for v in rec_segments.values())
    logger.info(f"Total valid utterances: {total_utts} "
                f"(filtered {total_filtered}) across {len(rec_segments)} recordings")

    num_shards = (total_utts + args.shard_size - 1) // args.shard_size
    logger.info(f"Will create ~{num_shards} shards of size {args.shard_size}")

    all_results = []
    total_errors = 0
    shard_idx = 0
    start_time = time.time()
    rec_count = 0

    for rec_id, segments in rec_segments.items():
        rec = rec_map[rec_id]
        audio_path = rec["sources"][0]["source"]
        file_sr = rec.get("sampling_rate", SAMPLE_RATE)
        rec_count += 1

        try:
            with sf.SoundFile(audio_path, "r") as f:
                audio_data = f.read(dtype="float32")
                sr = f.samplerate
        except Exception as e:
            total_errors += len(segments)
            if rec_count % 100 == 0:
                logger.warning(f"Failed to read {audio_path}: {e}")
            continue

        if len(audio_data) == 0:
            total_errors += len(segments)
            continue

        if len(audio_data.shape) == 1:
            full_waveform = torch.from_numpy(audio_data)
        else:
            full_waveform = torch.from_numpy(audio_data[:, 0])

        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            full_waveform = resampler(full_waveform.unsqueeze(0)).squeeze(0)
            effective_sr = SAMPLE_RATE
        else:
            effective_sr = sr

        total_samples = full_waveform.size(0)

        for sup in segments:
            seg_start = int(sup["start"] * effective_sr)
            seg_frames = int(sup["duration"] * effective_sr)
            seg_end = min(seg_start + seg_frames, total_samples)

            if seg_start >= total_samples or seg_end <= seg_start:
                total_errors += 1
                continue

            waveform = full_waveform[seg_start:seg_end].unsqueeze(0)

            features = fbank_transform(waveform).squeeze(0).transpose(0, 1)
            features = torch.log(features.clamp(min=1e-10))
            features = features.to(torch.float16)

            token_ids = sp.EncodeAsIds(sup["text"])

            all_results.append({
                "features": features,
                "targets": token_ids,
                "duration": sup["duration"],
                "utt_id": sup["id"],
            })

            if len(all_results) >= args.shard_size:
                shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
                _save_shard(all_results[:args.shard_size], shard_path)
                all_results = all_results[args.shard_size:]
                shard_idx += 1

                if shard_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    done = shard_idx * args.shard_size
                    speed = done / elapsed if elapsed > 0 else 0
                    eta = (total_utts - done) / speed if speed > 0 else 0
                    logger.info(
                        f"Shard {shard_idx} | "
                        f"recordings={rec_count}/{len(rec_segments)} | "
                        f"utts={done} errors={total_errors} | "
                        f"{speed:.0f} utt/s | ETA {eta/60:.1f}min"
                    )

        if rec_count % 500 == 0:
            elapsed = time.time() - start_time
            done = shard_idx * args.shard_size + len(all_results)
            logger.info(
                f"Progress: rec {rec_count}/{len(rec_segments)} | "
                f"utts={done} | shards={shard_idx} | "
                f"{elapsed/60:.1f}min elapsed"
            )

    if all_results:
        shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
        _save_shard(all_results, shard_path)
        shard_idx += 1

    elapsed = time.time() - start_time
    total_done = shard_idx * args.shard_size
    logger.info(f"Done! {shard_idx} shards in {elapsed/60:.1f} min")
    logger.info(f"Errors: {total_errors}")

    meta = {
        "num_utterances": total_utts - total_errors,
        "num_shards": shard_idx,
        "shard_size": args.shard_size,
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "dtype": "float16",
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def _save_shard(results, path):
    shard_data = {
        "features": [r["features"] for r in results],
        "targets": [
            torch.tensor(r["targets"], dtype=torch.int16) for r in results
        ],
        "durations": [r["duration"] for r in results],
        "utt_ids": [r["utt_id"] for r in results],
    }
    torch.save(shard_data, path)


if __name__ == "__main__":
    main()
