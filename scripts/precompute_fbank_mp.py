"""Precompute 80-dim log-mel fbank features from WenetSpeech opus files.

Strategy: group supervisions by recording, read each opus file ONCE into memory,
slice all segments, compute fbank, and write shards. Uses multiprocessing at the
recording-file level (each worker handles one full recording at a time).

Usage:
    python scripts/precompute_fbank_mp.py \
        --manifest-dir data/manifests/wenetspeech/train_ms \
        --output-dir data/fbank_shards/wenetspeech_ms \
        --tokenizer data/tokenizer/sp_6000.model \
        --num-workers 32 \
        --shard-size 1000

Dependencies: torch, torchaudio, sentencepiece, soundfile
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from multiprocessing import Process, Queue, Value
from pathlib import Path
from queue import Empty

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 512
WIN_LENGTH = int(0.025 * SAMPLE_RATE)
HOP_LENGTH = int(0.010 * SAMPLE_RATE)


def process_recording(rec_id, rec, segments, tokenizer_path):
    """Read one recording file fully, extract fbank for all its segments.

    Returns list of dicts with features/targets/duration/utt_id, or empty on error.
    """
    import soundfile as sf
    import sentencepiece as spm
    import torchaudio.transforms as T

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    fbank_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    audio_path = rec["sources"][0]["source"]

    try:
        with sf.SoundFile(audio_path, "r") as f:
            audio_data = f.read(dtype="float32")
            sr = f.samplerate
    except Exception:
        return [], len(segments)

    if len(audio_data) == 0:
        return [], len(segments)

    if len(audio_data.shape) == 1:
        full_waveform = torch.from_numpy(audio_data)
    else:
        full_waveform = torch.from_numpy(audio_data[:, 0])

    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        full_waveform = resampler(full_waveform.unsqueeze(0)).squeeze(0)

    total_samples = full_waveform.size(0)
    results = []
    errors = 0

    for sup in segments:
        seg_start = int(sup["start"] * SAMPLE_RATE)
        seg_frames = int(sup["duration"] * SAMPLE_RATE)
        seg_end = min(seg_start + seg_frames, total_samples)

        if seg_start >= total_samples or seg_end <= seg_start:
            errors += 1
            continue

        waveform = full_waveform[seg_start:seg_end].unsqueeze(0)
        features = fbank_transform(waveform).squeeze(0).transpose(0, 1)
        features = torch.log(features.clamp(min=1e-10))
        features = features.to(torch.float16)

        token_ids = sp.EncodeAsIds(sup["text"])

        results.append({
            "features": features,
            "targets": token_ids,
            "duration": sup["duration"],
            "utt_id": sup["id"],
        })

    return results, errors


def worker_fn(task_queue, result_queue, tokenizer_path):
    """Worker process: pull recording tasks, push segment results."""
    while True:
        try:
            item = task_queue.get(timeout=5)
        except Empty:
            break
        if item is None:
            break

        rec_id, rec, segments = item
        results, errors = process_recording(rec_id, rec, segments, tokenizer_path)
        result_queue.put((rec_id, results, errors))


def save_shard(results, shard_idx, output_dir):
    shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
    shard_data = {
        "features": [r["features"] for r in results],
        "targets": [
            torch.tensor(r["targets"], dtype=torch.int16) for r in results
        ],
        "durations": [r["duration"] for r in results],
        "utt_ids": [r["utt_id"] for r in results],
    }
    torch.save(shard_data, shard_path)


def main():
    parser = argparse.ArgumentParser(description="Precompute fbank features (multiprocessing)")
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
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    num_shards_est = (total_utts + args.shard_size - 1) // args.shard_size
    logger.info(f"Estimated shards: {num_shards_est}, shard_size={args.shard_size}")
    logger.info(f"Workers: {args.num_workers}")

    task_queue = Queue(maxsize=args.num_workers * 2)
    result_queue = Queue(maxsize=args.num_workers * 4)

    workers = []
    for _ in range(args.num_workers):
        p = Process(target=worker_fn, args=(task_queue, result_queue, args.tokenizer))
        p.daemon = True
        p.start()
        workers.append(p)

    # Feed tasks in a separate thread to avoid blocking main
    import threading

    def feeder():
        for rec_id, segments in rec_segments.items():
            rec = rec_map[rec_id]
            task_queue.put((rec_id, rec, segments))
        for _ in range(args.num_workers):
            task_queue.put(None)

    feed_thread = threading.Thread(target=feeder, daemon=True)
    feed_thread.start()

    # Collect results and write shards
    all_results = []
    shard_idx = 0
    total_processed = 0
    total_errors = 0
    recordings_done = 0
    start_time = time.time()

    total_recordings = len(rec_segments)

    pbar = tqdm(
        total=total_utts,
        desc="Extracting fbank",
        unit="utt",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    while recordings_done < total_recordings:
        try:
            rec_id, results, errors = result_queue.get(timeout=300)
        except Empty:
            logger.warning("Result queue timeout (300s), checking workers...")
            alive = sum(1 for p in workers if p.is_alive())
            if alive == 0:
                logger.error("All workers dead, exiting")
                break
            continue

        recordings_done += 1
        total_errors += errors
        all_results.extend(results)
        total_processed += len(results)
        pbar.update(len(results))
        pbar.set_postfix(
            recs=f"{recordings_done}/{total_recordings}",
            shards=shard_idx,
            errs=total_errors,
        )

        while len(all_results) >= args.shard_size:
            save_shard(all_results[:args.shard_size], shard_idx, output_dir)
            all_results = all_results[args.shard_size:]
            shard_idx += 1

            if shard_idx % 50 == 0:
                elapsed = time.time() - start_time
                speed = total_processed / elapsed if elapsed > 0 else 0
                eta = (total_utts - total_processed) / speed if speed > 0 else 0
                logger.info(
                    f"Shard {shard_idx} | "
                    f"recs={recordings_done}/{total_recordings} | "
                    f"utts={total_processed} errors={total_errors} | "
                    f"{speed:.0f} utt/s | ETA {eta/60:.1f}min"
                )

    pbar.close()

    # Save remaining
    if all_results:
        save_shard(all_results, shard_idx, output_dir)
        shard_idx += 1

    # Wait for workers
    for p in workers:
        p.join(timeout=10)

    elapsed = time.time() - start_time
    logger.info(f"Done! {shard_idx} shards, {total_processed} utts in {elapsed/60:.1f} min")
    logger.info(f"Errors: {total_errors}")

    meta = {
        "num_utterances": total_processed,
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
    logger.info(f"Metadata saved to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
