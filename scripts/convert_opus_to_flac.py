"""Batch convert opus files to flac for fast random access.

Reads the recordings manifest, converts each opus file to 16kHz mono flac
using ffmpeg in parallel, then writes an updated recordings manifest pointing
to the new flac files.

Usage:
    python scripts/convert_opus_to_flac.py \
        --manifest-dir data/manifests/wenetspeech/train_ms \
        --output-audio-dir /cmy/cmy/WenetSpeech/flac_16k \
        --num-workers 16 \
        --ffmpeg /cmy/cmy/miniconda3/envs/mfa/bin/ffmpeg
"""

import argparse
import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def convert_one(args_tuple):
    """Convert a single opus file to flac."""
    rec_id, src_path, dst_path, ffmpeg_bin = args_tuple
    dst = Path(dst_path)
    if dst.exists() and dst.stat().st_size > 0:
        return rec_id, dst_path, "skipped"

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin, "-y", "-i", src_path,
        "-ar", "16000", "-ac", "1",
        "-c:a", "flac",
        "-compression_level", "5",
        dst_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=600,
        )
        if result.returncode != 0:
            return rec_id, dst_path, f"error: {result.stderr[-200:]}"
        return rec_id, dst_path, "ok"
    except subprocess.TimeoutExpired:
        return rec_id, dst_path, "timeout"
    except Exception as e:
        return rec_id, dst_path, f"exception: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=str,
                        default="data/manifests/wenetspeech/train_ms")
    parser.add_argument("--output-audio-dir", type=str,
                        default="/cmy/cmy/WenetSpeech/flac_16k")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--ffmpeg", type=str,
                        default="/cmy/cmy/miniconda3/envs/mfa/bin/ffmpeg")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    output_audio_dir = Path(args.output_audio_dir)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading recordings manifest...")
    recordings = []
    with open(manifest_dir / "recordings.jsonl") as f:
        for line in f:
            if line.strip():
                recordings.append(json.loads(line))
    logger.info(f"Loaded {len(recordings)} recordings")

    tasks = []
    for rec in recordings:
        src_path = rec["sources"][0]["source"]
        rel_path = Path(src_path).relative_to("/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/audio")
        dst_path = str(output_audio_dir / rel_path.with_suffix(".flac"))
        tasks.append((rec["id"], src_path, dst_path, args.ffmpeg))

    logger.info(f"Starting conversion with {args.num_workers} workers...")
    start_time = time.time()
    completed = 0
    errors = 0
    skipped = 0

    updated_recordings = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(convert_one, t): t for t in tasks}

        for future in as_completed(futures):
            rec_id, dst_path, status = future.result()
            completed += 1

            if status == "ok":
                pass
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                if errors <= 20:
                    logger.warning(f"Error for {rec_id}: {status}")

            if completed % 500 == 0:
                elapsed = time.time() - start_time
                speed = completed / elapsed
                remaining = len(tasks) - completed
                eta = remaining / speed if speed > 0 else 0
                logger.info(
                    f"Progress: {completed}/{len(tasks)} | "
                    f"errors={errors} skipped={skipped} | "
                    f"{speed:.1f} files/s | ETA {eta/60:.1f}min"
                )

    elapsed = time.time() - start_time
    logger.info(f"Conversion done in {elapsed/60:.1f} min")
    logger.info(f"Completed={completed} errors={errors} skipped={skipped}")

    logger.info("Writing updated recordings manifest...")
    rec_map = {rec["id"]: rec for rec in recordings}
    output_manifest = manifest_dir / "recordings_flac.jsonl"

    with open(output_manifest, "w") as f:
        for rec in recordings:
            src_path = rec["sources"][0]["source"]
            rel_path = Path(src_path).relative_to(
                "/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/audio"
            )
            dst_path = str(output_audio_dir / rel_path.with_suffix(".flac"))

            new_rec = rec.copy()
            new_rec["sources"] = [{
                "type": "file",
                "channels": [0],
                "source": dst_path,
            }]
            new_rec["sampling_rate"] = 16000
            f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

    logger.info(f"Updated manifest: {output_manifest}")


if __name__ == "__main__":
    main()
