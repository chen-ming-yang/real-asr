"""Generate Lhotse-format JSONL manifests from dysarthric train.jsonl.

Usage:
    python data/prepare_dysarthric_manifest.py \
        --input /cmy/cmy/Qwen3-ASR/data/train.jsonl \
        --output-dir data/manifests/dysarthric/train \
        --max-duration 30.0
"""

import argparse
import json
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/cmy/cmy/Qwen3-ASR/data/train.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/manifests/dysarthric/train")
    parser.add_argument("--old-prefix", type=str, default="/cmy/after_catting")
    parser.add_argument("--new-prefix", type=str, default="/cmy/data/after_catting")
    parser.add_argument("--max-duration", type=float, default=30.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recordings = []
    supervisions = []
    skipped_missing = 0
    skipped_long = 0

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Total entries: {len(lines)}")

    for i, line in enumerate(lines):
        item = json.loads(line)
        utt_id = item["key"]
        audio_path = item["source"].replace(args.old_prefix, args.new_prefix)
        text = item["target"]
        speaker = item.get("speaker", "unknown")

        if not Path(audio_path).exists():
            skipped_missing += 1
            continue

        info = sf.info(audio_path)
        duration = info.frames / info.samplerate

        if duration > args.max_duration:
            skipped_long += 1
            continue

        rec = {
            "id": utt_id,
            "sources": [{"type": "file", "channels": [0], "source": audio_path}],
            "sampling_rate": info.samplerate,
            "num_samples": info.frames,
            "duration": round(duration, 2),
        }
        recordings.append(rec)

        sup = {
            "id": utt_id,
            "recording_id": utt_id,
            "start": 0.0,
            "duration": round(duration, 2),
            "channel": 0,
            "text": text,
            "language": "zh",
            "speaker": speaker,
        }
        supervisions.append(sup)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(lines)}...")

    rec_path = output_dir / "recordings.jsonl"
    sup_path = output_dir / "supervisions.jsonl"

    with open(rec_path, "w", encoding="utf-8") as f:
        for rec in recordings:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(sup_path, "w", encoding="utf-8") as f:
        for sup in supervisions:
            f.write(json.dumps(sup, ensure_ascii=False) + "\n")

    print(f"\nDone!")
    print(f"  Kept: {len(recordings)} utterances")
    print(f"  Skipped (missing): {skipped_missing}")
    print(f"  Skipped (>{args.max_duration}s): {skipped_long}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
