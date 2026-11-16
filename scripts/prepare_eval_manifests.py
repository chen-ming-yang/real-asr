"""Extract dev/test_net manifests from WenetSpeech.json.

The WenetSpeech.json is ~6.2GB, so we stream-parse using ijson to avoid OOM.
Produces lhotse-compatible recordings.jsonl and supervisions.jsonl.

Usage:
    python scripts/prepare_eval_manifests.py
"""

import json
import os
import sys
from pathlib import Path

# ijson may not be available, fall back to line-by-line scanning
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False


WENETSPEECH_JSON = "/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/WenetSpeech.json"
AUDIO_ROOT = "/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY"
OUTPUT_DIR = "/cmy/cmy/real-asr/data/manifests/wenetspeech"

TARGET_PREFIXES = {
    "dev": "audio/dev/",
    "test_net": "audio/test_net/",
    "test_meeting": "audio/test_meeting/",
}


def process_with_ijson():
    """Stream-parse WenetSpeech.json using ijson."""
    import ijson

    # Prepare output files
    writers = {}
    for split in TARGET_PREFIXES:
        out_dir = Path(OUTPUT_DIR) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        writers[split] = {
            "rec": open(out_dir / "recordings.jsonl", "w"),
            "sup": open(out_dir / "supervisions.jsonl", "w"),
            "count": 0,
        }

    print(f"Parsing {WENETSPEECH_JSON} (~6.2GB, may take a few minutes)...")

    with open(WENETSPEECH_JSON, "rb") as f:
        parser = ijson.items(f, "audios.item")

        for audio_idx, audio in enumerate(parser):
            if audio_idx % 10000 == 0:
                counts = {k: v["count"] for k, v in writers.items()}
                print(f"  Processed {audio_idx} audios... {counts}")

            path = audio["path"]
            matched_split = None
            for split, prefix in TARGET_PREFIXES.items():
                if path.startswith(prefix):
                    matched_split = split
                    break

            if matched_split is None:
                continue

            aid = audio["aid"]
            duration = float(audio["duration"])
            full_path = os.path.join(AUDIO_ROOT, path)

            # Write recording
            rec = {
                "id": aid,
                "sources": [{"type": "file", "channels": [0], "source": full_path}],
                "sampling_rate": 16000,
                "num_samples": int(duration * 16000),
                "duration": duration,
            }
            writers[matched_split]["rec"].write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Write supervisions
            for seg in audio.get("segments", []):
                begin = float(seg["begin_time"])
                end = float(seg["end_time"])
                sup = {
                    "id": seg["sid"],
                    "recording_id": aid,
                    "start": begin,
                    "duration": round(end - begin, 3),
                    "channel": 0,
                    "text": seg["text"],
                    "language": "zh",
                    "speaker": "unknown",
                }
                writers[matched_split]["sup"].write(json.dumps(sup, ensure_ascii=False) + "\n")
                writers[matched_split]["count"] += 1

    # Close files
    for split, w in writers.items():
        w["rec"].close()
        w["sup"].close()
        print(f"  {split}: {w['count']} segments")

    print("Done!")


def process_without_ijson():
    """Fallback: parse the full JSON (requires ~12GB RAM)."""
    print(f"Loading {WENETSPEECH_JSON} into memory (needs ~12GB RAM)...")
    with open(WENETSPEECH_JSON, "r") as f:
        data = json.load(f)

    writers = {}
    for split in TARGET_PREFIXES:
        out_dir = Path(OUTPUT_DIR) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        writers[split] = {
            "rec": open(out_dir / "recordings.jsonl", "w"),
            "sup": open(out_dir / "supervisions.jsonl", "w"),
            "count": 0,
        }

    print(f"Processing {len(data['audios'])} audios...")

    for audio in data["audios"]:
        path = audio["path"]
        matched_split = None
        for split, prefix in TARGET_PREFIXES.items():
            if path.startswith(prefix):
                matched_split = split
                break

        if matched_split is None:
            continue

        aid = audio["aid"]
        duration = audio["duration"]
        full_path = os.path.join(AUDIO_ROOT, path)

        rec = {
            "id": aid,
            "sources": [{"type": "file", "channels": [0], "source": full_path}],
            "sampling_rate": 16000,
            "num_samples": int(duration * 16000),
            "duration": duration,
        }
        writers[matched_split]["rec"].write(json.dumps(rec, ensure_ascii=False) + "\n")

        for seg in audio.get("segments", []):
            begin = seg["begin_time"]
            end = seg["end_time"]
            sup = {
                "id": seg["sid"],
                "recording_id": aid,
                "start": begin,
                "duration": round(end - begin, 3),
                "channel": 0,
                "text": seg["text"],
                "language": "zh",
                "speaker": "unknown",
            }
            writers[matched_split]["sup"].write(json.dumps(sup, ensure_ascii=False) + "\n")
            writers[matched_split]["count"] += 1

    for split, w in writers.items():
        w["rec"].close()
        w["sup"].close()
        print(f"  {split}: {w['count']} segments")

    print("Done!")


if __name__ == "__main__":
    if HAS_IJSON:
        process_with_ijson()
    else:
        process_without_ijson()
