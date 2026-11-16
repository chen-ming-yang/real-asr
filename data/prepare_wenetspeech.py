"""Convert WenetSpeech JSON annotation to Lhotse JSONL format.

Usage:
    python data/prepare_wenetspeech.py \
        --wenetspeech-json /cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/WenetSpeech.json \
        --audio-root /cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY \
        --output-dir data/manifests/wenetspeech \
        --subsets L \
        --splits train dev test_net test_meeting
"""

import argparse
import json
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Set

import ijson


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_split_from_path(audio_path: str) -> str:
    """Determine split (train/dev/test_net/test_meeting) from audio path."""
    if "/train/" in audio_path:
        return "train"
    elif "/dev/" in audio_path:
        return "dev"
    elif "/test_net/" in audio_path:
        return "test_net"
    elif "/test_meeting/" in audio_path:
        return "test_meeting"
    return "train"


def process_wenetspeech(
    wenetspeech_json: Path,
    audio_root: Path,
    output_dir: Path,
    subsets: Set[str],
    splits: Set[str],
):
    """Stream-parse WenetSpeech.json and output Lhotse JSONL manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open output files for each split
    writers = {}
    for split in splits:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        writers[split] = {
            "rec": open(split_dir / "recordings.jsonl", "w", encoding="utf-8"),
            "sup": open(split_dir / "supervisions.jsonl", "w", encoding="utf-8"),
        }

    total_recordings = 0
    total_supervisions = 0
    total_duration_hours = 0.0

    logger.info(f"Streaming WenetSpeech.json from {wenetspeech_json}")
    logger.info(f"Target subsets: {subsets}")
    logger.info(f"Target splits: {splits}")

    with open(wenetspeech_json, "rb") as f:
        for audio in ijson.items(f, "audios.item"):
            audio_path = audio["path"]
            split = get_split_from_path(audio_path)

            if split not in splits:
                continue

            full_audio_path = str(audio_root / audio_path)

            # Check if audio file exists
            if not os.path.exists(full_audio_path):
                continue

            aid = audio["aid"]
            duration = float(audio["duration"])

            # Filter segments by subset
            valid_segments = []
            for seg in audio.get("segments", []):
                seg_subsets = set(seg.get("subsets", []))
                if seg_subsets & subsets:
                    valid_segments.append(seg)

            if not valid_segments:
                continue

            # Write recording
            recording = {
                "id": aid,
                "sources": [{"type": "file", "channels": [0], "source": full_audio_path}],
                "sampling_rate": 16000,
                "num_samples": int(duration * 16000),
                "duration": round(duration, 4),
            }
            writers[split]["rec"].write(json.dumps(recording, ensure_ascii=False, cls=DecimalEncoder) + "\n")
            total_recordings += 1

            # Write supervisions
            for seg in valid_segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue

                seg_duration = float(seg["end_time"]) - float(seg["begin_time"])
                if seg_duration < 0.3 or seg_duration > 30.0:
                    continue

                supervision = {
                    "id": seg["sid"],
                    "recording_id": aid,
                    "start": round(float(seg["begin_time"]), 4),
                    "duration": round(seg_duration, 4),
                    "channel": 0,
                    "text": text,
                    "language": "zh",
                    "speaker": "unknown",
                    "custom": {
                        "confidence": float(seg.get("confidence", 0.0)),
                        "subsets": seg.get("subsets", []),
                    },
                }
                writers[split]["sup"].write(json.dumps(supervision, ensure_ascii=False, cls=DecimalEncoder) + "\n")
                total_supervisions += 1
                total_duration_hours += seg_duration / 3600.0

            if total_recordings % 1000 == 0:
                logger.info(
                    f"Processed {total_recordings} recordings, "
                    f"{total_supervisions} supervisions, "
                    f"{total_duration_hours:.1f} hours"
                )

    # Close all files
    for split in writers:
        writers[split]["rec"].close()
        writers[split]["sup"].close()

    logger.info("=" * 60)
    logger.info(f"Done! Total recordings: {total_recordings}")
    logger.info(f"Total supervisions: {total_supervisions}")
    logger.info(f"Total duration: {total_duration_hours:.1f} hours")
    for split in splits:
        split_dir = output_dir / split
        rec_count = sum(1 for _ in open(split_dir / "recordings.jsonl"))
        sup_count = sum(1 for _ in open(split_dir / "supervisions.jsonl"))
        logger.info(f"  {split}: {rec_count} recordings, {sup_count} supervisions")


def main():
    parser = argparse.ArgumentParser(description="Convert WenetSpeech to Lhotse JSONL format")
    parser.add_argument(
        "--wenetspeech-json",
        type=Path,
        default=Path("/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/WenetSpeech.json"),
        help="Path to WenetSpeech.json",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY"),
        help="Root directory of WenetSpeech audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests/wenetspeech"),
        help="Output directory for Lhotse manifests",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["L"],
        choices=["S", "M", "L"],
        help="WenetSpeech subsets to include (S=100h, M=1000h, L=10000h)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test_net", "test_meeting"],
        help="Data splits to generate",
    )
    args = parser.parse_args()

    process_wenetspeech(
        wenetspeech_json=args.wenetspeech_json,
        audio_root=args.audio_root,
        output_dir=args.output_dir,
        subsets=set(args.subsets),
        splits=set(args.splits),
    )


if __name__ == "__main__":
    main()
