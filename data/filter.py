"""Filter pseudo-labels for quality control.

Dysarthric speech has different characteristics than normal speech,
so filtering thresholds are adjusted accordingly:
- Slower speaking rate is normal (0.5 char/s minimum vs 1.0 for normal)
- Repetitions may be genuine disfluencies, not errors
- Allow longer silences

Usage:
    python data/filter.py \
        --input-dir data/manifests/pseudo_labeled \
        --output-dir data/manifests/pseudo_labeled_filtered \
        --min-char-rate 0.5 \
        --max-char-rate 12.0
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_manifests(manifest_dir: Path) -> Tuple[List[dict], List[dict]]:
    """Load recordings and supervisions from JSONL files."""
    rec_path = manifest_dir / "recordings.jsonl"
    sup_path = manifest_dir / "supervisions.jsonl"

    recordings = []
    with open(rec_path, "r", encoding="utf-8") as f:
        for line in f:
            recordings.append(json.loads(line))

    supervisions = []
    with open(sup_path, "r", encoding="utf-8") as f:
        for line in f:
            supervisions.append(json.loads(line))

    return recordings, supervisions


def check_repeated_chars(text: str, max_consecutive: int = 5) -> bool:
    """Check if text has excessive character repetitions.

    For dysarthric speech, we allow up to 5 consecutive repeats
    (higher threshold than normal speech) as repetitions can be
    genuine disfluencies.
    """
    pattern = re.compile(r"(.)\1{" + str(max_consecutive) + r",}")
    return bool(pattern.search(text))


def check_repeated_words(text: str, max_consecutive: int = 4) -> bool:
    """Check if text has excessive word/phrase repetitions."""
    chars = list(text.replace(" ", ""))
    if len(chars) < 4:
        return False

    for window in range(2, 6):
        for i in range(len(chars) - window * max_consecutive):
            pattern = chars[i : i + window]
            count = 1
            j = i + window
            while j + window <= len(chars):
                if chars[j : j + window] == pattern:
                    count += 1
                    j += window
                else:
                    break
            if count >= max_consecutive:
                return True
    return False


def check_text_quality(text: str) -> Tuple[bool, str]:
    """Check if transcription text is reasonable.

    Returns (is_valid, reason).
    """
    if not text or len(text.strip()) == 0:
        return False, "empty_text"

    # Remove punctuation for analysis
    clean = re.sub(r"[，。！？、；：""''（）\s]", "", text)

    if len(clean) == 0:
        return False, "only_punctuation"

    if check_repeated_chars(clean, max_consecutive=5):
        return False, "excessive_char_repeat"

    if check_repeated_words(clean, max_consecutive=4):
        return False, "excessive_word_repeat"

    # Check for garbled output (high ratio of rare/unusual characters)
    unusual_count = sum(1 for c in clean if ord(c) > 0x9FFF or ord(c) < 0x4E00)
    if len(clean) > 0 and unusual_count / len(clean) > 0.3:
        return False, "garbled_text"

    return True, "ok"


def filter_manifests(
    recordings: List[dict],
    supervisions: List[dict],
    min_duration: float = 0.3,
    max_duration: float = 30.0,
    min_char_rate: float = 0.5,
    max_char_rate: float = 12.0,
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    """Apply filtering rules to pseudo-labeled data.

    Args:
        recordings: list of recording dicts
        supervisions: list of supervision dicts
        min_duration: minimum segment duration (seconds)
        max_duration: maximum segment duration (seconds)
        min_char_rate: minimum characters per second (lower for dysarthric)
        max_char_rate: maximum characters per second

    Returns:
        filtered_recordings, filtered_supervisions, rejection_stats
    """
    rec_map = {r["id"]: r for r in recordings}
    rejection_stats: Dict[str, int] = {}

    def reject(reason: str):
        rejection_stats[reason] = rejection_stats.get(reason, 0) + 1

    filtered_recs = []
    filtered_sups = []

    for sup in supervisions:
        rec = rec_map.get(sup["recording_id"])
        if rec is None:
            reject("missing_recording")
            continue

        duration = sup["duration"]
        text = sup["text"]

        # Duration filter
        if duration < min_duration:
            reject("too_short")
            continue
        if duration > max_duration:
            reject("too_long")
            continue

        # Text quality
        is_valid, reason = check_text_quality(text)
        if not is_valid:
            reject(reason)
            continue

        # Character rate (adjusted for dysarthric speech)
        clean_text = re.sub(r"[，。！？、；：""''（）\s]", "", text)
        char_rate = len(clean_text) / duration if duration > 0 else 0

        if char_rate < min_char_rate:
            reject("char_rate_too_low")
            continue
        if char_rate > max_char_rate:
            reject("char_rate_too_high")
            continue

        filtered_recs.append(rec)
        filtered_sups.append(sup)

    return filtered_recs, filtered_sups, rejection_stats


def main():
    parser = argparse.ArgumentParser(description="Filter pseudo-labels for dysarthric ASR")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--min-char-rate", type=float, default=0.5)
    parser.add_argument("--max-char-rate", type=float, default=12.0)
    args = parser.parse_args()

    recordings, supervisions = load_manifests(args.input_dir)
    logger.info(f"Loaded {len(supervisions)} supervisions")

    filtered_recs, filtered_sups, stats = filter_manifests(
        recordings, supervisions,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_char_rate=args.min_char_rate,
        max_char_rate=args.max_char_rate,
    )

    keep_rate = len(filtered_sups) / max(len(supervisions), 1) * 100
    logger.info(f"Kept {len(filtered_sups)}/{len(supervisions)} ({keep_rate:.1f}%)")
    logger.info(f"Rejection stats: {json.dumps(stats, indent=2)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_dir / "recordings.jsonl", "w", encoding="utf-8") as f:
        for r in filtered_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.output_dir / "supervisions.jsonl", "w", encoding="utf-8") as f:
        for s in filtered_sups:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    total_hours = sum(s["duration"] for s in filtered_sups) / 3600
    logger.info(f"Filtered dataset: {total_hours:.1f} hours")


if __name__ == "__main__":
    main()
