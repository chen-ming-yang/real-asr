"""Prepare dysarthric speech data into Lhotse manifests.

Usage:
    python data/prepare_dysarthric.py \
        --data-dir /path/to/dysarthric_data \
        --output-dir data/manifests/dysarthric \
        --severity-map /path/to/severity_labels.json

Expected input directory structure:
    data_dir/
    ├── wavs/
    │   ├── spk001_001.wav
    │   ├── spk001_002.wav
    │   └── ...
    ├── transcripts.json   # {"spk001_001": "你好世界", ...}
    └── severity.json      # {"spk001": "moderate", ...} (optional)

Or alternatively:
    data_dir/
    ├── mild/
    │   ├── audio/ & transcripts.txt
    ├── moderate/
    │   ├── audio/ & transcripts.txt
    └── severe/
        ├── audio/ & transcripts.txt
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_transcripts(transcript_path: Path) -> Dict[str, str]:
    """Load transcripts from JSON or txt file.

    Supports:
    - JSON: {"utt_id": "text", ...}
    - TXT: utt_id<tab>text per line
    """
    if transcript_path.suffix == ".json":
        with open(transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        transcripts = {}
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", maxsplit=1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]
                else:
                    parts = line.split("|", maxsplit=1)
                    if len(parts) == 2:
                        transcripts[parts[0]] = parts[1]
        return transcripts


def load_severity_map(severity_path: Optional[Path]) -> Dict[str, str]:
    """Load speaker-to-severity mapping."""
    if severity_path is None or not severity_path.exists():
        return {}
    with open(severity_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_speaker_id(utt_id: str) -> str:
    """Extract speaker ID from utterance ID (assume format: spkXXX_NNN)."""
    parts = utt_id.rsplit("_", maxsplit=1)
    return parts[0] if len(parts) > 1 else utt_id


def prepare_manifests(
    data_dir: Path,
    output_dir: Path,
    severity_map: Dict[str, str],
):
    """Generate Lhotse-compatible manifests from raw data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = data_dir / "wavs"
    # Support both flat and severity-grouped layouts
    if wav_dir.exists():
        wav_files = sorted(wav_dir.glob("*.wav"))
    else:
        wav_files = sorted(data_dir.rglob("*.wav"))

    if not wav_files:
        logger.error(f"No wav files found in {data_dir}")
        return

    # Load transcripts
    transcript_path = data_dir / "transcripts.json"
    if not transcript_path.exists():
        transcript_path = data_dir / "transcripts.txt"
    if not transcript_path.exists():
        logger.error(f"No transcripts file found in {data_dir}")
        return

    transcripts = load_transcripts(transcript_path)
    logger.info(f"Loaded {len(transcripts)} transcripts")

    # Build recordings and supervisions in Lhotse JSONL format
    recordings = []
    supervisions = []
    skipped = 0

    for wav_path in wav_files:
        utt_id = wav_path.stem
        if utt_id not in transcripts:
            skipped += 1
            continue

        try:
            info = sf.info(str(wav_path))
        except Exception as e:
            logger.warning(f"Cannot read {wav_path}: {e}")
            skipped += 1
            continue

        duration = info.duration
        sample_rate = info.samplerate
        num_samples = info.frames

        if duration < 0.1 or duration > 60.0:
            skipped += 1
            continue

        speaker = get_speaker_id(utt_id)
        severity = severity_map.get(speaker, "unknown")

        recording = {
            "id": utt_id,
            "sources": [{"type": "file", "channels": [0], "source": str(wav_path.resolve())}],
            "sampling_rate": sample_rate,
            "num_samples": num_samples,
            "duration": round(duration, 4),
        }

        supervision = {
            "id": utt_id,
            "recording_id": utt_id,
            "start": 0.0,
            "duration": round(duration, 4),
            "channel": 0,
            "text": transcripts[utt_id],
            "language": "zh",
            "speaker": speaker,
            "custom": {"severity": severity},
        }

        recordings.append(recording)
        supervisions.append(supervision)

    logger.info(
        f"Prepared {len(recordings)} utterances, skipped {skipped}. "
        f"Total duration: {sum(r['duration'] for r in recordings) / 3600:.1f} hours"
    )

    # Write JSONL manifests
    rec_path = output_dir / "recordings.jsonl"
    sup_path = output_dir / "supervisions.jsonl"

    with open(rec_path, "w", encoding="utf-8") as f:
        for r in recordings:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(sup_path, "w", encoding="utf-8") as f:
        for s in supervisions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Saved recordings to {rec_path}")
    logger.info(f"Saved supervisions to {sup_path}")

    # Statistics by severity
    severity_stats = {}
    for s in supervisions:
        sev = s["custom"]["severity"]
        severity_stats[sev] = severity_stats.get(sev, 0) + 1
    logger.info(f"Severity distribution: {severity_stats}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dysarthric data manifests")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root directory of raw data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests/dysarthric"))
    parser.add_argument("--severity-map", type=Path, default=None, help="JSON: speaker -> severity")
    args = parser.parse_args()

    severity_map = load_severity_map(args.severity_map)
    if not severity_map:
        # Try loading from data_dir/severity.json
        default_sev = args.data_dir / "severity.json"
        severity_map = load_severity_map(default_sev)

    prepare_manifests(args.data_dir, args.output_dir, severity_map)


if __name__ == "__main__":
    main()
