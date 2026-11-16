"""Generate Lhotse-format JSONL manifests for AISHELL-1 train set.

Usage:
    python data/prepare_aishell_manifest.py \
        --wav-dir data/data_aishell/wav/train \
        --transcript data/data_aishell/transcript/aishell_transcript_v0.8.txt \
        --output-dir data/manifests/aishell/train
"""

import argparse
import json
import os
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", type=str, default="data/data_aishell/wav/train")
    parser.add_argument("--transcript", type=str, default="data/data_aishell/transcript/aishell_transcript_v0.8.txt")
    parser.add_argument("--output-dir", type=str, default="data/manifests/aishell/train")
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load transcript
    transcripts = {}
    with open(args.transcript, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt_id, text = parts
                transcripts[utt_id] = text.replace(" ", "")

    print(f"Loaded {len(transcripts)} transcripts")

    # Scan wav files
    recordings = []
    supervisions = []
    skipped = 0

    for speaker_dir in sorted(wav_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name

        for wav_file in sorted(speaker_dir.glob("*.wav")):
            utt_id = wav_file.stem

            if utt_id not in transcripts:
                skipped += 1
                continue

            info = sf.info(str(wav_file))
            num_samples = info.frames
            sample_rate = info.samplerate
            duration = num_samples / sample_rate

            rec = {
                "id": utt_id,
                "sources": [
                    {
                        "type": "file",
                        "channels": [0],
                        "source": str(wav_file.resolve()),
                    }
                ],
                "sampling_rate": sample_rate,
                "num_samples": num_samples,
                "duration": round(duration, 2),
            }
            recordings.append(rec)

            sup = {
                "id": utt_id,
                "recording_id": utt_id,
                "start": 0.0,
                "duration": round(duration, 2),
                "channel": 0,
                "text": transcripts[utt_id],
                "language": "zh",
                "speaker": speaker_id,
            }
            supervisions.append(sup)

    # Write manifests
    rec_path = output_dir / "recordings.jsonl"
    sup_path = output_dir / "supervisions.jsonl"

    with open(rec_path, "w", encoding="utf-8") as f:
        for rec in recordings:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(sup_path, "w", encoding="utf-8") as f:
        for sup in supervisions:
            f.write(json.dumps(sup, ensure_ascii=False) + "\n")

    print(f"Done! {len(recordings)} recordings, {len(supervisions)} supervisions")
    print(f"Skipped {skipped} wav files (no transcript)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
