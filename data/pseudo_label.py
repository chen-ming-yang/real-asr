"""Generate pseudo-labels using finetuned Qwen3-ASR teacher model.

Usage:
    python data/pseudo_label.py \
        --model-path /path/to/finetuned_qwen3_asr \
        --audio-dir /path/to/unlabeled_audio \
        --output-dir data/manifests/pseudo_labeled \
        --batch-size 32 \
        --num-gpus 1

This script:
1. Loads the finetuned Qwen3-ASR model via vLLM for efficient batch inference
2. Processes unlabeled audio files in batches
3. Outputs pseudo-labeled manifests in Lhotse JSONL format
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def segment_audio(wav_path: Path, max_duration: float = 30.0, min_duration: float = 0.5) -> List[dict]:
    """Segment long audio into chunks using energy-based VAD.

    Returns list of segments with start/end times.
    """
    data, sr = sf.read(str(wav_path))
    if len(data.shape) > 1:
        data = data[:, 0]

    total_duration = len(data) / sr
    if total_duration <= max_duration:
        return [{"start": 0.0, "end": total_duration, "audio": data, "sr": sr}]

    # Simple energy-based segmentation
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = []
    for i in range(0, len(data) - frame_length, hop_length):
        frame = data[i : i + frame_length]
        energy.append(np.sum(frame ** 2))
    energy = np.array(energy)

    # Find silence points (energy below threshold)
    threshold = np.percentile(energy, 20)
    silence_mask = energy < threshold

    segments = []
    segment_start = 0.0
    max_samples = int(max_duration * sr)

    i = 0
    while i < len(data):
        end_sample = min(i + max_samples, len(data))
        # Find nearest silence point near the end
        if end_sample < len(data):
            search_start = max(0, end_sample // hop_length - 50)
            search_end = min(len(silence_mask), end_sample // hop_length + 50)
            silence_points = np.where(silence_mask[search_start:search_end])[0]
            if len(silence_points) > 0:
                cut_frame = search_start + silence_points[len(silence_points) // 2]
                end_sample = cut_frame * hop_length

        seg_data = data[i:end_sample]
        duration = len(seg_data) / sr

        if duration >= min_duration:
            segments.append({
                "start": i / sr,
                "end": end_sample / sr,
                "audio": seg_data,
                "sr": sr,
            })

        i = end_sample

    return segments


def load_teacher_model(model_path: str, num_gpus: int = 1):
    """Load Qwen3-ASR teacher model using vLLM for efficient inference."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM is required. Install with: pip install vllm>=0.14.0")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    return llm


def transcribe_batch(
    llm,
    audio_segments: List[Tuple[str, np.ndarray, int]],
    language: str = "zh",
) -> List[str]:
    """Batch transcribe audio segments using vLLM.

    Args:
        llm: vLLM LLM instance
        audio_segments: list of (utt_id, audio_array, sample_rate)
        language: target language

    Returns:
        list of transcription strings
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        repetition_penalty=1.0,
    )

    # Build prompts with audio input
    prompts = []
    for utt_id, audio, sr in audio_segments:
        prompt = {
            "prompt": f"<|startoftranscript|><|{language}|><|transcribe|>",
            "multi_modal_data": {"audio": (audio, sr)},
        }
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params)
    transcriptions = [output.outputs[0].text.strip() for output in outputs]
    return transcriptions


def process_directory(
    model_path: str,
    audio_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    num_gpus: int = 1,
    max_duration: float = 30.0,
    min_duration: float = 0.5,
):
    """Process all audio files in directory and generate pseudo-labels."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_extensions = {".wav", ".flac", ".mp3", ".ogg"}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(sorted(audio_dir.rglob(f"*{ext}")))

    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio files")

    # Load teacher model
    logger.info(f"Loading teacher model from {model_path}")
    llm = load_teacher_model(model_path, num_gpus)

    # Process in batches
    recordings = []
    supervisions = []
    batch = []
    total_processed = 0

    for audio_path in audio_files:
        segments = segment_audio(audio_path, max_duration, min_duration)

        for idx, seg in enumerate(segments):
            utt_id = f"{audio_path.stem}_{idx:04d}" if len(segments) > 1 else audio_path.stem
            batch.append((utt_id, seg["audio"], seg["sr"], audio_path, seg["start"], seg["end"]))

            if len(batch) >= batch_size:
                _process_batch(llm, batch, recordings, supervisions)
                total_processed += len(batch)
                if total_processed % 500 == 0:
                    logger.info(f"Processed {total_processed} segments")
                batch = []

    # Process remaining
    if batch:
        _process_batch(llm, batch, recordings, supervisions)
        total_processed += len(batch)

    logger.info(f"Total processed: {total_processed} segments")

    # Write manifests
    rec_path = output_dir / "recordings.jsonl"
    sup_path = output_dir / "supervisions.jsonl"

    with open(rec_path, "w", encoding="utf-8") as f:
        for r in recordings:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(sup_path, "w", encoding="utf-8") as f:
        for s in supervisions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(recordings)} recordings to {rec_path}")
    logger.info(
        f"Total pseudo-labeled duration: "
        f"{sum(r['duration'] for r in recordings) / 3600:.1f} hours"
    )


def _process_batch(llm, batch, recordings, supervisions):
    """Process a batch of audio segments."""
    audio_inputs = [(utt_id, audio, sr) for utt_id, audio, sr, _, _, _ in batch]
    transcriptions = transcribe_batch(llm, audio_inputs)

    for (utt_id, audio, sr, audio_path, start, end), text in zip(batch, transcriptions):
        duration = len(audio) / sr
        recordings.append({
            "id": utt_id,
            "sources": [{"type": "file", "channels": [0], "source": str(audio_path.resolve())}],
            "sampling_rate": sr,
            "num_samples": len(audio),
            "duration": round(duration, 4),
        })
        supervisions.append({
            "id": utt_id,
            "recording_id": utt_id,
            "start": round(start, 4),
            "duration": round(duration, 4),
            "channel": 0,
            "text": text,
            "language": "zh",
            "speaker": "unknown",
            "custom": {"pseudo_label": True, "source_file": audio_path.name},
        })


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels with Qwen3-ASR teacher")
    parser.add_argument("--model-path", type=str, required=True, help="Path to finetuned Qwen3-ASR")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory with unlabeled audio")
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests/pseudo_labeled"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--min-duration", type=float, default=0.5)
    args = parser.parse_args()

    process_directory(
        model_path=args.model_path,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
    )


if __name__ == "__main__":
    main()
