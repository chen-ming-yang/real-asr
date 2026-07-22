"""Dysarthric speech data augmentation pipeline.

Augmentation strategies specific to dysarthric speech:
1. Speaking Rate Modification (SRM) - wider range due to variable speech rate
2. Pitch Modification - simulates vocal cord instability
3. Formant Perturbation - simulates articulatory imprecision
4. SpecAugment - standard, but more aggressive for robustness
5. Noise augmentation - environmental robustness

Usage:
    python data/augment.py \
        --input-dir data/manifests/dysarthric \
        --output-dir data/augmented \
        --config configs/train_stage3.yaml
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class DysarthricAugmenter:
    """Audio augmentation pipeline tailored for dysarthric speech."""

    def __init__(self, config: Dict):
        aug_cfg = config.get("augmentation", config)
        self.speed_factors = aug_cfg.get("speed_perturb", {}).get(
            "factors", [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        )
        self.pitch_range = aug_cfg.get("pitch_perturb", {}).get("range_semitones", [-3, 3])
        self.pitch_prob = aug_cfg.get("pitch_perturb", {}).get("prob", 0.3)
        self.formant_shift = aug_cfg.get("formant_perturb", {}).get("shift_ratio", 0.1)
        self.formant_prob = aug_cfg.get("formant_perturb", {}).get("prob", 0.3)
        self.noise_dir = aug_cfg.get("musan", {}).get("noise_dir", None)
        self.noise_prob = aug_cfg.get("musan", {}).get("prob", 0.5)

    def speed_perturb(self, audio: np.ndarray, sr: int) -> tuple:
        """Apply speed perturbation (changes duration and pitch together)."""
        factor = random.choice(self.speed_factors)
        if factor == 1.0:
            return audio, sr

        # Resample to change speed
        new_sr = int(sr * factor)
        return audio, new_sr  # Will be resampled to target SR later

    def pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Shift pitch without changing duration.

        Uses simple resampling approach; for production, consider
        librosa.effects.pitch_shift or pyworld.
        """
        if random.random() > self.pitch_prob:
            return audio

        semitones = random.uniform(self.pitch_range[0], self.pitch_range[1])
        # Pitch shift via resampling + time stretch
        ratio = 2 ** (semitones / 12.0)

        # Simple approach: resample then trim/pad
        indices = np.arange(0, len(audio), ratio)
        indices = indices[indices < len(audio)].astype(int)
        shifted = audio[indices]

        # Match original length
        if len(shifted) < len(audio):
            shifted = np.pad(shifted, (0, len(audio) - len(shifted)))
        else:
            shifted = shifted[: len(audio)]

        return shifted

    def formant_perturb(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Simple formant perturbation via spectral envelope modification.

        This is a simplified version. For production, use WORLD vocoder
        or Praat-based formant manipulation.
        """
        if random.random() > self.formant_prob:
            return audio

        shift = random.uniform(-self.formant_shift, self.formant_shift)

        # Simple spectral warping via interpolation in frequency domain
        n_fft = 2048
        hop = n_fft // 4
        frames = []

        for i in range(0, len(audio) - n_fft, hop):
            frame = audio[i : i + n_fft] * np.hanning(n_fft)
            spectrum = np.fft.rfft(frame)
            mag = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Warp frequency axis
            freq_bins = len(mag)
            new_indices = np.arange(freq_bins) * (1 + shift)
            new_indices = np.clip(new_indices, 0, freq_bins - 1)

            warped_mag = np.interp(np.arange(freq_bins), new_indices, mag)
            warped_spectrum = warped_mag * np.exp(1j * phase)
            frames.append(np.fft.irfft(warped_spectrum, n=n_fft))

        if not frames:
            return audio

        # Overlap-add reconstruction
        output = np.zeros(len(audio))
        for i, frame in enumerate(frames):
            start = i * hop
            end = min(start + n_fft, len(audio))
            output[start:end] += frame[: end - start]

        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * np.max(np.abs(audio))

        return output

    def add_noise(self, audio: np.ndarray, sr: int, snr_db: float = None) -> np.ndarray:
        """Add noise from MUSAN dataset or synthetic noise."""
        if random.random() > self.noise_prob:
            return audio

        if snr_db is None:
            snr_db = random.uniform(5, 20)

        if self.noise_dir and Path(self.noise_dir).exists():
            noise_files = list(Path(self.noise_dir).rglob("*.wav"))
            if noise_files:
                noise_path = random.choice(noise_files)
                noise, noise_sr = sf.read(str(noise_path))
                if len(noise.shape) > 1:
                    noise = noise[:, 0]
                # Resample if needed
                if noise_sr != sr:
                    ratio = sr / noise_sr
                    indices = np.arange(0, len(noise), 1 / ratio).astype(int)
                    indices = indices[indices < len(noise)]
                    noise = noise[indices]
            else:
                noise = np.random.randn(len(audio)) * 0.01
        else:
            noise = np.random.randn(len(audio)) * 0.01

        # Tile or trim noise to match audio length
        if len(noise) < len(audio):
            repeats = len(audio) // len(noise) + 1
            noise = np.tile(noise, repeats)
        noise = noise[: len(audio)]

        # Apply SNR
        audio_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power > 0:
            scale = np.sqrt(audio_power / (noise_power * (10 ** (snr_db / 10))))
            noisy = audio + noise * scale
        else:
            noisy = audio

        return noisy

    def augment(self, audio: np.ndarray, sr: int) -> tuple:
        """Apply full augmentation pipeline.

        Returns (augmented_audio, sample_rate).
        """
        # Speed perturbation
        audio, sr = self.speed_perturb(audio, sr)

        # Pitch shift
        audio = self.pitch_shift(audio, sr)

        # Formant perturbation
        audio = self.formant_perturb(audio, sr)

        # Noise addition
        audio = self.add_noise(audio, sr)

        return audio, sr


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    config: Dict,
    num_augmented_copies: int = 3,
):
    """Generate augmented copies of the dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    augmenter = DysarthricAugmenter(config)

    # Load manifests
    sup_path = input_dir / "supervisions.jsonl"
    supervisions = []
    with open(sup_path, "r", encoding="utf-8") as f:
        for line in f:
            supervisions.append(json.loads(line))

    logger.info(f"Loaded {len(supervisions)} utterances, generating {num_augmented_copies} copies each")

    aug_recordings = []
    aug_supervisions = []
    wav_output_dir = output_dir / "wavs"
    wav_output_dir.mkdir(parents=True, exist_ok=True)

    for sup in supervisions:
        source_path = None
        rec_path = input_dir / "recordings.jsonl"
        # Find recording source
        with open(rec_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["id"] == sup["recording_id"]:
                    source_path = rec["sources"][0]["source"]
                    break

        if source_path is None or not Path(source_path).exists():
            continue

        audio, sr = sf.read(source_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        for copy_idx in range(num_augmented_copies):
            aug_audio, aug_sr = augmenter.augment(audio.copy(), sr)

            # Resample back to 16kHz if speed perturb changed SR
            if aug_sr != 16000:
                ratio = 16000 / aug_sr
                indices = np.arange(0, len(aug_audio), 1 / ratio).astype(int)
                indices = indices[indices < len(aug_audio)]
                aug_audio = aug_audio[indices]
                aug_sr = 16000

            # Save augmented audio
            aug_id = f"{sup['id']}_aug{copy_idx}"
            aug_path = wav_output_dir / f"{aug_id}.wav"
            sf.write(str(aug_path), aug_audio, aug_sr)

            aug_recordings.append({
                "id": aug_id,
                "sources": [{"type": "file", "channels": [0], "source": str(aug_path.resolve())}],
                "sampling_rate": aug_sr,
                "num_samples": len(aug_audio),
                "duration": round(len(aug_audio) / aug_sr, 4),
            })

            aug_supervisions.append({
                "id": aug_id,
                "recording_id": aug_id,
                "start": 0.0,
                "duration": round(len(aug_audio) / aug_sr, 4),
                "channel": 0,
                "text": sup["text"],
                "language": sup.get("language", "zh"),
                "speaker": sup.get("speaker", "unknown"),
                "custom": {
                    "severity": sup.get("custom", {}).get("severity", "unknown"),
                    "augmented": True,
                    "original_id": sup["id"],
                    "copy_idx": copy_idx,
                },
            })

    # Save augmented manifests
    with open(output_dir / "recordings.jsonl", "w", encoding="utf-8") as f:
        for r in aug_recordings:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(output_dir / "supervisions.jsonl", "w", encoding="utf-8") as f:
        for s in aug_supervisions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    total_hours = sum(r["duration"] for r in aug_recordings) / 3600
    logger.info(f"Generated {len(aug_recordings)} augmented utterances ({total_hours:.1f} hours)")


def main():
    parser = argparse.ArgumentParser(description="Dysarthric speech data augmentation")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="YAML config with augmentation params")
    parser.add_argument("--num-copies", type=int, default=3)
    args = parser.parse_args()

    config = {}
    if args.config and args.config.exists():
        import yaml
        with open(args.config, "r") as f:
            full_cfg = yaml.safe_load(f)
            config = full_cfg.get("training", {}).get("augmentation", {})

    augment_dataset(args.input_dir, args.output_dir, config, args.num_copies)


if __name__ == "__main__":
    main()
