"""Quick inference script for epoch_8 model (trained with old char-hash tokenizer).

Usage:
    PYTHONPATH=/cmy/cmy/real-asr python infer_v7.py
"""
import sys
sys.path.insert(0, "/cmy/cmy/real-asr")

import json
import torch
import torchaudio
import soundfile as sf
import yaml
import numpy as np
from pathlib import Path

def load_model(checkpoint_path, config_path, device="cuda"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from train.train import build_model
    transducer, ctc_head = build_model(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model", ckpt)
    if any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}

    transducer.load_state_dict(model_state, strict=False)
    transducer = transducer.to(device).eval()

    ctc_state = ckpt.get("ctc_head")
    if ctc_state:
        ctc_head.load_state_dict(ctc_state)
    ctc_head = ctc_head.to(device).eval()

    return transducer, ctc_head


def extract_features(audio_path, start=None, duration=None, device="cuda"):
    """Extract log-mel features from audio."""
    sample_rate = 16000
    if start is not None and duration is not None:
        with sf.SoundFile(audio_path, "r") as f:
            file_sr = f.samplerate
            f.seek(int(start * file_sr))
            data = f.read(frames=int(duration * file_sr), dtype="float32")
            sr = file_sr
    else:
        data, sr = sf.read(audio_path, dtype="float32")

    if len(data.shape) == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)
    else:
        waveform = torch.from_numpy(data.T)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=512,
        win_length=int(0.025 * sample_rate),
        hop_length=int(0.010 * sample_rate),
        n_mels=80,
    )
    features = fbank(waveform).squeeze(0).transpose(0, 1)  # (T, 80)
    features = torch.log(features.clamp(min=1e-10))
    return features.unsqueeze(0).to(device)  # (1, T, 80)


def ctc_greedy_decode(log_probs, blank_id=0):
    """CTC greedy decoding: collapse repeated, remove blanks."""
    predicted_ids = log_probs.argmax(dim=-1).squeeze(0).tolist()
    decoded = []
    prev = -1
    for tid in predicted_ids:
        if tid != blank_id and tid != prev:
            decoded.append(tid)
        prev = tid
    return decoded


def decode_old_tokenizer(token_ids):
    """Reverse the old ord(char) % 5999 + 1 mapping (best effort).
    
    Since the mapping is lossy (many chars map to same id), we can't
    perfectly reverse it. Instead we show the raw token IDs and attempt
    a rough decode using the most common Chinese characters.
    """
    return f"[raw_ids: {token_ids[:30]}{'...' if len(token_ids) > 30 else ''}] ({len(token_ids)} tokens)"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "/cmy/cmy/real-asr/exp/stage1_pretrain/epoch_8.pt"
    config = "/cmy/cmy/real-asr/configs/model.yaml"

    print(f"Loading model from {checkpoint}...")
    transducer, ctc_head = load_model(checkpoint, config, device)
    print("Model loaded.")

    # Test samples from the training data
    samples = [
        {
            "audio": "/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus",
            "start": 46.0, "duration": 1.36,
            "ref": "我就搬出去住"
        },
        {
            "audio": "/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus",
            "start": 128.84, "duration": 5.36,
            "ref": "我就可以考虑回去了不急不急多住些日子没什么了不起的"
        },
        {
            "audio": "/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus",
            "start": 366.24, "duration": 4.6,
            "ref": "怎么能叫混呢我这叫来激发来感悟"
        },
    ]

    print(f"\n{'='*60}")
    print("CTC Greedy Decoding (epoch_8 model, old tokenizer)")
    print(f"{'='*60}\n")

    with torch.no_grad():
        for i, sample in enumerate(samples):
            features = extract_features(
                sample["audio"], sample["start"], sample["duration"], device
            )
            feature_lens = torch.tensor([features.size(1)], device=device)

            encoder_out, encoder_out_lens = transducer.encoder(features, feature_lens)
            log_probs = ctc_head(encoder_out)

            token_ids = ctc_greedy_decode(log_probs, blank_id=0)

            print(f"Sample {i+1}:")
            print(f"  Reference: {sample['ref']}")
            print(f"  Duration:  {sample['duration']:.2f}s")
            print(f"  Encoder frames: {encoder_out.size(1)}")
            print(f"  Decoded tokens: {len(token_ids)}")
            print(f"  Token IDs: {token_ids[:50]}")

            # Check output distribution
            probs = torch.softmax(log_probs.squeeze(0), dim=-1)
            blank_prob = probs[:, 0].mean().item()
            max_non_blank = probs[:, 1:].max(dim=-1).values.mean().item()
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            print(f"  Avg blank prob: {blank_prob:.4f}")
            print(f"  Avg max non-blank prob: {max_non_blank:.4f}")
            print(f"  Avg entropy: {entropy:.2f}")
            print()


if __name__ == "__main__":
    main()
