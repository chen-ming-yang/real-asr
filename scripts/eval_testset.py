"""Batch evaluation on WenetSpeech dev/test_net.

Runs CTC greedy decode on a full test set and reports overall CER.

Usage:
    PYTHONPATH=/cmy/cmy/real-asr python scripts/eval_testset.py \
        --checkpoint exp/stage1_icefall/epoch_50.pt \
        --config configs/train_stage1_icefall.yaml \
        --test-set test_net \
        --max-samples 500
"""

import sys
sys.path.insert(0, "/cmy/cmy/real-asr")

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import yaml
import sentencepiece as spm


def load_asr_model(checkpoint_path, config_path, device="cuda"):
    from train.train import build_model, load_config
    config = load_config(config_path)
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


def extract_features(audio_path, start, duration, device="cuda"):
    sample_rate = 16000
    try:
        with sf.SoundFile(audio_path, "r") as f:
            file_sr = f.samplerate
            f.seek(int(start * file_sr))
            data = f.read(frames=int(duration * file_sr), dtype="float32")
            sr = file_sr
    except Exception:
        return None

    if len(data) == 0:
        return None

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
    features = fbank(waveform).squeeze(0).transpose(0, 1)
    features = torch.log(features.clamp(min=1e-10))
    return features.unsqueeze(0).to(device)


def ctc_greedy_decode(log_probs, blank_id=0):
    predicted_ids = log_probs.argmax(dim=-1).squeeze(0).tolist()
    decoded = []
    prev = -1
    for tid in predicted_ids:
        if tid != blank_id and tid != prev:
            decoded.append(tid)
        prev = tid
    return decoded


def compute_cer_details(hyp, ref):
    """Returns (edits, ref_len)."""
    hyp = re.sub(r"[，。！？、；：\"\"\'\'\（\）\s]", "", hyp)
    ref = re.sub(r"[，。！？、；：\"\"\'\'\（\）\s]", "", ref)
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, max(n, 1)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m], n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="exp/stage1_icefall/epoch_50.pt")
    parser.add_argument("--config", type=str, default="configs/train_stage1_icefall.yaml")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/sp_6000.model")
    parser.add_argument("--test-set", type=str, default="test_net",
                        choices=["dev", "test_net", "test_meeting"])
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to evaluate (0=all)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    manifest_dir = Path(f"data/manifests/wenetspeech/{args.test_set}")

    print(f"Loading ASR model from {args.checkpoint}...")
    transducer, ctc_head = load_asr_model(args.checkpoint, args.config, device)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)

    # Load recordings index
    recs = {}
    with open(manifest_dir / "recordings.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            recs[rec["id"]] = rec

    # Load supervisions
    supervisions = []
    with open(manifest_dir / "supervisions.jsonl") as f:
        for line in f:
            supervisions.append(json.loads(line))

    if args.max_samples > 0:
        supervisions = supervisions[:args.max_samples]

    print(f"Test set: {args.test_set}, samples: {len(supervisions)}")
    print(f"Decode: CTC greedy")
    print()

    total_edits = 0
    total_ref_len = 0
    errors = 0
    start_time = time.time()

    with torch.no_grad():
        for i, sup in enumerate(supervisions):
            rec = recs.get(sup["recording_id"])
            if rec is None:
                errors += 1
                continue

            audio_path = rec["sources"][0]["source"]
            features = extract_features(audio_path, sup["start"], sup["duration"], device)
            if features is None:
                errors += 1
                continue

            feature_lens = torch.tensor([features.size(1)], device=device)
            encoder_out, encoder_out_lens = transducer.encoder(features, feature_lens)
            log_probs = ctc_head(encoder_out)
            ids = ctc_greedy_decode(log_probs, blank_id=0)
            hyp = sp.DecodeIds(ids)

            ref = sup["text"]
            edits, ref_len = compute_cer_details(hyp, ref)
            total_edits += edits
            total_ref_len += ref_len

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                running_cer = total_edits / max(total_ref_len, 1) * 100
                speed = (i + 1) / elapsed
                print(f"  [{i+1}/{len(supervisions)}] CER={running_cer:.2f}% "
                      f"({speed:.1f} samples/s, errors={errors})")

    elapsed = time.time() - start_time
    cer = total_edits / max(total_ref_len, 1) * 100

    print()
    print(f"{'=' * 60}")
    print(f"Test set:      {args.test_set}")
    print(f"Samples:       {len(supervisions)} (errors: {errors})")
    print(f"Decode:        CTC greedy")
    print(f"CER:           {cer:.2f}%")
    print(f"Total edits:   {total_edits}")
    print(f"Total ref len: {total_ref_len}")
    print(f"Time:          {elapsed:.1f}s ({len(supervisions)/elapsed:.1f} samples/s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
