"""Inference script for v8+ model (SentencePiece tokenizer).

Usage:
    PYTHONPATH=/cmy/cmy/real-asr python infer_sp.py
"""
import sys
sys.path.insert(0, "/cmy/cmy/real-asr")

import json
import torch
import torchaudio
import soundfile as sf
import yaml
import sentencepiece as spm
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


def greedy_rnnt_decode(transducer, encoder_out, max_symbols=100, blank_id=0):
    """Simple greedy RNN-T decoding."""
    B, T, D = encoder_out.shape
    device = encoder_out.device

    decoded = []
    t = 0
    context = [blank_id, blank_id]

    while t < T:
        encoder_frame = encoder_out[:, t:t+1, :]
        pred_input = torch.tensor([[context[-2], context[-1]]], device=device, dtype=torch.long)

        predictor_out = transducer.predictor(pred_input)
        predictor_frame = predictor_out[:, -1:, :]

        joiner_out = transducer.joiner(encoder_frame, predictor_frame)
        log_probs = torch.log_softmax(joiner_out.squeeze(1), dim=-1)
        y = log_probs.argmax(dim=-1).item()

        if y == blank_id:
            t += 1
        else:
            decoded.append(y)
            context.append(y)
            if len(decoded) >= max_symbols:
                break
    return decoded


def compute_cer(hyp, ref):
    import re
    hyp = re.sub(r"[，。！？、；：""''（）\s]", "", hyp)
    ref = re.sub(r"[，。！？、；：""''（）\s]", "", ref)
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m] / max(n, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "/cmy/cmy/real-asr/exp/stage1_pretrain/epoch_20.pt"
    config = "/cmy/cmy/real-asr/configs/model.yaml"
    tokenizer_path = "/cmy/cmy/real-asr/data/tokenizer/sp_6000.model"

    print(f"Loading model from {checkpoint}...")
    transducer, ctc_head = load_model(checkpoint, config, device)

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    print(f"Loaded tokenizer: vocab_size={sp.GetPieceSize()}")
    print(f"Model loaded.\n")

    # Test samples from M/S subset
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
        {
            "audio": "/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus",
            "start": 20.08, "duration": 4.32,
            "ref": "怎么样这些日子住得还习惯吧"
        },
        {
            "audio": "/cmy/cmy/WenetSpeech/DATA_UNTAR_DIRECTORY/audio/train/youtube/B00000/Y0000000000_--5llN02F84.opus",
            "start": 26.68, "duration": 3.8,
            "ref": "对了美静这段日子经常不和我们一起用餐"
        },
    ]

    # Also try dev set samples
    dev_sup_path = "/cmy/cmy/real-asr/data/manifests/wenetspeech/train_ms/supervisions.jsonl"
    dev_rec_path = "/cmy/cmy/real-asr/data/manifests/wenetspeech/train_ms/recordings.jsonl"

    # Load a few random samples from different parts of the dataset
    import random
    random.seed(42)
    with open(dev_sup_path) as f:
        lines = []
        for i, line in enumerate(f):
            if i in [100, 500, 1000, 5000, 10000]:
                lines.append(json.loads(line))
    
    recs = {}
    with open(dev_rec_path) as f:
        for line in f:
            rec = json.loads(line)
            recs[rec["id"]] = rec

    for item in lines:
        rec = recs.get(item["recording_id"])
        if rec:
            samples.append({
                "audio": rec["sources"][0]["source"],
                "start": item["start"],
                "duration": item["duration"],
                "ref": item["text"],
            })

    print(f"{'='*70}")
    print(f"Inference Test - epoch_20 (SentencePiece tokenizer, M+S data)")
    print(f"{'='*70}\n")

    total_cer_ctc = 0
    total_cer_rnnt = 0
    count = 0

    with torch.no_grad():
        for i, sample in enumerate(samples):
            features = extract_features(
                sample["audio"], sample["start"], sample["duration"], device
            )
            feature_lens = torch.tensor([features.size(1)], device=device)

            encoder_out, encoder_out_lens = transducer.encoder(features, feature_lens)

            # CTC decoding
            log_probs = ctc_head(encoder_out)
            ctc_ids = ctc_greedy_decode(log_probs, blank_id=0)
            ctc_text = sp.DecodeIds(ctc_ids)

            # RNN-T greedy decoding
            rnnt_ids = greedy_rnnt_decode(transducer, encoder_out, blank_id=0)
            rnnt_text = sp.DecodeIds(rnnt_ids)

            ref = sample["ref"]
            cer_ctc = compute_cer(ctc_text, ref)
            cer_rnnt = compute_cer(rnnt_text, ref)
            total_cer_ctc += cer_ctc
            total_cer_rnnt += cer_rnnt
            count += 1

            print(f"Sample {i+1} ({sample['duration']:.1f}s):")
            print(f"  REF:      {ref}")
            print(f"  CTC:      {ctc_text}")
            print(f"  RNN-T:    {rnnt_text}")
            print(f"  CER(CTC): {cer_ctc*100:.1f}%  CER(RNN-T): {cer_rnnt*100:.1f}%")
            print()

    print(f"{'='*70}")
    print(f"Average CER (CTC):   {total_cer_ctc/count*100:.1f}%")
    print(f"Average CER (RNN-T): {total_cer_rnnt/count*100:.1f}%")
    print(f"Samples tested: {count}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
