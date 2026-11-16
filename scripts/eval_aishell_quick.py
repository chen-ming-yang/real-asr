"""Quick AISHELL test eval with CTC greedy decode."""
import logging
logging.disable(logging.INFO)

import sys
sys.path.insert(0, "/cmy/cmy/real-asr")

import re
import time
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import sentencepiece as spm
from train.train import build_model, load_config


def main():
    device = "cuda"
    checkpoint_path = "exp/ctc_L/epoch_10.pt"
    config_path = "configs/train_ctc_L.yaml"
    tokenizer_path = "data/tokenizer/sp_6000.model"
    test_dir = Path("data/test")

    print(f"Loading model from {checkpoint_path}...", flush=True)
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

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    print("Model loaded.", flush=True)

    # Load AISHELL transcripts
    transcripts = {}
    with open(test_dir / "aishell_transcript_v0.8.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                transcripts[parts[0]] = "".join(parts[1:])

    wav_files = sorted(test_dir.glob("S*/*.wav"))
    print(f"AISHELL test: {len(wav_files)} wavs, {len(transcripts)} transcripts", flush=True)

    sample_rate = 16000
    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=512,
        win_length=int(0.025 * sample_rate),
        hop_length=int(0.010 * sample_rate),
        n_mels=80,
    ).to(device)

    total_edits = 0
    total_ref_len = 0
    errors = 0
    matched = 0
    start_time = time.time()

    with torch.no_grad():
        for i, wav_path in enumerate(wav_files):
            utt_id = wav_path.stem
            ref = transcripts.get(utt_id)
            if ref is None:
                errors += 1
                continue

            try:
                data, sr = sf.read(str(wav_path), dtype="float32")
            except Exception:
                errors += 1
                continue

            if len(data) == 0:
                errors += 1
                continue

            if len(data.shape) == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T)
            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.to(device)
            features = fbank(waveform).squeeze(0).transpose(0, 1)
            features = torch.log(features.clamp(min=1e-10)).unsqueeze(0)

            feature_lens = torch.tensor([features.size(1)], device=device)
            encoder_out, encoder_out_lens = transducer.encoder(features, feature_lens)
            log_probs = ctc_head(encoder_out)
            predicted_ids = log_probs.argmax(dim=-1).squeeze(0).tolist()

            decoded = []
            prev = -1
            for tid in predicted_ids:
                if tid != 0 and tid != prev:
                    decoded.append(tid)
                prev = tid
            hyp = sp.DecodeIds(decoded)

            ref_clean = re.sub(r"[，。！？、；：\s]", "", ref)
            hyp_clean = re.sub(r"[，。！？、；：\s]", "", hyp)
            n, m = len(ref_clean), len(hyp_clean)
            if n == 0:
                total_edits += m
                total_ref_len += max(n, 1)
            else:
                dp = [[0] * (m + 1) for _ in range(n + 1)]
                for ii in range(n + 1):
                    dp[ii][0] = ii
                for jj in range(m + 1):
                    dp[0][jj] = jj
                for ii in range(1, n + 1):
                    for jj in range(1, m + 1):
                        if ref_clean[ii - 1] == hyp_clean[jj - 1]:
                            dp[ii][jj] = dp[ii - 1][jj - 1]
                        else:
                            dp[ii][jj] = 1 + min(dp[ii - 1][jj], dp[ii][jj - 1], dp[ii - 1][jj - 1])
                total_edits += dp[n][m]
                total_ref_len += n

            matched += 1

            if (i + 1) % 200 == 0:
                elapsed = time.time() - start_time
                running_cer = total_edits / max(total_ref_len, 1) * 100
                print(f"  [{i+1}/{len(wav_files)}] CER={running_cer:.2f}% "
                      f"({(i+1)/elapsed:.1f} utt/s, errors={errors})", flush=True)

    elapsed = time.time() - start_time
    cer = total_edits / max(total_ref_len, 1) * 100
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"Test set:      AISHELL test", flush=True)
    print(f"Checkpoint:    {checkpoint_path}", flush=True)
    print(f"Samples:       {matched} (errors/skipped: {errors})", flush=True)
    print(f"Decode:        CTC greedy", flush=True)
    print(f"CER:           {cer:.2f}%", flush=True)
    print(f"Total edits:   {total_edits}", flush=True)
    print(f"Total ref len: {total_ref_len}", flush=True)
    print(f"Time:          {elapsed:.1f}s ({matched/elapsed:.1f} utt/s)", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
