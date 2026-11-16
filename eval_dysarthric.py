"""Evaluate CTC + LM on dysarthric speech test set.

Usage:
    PYTHONPATH=/cmy/cmy/real-asr python eval_dysarthric.py
"""

import json
import re
import time
from pathlib import Path

import sentencepiece as spm
import soundfile as sf
import torch
import torchaudio

from train.train import build_model, load_config
from scripts.decode_ctc_lm import (
    ctc_prefix_beam_search,
    load_lm,
    score_with_lm,
    log_add,
)


def compute_cer(hyp, ref):
    hyp = re.sub(r"[，。！？、；：\u201c\u201d\u2018\u2019（）\s]", "", hyp)
    ref = re.sub(r"[，。！？、；：\u201c\u201d\u2018\u2019（）\s]", "", ref)
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 1
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


def decode_with_lm_rescore(log_probs, lm, beam_size=10, lm_weight=0.3, device="cuda"):
    nbest = ctc_prefix_beam_search(log_probs, beam_size=beam_size, blank_id=0)
    if not nbest:
        return []
    best_score = float("-inf")
    best_ids = []
    for token_ids, ctc_score in nbest:
        if len(token_ids) == 0:
            continue
        lm_score = score_with_lm(lm, token_ids, device=device)
        ctc_score_norm = ctc_score / max(len(token_ids), 1)
        combined = ctc_score_norm + lm_weight * lm_score
        if combined > best_score:
            best_score = combined
            best_ids = token_ids
    return best_ids


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", default="/cmy/cmy/Qwen3-ASR/data/test.jsonl")
    parser.add_argument("--checkpoint", default="exp/finetune_aishell/epoch_20.pt")
    parser.add_argument("--config", default="configs/train_finetune_aishell.yaml")
    parser.add_argument("--lm-checkpoint", default="exp/transformer_lm/best.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer/sp_6000.model")
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--lm-weight", type=float, default=0.3)
    parser.add_argument("--audio-prefix-replace", default="/cmy/after_catting=/cmy/data/after_catting",
                        help="old=new path prefix replacement")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device

    # Parse audio prefix replacement
    old_prefix, new_prefix = args.audio_prefix_replace.split("=")

    # Load ASR model
    print(f"Loading ASR model: {args.checkpoint}")
    config = load_config(args.config)
    transducer, ctc_head = build_model(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    transducer.load_state_dict(ckpt["model"], strict=False)
    ctc_head.load_state_dict(ckpt["ctc_head"], strict=False)
    transducer = transducer.to(device).eval()
    ctc_head = ctc_head.to(device).eval()

    # Load LM
    print(f"Loading LM: {args.lm_checkpoint}")
    lm = load_lm(args.lm_checkpoint, device)

    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    print(f"Tokenizer vocab_size={sp.GetPieceSize()}")

    # Fbank
    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80
    ).to(device)

    # Load test set
    samples = []
    with open(args.test_jsonl, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Test set: {len(samples)} utterances")
    print(f"Beam size: {args.beam_size}, LM weight: {args.lm_weight}")
    print()

    # Per-speaker stats
    speaker_stats = {}
    total_edit_greedy = 0
    total_edit_lm = 0
    total_ref = 0
    skipped = 0
    start_time = time.time()

    with torch.no_grad():
        for idx, sample in enumerate(samples):
            audio_path = sample["source"].replace(old_prefix, new_prefix)
            ref = sample["target"]
            speaker = sample.get("speaker", "unknown")

            if not Path(audio_path).exists():
                skipped += 1
                continue

            data, sr = sf.read(audio_path, dtype="float32")
            waveform = torch.from_numpy(data).unsqueeze(0).to(device)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            features = fbank(waveform).squeeze(0).transpose(0, 1)
            features = torch.log(features.clamp(min=1e-10)).unsqueeze(0)
            feature_lens = torch.tensor([features.size(1)], device=device)

            encoder_out, _ = transducer.encoder(features, feature_lens)
            ctc_log_probs = ctc_head(encoder_out).squeeze(0)

            # CTC greedy
            predicted_ids = ctc_log_probs.argmax(dim=-1).tolist()
            greedy_ids = []
            prev = -1
            for tid in predicted_ids:
                if tid != 0 and tid != prev:
                    greedy_ids.append(tid)
                prev = tid
            greedy_text = sp.DecodeIds(greedy_ids)

            # CTC beam + LM
            lm_ids = decode_with_lm_rescore(
                ctc_log_probs, lm,
                beam_size=args.beam_size,
                lm_weight=args.lm_weight,
                device=device,
            )
            lm_text = sp.DecodeIds(lm_ids)

            edit_g, ref_len = compute_cer(greedy_text, ref)
            edit_l, _ = compute_cer(lm_text, ref)
            total_edit_greedy += edit_g
            total_edit_lm += edit_l
            total_ref += ref_len

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {"edit_g": 0, "edit_l": 0, "ref": 0, "count": 0}
            speaker_stats[speaker]["edit_g"] += edit_g
            speaker_stats[speaker]["edit_l"] += edit_l
            speaker_stats[speaker]["ref"] += ref_len
            speaker_stats[speaker]["count"] += 1

            if (idx + 1) % 200 == 0:
                cer_g = total_edit_greedy / max(total_ref, 1) * 100
                cer_l = total_edit_lm / max(total_ref, 1) * 100
                elapsed = time.time() - start_time
                print(f"  [{idx+1}/{len(samples)}] greedy={cer_g:.2f}% beam+LM={cer_l:.2f}% "
                      f"({(idx+1)/elapsed:.1f} utt/s, skipped={skipped})")

    elapsed = time.time() - start_time
    cer_greedy = total_edit_greedy / max(total_ref, 1) * 100
    cer_lm = total_edit_lm / max(total_ref, 1) * 100

    print()
    print("=" * 70)
    print("Dysarthric Speech Test Results")
    print("=" * 70)
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  LM:            {args.lm_checkpoint}")
    print(f"  Test set:      {args.test_jsonl}")
    print(f"  Utterances:    {len(samples)} (skipped: {skipped})")
    print(f"  Beam / LM wt:  {args.beam_size} / {args.lm_weight}")
    print(f"  CER (greedy):  {cer_greedy:.2f}%")
    print(f"  CER (beam+LM): {cer_lm:.2f}%")
    print(f"  Improvement:   {cer_greedy - cer_lm:.2f}% absolute")
    print(f"  Time:          {elapsed:.1f}s")
    print("=" * 70)

    # Per-speaker breakdown
    print()
    print(f"{'Speaker':<10} {'Count':<8} {'CER greedy':<14} {'CER beam+LM':<14}")
    print("-" * 50)
    for spk in sorted(speaker_stats.keys()):
        s = speaker_stats[spk]
        cer_g = s["edit_g"] / max(s["ref"], 1) * 100
        cer_l = s["edit_l"] / max(s["ref"], 1) * 100
        print(f"{spk:<10} {s['count']:<8} {cer_g:<14.2f} {cer_l:<14.2f}")


if __name__ == "__main__":
    main()
