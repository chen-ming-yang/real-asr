"""Evaluate CTC greedy + CTC beam + LM rescore on AISHELL test set.

Usage:
    PYTHONPATH=/cmy/cmy/real-asr python scripts/eval_aishell_ctc_lm.py \
        --checkpoint exp/stage1_icefall/epoch_50.pt \
        --config configs/train_rnnt_stageA.yaml \
        --lm-checkpoint exp/transformer_lm/best.pt \
        --test-dir data/test \
        --beam-size 10 --lm-weight 0.2
"""

import sys
sys.path.insert(0, "/cmy/cmy/real-asr")

import argparse
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


def load_lm(checkpoint_path, device="cuda"):
    from model.transformer_lm import TransformerLM
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model_cfg = config["model"]
    lm = TransformerLM(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=0.0,
        max_seq_len=model_cfg["max_seq_len"],
    )
    lm.load_state_dict(ckpt["model_state_dict"])
    lm = lm.to(device).eval()
    return lm


def extract_features(audio_path, device="cuda"):
    sample_rate = 16000
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


def log_add(a: float, b: float) -> float:
    import math
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


def ctc_prefix_beam_search(log_probs, beam_size=10, blank_id=0):
    T, V = log_probs.shape
    log_probs_np = log_probs.cpu().numpy()
    NEG_INF = float("-inf")

    beams = {(): (0.0, NEG_INF)}

    for t in range(T):
        new_beams = {}

        for prefix, (p_b, p_nb) in beams.items():
            p_total = log_add(p_b, p_nb)

            # Top-k pruning for speed
            top_k = min(beam_size * 2, V)
            top_indices = log_probs_np[t].argsort()[-top_k:][::-1]

            for c in top_indices:
                c = int(c)
                p_c = float(log_probs_np[t, c])

                if c == blank_id:
                    key = prefix
                    if key in new_beams:
                        old_b, old_nb = new_beams[key]
                        new_beams[key] = (log_add(old_b, p_total + p_c), old_nb)
                    else:
                        new_beams[key] = (p_total + p_c, NEG_INF)
                else:
                    if len(prefix) > 0 and prefix[-1] == c:
                        new_prefix = prefix + (c,)
                        if new_prefix in new_beams:
                            old_b, old_nb = new_beams[new_prefix]
                            new_beams[new_prefix] = (old_b, log_add(old_nb, p_b + p_c))
                        else:
                            new_beams[new_prefix] = (NEG_INF, p_b + p_c)
                        key = prefix
                        if key in new_beams:
                            old_b, old_nb = new_beams[key]
                            new_beams[key] = (old_b, log_add(old_nb, p_nb + p_c))
                        else:
                            new_beams[key] = (NEG_INF, p_nb + p_c)
                    else:
                        new_prefix = prefix + (c,)
                        if new_prefix in new_beams:
                            old_b, old_nb = new_beams[new_prefix]
                            new_beams[new_prefix] = (old_b, log_add(old_nb, p_total + p_c))
                        else:
                            new_beams[new_prefix] = (NEG_INF, p_total + p_c)

        scored = []
        for prefix, (p_b, p_nb) in new_beams.items():
            scored.append((prefix, log_add(p_b, p_nb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {}
        for prefix, score in scored[:beam_size]:
            beams[prefix] = new_beams[prefix]

    results = []
    for prefix, (p_b, p_nb) in beams.items():
        score = log_add(p_b, p_nb)
        results.append((list(prefix), score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def score_with_lm(lm, token_ids, device="cuda", bos_id=1):
    if len(token_ids) == 0:
        return 0.0
    input_ids = [bos_id] + token_ids
    input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
    with torch.no_grad():
        log_probs = lm.score(input_tensor)
    total_log_prob = 0.0
    for i, tid in enumerate(token_ids):
        total_log_prob += log_probs[0, i, tid].item()
    return total_log_prob / len(token_ids)


def decode_with_lm_rescore(log_probs, lm, beam_size=10, lm_weight=0.2, blank_id=0, device="cuda"):
    nbest = ctc_prefix_beam_search(log_probs, beam_size=beam_size, blank_id=blank_id)
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


def compute_cer_details(hyp, ref):
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


def load_transcript(transcript_path, test_dir):
    """Load AISHELL transcript, filtering to only utterances present in test_dir."""
    available_utts = set()
    test_dir = Path(test_dir)
    for spk_dir in test_dir.iterdir():
        if spk_dir.is_dir():
            for wav_file in spk_dir.glob("*.wav"):
                available_utts.add(wav_file.stem)

    transcript = {}
    with open(transcript_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            if utt_id in available_utts:
                text = parts[1].replace(" ", "")
                transcript[utt_id] = text

    return transcript


def find_wav_path(utt_id, test_dir):
    """Find wav path from utterance ID (e.g. BAC009S0764W0121 -> S0764/BAC009S0764W0121.wav)."""
    import re
    match = re.search(r"(S\d+)", utt_id)
    if match:
        spk = match.group(1)
        wav_path = Path(test_dir) / spk / f"{utt_id}.wav"
        if wav_path.exists():
            return str(wav_path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="exp/stage1_icefall/epoch_50.pt")
    parser.add_argument("--config", type=str, default="configs/train_rnnt_stageA.yaml")
    parser.add_argument("--lm-checkpoint", type=str, default="exp/transformer_lm/best.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/sp_6000.model")
    parser.add_argument("--test-dir", type=str, default="data/test")
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--lm-weight", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    test_dir = Path(args.test_dir)
    transcript_path = test_dir / "aishell_transcript_v0.8.txt"

    print(f"Loading ASR model from {args.checkpoint}...")
    transducer, ctc_head = load_asr_model(args.checkpoint, args.config, device)

    print(f"Loading LM from {args.lm_checkpoint}...")
    lm = load_lm(args.lm_checkpoint, device)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    print(f"Tokenizer: vocab_size={sp.GetPieceSize()}")

    print(f"Loading transcript from {transcript_path}...")
    transcript = load_transcript(transcript_path, test_dir)
    print(f"Test utterances: {len(transcript)}")
    print(f"Beam size: {args.beam_size}, LM weight: {args.lm_weight}")
    print()

    utt_ids = sorted(transcript.keys())
    if args.max_samples > 0:
        utt_ids = utt_ids[:args.max_samples]

    total_edits_greedy = 0
    total_edits_lm = 0
    total_ref_len = 0
    errors = 0
    start_time = time.time()

    print(f"{'='*70}")
    print(f"AISHELL Test Set Evaluation: CTC Greedy vs CTC Beam + LM Rescore")
    print(f"{'='*70}\n")

    with torch.no_grad():
        for i, utt_id in enumerate(utt_ids):
            wav_path = find_wav_path(utt_id, test_dir)
            if wav_path is None:
                errors += 1
                continue

            try:
                features = extract_features(wav_path, device)
            except Exception:
                errors += 1
                continue

            feature_lens = torch.tensor([features.size(1)], device=device)
            encoder_out, encoder_out_lens = transducer.encoder(features, feature_lens)
            ctc_log_probs = ctc_head(encoder_out).squeeze(0)

            # CTC greedy
            greedy_ids = ctc_greedy_decode(ctc_log_probs.unsqueeze(0), blank_id=0)
            greedy_text = sp.DecodeIds(greedy_ids)

            # CTC beam + LM
            lm_ids = decode_with_lm_rescore(
                ctc_log_probs, lm,
                beam_size=args.beam_size,
                lm_weight=args.lm_weight,
                blank_id=0,
                device=device,
            )
            lm_text = sp.DecodeIds(lm_ids)

            ref = transcript[utt_id]
            edits_g, ref_len = compute_cer_details(greedy_text, ref)
            edits_l, _ = compute_cer_details(lm_text, ref)
            total_edits_greedy += edits_g
            total_edits_lm += edits_l
            total_ref_len += ref_len

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                cer_g = total_edits_greedy / max(total_ref_len, 1) * 100
                cer_l = total_edits_lm / max(total_ref_len, 1) * 100
                speed = (i + 1) / elapsed
                print(f"  [{i+1}/{len(utt_ids)}] "
                      f"CER_greedy={cer_g:.2f}% CER_lm={cer_l:.2f}% "
                      f"({speed:.1f} utt/s, errors={errors})")

            if (i + 1) <= 5:
                print(f"  [{utt_id}]")
                print(f"    REF:     {ref}")
                print(f"    Greedy:  {greedy_text}")
                print(f"    Beam+LM: {lm_text}")
                print()

    elapsed = time.time() - start_time
    cer_greedy = total_edits_greedy / max(total_ref_len, 1) * 100
    cer_lm = total_edits_lm / max(total_ref_len, 1) * 100

    print(f"\n{'='*70}")
    print(f"AISHELL Test Set Results")
    print(f"{'='*70}")
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  LM:               {args.lm_checkpoint}")
    print(f"  Samples:          {len(utt_ids)} (errors: {errors})")
    print(f"  Beam size:        {args.beam_size}")
    print(f"  LM weight:        {args.lm_weight}")
    print(f"  CER (greedy):     {cer_greedy:.2f}%")
    print(f"  CER (beam+LM):    {cer_lm:.2f}%")
    print(f"  Improvement:      {cer_greedy - cer_lm:.2f}% absolute")
    print(f"  Time:             {elapsed:.1f}s ({len(utt_ids)/elapsed:.1f} utt/s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
