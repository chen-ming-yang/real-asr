"""CTC Beam Search with Transformer LM rescoring.

Usage:
    PYTHONPATH=/cmy/cmy/real-asr python scripts/decode_ctc_lm.py \
        --checkpoint exp/stage1_icefall/epoch_50.pt \
        --lm-checkpoint exp/transformer_lm/best.pt \
        --beam-size 10 --lm-weight 0.2
"""

import sys
sys.path.insert(0, "/cmy/cmy/real-asr")

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
import soundfile as sf
import yaml
import sentencepiece as spm


def load_asr_model(checkpoint_path, config_path, device="cuda"):
    from train.train import build_model
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Handle inheritance
    if "inherit" in config:
        base_dir = Path(config_path).parent
        base_path = base_dir / config.pop("inherit")
        with open(base_path) as f:
            base_config = yaml.safe_load(f)
        from train.train import deep_merge
        config = deep_merge(base_config, config)

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


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    beam_size: int = 10,
    blank_id: int = 0,
) -> List[Tuple[List[int], float]]:
    """CTC prefix beam search.

    Args:
        log_probs: (T, V) log probabilities from CTC head
        beam_size: beam width
        blank_id: blank token ID

    Returns:
        List of (token_ids, score) tuples, sorted by score descending
    """
    T, V = log_probs.shape
    log_probs = log_probs.cpu().numpy()

    # Each beam: (prefix_tuple, (p_blank, p_non_blank))
    # p_blank = log prob of all paths ending in blank with this prefix
    # p_non_blank = log prob of all paths ending in non-blank with this prefix
    NEG_INF = float("-inf")

    beams = {(): (0.0, NEG_INF)}  # (prefix) -> (p_blank, p_non_blank)

    for t in range(T):
        new_beams = {}

        for prefix, (p_b, p_nb) in beams.items():
            p_total = log_add(p_b, p_nb)

            for c in range(V):
                p_c = log_probs[t, c]

                if c == blank_id:
                    # Extend with blank
                    key = prefix
                    if key in new_beams:
                        old_b, old_nb = new_beams[key]
                        new_beams[key] = (log_add(old_b, p_total + p_c), old_nb)
                    else:
                        new_beams[key] = (p_total + p_c, NEG_INF)
                else:
                    # Extend with character c
                    if len(prefix) > 0 and prefix[-1] == c:
                        # Same as last char: only extend from blank ending
                        new_prefix = prefix + (c,)
                        if new_prefix in new_beams:
                            old_b, old_nb = new_beams[new_prefix]
                            new_beams[new_prefix] = (old_b, log_add(old_nb, p_b + p_c))
                        else:
                            new_beams[new_prefix] = (NEG_INF, p_b + p_c)

                        # Also: collapse (same char following non-blank)
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

        # Prune to beam_size
        scored = []
        for prefix, (p_b, p_nb) in new_beams.items():
            scored.append((prefix, log_add(p_b, p_nb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {}
        for prefix, score in scored[:beam_size]:
            beams[prefix] = new_beams[prefix]

    # Final results
    results = []
    for prefix, (p_b, p_nb) in beams.items():
        score = log_add(p_b, p_nb)
        results.append((list(prefix), score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    import math
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


def score_with_lm(lm, token_ids: List[int], device="cuda", bos_id=1) -> float:
    """Score a token sequence with the LM.

    Returns average log-probability per token.
    """
    if len(token_ids) == 0:
        return 0.0

    input_ids = [bos_id] + token_ids
    input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)

    with torch.no_grad():
        log_probs = lm.score(input_tensor)  # (1, seq_len, vocab)

    total_log_prob = 0.0
    for i, tid in enumerate(token_ids):
        total_log_prob += log_probs[0, i, tid].item()

    return total_log_prob / len(token_ids)


def decode_with_lm_rescore(
    log_probs: torch.Tensor,
    lm,
    beam_size: int = 10,
    lm_weight: float = 0.2,
    blank_id: int = 0,
    device: str = "cuda",
) -> List[int]:
    """CTC beam search + LM rescoring.

    1. Run CTC prefix beam search to get N-best hypotheses
    2. Rescore each hypothesis with LM
    3. Return the best combined score
    """
    # Get N-best from CTC beam search
    nbest = ctc_prefix_beam_search(log_probs, beam_size=beam_size, blank_id=blank_id)

    if not nbest:
        return []

    # Rescore with LM
    best_score = float("-inf")
    best_ids = []

    for token_ids, ctc_score in nbest:
        if len(token_ids) == 0:
            continue
        lm_score = score_with_lm(lm, token_ids, device=device)
        # Normalize CTC score by length
        ctc_score_norm = ctc_score / max(len(token_ids), 1)
        combined = ctc_score_norm + lm_weight * lm_score

        if combined > best_score:
            best_score = combined
            best_ids = token_ids

    return best_ids


def compute_cer(hyp, ref):
    hyp = re.sub(r"[，。！？、；：""''（）\s]", "", hyp)
    ref = re.sub(r"[，。！？、；：""''（）\s]", "", ref)
    n, m = len(ref), len(hyp)
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
    return dp[n][m] / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="exp/stage1_icefall/epoch_50.pt")
    parser.add_argument("--config", type=str, default="configs/train_stage1_icefall.yaml")
    parser.add_argument("--lm-checkpoint", type=str, default="exp/transformer_lm/best.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/sp_6000.model")
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--lm-weight", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    print(f"Loading ASR model from {args.checkpoint}...")
    transducer, ctc_head = load_asr_model(args.checkpoint, args.config, device)

    print(f"Loading LM from {args.lm_checkpoint}...")
    lm = load_lm(args.lm_checkpoint, device)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    print(f"Tokenizer: vocab_size={sp.GetPieceSize()}")
    print(f"Beam size: {args.beam_size}, LM weight: {args.lm_weight}")
    print()

    # Test samples
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

    # Add more samples from manifest
    sup_path = "/cmy/cmy/real-asr/data/manifests/wenetspeech/train_ms/supervisions.jsonl"
    rec_path = "/cmy/cmy/real-asr/data/manifests/wenetspeech/train_ms/recordings.jsonl"
    try:
        with open(sup_path) as f:
            extra_sups = []
            for i, line in enumerate(f):
                if i in [100, 500, 1000, 5000, 10000, 50000, 100000, 200000, 300000]:
                    extra_sups.append(json.loads(line))
        recs = {}
        with open(rec_path) as f:
            for line in f:
                rec = json.loads(line)
                recs[rec["id"]] = rec
        for item in extra_sups:
            rec = recs.get(item["recording_id"])
            if rec:
                samples.append({
                    "audio": rec["sources"][0]["source"],
                    "start": item["start"],
                    "duration": item["duration"],
                    "ref": item["text"],
                })
    except Exception as e:
        print(f"Warning: Could not load extra samples: {e}")

    print(f"{'=' * 70}")
    print(f"CTC Beam Search + LM Rescore Decode")
    print(f"beam_size={args.beam_size}, lm_weight={args.lm_weight}")
    print(f"{'=' * 70}\n")

    total_cer_greedy = 0.0
    total_cer_lm = 0.0
    count = 0

    with torch.no_grad():
        for i, sample in enumerate(samples):
            features = extract_features(
                sample["audio"], sample["start"], sample["duration"], device
            )
            feature_lens = torch.tensor([features.size(1)], device=device)
            encoder_out, encoder_out_lens = transducer.encoder(features, feature_lens)
            ctc_log_probs = ctc_head(encoder_out).squeeze(0)  # (T, V)

            # CTC greedy
            greedy_ids = ctc_greedy_decode(ctc_log_probs.unsqueeze(0), blank_id=0)
            greedy_text = sp.DecodeIds(greedy_ids)

            # CTC beam + LM rescore
            lm_ids = decode_with_lm_rescore(
                ctc_log_probs, lm,
                beam_size=args.beam_size,
                lm_weight=args.lm_weight,
                blank_id=0,
                device=device,
            )
            lm_text = sp.DecodeIds(lm_ids)

            ref = sample["ref"]
            cer_greedy = compute_cer(greedy_text, ref)
            cer_lm = compute_cer(lm_text, ref)
            total_cer_greedy += cer_greedy
            total_cer_lm += cer_lm
            count += 1

            print(f"Sample {i + 1} ({sample['duration']:.1f}s):")
            print(f"  REF:       {ref}")
            print(f"  Greedy:    {greedy_text} (CER: {cer_greedy * 100:.1f}%)")
            print(f"  Beam+LM:   {lm_text} (CER: {cer_lm * 100:.1f}%)")
            print()

    print(f"{'=' * 70}")
    print(f"Average CER (CTC greedy):     {total_cer_greedy / count * 100:.1f}%")
    print(f"Average CER (CTC beam + LM):  {total_cer_lm / count * 100:.1f}%")
    print(f"Improvement:                   {(total_cer_greedy - total_cer_lm) / count * 100:.1f}% absolute")
    print(f"Samples tested: {count}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
