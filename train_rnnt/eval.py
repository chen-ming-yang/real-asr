"""Evaluate RNN-T model: greedy + beam search, optional averaged checkpoint."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _predictor_input(tokens, blank_id=0, device="cuda"):
    """Build predictor input: blank + tokens (or just blank if empty)."""
    y = [blank_id] + list(tokens)
    return torch.tensor([y], dtype=torch.long, device=device)


def greedy_decode(model, encoder_out, encoder_out_lens, max_symbols=5):
    """Proper RNN-T greedy: stay on frame until blank, then advance."""
    B = encoder_out.size(0)
    device = encoder_out.device
    blank_id = 0
    results = []

    for b in range(B):
        T = int(encoder_out_lens[b].item())
        enc = encoder_out[b, :T]
        hyp = []
        t = 0
        while t < T:
            symbols = 0
            while symbols < max_symbols:
                y = _predictor_input(hyp, blank_id, device)
                pred = model.predictor(y)[:, -1:, :]
                enc_t = enc[t:t+1].unsqueeze(0)
                logits = model.joiner(enc_t, pred).squeeze()
                token = int(logits.argmax().item())
                if token == blank_id:
                    break
                hyp.append(token)
                symbols += 1
            t += 1
        results.append(hyp)
    return results


def beam_search_decode(model, encoder_out, encoder_out_lens, beam_size=8, max_symbols=1):
    """Fast frame-synchronous RNN-T beam search (one emit-or-blank per frame)."""
    B = encoder_out.size(0)
    device = encoder_out.device
    blank_id = 0
    results = []
    neg_inf = -1e9

    for b in range(B):
        T = int(encoder_out_lens[b].item())
        enc = encoder_out[b, :T]
        beams = {(): 0.0}  # hyp -> score

        for t in range(T):
            enc_t = enc[t:t+1].unsqueeze(0)
            candidates = {}
            for hyp, score in beams.items():
                y = _predictor_input(hyp, blank_id, device)
                pred = model.predictor(y)[:, -1:, :]
                logits = model.joiner(enc_t, pred).squeeze()
                log_probs = F.log_softmax(logits.float(), dim=-1)

                # blank: keep hyp
                ns = score + float(log_probs[blank_id])
                candidates[hyp] = max(candidates.get(hyp, neg_inf), ns)

                # top non-blanks
                vals, idxs = torch.topk(log_probs, k=min(beam_size + 1, log_probs.size(0)))
                for val, idx in zip(vals.tolist(), idxs.tolist()):
                    if idx == blank_id:
                        continue
                    nh = hyp + (idx,)
                    ns = score + val
                    candidates[nh] = max(candidates.get(nh, neg_inf), ns)

            beams = dict(
                sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:beam_size]
            )

        best = max(beams.items(), key=lambda x: x[1])[0] if beams else ()
        results.append(list(best))
    return results


def compute_cer(hyps, refs):
    total_edits = 0
    total_chars = 0
    for hyp, ref in zip(hyps, refs):
        total_edits += edit_distance(hyp, ref)
        total_chars += len(ref)
    return total_edits / max(total_chars, 1)


def edit_distance(a, b):
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = tmp
    return dp[lb]


def load_model(config_path, checkpoint):
    from train_rnnt.train import load_config, build_model
    config = load_config(config_path)
    model = build_model(config)
    model = model.cuda().eval()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", "?")
    return model, epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="train_rnnt/config_aishell.yaml")
    parser.add_argument("--test-manifest", type=str, default="data/manifests/aishell/test")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/sp_6000.model")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--decode", type=str, default="both",
                        choices=["greedy", "beam", "both"])
    args = parser.parse_args()

    model, epoch = load_model(args.config, args.checkpoint)
    print(f"Loaded: {args.checkpoint}, epoch={epoch}")

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)

    manifest_dir = Path(args.test_manifest)
    sups = {}
    with open(manifest_dir / "supervisions.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            sups[obj["recording_id"]] = obj["text"]

    from lhotse import RecordingSet
    import torchaudio.compliance.kaldi as kaldi
    recordings = RecordingSet.from_jsonl(manifest_dir / "recordings.jsonl")

    greedy_hyps, beam_hyps, refs = [], [], []
    n = 0

    with torch.no_grad():
        for rec in recordings:
            ref_text = sups.get(rec.id, "")
            if not ref_text:
                continue
            audio_np = rec.load_audio()
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            if audio_np.shape[1] < 400:
                continue
            audio = torch.tensor(audio_np, dtype=torch.float32).cuda()
            if audio.shape[0] > 1:
                audio = audio[:1]

            features = kaldi.fbank(audio, num_mel_bins=80, sample_frequency=16000).unsqueeze(0)
            feat_lens = torch.tensor([features.size(1)], device=features.device)
            enc_out, enc_lens, _ = model(features, feat_lens)

            ref_chars = list(ref_text.replace(" ", ""))
            refs.append(ref_chars)

            if args.decode in ("greedy", "both"):
                tokens = greedy_decode(model, enc_out, enc_lens)
                hyp_text = sp.DecodeIds(tokens[0])
                greedy_hyps.append(list(hyp_text.replace(" ", "")))

            if args.decode in ("beam", "both"):
                tokens = beam_search_decode(
                    model, enc_out, enc_lens, beam_size=args.beam_size
                )
                hyp_text = sp.DecodeIds(tokens[0])
                beam_hyps.append(list(hyp_text.replace(" ", "")))

            n += 1
            if n % 500 == 0:
                msg = f"  [{n}]"
                if greedy_hyps:
                    msg += f" greedy={compute_cer(greedy_hyps, refs)*100:.2f}%"
                if beam_hyps:
                    msg += f" beam={compute_cer(beam_hyps, refs)*100:.2f}%"
                print(msg, flush=True)

    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples:    {len(refs)}")
    if greedy_hyps:
        print(f"CER greedy: {compute_cer(greedy_hyps, refs)*100:.2f}%")
    if beam_hyps:
        print(f"CER beam({args.beam_size}): {compute_cer(beam_hyps, refs)*100:.2f}%")


if __name__ == "__main__":
    main()
