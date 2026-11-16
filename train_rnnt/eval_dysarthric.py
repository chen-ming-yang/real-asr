"""Evaluate RNN-T on dysarthric test.jsonl (same format as CTC eval_dysarthric.py)."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import sentencepiece as spm
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_rnnt.eval import greedy_decode, beam_search_decode, compute_cer, edit_distance


def normalize(text: str) -> str:
    return re.sub(r"[，。！？、；：\u201c\u201d\u2018\u2019（）\s]", "", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", default="/cmy/cmy/Qwen3-ASR/data/test.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="train_rnnt/config_dysarthric.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer/sp_6000.model")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--decode", default="both", choices=["greedy", "beam", "both"])
    parser.add_argument("--audio-prefix-replace",
                        default="/cmy/after_catting=/cmy/data/after_catting")
    args = parser.parse_args()

    old_prefix, new_prefix = args.audio_prefix_replace.split("=")

    from train_rnnt.train import load_config, build_model
    config = load_config(args.config)
    model = build_model(config).cuda().eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded: {args.checkpoint}, epoch={ckpt.get('epoch', '?')}")

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)

    samples = []
    with open(args.test_jsonl) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Test set: {len(samples)} utterances")
    print(f"Decode: {args.decode}, beam={args.beam_size}")
    print()

    speaker_stats = {}
    greedy_hyps, beam_hyps, refs = [], [], []
    errors = 0
    t0 = time.time()

    with torch.no_grad():
        for i, sample in enumerate(samples):
            audio_path = sample["source"].replace(old_prefix, new_prefix)
            ref_text = normalize(sample["target"])
            speaker = str(sample.get("speaker", "unk"))

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "n": 0, "edit_g": 0, "chars": 0, "edit_b": 0
                }

            try:
                audio, sr = sf.read(audio_path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    import torchaudio
                    wav = torch.from_numpy(audio).unsqueeze(0)
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                    audio = wav.squeeze(0).numpy()
                if len(audio) < 400:
                    continue

                wav = torch.from_numpy(audio).unsqueeze(0).cuda()
                features = kaldi.fbank(
                    wav, num_mel_bins=80, sample_frequency=16000
                ).unsqueeze(0)
                feat_lens = torch.tensor([features.size(1)], device=features.device)
                enc_out, enc_lens, _ = model(features, feat_lens)

                ref_chars = list(ref_text)
                refs.append(ref_chars)
                speaker_stats[speaker]["n"] += 1
                speaker_stats[speaker]["chars"] += len(ref_chars)

                if args.decode in ("greedy", "both"):
                    tokens = greedy_decode(model, enc_out, enc_lens)
                    hyp = normalize(sp.DecodeIds(tokens[0]))
                    hyp_chars = list(hyp)
                    greedy_hyps.append(hyp_chars)
                    eg = edit_distance(hyp_chars, ref_chars)
                    speaker_stats[speaker]["edit_g"] += eg

                if args.decode in ("beam", "both"):
                    tokens = beam_search_decode(
                        model, enc_out, enc_lens, beam_size=args.beam_size
                    )
                    hyp = normalize(sp.DecodeIds(tokens[0]))
                    hyp_chars = list(hyp)
                    beam_hyps.append(hyp_chars)
                    eb = edit_distance(hyp_chars, ref_chars)
                    speaker_stats[speaker]["edit_b"] += eb

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR {sample.get('key')}: {e}")
                continue

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                speed = (i + 1) / elapsed
                msg = f"  [{i+1}/{len(samples)}]"
                if greedy_hyps:
                    msg += f" greedy={compute_cer(greedy_hyps, refs)*100:.2f}%"
                if beam_hyps:
                    msg += f" beam={compute_cer(beam_hyps, refs)*100:.2f}%"
                msg += f" ({speed:.1f} utt/s, errors={errors})"
                print(msg, flush=True)

    print()
    print("=" * 60)
    print("Dysarthric Test Results (RNN-T)")
    print("=" * 60)
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Samples:       {len(refs)} (errors: {errors})")
    if greedy_hyps:
        print(f"  CER (greedy):  {compute_cer(greedy_hyps, refs)*100:.2f}%")
    if beam_hyps:
        print(f"  CER (beam={args.beam_size}): {compute_cer(beam_hyps, refs)*100:.2f}%")
    print()
    print(f"{'Speaker':<10} {'Count':<8} {'CER greedy':<14} {'CER beam':<14}")
    print("-" * 50)
    for spk in sorted(speaker_stats.keys()):
        st = speaker_stats[spk]
        if st["chars"] == 0:
            continue
        cg = st["edit_g"] / st["chars"] * 100 if greedy_hyps else 0
        cb = st["edit_b"] / st["chars"] * 100 if beam_hyps else 0
        print(f"{spk:<10} {st['n']:<8} {cg:<14.2f} {cb:<14.2f}")


if __name__ == "__main__":
    main()
