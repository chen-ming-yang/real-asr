"""Evaluate CTC model on AISHELL test set with SentencePiece decoding.

Usage:
    python eval_ctc.py \
        --checkpoint exp/finetune_aishell/epoch_20.pt \
        --test-manifest data/manifests/aishell/test \
        --config configs/train_finetune_aishell.yaml
"""

import argparse
import json
import logging
import re
from pathlib import Path

import sentencepiece as spm
import soundfile as sf
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_cer(hypothesis: str, reference: str):
    hyp = re.sub(r"[，。！？、；：""''（）\s]", "", hypothesis)
    ref = re.sub(r"[，。！？、；：""''（）\s]", "", reference)
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    n, m = len(ref_chars), len(hyp_chars)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m], n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_finetune_aishell.yaml")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/sp_6000.model")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    import yaml
    from train.train import build_model, load_config

    config = load_config(args.config)

    transducer, ctc_head = build_model(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    transducer.load_state_dict(ckpt["model"], strict=False)
    ctc_head.load_state_dict(ckpt["ctc_head"], strict=False)
    transducer = transducer.to(args.device).eval()
    ctc_head = ctc_head.to(args.device).eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    logger.info(f"Tokenizer vocab_size={sp.GetPieceSize()}")

    # Fbank
    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=512,
        win_length=400, hop_length=160, n_mels=80,
    ).to(args.device)

    # Load test manifest
    manifest_dir = Path(args.test_manifest)
    recordings = {}
    with open(manifest_dir / "recordings.jsonl", "r") as f:
        for line in f:
            rec = json.loads(line)
            recordings[rec["id"]] = rec

    supervisions = []
    with open(manifest_dir / "supervisions.jsonl", "r") as f:
        for line in f:
            supervisions.append(json.loads(line))

    logger.info(f"Test set: {len(supervisions)} utterances")

    total_edit = 0
    total_ref = 0
    errors = []

    with torch.no_grad():
        for idx, sup in enumerate(supervisions):
            rec = recordings[sup["recording_id"]]
            audio_path = rec["sources"][0]["source"]

            data, sr = sf.read(audio_path, dtype="float32")
            waveform = torch.from_numpy(data).unsqueeze(0).to(args.device)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            features = fbank(waveform).squeeze(0).transpose(0, 1)
            features = torch.log(features.clamp(min=1e-10))
            features = features.unsqueeze(0)
            feature_lens = torch.tensor([features.size(1)], device=args.device)

            encoder_out, encoder_out_lens = transducer.encoder(features, feature_lens)
            log_probs = ctc_head(encoder_out)
            predicted_ids = log_probs.argmax(dim=-1).squeeze(0).tolist()

            # CTC collapse
            decoded_ids = []
            prev_id = -1
            for tid in predicted_ids:
                if tid != 0 and tid != prev_id:
                    decoded_ids.append(tid)
                prev_id = tid

            hypothesis = sp.DecodeIds(decoded_ids)
            reference = sup["text"]

            edit, ref_len = compute_cer(hypothesis, reference)
            total_edit += edit
            total_ref += ref_len

            if edit > 0 and len(errors) < 20:
                errors.append((sup["id"], reference, hypothesis, edit / max(ref_len, 1)))

            if (idx + 1) % 500 == 0:
                running_cer = total_edit / max(total_ref, 1) * 100
                logger.info(f"[{idx+1}/{len(supervisions)}] Running CER: {running_cer:.2f}%")

    cer = total_edit / max(total_ref, 1) * 100
    logger.info("=" * 50)
    logger.info(f"AISHELL Test CER: {cer:.2f}% ({len(supervisions)} utterances)")
    logger.info(f"Total edits: {total_edit}, Total ref chars: {total_ref}")
    logger.info("=" * 50)

    if errors:
        logger.info("\nSample errors:")
        for utt_id, ref, hyp, utt_cer in errors[:10]:
            logger.info(f"  [{utt_id}] CER={utt_cer:.2%}")
            logger.info(f"    REF: {ref}")
            logger.info(f"    HYP: {hyp}")


if __name__ == "__main__":
    main()
