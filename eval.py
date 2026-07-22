"""Evaluation script for Real-ASR model.

Computes CER/WER on test sets, with breakdown by severity level.

Usage:
    python eval.py \
        --checkpoint exp/stage3_finetune/best.pt \
        --test-manifest data/manifests/dysarthric_test \
        --config configs/model.yaml
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_cer(hypothesis: str, reference: str) -> Dict[str, int]:
    """Compute character-level edit distance metrics.

    Returns dict with: substitutions, insertions, deletions, total_ref_chars
    """
    # Remove spaces and punctuation for CER calculation
    import re
    hyp = re.sub(r"[，。！？、；：""''（）\s]", "", hypothesis)
    ref = re.sub(r"[，。！？、；：""''（）\s]", "", reference)

    ref_chars = list(ref)
    hyp_chars = list(hyp)

    # Dynamic programming for edit distance
    n = len(ref_chars)
    m = len(hyp_chars)
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

    edit_distance = dp[n][m]
    return {
        "edit_distance": edit_distance,
        "ref_len": max(n, 1),
        "cer": edit_distance / max(n, 1),
    }


def evaluate(
    checkpoint_path: str,
    test_manifest_dir: str,
    config_path: str,
    device: str = "cuda",
    streaming: bool = True,
):
    """Run evaluation on test set."""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Build model
    from train.train import build_model
    transducer, ctc_head = build_model(config["model"])

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_state = ckpt.get("model", ckpt)
    if any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    if any(k.startswith("student.") for k in model_state.keys()):
        model_state = {
            k.replace("student.", ""): v
            for k, v in model_state.items()
            if k.startswith("student.")
        }

    transducer.load_state_dict(model_state, strict=False)
    transducer = transducer.to(device).eval()
    ctc_head = ctc_head.to(device).eval()

    logger.info(f"Loaded model from {checkpoint_path}")

    # Load test data
    manifest_dir = Path(test_manifest_dir)
    supervisions = []
    with open(manifest_dir / "supervisions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            supervisions.append(json.loads(line))

    recordings = {}
    with open(manifest_dir / "recordings.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            recordings[rec["id"]] = rec

    # Evaluate
    results_by_severity = defaultdict(lambda: {"total_edit": 0, "total_ref": 0, "count": 0})
    all_results = []

    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=512,
        win_length=400, hop_length=160,
        n_mels=80,
    ).to(device)

    with torch.no_grad():
        for idx, sup in enumerate(supervisions):
            rec = recordings[sup["recording_id"]]
            audio_path = rec["sources"][0]["source"]

            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            waveform = waveform.to(device)
            features = fbank(waveform).squeeze(0).transpose(0, 1)  # (T, 80)
            features = torch.log(features.clamp(min=1e-10))
            features = features.unsqueeze(0)  # (1, T, 80)
            feature_lens = torch.tensor([features.size(1)], device=device)

            # Greedy decoding via CTC
            encoder_out, encoder_out_lens = transducer.encoder(
                features, feature_lens, streaming=streaming
            )
            log_probs = ctc_head(encoder_out)  # (1, T, V)
            predicted_ids = log_probs.argmax(dim=-1).squeeze(0)  # (T,)

            # CTC decode (collapse repeated, remove blanks)
            decoded_ids = []
            prev_id = -1
            for token_id in predicted_ids.tolist():
                if token_id != 0 and token_id != prev_id:
                    decoded_ids.append(token_id)
                prev_id = token_id

            # Convert to text (placeholder: chr mapping)
            hypothesis = "".join(chr(t) if t < 128 else chr(0x4E00 + t % 0x5200) for t in decoded_ids)

            reference = sup["text"]
            severity = sup.get("custom", {}).get("severity", "unknown")

            cer_result = compute_cer(hypothesis, reference)

            results_by_severity[severity]["total_edit"] += cer_result["edit_distance"]
            results_by_severity[severity]["total_ref"] += cer_result["ref_len"]
            results_by_severity[severity]["count"] += 1

            results_by_severity["overall"]["total_edit"] += cer_result["edit_distance"]
            results_by_severity["overall"]["total_ref"] += cer_result["ref_len"]
            results_by_severity["overall"]["count"] += 1

            all_results.append({
                "utt_id": sup["id"],
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": cer_result["cer"],
                "severity": severity,
            })

            if (idx + 1) % 100 == 0:
                overall = results_by_severity["overall"]
                running_cer = overall["total_edit"] / max(overall["total_ref"], 1)
                logger.info(f"Progress: {idx + 1}/{len(supervisions)}, Running CER: {running_cer:.4f}")

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    for severity in ["mild", "moderate", "severe", "unknown", "overall"]:
        if severity not in results_by_severity:
            continue
        stats = results_by_severity[severity]
        cer = stats["total_edit"] / max(stats["total_ref"], 1) * 100
        logger.info(f"  {severity:>10}: CER = {cer:.2f}% ({stats['count']} utterances)")

    # Save detailed results
    output_path = Path(checkpoint_path).parent / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                k: {"cer": v["total_edit"] / max(v["total_ref"], 1), "count": v["count"]}
                for k, v in results_by_severity.items()
            },
            "details": all_results[:100],  # Save first 100 for inspection
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Detailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Real-ASR model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-streaming", action="store_true", help="Use offline mode")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        test_manifest_dir=args.test_manifest,
        config_path=args.config,
        device=args.device,
        streaming=not args.no_streaming,
    )


if __name__ == "__main__":
    main()
