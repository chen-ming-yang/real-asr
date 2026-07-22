"""Export Zipformer + Transducer model to ONNX for deployment.

Exports three ONNX models:
1. encoder.onnx - Zipformer encoder (processes audio features)
2. predictor.onnx - Stateless predictor (processes text history)
3. joiner.onnx - Joiner (combines encoder + predictor outputs)

These can be loaded by sherpa-onnx for streaming inference.

Usage:
    python export/export_onnx.py \
        --checkpoint exp/stage3_finetune/best.pt \
        --config configs/model.yaml \
        --output-dir exported_models
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def export_encoder(encoder: nn.Module, output_path: Path, num_mel_bins: int = 80):
    """Export encoder to ONNX."""
    encoder.eval()

    # Example input
    batch_size = 1
    seq_len = 200  # ~2 seconds of audio at 10ms shift
    x = torch.randn(batch_size, seq_len, num_mel_bins)
    x_lens = torch.tensor([seq_len], dtype=torch.long)

    # Trace
    torch.onnx.export(
        encoder,
        (x, x_lens, True),  # streaming=True
        str(output_path),
        input_names=["features", "feature_lens"],
        output_names=["encoder_out", "encoder_out_lens"],
        dynamic_axes={
            "features": {0: "batch", 1: "time"},
            "feature_lens": {0: "batch"},
            "encoder_out": {0: "batch", 1: "time"},
            "encoder_out_lens": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info(f"Exported encoder to {output_path}")


def export_predictor(predictor: nn.Module, output_path: Path, vocab_size: int):
    """Export predictor to ONNX."""
    predictor.eval()

    # Example input: token history
    batch_size = 1
    context_size = predictor.context_size if hasattr(predictor, "context_size") else 2
    y = torch.randint(0, vocab_size, (batch_size, context_size))

    torch.onnx.export(
        predictor,
        (y,),
        str(output_path),
        input_names=["tokens"],
        output_names=["predictor_out"],
        dynamic_axes={
            "tokens": {0: "batch", 1: "seq_len"},
            "predictor_out": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info(f"Exported predictor to {output_path}")


def export_joiner(joiner: nn.Module, output_path: Path, encoder_dim: int, predictor_dim: int):
    """Export joiner to ONNX."""
    joiner.eval()

    # Example inputs (single frame)
    encoder_out = torch.randn(1, 1, 1, encoder_dim)
    predictor_out = torch.randn(1, 1, 1, predictor_dim)

    torch.onnx.export(
        joiner,
        (encoder_out, predictor_out),
        str(output_path),
        input_names=["encoder_out", "predictor_out"],
        output_names=["logits"],
        dynamic_axes={
            "encoder_out": {0: "batch"},
            "predictor_out": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info(f"Exported joiner to {output_path}")


def verify_onnx(onnx_path: Path):
    """Verify ONNX model is valid."""
    import onnx
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    logger.info(f"ONNX model verified: {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Real-ASR model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    parser.add_argument("--output-dir", type=str, default="exported_models")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX models after export")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    from train.train import build_model
    transducer, ctc_head = build_model(config["model"])

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_state = ckpt.get("model", ckpt)

    # Handle DDP state dict
    if any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}

    # Handle distill wrapper state dict
    if any(k.startswith("student.") for k in model_state.keys()):
        model_state = {k.replace("student.", ""): v for k, v in model_state.items() if k.startswith("student.")}

    transducer.load_state_dict(model_state, strict=False)
    transducer.eval()

    logger.info(f"Loaded checkpoint from {args.checkpoint}")
    total_params = sum(p.numel() for p in transducer.parameters())
    logger.info(f"Total parameters: {total_params / 1e6:.1f}M")

    # Export each component
    num_mel_bins = config["model"]["audio"]["num_mel_bins"]
    encoder_dim = config["model"].get("encoder_output_dim", 512)
    predictor_dim = config["model"]["predictor"]["embedding_dim"]
    vocab_size = config["model"]["joiner"]["output_dim"]

    export_encoder(transducer.encoder, output_dir / "encoder.onnx", num_mel_bins)
    export_predictor(transducer.predictor, output_dir / "predictor.onnx", vocab_size)
    export_joiner(transducer.joiner, output_dir / "joiner.onnx", encoder_dim, predictor_dim)

    # Save tokenizer config alongside models
    tokenizer_cfg = config["model"].get("tokenizer", {})
    with open(output_dir / "tokenizer_config.json", "w") as f:
        import json
        json.dump(tokenizer_cfg, f, indent=2)

    if args.verify:
        for onnx_file in output_dir.glob("*.onnx"):
            verify_onnx(onnx_file)

    logger.info(f"Export complete. Models saved to {output_dir}/")
    logger.info("To deploy with sherpa-onnx:")
    logger.info(f"  sherpa-onnx-offline --encoder {output_dir}/encoder.onnx "
                f"--decoder {output_dir}/predictor.onnx --joiner {output_dir}/joiner.onnx "
                f"--tokens {output_dir}/tokens.txt audio.wav")


if __name__ == "__main__":
    main()
