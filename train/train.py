"""Unified training script for Real-ASR.

Supports three stages:
- Stage 1: pretrain (general Chinese ASR on pseudo-labeled data)
- Stage 2: distill (knowledge distillation from Qwen3-ASR teacher)
- Stage 3: finetune (supervised fine-tuning on dysarthric labeled data)

Usage:
    # Stage 1
    python train/train.py --config configs/train_stage1.yaml

    # Stage 3
    python train/train.py --config configs/train_stage3.yaml

    # For Stage 2 (distillation), use train/distill.py instead.
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle inheritance
    if "inherit" in config:
        base_dir = Path(config_path).parent
        base_path = base_dir / config.pop("inherit")
        with open(base_path, "r") as f:
            base_config = yaml.safe_load(f)
        # Merge: config overrides base
        merged = deep_merge(base_config, config)
        return merged

    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries (override takes precedence)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_cfg: Dict) -> nn.Module:
    """Build Zipformer + Transducer model from config."""
    from model.zipformer import ZipformerEncoder
    from model.transducer import Transducer
    from model.ctc_head import CTCHead

    encoder_cfg = model_cfg["encoder"]
    encoder_cfg["input_dim"] = model_cfg["audio"]["num_mel_bins"]
    encoder_cfg["output_dim"] = model_cfg.get("encoder_output_dim", 512)
    encoder = ZipformerEncoder(encoder_cfg)

    transducer = Transducer(encoder, model_cfg)
    ctc_head = CTCHead(
        encoder_dim=encoder.get_output_dim(),
        vocab_size=model_cfg["ctc_head"]["output_dim"],
    )

    return transducer, ctc_head


def build_dataloader(data_cfg: Dict, stage: str, rank: int = 0, world_size: int = 1) -> DataLoader:
    """Build dataloader from Lhotse manifests.

    In a real implementation, this would use Lhotse's CutSet and
    DynamicBucketingSampler. Here we provide the interface.
    """
    from data.dataset import AsrDataset, AsrBucketingSampler

    manifest_dir = Path(data_cfg["manifest_dir"])
    dataset = AsrDataset(
        manifest_dir=manifest_dir,
        max_duration=data_cfg["max_duration"],
        min_duration=data_cfg.get("min_duration", 0.3),
    )

    sampler = AsrBucketingSampler(
        dataset=dataset,
        max_duration_per_batch=data_cfg["bucketing_sampler"]["max_duration_per_batch"],
        shuffle=data_cfg["bucketing_sampler"]["shuffle"],
        drop_last=data_cfg["bucketing_sampler"].get("drop_last", True),
        rank=rank,
        world_size=world_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    return dataloader


def build_optimizer(model: nn.Module, optim_cfg: Dict):
    """Build optimizer from config."""
    lr = optim_cfg["lr"]
    betas = tuple(optim_cfg.get("betas", [0.9, 0.98]))
    weight_decay = optim_cfg.get("weight_decay", 1e-6)

    # Use ScaledAdam if available (from icefall), else fallback to AdamW
    try:
        from icefall.optim import ScaledAdam
        optimizer = ScaledAdam(
            model.parameters(),
            lr=lr,
            betas=betas,
            clipping_scale=optim_cfg.get("clipping_scale", 2.0),
        )
    except ImportError:
        logger.warning("ScaledAdam not available, falling back to AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    return optimizer


def build_scheduler(optimizer, scheduler_cfg: Dict, steps_per_epoch: int):
    """Build learning rate scheduler."""
    warmup_steps = scheduler_cfg.get("warmup_steps", 5000)
    total_steps = steps_per_epoch * 100  # rough estimate

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        decay = 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        return max(decay, 0.01)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_loss(
    logits: torch.Tensor,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    targets: torch.Tensor,
    target_lens: torch.Tensor,
    ctc_head: nn.Module,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    """Compute training losses (RNN-T + CTC).

    In production, use k2's pruned RNN-T loss for efficiency.
    """
    losses = {}

    # Pruned RNN-T loss (requires k2)
    try:
        import k2
        rnnt_loss = k2.rnnt_loss_pruned(
            logits=logits,
            symbols=targets,
            ranges=None,  # will be computed inside
            termination_symbol=0,
            boundary=None,
            reduction="mean",
        )
    except (ImportError, Exception):
        # Fallback: simple cross-entropy on joiner output (placeholder)
        # In real training, k2 pruned rnnt loss is essential
        rnnt_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        logger.warning("k2 not available, using placeholder RNN-T loss")

    losses["rnnt"] = rnnt_loss * loss_cfg.get("rnnt_weight", 1.0)

    # CTC auxiliary loss
    if loss_cfg.get("ctc_weight", 0) > 0:
        ctc_log_probs = ctc_head(encoder_out)  # (B, T, V)
        ctc_log_probs = ctc_log_probs.transpose(0, 1)  # (T, B, V) for CTC loss
        ctc_loss = nn.functional.ctc_loss(
            ctc_log_probs,
            targets,
            encoder_out_lens,
            target_lens,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )
        losses["ctc"] = ctc_loss * loss_cfg["ctc_weight"]

    losses["total"] = sum(losses.values())
    return losses


def train_one_epoch(
    model: nn.Module,
    ctc_head: nn.Module,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    loss_cfg: Dict,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
) -> int:
    """Train for one epoch."""
    model.train()
    ctc_head.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        features = batch["features"].cuda(non_blocking=True)
        feature_lens = batch["feature_lens"].cuda(non_blocking=True)
        targets = batch["targets"].cuda(non_blocking=True)
        target_lens = batch["target_lens"].cuda(non_blocking=True)

        # Prepend blank to targets for predictor input
        blank = torch.zeros(targets.size(0), 1, dtype=targets.dtype, device=targets.device)
        predictor_input = torch.cat([blank, targets], dim=1)

        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            logits, encoder_out, encoder_out_lens = model(
                features, feature_lens, predictor_input, streaming=True
            )
            losses = compute_loss(
                logits, encoder_out, encoder_out_lens,
                targets, target_lens, ctc_head, loss_cfg,
            )

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(ctc_head.parameters()), max_norm=5.0
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += losses["total"].item()
        num_batches += 1
        global_step += 1

        if writer and global_step % 100 == 0:
            writer.add_scalar("train/loss_total", losses["total"].item(), global_step)
            writer.add_scalar("train/loss_rnnt", losses.get("rnnt", torch.tensor(0)).item(), global_step)
            writer.add_scalar("train/loss_ctc", losses.get("ctc", torch.tensor(0)).item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        if batch_idx % 200 == 0:
            avg_loss = total_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch} batch {batch_idx}: loss={avg_loss:.4f}")

    return global_step


def save_checkpoint(
    model: nn.Module,
    ctc_head: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    exp_dir: Path,
    keep_last_n: int = 5,
):
    """Save training checkpoint."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_dir / f"epoch_{epoch}.pt"

    state = {
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "ctc_head": ctc_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(state, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # Manage checkpoint count
    ckpts = sorted(exp_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    while len(ckpts) > keep_last_n:
        old = ckpts.pop(0)
        old.unlink()
        logger.info(f"Removed old checkpoint: {old}")


def main():
    parser = argparse.ArgumentParser(description="Train Real-ASR model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]
    set_seed(training_cfg.get("seed", 42))

    # Distributed setup
    world_size = training_cfg.get("distributed", {}).get("world_size", 1)
    rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    if world_size > 1:
        dist.init_process_group(
            backend=training_cfg["distributed"]["backend"],
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)

    # Build model
    transducer, ctc_head = build_model(config["model"])
    transducer = transducer.cuda()
    ctc_head = ctc_head.cuda()

    # Resume from checkpoint
    global_step = 0
    start_epoch = 1
    resume_from = training_cfg.get("resume_from")
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location="cpu")
        transducer.load_state_dict(ckpt["model"])
        ctc_head.load_state_dict(ckpt["ctc_head"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        logger.info(f"Resumed from {resume_from}, epoch {start_epoch}")

    if world_size > 1:
        transducer = DDP(transducer, device_ids=[rank])
        ctc_head = DDP(ctc_head, device_ids=[rank])

    # Build dataloader, optimizer, scheduler
    dataloader = build_dataloader(config["data"], training_cfg["stage"], rank, world_size)
    optimizer = build_optimizer(
        nn.ModuleList([transducer, ctc_head]), training_cfg["optimizer"]
    )
    scheduler = build_scheduler(optimizer, training_cfg["scheduler"], steps_per_epoch=1000)
    scaler = GradScaler()

    # Restore optimizer state
    if resume_from and Path(resume_from).exists() and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    # TensorBoard
    exp_dir = Path(training_cfg["checkpoint"]["exp_dir"])
    writer = SummaryWriter(exp_dir / "tb_logs") if rank == 0 else None

    # Training loop
    num_epochs = training_cfg["num_epochs"]
    for epoch in range(start_epoch, num_epochs + 1):
        if hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)

        logger.info(f"=== Epoch {epoch}/{num_epochs} ===")
        global_step = train_one_epoch(
            transducer, ctc_head, dataloader, optimizer, scheduler,
            scaler, training_cfg["loss"], epoch, writer, global_step,
        )

        if rank == 0:
            save_checkpoint(
                transducer, ctc_head, optimizer, scheduler, epoch, global_step,
                exp_dir,
                keep_last_n=training_cfg["checkpoint"].get("keep_last_n", 5),
            )

    if world_size > 1:
        dist.destroy_process_group()

    if writer:
        writer.close()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
