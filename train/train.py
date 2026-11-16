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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.cuda.amp import GradScaler  # kept for ckpt compat
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


def build_model(config: Dict) -> nn.Module:
    """Build Zipformer + Transducer model from config."""
    from model.transducer import Transducer
    from model.ctc_head import CTCHead

    model_cfg = config["model"]
    encoder_cfg = model_cfg["encoder"]
    encoder_cfg["input_dim"] = config["audio"]["num_mel_bins"]
    encoder_cfg["output_dim"] = model_cfg.get("encoder_output_dim", 512)

    encoder_type = encoder_cfg.get("type", "zipformer")
    if encoder_type == "icefall_zipformer2":
        from model.icefall_encoder_wrapper import IcefallZipformerEncoder
        encoder = IcefallZipformerEncoder(encoder_cfg)
    else:
        from model.zipformer import ZipformerEncoder
        encoder = ZipformerEncoder(encoder_cfg)

    transducer = Transducer(encoder, model_cfg)
    ctc_head = CTCHead(
        encoder_dim=encoder.get_output_dim(),
        vocab_size=model_cfg["ctc_head"]["output_dim"],
    )

    return transducer, ctc_head


def build_dataloader(data_cfg: Dict, stage: str, rank: int = 0, world_size: int = 1) -> DataLoader:
    """Build dataloader — uses precomputed fbank shards if shard_dir is set,
    otherwise falls back to on-the-fly extraction from Lhotse manifests.
    """
    shard_dir = data_cfg.get("shard_dir")

    if shard_dir:
        from data.dataset import PrecomputedShardDataset, ShardBucketingSampler
        dataset = PrecomputedShardDataset(shard_dir=shard_dir)
        sampler = ShardBucketingSampler(
            dataset=dataset,
            max_duration_per_batch=data_cfg["bucketing_sampler"]["max_duration_per_batch"],
            shuffle=data_cfg["bucketing_sampler"]["shuffle"],
            drop_last=data_cfg["bucketing_sampler"].get("drop_last", True),
            rank=rank,
            world_size=world_size,
        )
    else:
        from data.dataset import AsrDataset, AsrBucketingSampler
        manifest_dir = Path(data_cfg["manifest_dir"])
        dataset = AsrDataset(
            manifest_dir=manifest_dir,
            tokenizer_path=data_cfg.get("tokenizer_path"),
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

    num_workers = data_cfg.get("num_workers", 4)
    dl_kwargs = dict(
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4
    dataloader = DataLoader(dataset, **dl_kwargs)

    return dataloader


def build_optimizer(model: nn.Module, optim_cfg: Dict):
    """Build optimizer from config. Only includes parameters with requires_grad=True."""
    lr = optim_cfg["lr"]
    betas = tuple(optim_cfg.get("betas", [0.9, 0.98]))
    weight_decay = optim_cfg.get("weight_decay", 1e-6)
    optim_type = optim_cfg.get("type", "AdamW")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Optimizer: {len(trainable_params)} param groups, "
                f"{sum(p.numel() for p in trainable_params)/1e6:.1f}M params")

    if optim_type == "ScaledAdam":
        try:
            from icefall.optim import ScaledAdam
            optimizer = ScaledAdam(
                trainable_params,
                lr=lr,
                betas=betas,
                clipping_scale=optim_cfg.get("clipping_scale", 2.0),
            )
            return optimizer
        except ImportError:
            logger.warning("ScaledAdam not available, falling back to AdamW")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    return optimizer


def build_scheduler(optimizer, scheduler_cfg: Dict, steps_per_epoch: int):
    """Build LR scheduler with linear warmup + cosine decay."""
    warmup_steps = scheduler_cfg.get("warmup_steps", 5000)
    num_epochs = scheduler_cfg.get("num_epochs", 50)
    min_lr_ratio = scheduler_cfg.get("min_lr_ratio", 0.01)
    total_steps = steps_per_epoch * num_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return max(cosine_decay, min_lr_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_loss(
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    predictor_out: torch.Tensor,
    targets: torch.Tensor,
    target_lens: torch.Tensor,
    joiner: nn.Module,
    ctc_head: nn.Module,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    """Compute training losses using k2 pruned RNN-T + CTC.

    Uses the efficient k2 pruned loss workflow:
    1. simple_loss with return_grad to get prune ranges
    2. do_rnnt_pruning to prune am/lm
    3. joiner.forward_pruned to get pruned logits
    4. rnnt_loss_pruned for final loss
    """
    losses = {}
    device = encoder_out.device
    B = encoder_out.size(0)
    T = encoder_out_lens  # (B,)
    S = target_lens  # (B,)
    s_range = loss_cfg.get("prune_range", 5)

    boundary = torch.zeros(B, 4, dtype=torch.long, device=device)
    boundary[:, 2] = S
    boundary[:, 3] = T

    try:
        import k2
        am_proj = joiner.simple_am_proj(encoder_out)
        lm_proj = joiner.simple_lm_proj(predictor_out)

        simple_loss, (px_grad, py_grad) = k2.rnnt_loss_simple(
            lm=lm_proj.float(),
            am=am_proj.float(),
            symbols=targets.long(),
            termination_symbol=0,
            boundary=boundary,
            reduction="mean",
            return_grad=True,
        )

        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=s_range,
        )

        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=encoder_out, lm=predictor_out, ranges=ranges
        )

        pruned_logits = joiner.forward_pruned(am_pruned, lm_pruned)

        pruned_loss = k2.rnnt_loss_pruned(
            logits=pruned_logits.float(),
            symbols=targets.long(),
            ranges=ranges,
            termination_symbol=0,
            boundary=boundary,
            reduction="mean",
        )

        losses["rnnt"] = (
            simple_loss * loss_cfg.get("simple_loss_scale", 0.5)
            + pruned_loss
        ) * loss_cfg.get("rnnt_weight", 1.0)

    except Exception as e:
        import traceback
        logger.warning(
            f"k2 pruned RNN-T loss failed: {e}\n{traceback.format_exc()}"
            "Falling back to CTC only."
        )
        losses["rnnt"] = torch.tensor(
            0.0, device=device, requires_grad=True
        )

    # CTC auxiliary loss
    if loss_cfg.get("ctc_weight", 0) > 0:
        ctc_log_probs = ctc_head(encoder_out)
        ctc_log_probs = ctc_log_probs.transpose(0, 1)
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
    loss_cfg: Dict,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
    grad_accum_steps: int = 1,
) -> int:
    """Train for one epoch."""
    model.train()
    ctc_head.train()

    total_loss = 0.0
    num_batches = 0
    nan_count = 0
    total_batches = len(dataloader)
    import time
    from collections import deque
    epoch_start = time.time()
    recent_losses = deque(maxlen=100)
    last_good_state = None
    save_state_every = 500

    data_time_total = 0.0
    compute_time_total = 0.0
    t_batch_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        data_time = time.time() - t_batch_start
        data_time_total += data_time

        # Update global batch count for ScheduledFloat in icefall scaling
        try:
            from model.scaling import set_batch_count
            set_batch_count(global_step + batch_idx)
        except ImportError:
            pass

        t_compute_start = time.time()
        features = batch["features"].cuda(non_blocking=True)
        feature_lens = batch["feature_lens"].cuda(non_blocking=True)
        targets = batch["targets"].cuda(non_blocking=True)
        target_lens = batch["target_lens"].cuda(non_blocking=True)

        # Prepend blank to targets for predictor input
        blank = torch.zeros(targets.size(0), 1, dtype=targets.dtype, device=targets.device)
        predictor_input = torch.cat([blank, targets], dim=1)

        # Only zero grad at accumulation boundary
        if batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.float32):
            encoder_out, encoder_out_lens, predictor_out = model(
                features, feature_lens, predictor_input, streaming=True
            )
            losses = compute_loss(
                encoder_out, encoder_out_lens, predictor_out,
                targets, target_lens, model.joiner, ctc_head, loss_cfg,
            )

        loss_val = losses["total"] / grad_accum_steps

        # NaN/Inf detection with parameter rollback
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            nan_count += 1
            if nan_count <= 5:
                logger.warning(f"Epoch {epoch} batch {batch_idx}: NaN/Inf loss, skipping (#{nan_count})")
            if nan_count >= 10 and last_good_state is not None:
                logger.warning(f"Epoch {epoch}: {nan_count} consecutive NaN, rolling back parameters")
                model.load_state_dict(last_good_state["model"])
                ctc_head.load_state_dict(last_good_state["ctc_head"])
                optimizer.load_state_dict(last_good_state["optimizer"])
                nan_count = 0
            optimizer.zero_grad()
            continue

        nan_count = 0
        loss_val.backward()

        # Only step optimizer at accumulation boundary
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(ctc_head.parameters()), max_norm=1.0
            )

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.warning(f"Epoch {epoch} batch {batch_idx}: NaN grad_norm, skipping update")
                optimizer.zero_grad()
                continue

            optimizer.step()
            scheduler.step()

        cur_loss = loss_val.item() * grad_accum_steps
        compute_time = time.time() - t_compute_start
        compute_time_total += compute_time
        total_loss += cur_loss
        recent_losses.append(cur_loss)
        num_batches += 1
        global_step += 1

        # Save good state periodically for rollback
        if num_batches % save_state_every == 0:
            last_good_state = {
                "model": {k: v.clone() for k, v in model.state_dict().items()},
                "ctc_head": {k: v.clone() for k, v in ctc_head.state_dict().items()},
                "optimizer": optimizer.state_dict(),
            }

        if writer and global_step % 100 == 0:
            writer.add_scalar("train/loss_total", cur_loss, global_step)
            writer.add_scalar("train/loss_rnnt", losses.get("rnnt", torch.tensor(0)).item(), global_step)
            writer.add_scalar("train/loss_ctc", losses.get("ctc", torch.tensor(0)).item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("train/grad_norm", grad_norm, global_step)

        if batch_idx % 10 == 0:
            smooth_loss = sum(recent_losses) / len(recent_losses) if recent_losses else cur_loss
            elapsed = time.time() - epoch_start
            speed = num_batches / elapsed if elapsed > 0 else 0
            avg_data_t = data_time_total / max(num_batches, 1)
            avg_compute_t = compute_time_total / max(num_batches, 1)
            batch_size = features.size(0)
            feat_frames = feature_lens.sum().item()
            if total_batches > 0 and speed > 0:
                eta = (total_batches - num_batches) / speed
                eta_min, eta_sec = int(eta // 60), int(eta % 60)
                pct = 100.0 * num_batches / total_batches
                logger.info(
                    f"Epoch {epoch} [{num_batches}/{total_batches} ({pct:.1f}%)] "
                    f"loss={smooth_loss:.4f} | {speed:.1f} batch/s | ETA {eta_min}m{eta_sec:02d}s"
                    f" | data={avg_data_t:.2f}s compute={avg_compute_t:.2f}s"
                    f" | bs={batch_size} frames={feat_frames:.0f}"
                )
            else:
                logger.info(f"Epoch {epoch} batch {batch_idx}: loss={smooth_loss:.4f}")

        t_batch_start = time.time()

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


def apply_overrides(config: Dict, overrides: list) -> Dict:
    """Apply dot-notation overrides to config dict.

    Example: 'data.num_workers=4' sets config["data"]["num_workers"] = 4
    """
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        # Auto-convert types
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
        d[keys[-1]] = value
    return config


def main():
    parser = argparse.ArgumentParser(description="Train Real-ASR model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Override config values, e.g. data.num_workers=4")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)
    if args.resume:
        config.setdefault("training", {})["resume_from"] = args.resume
    training_cfg = config["training"]
    set_seed(training_cfg.get("seed", 42))

    # Distributed setup
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(
            backend=training_cfg.get("distributed", {}).get("backend", "nccl"),
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)

    # Build model
    transducer, ctc_head = build_model(config)
    transducer = transducer.cuda()
    ctc_head = ctc_head.cuda()

    # Resume from checkpoint
    global_step = 0
    start_epoch = 1
    resume_from = training_cfg.get("resume_from")
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        transducer.load_state_dict(ckpt["model"])
        ctc_head.load_state_dict(ckpt["ctc_head"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        logger.info(f"Resumed from {resume_from}, epoch {start_epoch}")

    # Load encoder weights from a separate checkpoint (for RNN-T retraining)
    resume_encoder_from = training_cfg.get("resume_encoder_from")
    if resume_encoder_from and Path(resume_encoder_from).exists():
        ckpt = torch.load(resume_encoder_from, map_location="cpu", weights_only=False)
        encoder_state = {}
        for k, v in ckpt["model"].items():
            if k.startswith("encoder."):
                encoder_state[k] = v
        missing, unexpected = transducer.load_state_dict(encoder_state, strict=False)
        logger.info(
            f"Loaded encoder from {resume_encoder_from}: "
            f"{len(encoder_state)} keys loaded, "
            f"{len(missing)} missing (predictor/joiner), "
            f"{len(unexpected)} unexpected"
        )

    # Freeze encoder if configured
    freeze_encoder = training_cfg.get("freeze_encoder", False)
    if freeze_encoder:
        frozen_count = 0
        for name, param in transducer.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = False
                frozen_count += 1
        trainable = sum(p.numel() for p in transducer.parameters() if p.requires_grad)
        trainable += sum(p.numel() for p in ctc_head.parameters() if p.requires_grad)
        logger.info(
            f"Encoder frozen: {frozen_count} params frozen, "
            f"{trainable/1e6:.1f}M trainable params remaining"
        )

    if world_size > 1:
        find_unused = freeze_encoder  # frozen encoder params are "unused" in DDP
        transducer = DDP(transducer, device_ids=[rank], find_unused_parameters=find_unused)
        ctc_head = DDP(ctc_head, device_ids=[rank])

    # Build dataloader, optimizer, scheduler
    dataloader = build_dataloader(config["data"], training_cfg["stage"], rank, world_size)
    steps_per_epoch = len(dataloader)
    logger.info(f"Steps per epoch (estimated): {steps_per_epoch}")

    optimizer = build_optimizer(
        nn.ModuleList([transducer, ctc_head]), training_cfg["optimizer"]
    )
    scheduler_cfg = training_cfg["scheduler"].copy()
    scheduler_cfg["num_epochs"] = training_cfg.get("num_epochs", 50)
    scheduler = build_scheduler(optimizer, scheduler_cfg, steps_per_epoch=steps_per_epoch)

    # Restore optimizer state (skip to use fresh optimizer with fixed LR schedule)
    # if resume_from and Path(resume_from).exists() and "optimizer" in ckpt:
    #     optimizer.load_state_dict(ckpt["optimizer"])
    #     scheduler.load_state_dict(ckpt["scheduler"])

    # TensorBoard
    exp_dir = Path(training_cfg["checkpoint"]["exp_dir"])
    writer = SummaryWriter(exp_dir / "tb_logs") if rank == 0 else None

    # Training loop
    num_epochs = training_cfg["num_epochs"]
    for epoch in range(start_epoch, num_epochs + 1):
        if hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)

        logger.info(f"=== Epoch {epoch}/{num_epochs} ===")
        grad_accum = training_cfg.get("optimizer", {}).get("grad_accum_steps", 1)
        global_step = train_one_epoch(
            transducer, ctc_head, dataloader, optimizer, scheduler,
            training_cfg["loss"], epoch, writer, global_step, grad_accum,
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
