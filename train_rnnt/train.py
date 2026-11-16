"""Training script for RNN-T (from scratch, joint CTC + RNN-T).

Architecture: Icefall Zipformer2 encoder + Stateless Predictor + Additive Joiner
Loss: k2 pruned RNN-T + auxiliary CTC
Strategy: Full joint training (encoder NOT frozen) to prevent blank dominance.
"""

import argparse
import logging
import os
import sys
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_encoder(encoder_cfg: dict):
    """Build Zipformer2 encoder from config."""
    from model.icefall_encoder_wrapper import IcefallZipformerEncoder
    return IcefallZipformerEncoder(encoder_cfg)


def build_model(config: Dict):
    """Build RNNTModel from config."""
    from train_rnnt.model import RNNTModel

    model_cfg = config["model"]
    encoder_cfg = model_cfg["encoder"].copy()
    encoder_cfg["input_dim"] = config["audio"]["num_mel_bins"]
    encoder_cfg["output_dim"] = model_cfg["encoder_output_dim"]

    encoder = build_encoder(encoder_cfg)
    encoder_dim = encoder.get_output_dim()
    vocab_size = model_cfg["vocab_size"]

    rnnt_cfg = {
        "predictor_dim": model_cfg.get("predictor_dim", 512),
        "joiner_dim": model_cfg.get("joiner_dim", 512),
        "context_size": model_cfg.get("context_size", 2),
    }
    model = RNNTModel(encoder, encoder_dim, vocab_size, rnnt_cfg)
    return model


def build_dataloader(data_cfg: Dict, rank: int = 0, world_size: int = 1) -> DataLoader:
    """Build dataloader - supports both shard mode and online manifest mode."""
    shard_dir = data_cfg.get("shard_dir")

    if shard_dir:
        from data.dataset import PrecomputedShardDataset
        from train_rnnt.shard_sampler import ShardLocalSampler

        dataset = PrecomputedShardDataset(shard_dir=shard_dir)
        sampler = ShardLocalSampler(
            dataset=dataset,
            max_duration_per_batch=data_cfg["max_duration_per_batch"],
            shuffle=True,
            drop_last=True,
            rank=rank,
            world_size=world_size,
        )
    else:
        from data.dataset import AsrDataset, AsrBucketingSampler
        dataset = AsrDataset(
            manifest_dir=Path(data_cfg["manifest_dir"]),
            tokenizer_path=data_cfg.get("tokenizer_path"),
            max_duration=data_cfg.get("max_duration", 20.0),
            min_duration=data_cfg.get("min_duration", 0.5),
        )
        sampler = AsrBucketingSampler(
            dataset=dataset,
            max_duration_per_batch=data_cfg["max_duration_per_batch"],
            shuffle=True,
            drop_last=True,
            rank=rank,
            world_size=world_size,
        )

    num_workers = data_cfg.get("num_workers", 0)
    dl_kwargs = dict(
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **dl_kwargs)


def compute_rnnt_loss(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    predictor_out: torch.Tensor,
    targets: torch.Tensor,
    target_lens: torch.Tensor,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    """Compute k2 pruned RNN-T loss + CTC auxiliary loss."""
    import k2

    device = encoder_out.device
    B = encoder_out.size(0)
    T = encoder_out_lens
    S = target_lens
    s_range = loss_cfg.get("prune_range", 5)

    boundary = torch.zeros(B, 4, dtype=torch.long, device=device)
    boundary[:, 2] = S
    boundary[:, 3] = T

    losses = {}

    # Simple loss for pruning ranges
    am_proj = model.joiner.simple_am_proj(encoder_out)
    lm_proj = model.joiner.simple_lm_proj(predictor_out)

    simple_loss, (px_grad, py_grad) = k2.rnnt_loss_simple(
        lm=lm_proj.float(),
        am=am_proj.float(),
        symbols=targets.long(),
        termination_symbol=0,
        boundary=boundary,
        reduction="mean",
        return_grad=True,
    )

    # Prune and compute full loss
    ranges = k2.get_rnnt_prune_ranges(
        px_grad=px_grad,
        py_grad=py_grad,
        boundary=boundary,
        s_range=s_range,
    )
    am_pruned, lm_pruned = k2.do_rnnt_pruning(
        am=encoder_out, lm=predictor_out, ranges=ranges
    )
    pruned_logits = model.joiner.forward_pruned(am_pruned, lm_pruned)

    pruned_loss = k2.rnnt_loss_pruned(
        logits=pruned_logits.float(),
        symbols=targets.long(),
        ranges=ranges,
        termination_symbol=0,
        boundary=boundary,
        reduction="mean",
    )

    simple_scale = loss_cfg.get("simple_loss_scale", 0.5)
    rnnt_weight = loss_cfg.get("rnnt_weight", 1.0)
    losses["rnnt"] = (simple_loss * simple_scale + pruned_loss) * rnnt_weight

    # CTC auxiliary
    ctc_weight = loss_cfg.get("ctc_weight", 0.3)
    if ctc_weight > 0:
        ctc_log_probs = model.get_ctc_log_probs(encoder_out)
        ctc_log_probs_t = ctc_log_probs.transpose(0, 1)
        ctc_loss = nn.functional.ctc_loss(
            ctc_log_probs_t, targets, encoder_out_lens, target_lens,
            blank=0, reduction="mean", zero_infinity=True,
        )
        losses["ctc"] = ctc_loss * ctc_weight

    losses["total"] = sum(losses.values())
    return losses


def train_one_epoch(
    model, dataloader, optimizer, scheduler,
    loss_cfg, epoch, writer, global_step, grad_accum_steps=1,
):
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(dataloader)
    recent_losses = deque(maxlen=100)
    epoch_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        features = batch["features"].cuda(non_blocking=True)
        feature_lens = batch["feature_lens"].cuda(non_blocking=True)
        targets = batch["targets"].cuda(non_blocking=True)
        target_lens = batch["target_lens"].cuda(non_blocking=True)

        blank = torch.zeros(
            targets.size(0), 1, dtype=targets.dtype, device=targets.device
        )
        predictor_input = torch.cat([blank, targets], dim=1)

        if batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad()

        encoder_out, encoder_out_lens, predictor_out = model(
            features, feature_lens, predictor_input, streaming=True
        )
        losses = compute_rnnt_loss(
            model.module if isinstance(model, DDP) else model,
            encoder_out, encoder_out_lens, predictor_out,
            targets, target_lens, loss_cfg,
        )
        loss_val = losses["total"] / grad_accum_steps

        if torch.isnan(loss_val) or torch.isinf(loss_val):
            logger.warning(f"Epoch {epoch} batch {batch_idx}: NaN/Inf loss")
            optimizer.zero_grad()
            continue

        loss_val.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        cur_loss = loss_val.item() * grad_accum_steps
        total_loss += cur_loss
        recent_losses.append(cur_loss)
        num_batches += 1
        global_step += 1

        if writer and global_step % 100 == 0:
            writer.add_scalar("train/loss", cur_loss, global_step)
            writer.add_scalar("train/rnnt", losses["rnnt"].item(), global_step)
            ctc_v = losses.get("ctc", torch.tensor(0)).item()
            writer.add_scalar("train/ctc", ctc_v, global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        if batch_idx % 10 == 0:
            smooth = sum(recent_losses) / len(recent_losses)
            elapsed = time.time() - epoch_start
            speed = num_batches / elapsed if elapsed > 0 else 0
            if total_batches > 0 and speed > 0:
                eta = (total_batches - num_batches) / speed
                pct = 100.0 * num_batches / total_batches
                logger.info(
                    f"Epoch {epoch} [{num_batches}/{total_batches} ({pct:.1f}%)] "
                    f"loss={smooth:.4f} | {speed:.1f} batch/s | ETA {int(eta//60)}m"
                )

    return global_step


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, exp_dir, keep_last_n=5):
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_dir / f"epoch_{epoch}.pt"

    state = {
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(state, ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")

    ckpts = sorted(exp_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    while len(ckpts) > keep_last_n:
        old = ckpts.pop(0)
        old.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]
    set_seed(training_cfg.get("seed", 42))

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)

    model = build_model(config)
    model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model built: {total_params:.1f}M parameters (all trainable)")

    # Resume
    global_step = 0
    start_epoch = 1
    resume_path = args.resume or training_cfg.get("resume_from")
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ep = ckpt.get("epoch", 0)
        reset_epoch = training_cfg.get("reset_epoch", False) or not isinstance(ep, int)
        if reset_epoch:
            start_epoch = 1
            global_step = 0
            logger.info(f"Loaded weights from {resume_path} (reset epoch counter)")
        else:
            start_epoch = ep + 1
            global_step = ckpt.get("global_step", 0)
            logger.info(f"Resumed from {resume_path}, epoch {start_epoch}")

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    dataloader = build_dataloader(config["data"], rank, world_size)
    steps_per_epoch = len(dataloader)
    logger.info(f"Steps/epoch: {steps_per_epoch}")

    lr = training_cfg["lr"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        betas=(0.9, 0.98), weight_decay=0.01,
    )

    warmup_steps = training_cfg.get("warmup_steps", 5000)
    num_epochs = training_cfg["num_epochs"]
    total_steps = steps_per_epoch * num_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.5 * (1 + np.cos(np.pi * min(progress, 1.0))), 0.01)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    exp_dir = Path(training_cfg["exp_dir"])
    writer = SummaryWriter(exp_dir / "tb_logs") if rank == 0 else None

    grad_accum = training_cfg.get("grad_accum_steps", 4)
    loss_cfg = config["loss"]

    for epoch in range(start_epoch, num_epochs + 1):
        if hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)
        logger.info(f"=== Epoch {epoch}/{num_epochs} ===")

        global_step = train_one_epoch(
            model, dataloader, optimizer, scheduler,
            loss_cfg, epoch, writer, global_step, grad_accum,
        )

        if rank == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                exp_dir, keep_last_n=training_cfg.get("keep_last_n", 5),
            )

    if world_size > 1:
        dist.destroy_process_group()
    if writer:
        writer.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
