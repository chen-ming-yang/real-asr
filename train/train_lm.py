"""Training script for Transformer Language Model.

Usage:
    python train/train_lm.py --config configs/train_lm.yaml
"""

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import yaml
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer_lm import TransformerLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class TokenDataset(Dataset):
    """Dataset that reads pre-tokenized text (one sentence per line, space-separated IDs)."""

    def __init__(self, token_file: str, seq_len: int, bos_id: int = 1, eos_id: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id

        logger.info(f"Loading tokens from {token_file}...")
        self.sentences = []
        with open(token_file, "r") as f:
            for line in f:
                ids = list(map(int, line.strip().split()))
                if len(ids) >= 2:
                    self.sentences.append(ids)

        logger.info(f"Loaded {len(self.sentences)} sentences")

        # Concatenate all into a long sequence for efficient batching
        all_tokens = []
        for ids in self.sentences:
            all_tokens.extend([self.bos_id] + ids + [self.eos_id])

        self.data = torch.tensor(all_tokens, dtype=torch.long)
        self.num_chunks = (len(self.data) - 1) // seq_len
        logger.info(f"Total tokens: {len(self.data)}, chunks: {self.num_chunks}")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        if len(chunk) < self.seq_len + 1:
            pad = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        return input_ids, target_ids


def build_model(config: Dict) -> TransformerLM:
    model_cfg = config["model"]
    model = TransformerLM(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        max_seq_len=model_cfg["max_seq_len"],
    )
    return model


def build_optimizer(model: nn.Module, optim_cfg: Dict):
    return torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg["lr"],
        betas=tuple(optim_cfg.get("betas", [0.9, 0.98])),
        weight_decay=optim_cfg.get("weight_decay", 0.01),
    )


def build_scheduler(optimizer, scheduler_cfg: Dict, total_steps: int):
    warmup_steps = scheduler_cfg.get("warmup_steps", 2000)
    min_lr_ratio = scheduler_cfg.get("min_lr_ratio", 0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return max(cosine_decay, min_lr_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, scheduler, grad_clip, device, epoch, writer):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        output = model(input_ids, target_ids)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        batch_tokens = (target_ids != -1).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        if (batch_idx + 1) % 500 == 0:
            avg_loss = total_loss / max(total_tokens, 1)
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"loss={avg_loss:.4f} ppl={ppl:.1f} lr={lr:.6f} "
                f"elapsed={elapsed:.0f}s"
            )
            global_step = (epoch - 1) * len(dataloader) + batch_idx
            writer.add_scalar("train/loss", avg_loss, global_step)
            writer.add_scalar("train/ppl", ppl, global_step)
            writer.add_scalar("train/lr", lr, global_step)

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        output = model(input_ids, target_ids)
        loss = output["loss"]

        batch_tokens = (target_ids != -1).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["training"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_cfg = config["data"]
    training_cfg = config["training"]
    ckpt_cfg = training_cfg["checkpoint"]

    exp_dir = Path(ckpt_cfg["exp_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(exp_dir / "tb_logs"))

    train_dataset = TokenDataset(data_cfg["train_tokens"], seq_len=data_cfg["seq_len"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if data_cfg.get("val_tokens") and os.path.exists(data_cfg["val_tokens"]):
        val_dataset = TokenDataset(data_cfg["val_tokens"], seq_len=data_cfg["seq_len"])
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_cfg["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    model = build_model(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {num_params / 1e6:.1f}M")

    optimizer = build_optimizer(model, training_cfg["optimizer"])
    total_steps = len(train_loader) * training_cfg["num_epochs"]
    scheduler = build_scheduler(optimizer, training_cfg["scheduler"], total_steps)
    grad_clip = training_cfg.get("grad_clip_norm", 1.0)

    logger.info(f"Steps per epoch: {len(train_loader)}, total steps: {total_steps}")

    best_val_loss = float("inf")
    for epoch in range(1, training_cfg["num_epochs"] + 1):
        logger.info(f"=== Epoch {epoch} ===")
        train_loss, train_ppl = train_one_epoch(
            model, train_loader, optimizer, scheduler, grad_clip, device, epoch, writer
        )
        logger.info(f"Epoch {epoch} train: loss={train_loss:.4f}, ppl={train_ppl:.1f}")

        if val_loader is not None:
            val_loss, val_ppl = validate(model, val_loader, device)
            logger.info(f"Epoch {epoch} val: loss={val_loss:.4f}, ppl={val_ppl:.1f}")
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/ppl", val_ppl, epoch)
        else:
            val_loss = train_loss

        # Save checkpoint
        ckpt_path = exp_dir / f"epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config,
        }, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = exp_dir / "best.pt"
            torch.save({"model_state_dict": model.state_dict(), "config": config}, best_path)
            logger.info(f"New best model: val_loss={val_loss:.4f}")

        # Cleanup old checkpoints
        keep_n = ckpt_cfg.get("keep_last_n", 3)
        ckpts = sorted(exp_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
        while len(ckpts) > keep_n:
            old = ckpts.pop(0)
            old.unlink()
            logger.info(f"Removed old checkpoint: {old}")

    writer.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
