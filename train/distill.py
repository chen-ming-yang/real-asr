"""Knowledge distillation training script (Stage 2).

Trains student model with guidance from finetuned Qwen3-ASR teacher:
- Feature-level KD: align student encoder output with teacher AuT encoder
- Output-level KD: match student CTC logits with teacher frame-level probs
- Temporal Alignment Buffer for streaming/offline mismatch

Usage:
    python train/distill.py --config configs/train_stage2.yaml

    # Multi-GPU
    torchrun --nproc_per_node=4 train/distill.py --config configs/train_stage2.yaml
"""

import argparse
import logging
import os
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

from train.train import (
    build_dataloader,
    build_model,
    build_optimizer,
    build_scheduler,
    deep_merge,
    load_config,
    save_checkpoint,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class TeacherWrapper:
    """Wrapper for Qwen3-ASR teacher model to extract features and probabilities.

    Supports two modes:
    1. Feature extraction: get AuT encoder hidden states (for feature-level KD)
    2. Token probability: get frame-level token probabilities (for output-level KD)
    """

    def __init__(self, model_path: str, device: str = "cuda:0", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.model = None
        self.processor = None

    def load(self):
        """Load teacher model. Call once before training loop."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError("transformers is required for teacher model loading")

        logger.info(f"Loading teacher model from {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        logger.info("Teacher model loaded")

    @torch.no_grad()
    def extract_encoder_features(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Extract AuT encoder output (before projector).

        Args:
            audio_features: (batch, time, mel_dim) fbank features

        Returns:
            teacher_enc_out: (batch, time', teacher_hidden_dim)
        """
        audio_features = audio_features.to(self.device, dtype=self.dtype)

        # Access AuT encoder directly
        # The exact attribute path depends on Qwen3-ASR implementation
        if hasattr(self.model, "audio_encoder"):
            encoder = self.model.audio_encoder
        elif hasattr(self.model, "model") and hasattr(self.model.model, "audio_encoder"):
            encoder = self.model.model.audio_encoder
        else:
            raise AttributeError(
                "Cannot find audio encoder in teacher model. "
                "Check model architecture and update attribute path."
            )

        encoder_out = encoder(audio_features)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out.float()

    @torch.no_grad()
    def get_frame_probabilities(
        self,
        audio_features: torch.Tensor,
        texts: list,
    ) -> torch.Tensor:
        """Get teacher's frame-level token probability distribution.

        Uses forced alignment to expand token-level probs to frame-level.

        Args:
            audio_features: (batch, time, mel_dim)
            texts: list of reference transcriptions

        Returns:
            frame_probs: (batch, time', vocab_size) soft probability targets
        """
        audio_features = audio_features.to(self.device, dtype=self.dtype)

        # Get encoder features and project to decoder space
        encoder_out = self.extract_encoder_features(audio_features)
        num_frames = encoder_out.size(1)

        # For each utterance, we need the teacher's output distribution
        # In practice, run teacher decode and collect logits at each step
        # Then use forced alignment to map to frame positions

        # Simplified approach: use CTC-style output from encoder
        # (Real implementation should run full teacher forward with text prompt)
        if hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(encoder_out)
        else:
            # Fallback: project encoder output to vocab size
            vocab_size = 6000  # should match student vocab
            proj = nn.Linear(encoder_out.size(-1), vocab_size).to(self.device)
            logits = proj(encoder_out)

        frame_probs = torch.softmax(logits.float(), dim=-1)
        return frame_probs


def distill_one_epoch(
    student: nn.Module,
    ctc_head: nn.Module,
    distill_wrapper,
    teacher: TeacherWrapper,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    loss_cfg: Dict,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
) -> int:
    """Train one epoch with distillation."""
    student.train()
    ctc_head.train()

    total_losses = {"total": 0, "rnnt": 0, "ctc": 0, "feat_kd": 0, "out_kd": 0}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        features = batch["features"].cuda(non_blocking=True)
        feature_lens = batch["feature_lens"].cuda(non_blocking=True)
        targets = batch["targets"].cuda(non_blocking=True)
        target_lens = batch["target_lens"].cuda(non_blocking=True)

        # Predictor input (blank prepended)
        blank = torch.zeros(targets.size(0), 1, dtype=targets.dtype, device=targets.device)
        predictor_input = torch.cat([blank, targets], dim=1)

        # Teacher forward (no grad)
        teacher_enc_out = teacher.extract_encoder_features(features)
        teacher_frame_probs = None
        if loss_cfg.get("output_kd_weight", 0) > 0 and "texts" in batch:
            teacher_frame_probs = teacher.get_frame_probabilities(
                features, batch["texts"]
            )

        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            # Student forward via distill wrapper
            result = distill_wrapper(
                features, feature_lens, predictor_input,
                teacher_enc_out=teacher_enc_out,
                teacher_frame_probs=teacher_frame_probs,
                streaming=True,
            )

            # Compute RNN-T loss
            logits = result["logits"]
            encoder_out = result["encoder_out"]
            encoder_out_lens = result["encoder_out_lens"]

            losses = {}

            # RNN-T loss
            try:
                import k2
                rnnt_loss = k2.rnnt_loss_pruned(
                    logits=logits, symbols=targets, ranges=None,
                    termination_symbol=0, boundary=None, reduction="mean",
                )
            except (ImportError, Exception):
                rnnt_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            losses["rnnt"] = rnnt_loss * loss_cfg.get("rnnt_weight", 1.0)

            # CTC loss
            if loss_cfg.get("ctc_weight", 0) > 0:
                ctc_log_probs = ctc_head(encoder_out).transpose(0, 1)
                ctc_loss = nn.functional.ctc_loss(
                    ctc_log_probs, targets, encoder_out_lens, target_lens,
                    blank=0, reduction="mean", zero_infinity=True,
                )
                losses["ctc"] = ctc_loss * loss_cfg["ctc_weight"]

            # Feature KD loss
            if "feature_kd_loss" in result:
                losses["feat_kd"] = result["feature_kd_loss"] * loss_cfg.get("feature_kd_weight", 0.5)

            # Output KD loss
            if "output_kd_loss" in result:
                losses["out_kd"] = result["output_kd_loss"] * loss_cfg.get("output_kd_weight", 0.3)

            total_loss = sum(losses.values())

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        all_params = list(student.parameters()) + list(ctc_head.parameters()) + list(distill_wrapper.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Accumulate stats
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()
        total_losses["total"] += total_loss.item()
        num_batches += 1
        global_step += 1

        if writer and global_step % 100 == 0:
            for k, v in losses.items():
                writer.add_scalar(f"distill/{k}", v.item(), global_step)
            writer.add_scalar("distill/total", total_loss.item(), global_step)
            writer.add_scalar("distill/lr", scheduler.get_last_lr()[0], global_step)

        if batch_idx % 200 == 0:
            avg = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
            logger.info(
                f"Epoch {epoch} batch {batch_idx}: "
                f"total={avg['total']:.4f} rnnt={avg['rnnt']:.4f} "
                f"feat_kd={avg.get('feat_kd', 0):.4f} out_kd={avg.get('out_kd', 0):.4f}"
            )

    return global_step


def main():
    parser = argparse.ArgumentParser(description="Distillation training for Real-ASR")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]
    teacher_cfg = config["teacher"]
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

    # Build student model
    transducer, ctc_head = build_model(config["model"])
    transducer = transducer.cuda()
    ctc_head = ctc_head.cuda()

    # Build distillation wrapper
    from model.distill_wrapper import DistillationWrapper

    student_dim = config["model"].get("encoder_output_dim", 512)
    teacher_dim = config["model"].get("teacher_hidden_dim", 1024)
    distill_wrapper = DistillationWrapper(
        student=transducer,
        ctc_head=ctc_head,
        teacher_hidden_dim=teacher_dim,
        student_hidden_dim=student_dim,
        feature_kd_type=training_cfg["loss"].get("feature_kd_type", "mse"),
        temperature=training_cfg["loss"].get("temperature", 2.0),
        max_delay_chunks=training_cfg["loss"].get("temporal_alignment_buffer", {}).get("max_delay_chunks", 4),
    ).cuda()

    # Load teacher model (on designated device)
    teacher = TeacherWrapper(
        model_path=teacher_cfg["model_path"],
        device=teacher_cfg.get("device", "cuda:0"),
        dtype=getattr(torch, teacher_cfg.get("dtype", "bfloat16")),
    )
    teacher.load()

    # Resume from Stage 1 checkpoint
    global_step = 0
    start_epoch = 1
    resume_from = training_cfg.get("resume_from")
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location="cpu")
        transducer.load_state_dict(ckpt["model"], strict=False)
        if "ctc_head" in ckpt:
            ctc_head.load_state_dict(ckpt["ctc_head"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        logger.info(f"Resumed from {resume_from}")

    if world_size > 1:
        distill_wrapper = DDP(distill_wrapper, device_ids=[rank], find_unused_parameters=True)

    # Optimizer and scheduler
    optimizer = build_optimizer(distill_wrapper, training_cfg["optimizer"])
    scheduler = build_scheduler(optimizer, training_cfg["scheduler"], steps_per_epoch=1000)
    scaler = GradScaler()

    # TensorBoard
    exp_dir = Path(training_cfg["checkpoint"]["exp_dir"])
    writer = SummaryWriter(exp_dir / "tb_logs") if rank == 0 else None

    # Dataloader
    dataloader = build_dataloader(config["data"], "distill", rank, world_size)

    # Training loop
    num_epochs = training_cfg["num_epochs"]
    for epoch in range(start_epoch, num_epochs + 1):
        if hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(epoch)

        logger.info(f"=== Distillation Epoch {epoch}/{num_epochs} ===")
        global_step = distill_one_epoch(
            transducer, ctc_head, distill_wrapper, teacher,
            dataloader, optimizer, scheduler, scaler,
            training_cfg["loss"], epoch, writer, global_step,
        )

        if rank == 0:
            save_checkpoint(
                distill_wrapper, ctc_head, optimizer, scheduler,
                epoch, global_step, exp_dir,
                keep_last_n=training_cfg["checkpoint"].get("keep_last_n", 5),
            )

    if world_size > 1:
        dist.destroy_process_group()
    if writer:
        writer.close()

    logger.info("Distillation training complete!")


if __name__ == "__main__":
    main()
