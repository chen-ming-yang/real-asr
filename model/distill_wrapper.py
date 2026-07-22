"""Knowledge distillation wrapper for teacher-student training.

Handles:
- Feature-level KD (MSE/cosine between encoder outputs)
- Output-level KD (KL divergence on CTC logits)
- Temporal Alignment Buffer for streaming student vs offline teacher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationWrapper(nn.Module):
    """Wraps student model with distillation loss computation."""

    def __init__(
        self,
        student: nn.Module,
        ctc_head: nn.Module,
        teacher_hidden_dim: int,
        student_hidden_dim: int,
        feature_kd_type: str = "mse",
        temperature: float = 2.0,
        max_delay_chunks: int = 4,
    ):
        super().__init__()
        self.student = student
        self.ctc_head = ctc_head
        self.temperature = temperature
        self.max_delay_chunks = max_delay_chunks
        self.feature_kd_type = feature_kd_type

        # Projection to align student encoder dim -> teacher encoder dim
        self.feature_proj = nn.Linear(student_hidden_dim, teacher_hidden_dim)

    def compute_feature_kd_loss(
        self,
        student_enc_out: torch.Tensor,
        teacher_enc_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Feature-level KD: align student encoder output to teacher AuT output.

        Args:
            student_enc_out: (batch, T, student_dim)
            teacher_enc_out: (batch, T, teacher_dim) from Qwen3-ASR AuT
            encoder_out_lens: (batch,) valid lengths
        """
        projected = self.feature_proj(student_enc_out)

        # Truncate to minimum length
        min_len = min(projected.size(1), teacher_enc_out.size(1))
        projected = projected[:, :min_len]
        teacher_enc_out = teacher_enc_out[:, :min_len]

        # Create mask for valid positions
        mask = torch.arange(min_len, device=projected.device).unsqueeze(0)
        mask = mask < encoder_out_lens.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()

        if self.feature_kd_type == "mse":
            loss = F.mse_loss(projected * mask, teacher_enc_out * mask, reduction="sum")
            loss = loss / mask.sum().clamp(min=1)
        elif self.feature_kd_type == "cosine":
            cos_sim = F.cosine_similarity(projected, teacher_enc_out, dim=-1)
            loss = (1.0 - cos_sim) * mask.squeeze(-1)
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            raise ValueError(f"Unknown feature_kd_type: {self.feature_kd_type}")

        return loss

    def compute_output_kd_loss(
        self,
        student_enc_out: torch.Tensor,
        teacher_frame_probs: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Output-level KD with Temporal Alignment Buffer.

        Uses student CTC logits vs teacher frame-level token probabilities.
        Searches over delay offsets to find best alignment.

        Args:
            student_enc_out: (batch, T, student_dim)
            teacher_frame_probs: (batch, T, vocab_size) teacher soft targets
            encoder_out_lens: (batch,)
        """
        student_logits = self.ctc_head.get_logits(student_enc_out)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_frame_probs / self.temperature, dim=-1)

        min_len = min(student_log_probs.size(1), teacher_soft.size(1))

        # Temporal Alignment Buffer: search for best delay
        best_loss = None
        chunk_size = self.student.encoder.chunk_size if hasattr(self.student.encoder, "chunk_size") else 32

        for delay in range(self.max_delay_chunks + 1):
            offset = delay * chunk_size
            if offset >= min_len:
                break

            effective_len = min_len - offset
            s_probs = student_log_probs[:, offset:offset + effective_len]
            t_probs = teacher_soft[:, :effective_len]

            # KL divergence
            kl = F.kl_div(s_probs, t_probs, reduction="none").sum(dim=-1)

            # Mask invalid positions
            mask = torch.arange(effective_len, device=kl.device).unsqueeze(0)
            mask = (mask < (encoder_out_lens.unsqueeze(1) - offset)).float()
            kl = (kl * mask).sum() / mask.sum().clamp(min=1)

            if best_loss is None or kl < best_loss:
                best_loss = kl

        return best_loss * (self.temperature ** 2)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        teacher_enc_out: torch.Tensor = None,
        teacher_frame_probs: torch.Tensor = None,
        streaming: bool = True,
    ) -> dict:
        """Forward pass returning all losses.

        Returns:
            dict with keys: logits, encoder_out, encoder_out_lens,
                           feature_kd_loss, output_kd_loss
        """
        logits, encoder_out, encoder_out_lens = self.student(
            x, x_lens, y, streaming=streaming
        )

        result = {
            "logits": logits,
            "encoder_out": encoder_out,
            "encoder_out_lens": encoder_out_lens,
        }

        if teacher_enc_out is not None:
            result["feature_kd_loss"] = self.compute_feature_kd_loss(
                encoder_out, teacher_enc_out, encoder_out_lens
            )

        if teacher_frame_probs is not None:
            result["output_kd_loss"] = self.compute_output_kd_loss(
                encoder_out, teacher_frame_probs, encoder_out_lens
            )

        return result
