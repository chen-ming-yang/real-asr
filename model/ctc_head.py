"""CTC auxiliary head for Zipformer encoder.

Used during training for:
1. Auxiliary CTC loss to stabilize encoder training
2. Output-level knowledge distillation (frame-level KL with teacher)
"""

import torch
import torch.nn as nn


class CTCHead(nn.Module):
    """Linear CTC projection head on top of encoder output."""

    def __init__(self, encoder_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, vocab_size)

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_out: (batch, time, encoder_dim)

        Returns:
            log_probs: (batch, time, vocab_size) log probabilities
        """
        return torch.log_softmax(self.proj(encoder_out), dim=-1)

    def get_logits(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Return raw logits (for KD temperature scaling)."""
        return self.proj(encoder_out)
