"""Icefall Zipformer2 Encoder wrapper.

Wraps the official icefall Zipformer2 with a Conv2dSubsampling frontend
to match the interface expected by our Transducer training pipeline.

Interface:
    Input:  (batch, time, num_mel_bins)  +  x_lens: (batch,)
    Output: (batch, T', encoder_dim)     +  output_lens: (batch,)
"""

import sys
import os
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from icefall_zipformer import Zipformer2
from scaling import ScaledLinear, ScheduledFloat


class Conv2dSubsampling(nn.Module):
    """Conv2d subsampling module (icefall style).

    Applies two Conv2d layers with stride 2 each, giving 4x time reduction.
    Input: (batch, time, mel_dim)
    Output: (batch, time//4, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, output_dim, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, 3, stride=2, padding=1)
        self.out_proj = nn.Linear(output_dim * (input_dim // 4), output_dim)
        self.activation = nn.SiLU()

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, mel_dim)
            x_lens: (batch,)
        Returns:
            out: (batch, time//4, output_dim)
            out_lens: (batch,)
        """
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.activation(self.conv1(x))  # (B, C, T/2, F/2)
        x = self.activation(self.conv2(x))  # (B, C, T/4, F/4)

        B, C, T, F_out = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F_out)  # (B, T, C*F)
        x = self.out_proj(x)  # (B, T, output_dim)

        out_lens = ((x_lens - 1) // 2 - 1) // 2
        out_lens = out_lens.clamp(min=1)
        return x, out_lens


class IcefallZipformerEncoder(nn.Module):
    """Wrapper around icefall's Zipformer2 for our training pipeline.

    This provides the same interface as our custom ZipformerEncoder:
        forward(x, x_lens, streaming=True) -> (encoder_out, encoder_out_lens)
        get_output_dim() -> int
    """

    def __init__(self, cfg: dict):
        super().__init__()
        input_dim = cfg.get("input_dim", 80)
        output_dim = cfg.get("output_dim", 512)

        encoder_dim = cfg.get("encoder_dim", (384, 512))
        downsampling_factor = cfg.get("downsampling_factor", (2, 4))
        num_encoder_layers = cfg.get("num_encoder_layers", (4, 4))
        num_heads = cfg.get("num_heads", (8, 8))
        feedforward_dim = cfg.get("feedforward_dim", (1536, 2048))
        cnn_module_kernel = cfg.get("cnn_module_kernel", (31, 31))
        query_head_dim = cfg.get("query_head_dim", (24, 24))
        pos_head_dim = cfg.get("pos_head_dim", (4, 4))
        value_head_dim = cfg.get("value_head_dim", (12, 12))
        encoder_unmasked_dim = cfg.get("encoder_unmasked_dim", (256, 256))
        pos_dim = cfg.get("pos_dim", 192)
        output_downsampling_factor = cfg.get("output_downsampling_factor", 2)
        warmup_batches = cfg.get("warmup_batches", 4000.0)
        dropout = cfg.get("dropout", None)
        causal = cfg.get("causal", False)

        first_encoder_dim = encoder_dim[0] if isinstance(encoder_dim, (list, tuple)) else encoder_dim

        self.encoder_embed = Conv2dSubsampling(input_dim, first_encoder_dim)

        if dropout is not None:
            dropout_val = ScheduledFloat((0.0, dropout), (20000.0, dropout * 0.33))
        else:
            dropout_val = None

        self.zipformer = Zipformer2(
            output_downsampling_factor=output_downsampling_factor,
            downsampling_factor=tuple(downsampling_factor),
            encoder_dim=tuple(encoder_dim),
            num_encoder_layers=tuple(num_encoder_layers),
            encoder_unmasked_dim=tuple(encoder_unmasked_dim),
            query_head_dim=tuple(query_head_dim),
            pos_head_dim=tuple(pos_head_dim),
            value_head_dim=tuple(value_head_dim),
            num_heads=tuple(num_heads),
            feedforward_dim=tuple(feedforward_dim),
            cnn_module_kernel=tuple(cnn_module_kernel),
            pos_dim=pos_dim,
            dropout=dropout_val,
            warmup_batches=warmup_batches,
            causal=causal,
        )

        max_encoder_dim = max(encoder_dim) if isinstance(encoder_dim, (list, tuple)) else encoder_dim
        self._output_dim = output_dim
        self.output_proj = nn.Linear(max_encoder_dim, output_dim)

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        streaming: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, num_mel_bins)
            x_lens: (batch,)
            streaming: unused (kept for API compat)
        Returns:
            output: (batch, T', output_dim)
            output_lens: (batch,)
        """
        x, x_lens = self.encoder_embed(x, x_lens)

        # Zipformer2 expects (seq_len, batch, dim)
        x = x.permute(1, 0, 2)  # (T, B, D)

        # Create padding mask
        B = x.size(1)
        T = x.size(0)
        src_key_padding_mask = None
        if x_lens is not None:
            src_key_padding_mask = torch.arange(T, device=x.device).unsqueeze(0) >= x_lens.unsqueeze(1)

        x, out_lens = self.zipformer(x, x_lens, src_key_padding_mask)

        # Back to (batch, time, dim)
        x = x.permute(1, 0, 2)  # (B, T', D)

        x = self.output_proj(x)

        return x, out_lens
