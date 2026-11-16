"""Zipformer encoder for streaming ASR.

Adapted from k2-fsa/icefall and pkufool/zipformer.
Multi-scale U-Net structure with chunked attention for streaming.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding with learnable parameters."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_emb = self.pe[:, : x.size(1)]
        return x, pos_emb


class ConvolutionModule(nn.Module):
    """Convolution module in Zipformer block."""

    def __init__(self, d_model: int, kernel_size: int = 31, bias: bool = True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model, bias=bias,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * torch.sigmoid(x2)  # GLU
        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        return x


class ZipformerBlock(nn.Module):
    """Single Zipformer block with self-attention + conv module."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        cnn_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.conv_module = ConvolutionModule(d_model, cnn_kernel_size)
        self.feed_forward1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, feedforward_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )
        self.feed_forward2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, feedforward_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm_final = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = residual + 0.5 * self.feed_forward1(x)

        residual = x
        x_norm = self.norm_final(x)  # pre-norm for attention
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = residual + self.dropout(x_attn)

        x = x + self.conv_module(x)
        x = x + 0.5 * self.feed_forward2(x)
        x = self.norm_final(x)
        return x


class ZipformerStage(nn.Module):
    """One stage of Zipformer (operates at a specific time resolution)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        cnn_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ZipformerBlock(d_model, num_heads, feedforward_dim, cnn_kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x


class DownsampleLayer(nn.Module):
    """Downsample by factor using Conv1d."""

    def __init__(self, in_dim: int, out_dim: int, factor: int):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=factor, stride=factor)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class UpsampleLayer(nn.Module):
    """Upsample by factor using ConvTranspose1d."""

    def __init__(self, in_dim: int, out_dim: int, factor: int):
        super().__init__()
        self.factor = factor
        self.conv_t = nn.ConvTranspose1d(
            in_dim, out_dim, kernel_size=factor, stride=factor
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_t(x)
        x = x[:, :, :target_len]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class ZipformerEncoder(nn.Module):
    """Multi-scale Zipformer encoder with U-Net structure.

    Supports streaming via chunked attention masking.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.num_stages = cfg.get("num_stages", 6)
        self.hidden_dims = cfg["hidden_dims"]
        self.num_heads = cfg["num_heads"]
        self.num_encoder_layers = cfg["num_encoder_layers"]
        self.downsampling_factors = cfg["downsampling_factors"]
        self.feedforward_dims = cfg["feedforward_dims"]
        self.cnn_module_kernels = cfg["cnn_module_kernels"]
        self.chunk_size = cfg.get("chunk_size", 32)
        self.left_context = cfg.get("left_context_frames", 64)
        self.right_context = cfg.get("right_context_frames", 8)
        dropout = cfg.get("dropout", 0.1)

        num_mel_bins = cfg.get("input_dim", 80)

        # Input projection: Fbank -> first stage dim
        self.input_proj = nn.Sequential(
            nn.Linear(num_mel_bins, self.hidden_dims[0]),
            nn.LayerNorm(self.hidden_dims[0]),
            nn.Dropout(dropout),
        )
        self.pos_enc = RelPositionalEncoding(self.hidden_dims[0])

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for i in range(self.num_stages):
            self.stages.append(
                ZipformerStage(
                    d_model=self.hidden_dims[i],
                    num_heads=self.num_heads[i],
                    num_layers=self.num_encoder_layers[i],
                    feedforward_dim=self.feedforward_dims[i],
                    cnn_kernel_size=self.cnn_module_kernels[i],
                    dropout=dropout,
                )
            )

        # Downsampling between stages using Conv1d stride
        # stages: 0(1x) -> 1(2x) -> 2(4x) -> 3(8x bottleneck)
        half = self.num_stages // 2  # 3
        for i in range(half):
            next_dim = self.hidden_dims[i + 1]
            curr_dim = self.hidden_dims[i]
            # Compute factor between consecutive stages
            curr_factor = self.downsampling_factors[i]
            next_factor = self.downsampling_factors[i + 1]
            stride = next_factor // curr_factor
            if stride > 1:
                self.downsamplers.append(
                    DownsampleLayer(curr_dim, next_dim, stride)
                )
            else:
                self.downsamplers.append(
                    nn.Sequential(
                        nn.Linear(curr_dim, next_dim),
                        nn.LayerNorm(next_dim),
                    )
                )

        # Upsampling for decoder side of U-Net
        # stages: 3(8x) -> 4(4x) -> 5(2x)
        for i in range(half - 1):
            src_stage = half + i      # 3, 4
            tgt_stage = half + i + 1  # 4, 5
            src_dim = self.hidden_dims[src_stage]
            tgt_dim = self.hidden_dims[tgt_stage]
            src_factor = self.downsampling_factors[src_stage]
            tgt_factor = self.downsampling_factors[tgt_stage]
            stride = src_factor // tgt_factor
            if stride > 1:
                self.upsamplers.append(
                    UpsampleLayer(src_dim, tgt_dim, stride)
                )
            else:
                self.upsamplers.append(
                    nn.Sequential(
                        nn.Linear(src_dim, tgt_dim),
                        nn.LayerNorm(tgt_dim),
                    )
                )

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dims[-1], cfg.get("output_dim", 512))
        self.output_norm = nn.LayerNorm(cfg.get("output_dim", 512))

    def _create_chunk_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create attention mask for chunked streaming inference."""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            chunk_start = max(0, (i // self.chunk_size) * self.chunk_size - self.left_context)
            chunk_end = min(seq_len, ((i // self.chunk_size) + 1) * self.chunk_size + self.right_context)
            mask[i, chunk_start:chunk_end] = False
        return mask  # True = masked

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
            streaming: if True, apply chunk mask

        Returns:
            output: (batch, time', output_dim) where time' = time // 4
            output_lens: (batch,)
        """
        x = self.input_proj(x)
        x, pos_emb = self.pos_enc(x)

        attn_mask = None
        if streaming:
            attn_mask = self._create_chunk_mask(x.size(1), x.device)

        half = self.num_stages // 2  # 3
        intermediates: List[Tuple[torch.Tensor, int]] = []

        # Encoder path (stages 0,1,2 with downsampling between)
        for i in range(half):
            x = self.stages[i](x, attn_mask=attn_mask)
            intermediates.append((x, x.size(1)))
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
                if streaming:
                    attn_mask = self._create_chunk_mask(x.size(1), x.device)

        # Bottleneck (stage 3)
        x = self.stages[half](x, attn_mask=attn_mask)

        # Decoder path (stages 4,5 with upsampling)
        for i in range(len(self.upsamplers)):
            stage_idx = half + 1 + i  # 4, 5
            # Skip connection: stage 4 connects to stage 2, stage 5 to stage 1
            skip_idx = half - 1 - i   # 2, 1

            if isinstance(self.upsamplers[i], UpsampleLayer):
                skip_len = intermediates[skip_idx][1]
                x = self.upsamplers[i](x, skip_len)
            else:
                x = self.upsamplers[i](x)

            # Skip connection (add)
            skip_x = intermediates[skip_idx][0]
            if x.size(1) != skip_x.size(1):
                min_len = min(x.size(1), skip_x.size(1))
                x = x[:, :min_len, :]
                skip_x = skip_x[:, :min_len, :]
            x = x + skip_x

            if streaming:
                attn_mask = self._create_chunk_mask(x.size(1), x.device)
            x = self.stages[stage_idx](x, attn_mask=attn_mask)

        x = self.output_proj(x)
        x = self.output_norm(x)

        # Final output is at 2x downsample (stage 5). Apply additional 2x
        # downsample to get 4x total (40ms/frame, standard for ASR).
        # Use simple mean-pooling for the final 2x reduction.
        B, T_out, D = x.shape
        if T_out % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))  # pad time dim
            T_out += 1
        x = x.view(B, T_out // 2, 2, D).mean(dim=2)  # (B, T/2, D)

        # Compute actual output lengths based on real tensor size
        # Each input frame goes through: 2x (U-Net) * 2x (pool) = 4x total
        output_lens = torch.clamp(x_lens // 4, max=x.size(1))

        return x, output_lens

    def get_output_dim(self) -> int:
        return self.output_proj.out_features
