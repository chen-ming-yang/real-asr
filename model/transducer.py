"""Stateless Transducer: Predictor + Joiner for RNN-T decoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StatelessPredictor(nn.Module):
    """Embedding-based stateless predictor (no LSTM).

    Uses embedding lookup + Conv1D for context modeling.
    Much faster than LSTM-based predictors at inference time.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        context_size: int = 2,
        conv_layers: int = 2,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.context_size = context_size

        conv_blocks = []
        for _ in range(conv_layers):
            conv_blocks.extend([
                nn.Conv1d(
                    embedding_dim, embedding_dim,
                    kernel_size=conv_kernel_size,
                    padding=(conv_kernel_size - 1) // 2,
                ),
                nn.SiLU(),
            ])
        self.conv = nn.Sequential(*conv_blocks)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (batch, seq_len) token indices, with blank_id=0 prepended

        Returns:
            predictor_out: (batch, seq_len, embedding_dim)
        """
        emb = self.embedding(y)  # (B, U, D)
        conv_in = emb.transpose(1, 2)  # (B, D, U)
        conv_out = self.conv(conv_in)  # (B, D, U)
        conv_out = conv_out.transpose(1, 2)  # (B, U, D)
        out = self.output_proj(conv_out + emb)  # residual
        out = self.norm(out)
        return out


class Joiner(nn.Module):
    """Joiner that combines encoder and predictor outputs.

    Uses LayerNorm before activation to prevent saturation,
    and ReLU instead of tanh for better gradient flow.
    """

    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int,
        hidden_dim: int,
        output_dim: int,  # vocab_size
    ):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        self.predictor_proj = nn.Linear(predictor_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.simple_lm_proj = nn.Linear(predictor_dim, output_dim)
        self.simple_am_proj = nn.Linear(encoder_dim, output_dim)

    def forward(
        self,
        encoder_out: torch.Tensor,
        predictor_out: torch.Tensor,
    ) -> torch.Tensor:
        """Full lattice forward: (B,T,D) x (B,U,D) -> (B,T,U,V)"""
        enc = self.encoder_proj(encoder_out)
        pred = self.predictor_proj(predictor_out)

        if enc.dim() == 3:
            enc = enc.unsqueeze(2)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)

        combined = enc + pred
        combined = self.norm(combined)
        combined = self.activation(combined)
        logits = self.output_proj(combined)
        return logits

    def forward_pruned(
        self,
        am_pruned: torch.Tensor,
        lm_pruned: torch.Tensor,
    ) -> torch.Tensor:
        """Pruned forward: (B,T,s_range,D) x (B,T,s_range,D) -> (B,T,s_range,V)"""
        enc = self.encoder_proj(am_pruned)
        pred = self.predictor_proj(lm_pruned)
        combined = enc + pred
        combined = self.norm(combined)
        combined = self.activation(combined)
        logits = self.output_proj(combined)
        return logits


class Transducer(nn.Module):
    """Complete Transducer model: Encoder + Predictor + Joiner."""

    def __init__(self, encoder: nn.Module, cfg: dict):
        super().__init__()
        self.encoder = encoder
        encoder_dim = encoder.get_output_dim()
        vocab_size = cfg["joiner"]["output_dim"]

        self.predictor = StatelessPredictor(
            vocab_size=vocab_size,
            embedding_dim=cfg["predictor"]["embedding_dim"],
            context_size=cfg["predictor"]["context_size"],
            conv_layers=cfg["predictor"]["conv_layers"],
            conv_kernel_size=cfg["predictor"]["conv_kernel_size"],
        )
        self.joiner = Joiner(
            encoder_dim=encoder_dim,
            predictor_dim=cfg["predictor"]["embedding_dim"],
            hidden_dim=cfg["joiner"]["hidden_dim"],
            output_dim=vocab_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        streaming: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, time, feat_dim) audio features
            x_lens: (batch,) lengths
            y: (batch, label_len) target token ids (blank prepended)
            streaming: whether to use chunk mask

        Returns:
            encoder_out: (B, T, encoder_dim)
            encoder_out_lens: (B,)
            predictor_out: (B, U, predictor_dim)
        """
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, streaming=streaming)
        predictor_out = self.predictor(y)
        return encoder_out, encoder_out_lens, predictor_out
