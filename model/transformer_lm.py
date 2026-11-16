"""Transformer Language Model for ASR rescoring.

A standard autoregressive Transformer decoder-only LM.
Used for CTC beam search rescoring and RNN-T shallow fusion.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLM(nn.Module):
    """Decoder-only Transformer Language Model.

    Args:
        vocab_size: Size of vocabulary (including blank and special tokens)
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: FFN inner dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        vocab_size: int = 6000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.output_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            target_ids: (batch, seq_len) target IDs for loss computation
                        If None, only returns logits.

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        B, T = input_ids.shape

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)

        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Use decoder with self-attention only (no cross-attention memory)
        memory = torch.zeros(B, 1, self.d_model, device=x.device)
        x = self.transformer(x, memory, tgt_mask=causal_mask)
        x = self.norm(x)
        logits = self.output_proj(x)

        result = {"logits": logits}

        if target_ids is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                target_ids.reshape(-1),
                ignore_index=-1,
            )
            result["loss"] = loss

        return result

    def score(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for next token prediction.

        Args:
            input_ids: (batch, seq_len) token IDs

        Returns:
            log_probs: (batch, seq_len, vocab_size)
        """
        output = self.forward(input_ids)
        return F.log_softmax(output["logits"], dim=-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
