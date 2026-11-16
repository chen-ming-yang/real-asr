"""RNN-T Model: Joint CTC + Transducer (from scratch).

Key differences from the old approach:
- Encoder, Predictor, Joiner are all trained jointly (no frozen encoder)
- CTC loss is used as an auxiliary loss to guide encoder output
- k2 pruned RNN-T loss for efficient transducer training
- Proper warmup prevents blank dominance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPredictor(nn.Module):
    """Stateless predictor using embedding + convolution.

    Similar to icefall's stateless predictor: uses embedding lookup
    followed by Conv1d layers for context aggregation.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 512, context_size: int = 2):
        super().__init__()
        self.context_size = context_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=context_size,
            padding=0,
            groups=embed_dim // 4,
        )
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (B, U) token ids (with blank=0 prepended for teacher forcing)
        Returns:
            (B, U, embed_dim)
        """
        emb = self.embedding(y)
        # Pad left for causal conv
        emb_t = emb.transpose(1, 2)  # (B, D, U)
        emb_t = F.pad(emb_t, (self.context_size - 1, 0))
        conv_out = self.conv(emb_t).transpose(1, 2)  # (B, U, D)
        return self.output_proj(conv_out + emb)


class Joiner(nn.Module):
    """Standard additive joiner with projections."""

    def __init__(self, encoder_dim: int, predictor_dim: int, joiner_dim: int, vocab_size: int):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, joiner_dim)
        self.predictor_proj = nn.Linear(predictor_dim, joiner_dim)
        self.output_proj = nn.Linear(joiner_dim, vocab_size)
        # For k2 simple loss
        self.simple_am_proj = nn.Linear(encoder_dim, vocab_size)
        self.simple_lm_proj = nn.Linear(predictor_dim, vocab_size)

    def forward(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Full-lattice forward: (B,T,D) x (B,U,D) -> (B,T,U,V)"""
        e = self.encoder_proj(enc).unsqueeze(2)   # (B, T, 1, J)
        p = self.predictor_proj(pred).unsqueeze(1) # (B, 1, U, J)
        logits = self.output_proj(torch.tanh(e + p))
        return logits

    def forward_pruned(self, am_pruned: torch.Tensor, lm_pruned: torch.Tensor) -> torch.Tensor:
        """Pruned forward: (B,T,s,D) x (B,T,s,D) -> (B,T,s,V)"""
        e = self.encoder_proj(am_pruned)
        p = self.predictor_proj(lm_pruned)
        logits = self.output_proj(torch.tanh(e + p))
        return logits


class RNNTModel(nn.Module):
    """Joint CTC + RNN-T model for from-scratch training."""

    def __init__(self, encoder: nn.Module, encoder_dim: int, vocab_size: int, cfg: dict):
        super().__init__()
        self.encoder = encoder
        predictor_dim = cfg.get("predictor_dim", 512)
        joiner_dim = cfg.get("joiner_dim", 512)
        context_size = cfg.get("context_size", 2)

        self.predictor = ConvPredictor(
            vocab_size=vocab_size,
            embed_dim=predictor_dim,
            context_size=context_size,
        )
        self.joiner = Joiner(
            encoder_dim=encoder_dim,
            predictor_dim=predictor_dim,
            joiner_dim=joiner_dim,
            vocab_size=vocab_size,
        )
        self.ctc_proj = nn.Linear(encoder_dim, vocab_size)

    def forward(self, x, x_lens, y=None, streaming=True):
        """
        Returns:
            encoder_out: (B, T, D)
            encoder_out_lens: (B,)
            predictor_out: (B, U, D) or None if y is None
        """
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, streaming=streaming)
        predictor_out = None
        if y is not None:
            predictor_out = self.predictor(y)
        return encoder_out, encoder_out_lens, predictor_out

    def get_ctc_log_probs(self, encoder_out: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.ctc_proj(encoder_out), dim=-1)
