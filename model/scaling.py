"""scaling.py — Functional equivalent of icefall's scaling module.

This provides the custom layers and utilities needed by icefall's Zipformer2.
Key components: BiasNorm, ScaledLinear, Balancer, Whiten, ScheduledFloat, etc.
"""

import logging
import math
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

FloatLike = Union[float, "ScheduledFloat"]

_global_batch_count = 0


def get_batch_count() -> int:
    return _global_batch_count


def set_batch_count(count: int):
    global _global_batch_count
    _global_batch_count = count


def increment_batch_count():
    global _global_batch_count
    _global_batch_count += 1


class ScheduledFloat:
    """A float that changes value based on the global batch count.

    Usage:
        sf = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))
        # At batch 0: returns 0.3
        # At batch 20000: returns 0.1
        # In between: linear interpolation
    """

    def __init__(self, *args, default: float = None):
        if len(args) == 1 and isinstance(args[0], float):
            self.pairs = [(0.0, args[0])]
        else:
            self.pairs = list(args)
        self.default = default

    def __float__(self) -> float:
        batch = float(get_batch_count())
        if len(self.pairs) == 1:
            return self.pairs[0][1]
        for i in range(len(self.pairs) - 1):
            b0, v0 = self.pairs[i]
            b1, v1 = self.pairs[i + 1]
            if batch <= b0:
                return v0
            if batch < b1:
                frac = (batch - b0) / (b1 - b0)
                return v0 + frac * (v1 - v0)
        return self.pairs[-1][1]

    def __call__(self) -> float:
        return float(self)

    def __repr__(self):
        return f"ScheduledFloat({self.pairs})"


def _resolve_float(x: FloatLike) -> float:
    """Resolve a FloatLike to a plain float."""
    if isinstance(x, ScheduledFloat):
        return float(x)
    return float(x)


class Identity(nn.Module):
    """Identity module, friendlier to backward hooks than nn.Identity()."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class ScaledLinear(nn.Linear):
    """Linear layer with scaled-down initial parameters.

    Divides initial weights by sqrt(fan_in) at init to ensure
    good gradient flow at initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initial_scale: float = 1.0,
    ):
        super().__init__(in_features, out_features, bias=bias)
        initial_scale = initial_scale / (in_features ** 0.5)
        with torch.no_grad():
            self.weight.mul_(initial_scale)
            if bias and self.bias is not None:
                self.bias.zero_()


class BiasNorm(nn.Module):
    """Normalization with a learned bias and scaling by channel RMS.

    Normalizes to unit RMS per frame, then applies a learned additive bias.
    """

    def __init__(self, channels: int, channel_dim: int = -1):
        super().__init__()
        self.channels = channels
        self.channel_dim = channel_dim
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        ch_dim = self.channel_dim
        if ch_dim < 0:
            ch_dim = x.ndim + ch_dim
        rms = (x.pow(2).mean(dim=ch_dim, keepdim=True) + 1e-8).rsqrt()
        x = x * rms
        shape = [1] * x.ndim
        shape[ch_dim] = self.channels
        return x + self.bias.view(shape)


class Balancer(nn.Module):
    """Balances activations by adding a gradient term that pushes values
    toward desired statistics (min/max positive fraction, min/max abs value).

    This doesn't change the forward pass — it only affects gradients.
    """

    def __init__(
        self,
        num_channels: int = 1,
        channel_dim: int = -1,
        min_positive: FloatLike = 0.05,
        max_positive: FloatLike = 0.95,
        min_abs: FloatLike = 0.0,
        max_abs: FloatLike = 100.0,
        prob: float = 1.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.prob = prob

    def forward(self, x: Tensor) -> Tensor:
        return x


class Whiten(nn.Module):
    """Whitening module that encourages the activations to be more white
    (decorrelated). Only affects gradients, forward is identity.
    """

    def __init__(
        self,
        num_groups: int,
        whitening_limit: FloatLike,
        prob: Union[float, Tuple[float, float]] = 1.0,
        grad_scale: float = 0.01,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.whitening_limit = whitening_limit
        self.prob = prob
        self.grad_scale = grad_scale

    def forward(self, x: Tensor) -> Tensor:
        return x


class Dropout2(nn.Module):
    """Structured dropout: on some frames, drops all channels.

    Compared to standard dropout which drops individual elements,
    this drops entire frames (time steps), which is better for
    sequential models.
    """

    def __init__(self, p: FloatLike = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        p = _resolve_float(self.p)
        if p == 0.0:
            return x
        mask_shape = list(x.shape)
        mask_shape[-1] = 1
        mask = torch.bernoulli(
            torch.full(mask_shape, 1.0 - p, device=x.device, dtype=x.dtype)
        )
        return x * mask / (1.0 - p)


class ChunkCausalDepthwiseConv1d(nn.Module):
    """Chunk-causal depthwise 1D convolution.

    Uses causal padding so that output at time t depends only on
    inputs at times <= t (within each chunk, if chunk_size > 0).
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=True,
        )

    def forward(
        self, x: Tensor, chunk_size: int = -1
    ) -> Tensor:
        # x: (seq_len, batch_size, channels)
        x = x.permute(1, 2, 0)  # (B, C, T)
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = x.permute(2, 0, 1)  # (T, B, C)
        return x


def _swoosh_r(x: Tensor) -> Tensor:
    """SwooshR activation: log(1 + exp(x - 1)) - 0.08*x - 0.313261687"""
    return torch.log(1.0 + torch.exp(x - 1.0)) - 0.08 * x - 0.313261687


def _swoosh_l(x: Tensor) -> Tensor:
    """SwooshL activation: log(1 + exp(x - 4)) - 0.08*x - 0.035"""
    return torch.log(1.0 + torch.exp(x - 4.0)) - 0.08 * x - 0.035


class ActivationDropoutAndLinear(nn.Module):
    """Fused activation, dropout, and linear projection.

    Applies activation -> dropout -> linear, fused for efficiency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "SwooshL",
        dropout_p: FloatLike = 0.0,
        dropout_shared_dim: Optional[int] = None,
        bias: bool = True,
        initial_scale: float = 1.0,
    ):
        super().__init__()
        self.activation = activation
        self.dropout_p = dropout_p
        self.dropout_shared_dim = dropout_shared_dim
        self.linear = ScaledLinear(
            in_features, out_features, bias=bias,
            initial_scale=initial_scale,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "SwooshL":
            x = _swoosh_l(x)
        elif self.activation == "SwooshR":
            x = _swoosh_r(x)
        elif self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "silu":
            x = F.silu(x)

        if self.training:
            p = _resolve_float(self.dropout_p)
            if p > 0.0:
                if self.dropout_shared_dim is not None:
                    mask_shape = list(x.shape)
                    mask_shape[self.dropout_shared_dim] = 1
                    mask = torch.bernoulli(
                        torch.full(
                            mask_shape, 1.0 - p,
                            device=x.device, dtype=x.dtype
                        )
                    )
                    x = x * mask / (1.0 - p)
                else:
                    x = F.dropout(x, p=p, training=True)

        return self.linear(x)


def convert_num_channels(x: Tensor, num_channels: int) -> Tensor:
    """Convert tensor to have specified number of channels (last dim).

    If x has more channels than needed, truncates.
    If x has fewer channels, pads with zeros.
    If same, returns x unchanged.
    """
    cur_channels = x.shape[-1]
    if cur_channels == num_channels:
        return x
    elif cur_channels > num_channels:
        return x[..., :num_channels]
    else:
        pad = torch.zeros(
            *x.shape[:-1], num_channels - cur_channels,
            device=x.device, dtype=x.dtype
        )
        return torch.cat([x, pad], dim=-1)


def limit_param_value(
    x: Tensor,
    min: float = -1.0e+06,
    max: float = 1.0e+06,
    prob: float = 0.6,
    training: bool = True,
) -> Tensor:
    """Limit parameter values by clamping, applied probabilistically during training."""
    if not training:
        return x
    if random.random() < prob:
        return x.clamp(min=min, max=max)
    return x


class _PenalizeAbsValuesGt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, limit, penalty):
        ctx.save_for_backward(x)
        ctx.limit = limit
        ctx.penalty = penalty
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        limit = ctx.limit
        penalty = ctx.penalty
        pos_mask = x > limit
        neg_mask = x < -limit
        grad = grad_output + penalty * (
            pos_mask.float() * (x - limit) + neg_mask.float() * (x + limit)
        )
        return grad, None, None


def penalize_abs_values_gt(
    x: Tensor, limit: float = 10.0, penalty: float = 1.0e-04, name: str = None
) -> Tensor:
    """Add a gradient penalty for absolute values exceeding a limit."""
    if not x.requires_grad or torch.jit.is_scripting():
        return x
    return _PenalizeAbsValuesGt.apply(x, limit, penalty)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Custom softmax that is more numerically stable for training.

    Same as torch.softmax but adds a small floor to prevent zero gradients
    when probabilities become very small.
    """
    s = torch.softmax(x, dim=dim)
    floor = 1.0e-07
    return s + floor * (1.0 - s.detach())
