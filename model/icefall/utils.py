"""Minimal icefall.utils — only what zipformer needs."""

import torch
from contextlib import contextmanager


@contextmanager
def torch_autocast(enabled: bool = True, dtype=torch.float16, cache_enabled: bool = True):
    """Context manager for torch autocast, compatible with icefall Zipformer2 usage."""
    if enabled:
        with torch.amp.autocast("cuda", dtype=dtype, cache_enabled=cache_enabled):
            yield
    else:
        yield
