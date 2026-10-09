"""Token-chunked pointwise layers.

The shape DiT MLP expands 1536 channels to 8192 and GELU upcasts that
activation to fp32. On a 4 GB card that spike, on top of a second classifier-free
guidance pass, is what pushes the driver into `device not ready`. The MLP is
pointwise, so slicing the token axis does not change the result.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import torch

from .offload import gpu_total_memory_gb


def mlp_chunk_size(n_tokens: int) -> Optional[int]:
    """Return a token chunk, or None to run the layer in one launch."""
    env = os.environ.get("TRELLIS_MLP_CHUNK", "auto").strip().lower()
    if env in ("0", "off", "none"):
        return None
    if env not in ("", "auto"):
        chunk = int(env)
    else:
        total = gpu_total_memory_gb()
        if not total or total >= 8:
            return None
        chunk = 2048
    if chunk <= 0 or n_tokens <= chunk:
        return None
    return chunk


def chunked_token_apply(feats: torch.Tensor, fn: Callable, chunk: Optional[int]) -> torch.Tensor:
    """Apply a pointwise `fn` along dim 0 in chunks. Exact, just less VRAM."""
    if chunk is None or feats.shape[0] <= chunk:
        return fn(feats)
    parts = []
    for start in range(0, feats.shape[0], chunk):
        parts.append(fn(feats[start:start + chunk].contiguous()))
    return torch.cat(parts, dim=0)
