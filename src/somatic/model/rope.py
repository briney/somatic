"""Rotary Position Embeddings (RoPE) implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformer attention.

    RoPE encodes position information by rotating query and key vectors
    in 2D subspaces, enabling relative position awareness without
    explicit position embeddings in the input.

    Args:
        dim: Dimension of each attention head (must be even)
        max_seq_len: Maximum sequence length to precompute
        base: Base for the geometric progression of frequencies
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 512,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute sin/cos cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build sin/cos cache for given sequence length."""
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)

        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, heads, seq_len, head_dim)
            k: Key tensor of shape (batch, heads, seq_len, head_dim)
            position_ids: Optional position indices of shape (batch, seq_len)

        Returns:
            Tuple of rotated (query, key) tensors
        """
        seq_len = q.shape[2]

        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        if position_ids is None:
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
        else:
            cos = self.cos_cached.squeeze(0).squeeze(0)
            sin = self.sin_cached.squeeze(0).squeeze(0)
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed
