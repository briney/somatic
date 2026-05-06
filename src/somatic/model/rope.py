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
        fraction: Fraction of `dim` to rotate. 1.0 = full RoPE, 0.0 = NoPE,
            values in between = partial RoPE (rotates the first
            `int(dim * fraction)` rounded down to even, leaves the rest
            un-rotated).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 512,
        base: float = 10000.0,
        fraction: float = 1.0,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"RoPE fraction must be in [0.0, 1.0], got {fraction}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.fraction = fraction

        # Round down to nearest even — _rotate_half uses chunk(2) which needs
        # an even-sized last dimension.
        rotated_dim = int(dim * fraction)
        rotated_dim -= rotated_dim % 2
        self.rotated_dim = rotated_dim

        if rotated_dim == 0:
            # NoPE: no cache needed, forward returns inputs unchanged.
            self.register_buffer("inv_freq", torch.zeros(0), persistent=False)
        else:
            # Precompute frequency bands sized to rotated_dim.
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotated_dim, 2).float() / rotated_dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
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
        # NoPE fast path
        if self.rotated_dim == 0:
            return q, k

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

        # Partial rotation. When rotated_dim == dim (full RoPE), q_pass/k_pass
        # are zero-width and the concat is a no-op.
        q_rot, q_pass = q[..., : self.rotated_dim], q[..., self.rotated_dim :]
        k_rot, k_pass = k[..., : self.rotated_dim], k[..., self.rotated_dim :]

        q_rot = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

        return (
            torch.cat([q_rot, q_pass], dim=-1),
            torch.cat([k_rot, k_pass], dim=-1),
        )
