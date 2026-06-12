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

    The sin/cos tables are precomputed buffers up to ``max_seq_len`` and
    auto-extended when a longer sequence arrives at inference. All three buffers
    (``inv_freq``, ``cos_cached``, ``sin_cached``) are non-persistent: they are
    derived from ``base``/``rotated_dim`` and never saved. Under HuggingFace's
    meta-device fast init they re-materialize as uninitialized memory, so the
    cache is flagged for rebuild on the first real forward (see
    ``_maybe_build_cache``).

    Args:
        dim: Dimension of each attention head (must be even)
        max_seq_len: Maximum sequence length to precompute
        base: Base for the geometric progression of frequencies
        fraction: Fraction of `dim` to rotate. 1.0 = full RoPE, 0.0 = NoPE,
            values in between = partial RoPE (rotates the first
            `int(dim * fraction)` rounded down to even, leaves the rest
            un-rotated).
    """

    # Registered buffers (declared for static typing; created in __init__).
    inv_freq: Tensor
    cos_cached: Tensor
    sin_cached: Tensor

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
            # NoPE: forward returns inputs unchanged. Register empty buffers so
            # attribute access stays valid regardless of config.
            self.register_buffer("inv_freq", torch.zeros(0), persistent=False)
            self.register_buffer("cos_cached", torch.zeros(1, 1, max_seq_len, 0), persistent=False)
            self.register_buffer("sin_cached", torch.zeros(1, 1, max_seq_len, 0), persistent=False)
            self._cache_initialized = True
            return

        inv_freq = self._compute_inv_freq(device=None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos, sin = self._compute_cos_sin(max_seq_len, device=inv_freq.device)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        # False under meta-device fast init (buffers are not real yet); the first
        # real forward rebuilds them. A normal CPU/GPU construction is valid now.
        self._cache_initialized = not self.cos_cached.is_meta

    def _compute_inv_freq(self, device: torch.device | None) -> Tensor:
        """Derive ``inv_freq`` from ``base``/``rotated_dim`` (plain attributes).

        Recomputed rather than read from the buffer so a meta-device-corrupted
        buffer can never leak into the rotation tables.
        """
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotated_dim, 2, dtype=torch.float32, device=device)
                / self.rotated_dim
            )
        )

    def _compute_cos_sin(self, seq_len: int, device: torch.device | None) -> tuple[Tensor, Tensor]:
        """Compute cos/sin caches of shape ``(1, 1, seq_len, rotated_dim)``."""
        inv_freq = self._compute_inv_freq(device=device)
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

    def _maybe_build_cache(self, seq_len: int, device: torch.device) -> None:
        """Rebuild the cos/sin tables when needed.

        Rebuilds when the cache has not been built on a real device (post
        meta-init load), when ``seq_len`` exceeds the cached length, or when the
        cache lives on a different device than the incoming tensors.
        """
        cached_len = self.cos_cached.shape[2]
        same_device = self.cos_cached.device == device
        if self._cache_initialized and seq_len <= cached_len and same_device:
            return
        new_len = max(seq_len, cached_len)
        cos, sin = self._compute_cos_sin(new_len, device=device)
        self.inv_freq = self._compute_inv_freq(device=device)
        self.cos_cached = cos
        self.sin_cached = sin
        self.max_seq_len = new_len
        self._cache_initialized = True

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
        self._maybe_build_cache(seq_len, device=q.device)

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
