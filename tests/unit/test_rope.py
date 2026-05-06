"""Tests for Rotary Position Embeddings."""

import pytest
import torch

from somatic.model.rope import RotaryPositionEmbedding


class TestRotaryPositionEmbedding:
    @pytest.fixture
    def rope(self):
        return RotaryPositionEmbedding(dim=64, max_seq_len=128)

    def test_init(self, rope):
        assert rope.dim == 64
        assert rope.max_seq_len == 128

    def test_even_dim_required(self):
        with pytest.raises(AssertionError):
            RotaryPositionEmbedding(dim=63)

    def test_forward_shape(self, rope):
        batch, heads, seq_len, head_dim = 2, 4, 32, 64
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_cache_extension(self, rope):
        """Test that cache extends automatically for longer sequences."""
        batch, heads, seq_len, head_dim = 2, 4, 200, 64  # Longer than max_seq_len
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert rope.max_seq_len >= seq_len

    def test_with_position_ids(self, rope):
        batch, heads, seq_len, head_dim = 2, 4, 32, 64
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

        q_rot, k_rot = rope(q, k, position_ids=position_ids)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rotation_changes_values(self, rope):
        batch, heads, seq_len, head_dim = 2, 4, 32, 64
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)

        q_rot, k_rot = rope(q, k)

        # Values should be different after rotation
        assert not torch.allclose(q, q_rot)
        assert not torch.allclose(k, k_rot)

    def test_fraction_one_matches_full_rope(self):
        """fraction=1.0 must match the original full-RoPE math bit-for-bit."""
        torch.manual_seed(0)
        head_dim, seq_len = 64, 32
        rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=128, fraction=1.0)
        q = torch.randn(2, 4, seq_len, head_dim)
        k = torch.randn(2, 4, seq_len, head_dim)

        q_out, k_out = rope(q, k)

        cos = rope.cos_cached[:, :, :seq_len, :]
        sin = rope.sin_cached[:, :, :seq_len, :]
        q_ref = (q * cos) + (rope._rotate_half(q) * sin)
        k_ref = (k * cos) + (rope._rotate_half(k) * sin)
        assert torch.equal(q_out, q_ref)
        assert torch.equal(k_out, k_ref)
        assert rope.rotated_dim == head_dim

    def test_fraction_zero_is_identity(self):
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=128, fraction=0.0)
        q = torch.randn(2, 4, 32, 64)
        k = torch.randn(2, 4, 32, 64)
        q_out, k_out = rope(q, k)
        assert torch.equal(q_out, q)
        assert torch.equal(k_out, k)
        assert rope.rotated_dim == 0

    def test_partial_fraction_leaves_tail_untouched(self):
        head_dim = 64
        rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=128, fraction=0.5)
        assert rope.rotated_dim == 32
        q = torch.randn(2, 4, 32, head_dim)
        k = torch.randn(2, 4, 32, head_dim)
        q_out, k_out = rope(q, k)
        assert torch.equal(q_out[..., 32:], q[..., 32:])
        assert torch.equal(k_out[..., 32:], k[..., 32:])
        assert not torch.allclose(q_out[..., :32], q[..., :32])
        assert not torch.allclose(k_out[..., :32], k[..., :32])

    def test_odd_rotated_dim_rounds_down_to_even(self):
        # dim=64 * 0.55 = 35.2 -> int=35 (odd) -> rounded to 34
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=64, fraction=0.55)
        assert rope.rotated_dim == 34
        # Forward must still work and leave dims [34:] untouched.
        q = torch.randn(1, 1, 8, 64)
        k = torch.randn(1, 1, 8, 64)
        q_out, _ = rope(q, k)
        assert q_out.shape == q.shape
        assert torch.equal(q_out[..., 34:], q[..., 34:])

    def test_fraction_validation(self):
        with pytest.raises(ValueError):
            RotaryPositionEmbedding(dim=64, fraction=-0.1)
        with pytest.raises(ValueError):
            RotaryPositionEmbedding(dim=64, fraction=1.1)

    def test_position_ids_with_partial_fraction(self):
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=128, fraction=0.5)
        q = torch.randn(2, 4, 32, 64)
        k = torch.randn(2, 4, 32, 64)
        pos = torch.arange(32).unsqueeze(0).expand(2, -1)
        q_out, k_out = rope(q, k, position_ids=pos)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert torch.equal(q_out[..., 32:], q[..., 32:])
        assert torch.equal(k_out[..., 32:], k[..., 32:])
