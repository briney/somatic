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
