"""Tests for attention modules."""

import pytest
import torch

from somatic.model.attention import ChainAwareAttention, MultiHeadAttention
from somatic.model.normalization import LearnedQKScale, QKNormModule


class TestMultiHeadAttention:
    @pytest.fixture
    def attention(self):
        return MultiHeadAttention(
            d_model=64, n_heads=4, head_dim=16, dropout=0.0, max_seq_len=128
        )

    def test_forward_shape(self, attention):
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()  # Ignored but required

        out = attention(x, chain_ids)
        assert out.shape == x.shape

    def test_with_attention_mask(self, attention):
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()
        attention_mask = torch.ones(batch, seq_len)
        attention_mask[:, -5:] = 0  # Mask last 5 positions

        out = attention(x, chain_ids, attention_mask=attention_mask)
        assert out.shape == x.shape

    def test_deterministic_without_dropout(self, attention):
        """Test that output is deterministic without dropout."""
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        attention.eval()
        out1 = attention(x, chain_ids)
        out2 = attention(x, chain_ids)

        assert torch.allclose(out1, out2)

    def test_need_weights_returns_attention(self, attention):
        """Test that need_weights=True returns attention weights."""
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        out, attn_weights = attention(x, chain_ids, need_weights=True)

        assert out.shape == x.shape
        n_heads = 4
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self, attention):
        """Test that attention weights sum to 1 for each query position."""
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        attention.eval()
        _, attn_weights = attention(x, chain_ids, need_weights=True)

        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_sdpa_and_manual_consistency(self, attention):
        """Test that SDPA and manual attention produce similar results."""
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        attention.eval()

        # SDPA path (need_weights=False)
        out_sdpa = attention(x, chain_ids, need_weights=False)

        # Manual path (need_weights=True)
        out_manual, _ = attention(x, chain_ids, need_weights=True)

        # Should be very close
        assert torch.allclose(out_sdpa, out_manual, atol=1e-5)


class TestChainAwareAttention:
    @pytest.fixture
    def attention(self):
        return ChainAwareAttention(
            d_model=64, n_heads=4, head_dim=16, dropout=0.0, max_seq_len=128
        )

    def test_forward_shape(self, attention):
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.cat(
            [torch.zeros(batch, seq_len // 2), torch.ones(batch, seq_len // 2)], dim=1
        ).long()

        out = attention(x, chain_ids)
        assert out.shape == x.shape

    def test_with_attention_mask(self, attention):
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()
        attention_mask = torch.ones(batch, seq_len)
        attention_mask[:, -5:] = 0  # Mask last 5 positions

        out = attention(x, chain_ids, attention_mask=attention_mask)
        assert out.shape == x.shape

    def test_single_chain(self, attention):
        """Test with single chain (all same chain_id)."""
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        out = attention(x, chain_ids)
        assert out.shape == x.shape

    def test_multiple_chains(self, attention):
        """Test with multiple chains."""
        batch, seq_len, d_model = 2, 30, 64
        x = torch.randn(batch, seq_len, d_model)
        # Three chains of 10 tokens each
        chain_ids = torch.cat(
            [
                torch.zeros(batch, 10),
                torch.ones(batch, 10),
                torch.full((batch, 10), 2),
            ],
            dim=1,
        ).long()

        out = attention(x, chain_ids)
        assert out.shape == x.shape

    def test_deterministic_without_dropout(self, attention):
        """Test that output is deterministic without dropout."""
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        attention.eval()
        out1 = attention(x, chain_ids)
        out2 = attention(x, chain_ids)

        assert torch.allclose(out1, out2)

    def test_need_weights_returns_attention(self, attention):
        """Test that need_weights=True returns attention weights."""
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.cat(
            [torch.zeros(batch, seq_len // 2), torch.ones(batch, seq_len // 2)], dim=1
        ).long()

        out, attn_weights = attention(x, chain_ids, need_weights=True)

        assert out.shape == x.shape
        n_heads = 4
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self, attention):
        """Test that attention weights sum to 1 for each query position."""
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.cat(
            [torch.zeros(batch, seq_len // 2), torch.ones(batch, seq_len // 2)], dim=1
        ).long()

        attention.eval()
        _, attn_weights = attention(x, chain_ids, need_weights=True)

        # Attention weights should sum to 1 for each query position
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestMultiHeadAttentionQKNorm:
    """Tests for MultiHeadAttention with QK normalization."""

    def test_qk_norm_none(self):
        """Test that qk_norm='none' produces no QK norm module."""
        attn = MultiHeadAttention(
            d_model=64, n_heads=4, head_dim=16, qk_norm="none"
        )
        assert attn.qk_norm_module is None

    def test_qk_norm_layernorm(self):
        """Test forward pass with qk_norm='norm' and LayerNorm."""
        attn = MultiHeadAttention(
            d_model=64, n_heads=4, head_dim=16,
            qk_norm="norm", norm_type="layernorm"
        )
        assert isinstance(attn.qk_norm_module, QKNormModule)

        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        out = attn(x, chain_ids)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_qk_norm_rmsnorm(self):
        """Test forward pass with qk_norm='norm' and RMSNorm."""
        attn = MultiHeadAttention(
            d_model=64, n_heads=4, head_dim=16,
            qk_norm="norm", norm_type="rmsnorm"
        )
        assert isinstance(attn.qk_norm_module, QKNormModule)

        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        out = attn(x, chain_ids)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_qk_norm_learned_scale(self):
        """Test forward pass with qk_norm='learned_scale'."""
        attn = MultiHeadAttention(
            d_model=64, n_heads=4, head_dim=16,
            qk_norm="learned_scale"
        )
        assert isinstance(attn.qk_norm_module, LearnedQKScale)

        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        out = attn(x, chain_ids)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_qk_norm_with_attention_weights(self):
        """Test that QK norm works with attention weight output."""
        attn = MultiHeadAttention(
            d_model=64, n_heads=4, head_dim=16,
            qk_norm="norm", norm_type="layernorm"
        )

        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        attn.eval()
        out, attn_weights = attn(x, chain_ids, need_weights=True)

        assert out.shape == x.shape
        assert attn_weights.shape == (batch, 4, seq_len, seq_len)
        # Weights should still sum to 1
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestChainAwareAttentionQKNorm:
    """Tests for ChainAwareAttention with QK normalization."""

    def test_qk_norm_none(self):
        """Test that qk_norm='none' produces no QK norm modules."""
        attn = ChainAwareAttention(
            d_model=64, n_heads=4, head_dim=16, qk_norm="none"
        )
        assert attn.qk_norm_self is None
        assert attn.qk_norm_cross is None

    def test_qk_norm_creates_separate_modules(self):
        """Test that ChainAwareAttention creates separate QK norm for self/cross."""
        attn = ChainAwareAttention(
            d_model=64, n_heads=4, head_dim=16,
            qk_norm="norm", norm_type="layernorm"
        )
        assert attn.qk_norm_self is not None
        assert attn.qk_norm_cross is not None
        # Should be separate instances
        assert attn.qk_norm_self is not attn.qk_norm_cross

    def test_qk_norm_layernorm(self):
        """Test forward pass with qk_norm='norm' and LayerNorm."""
        attn = ChainAwareAttention(
            d_model=64, n_heads=4, head_dim=16,
            qk_norm="norm", norm_type="layernorm"
        )

        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.cat(
            [torch.zeros(batch, seq_len // 2), torch.ones(batch, seq_len // 2)],
            dim=1,
        ).long()

        out = attn(x, chain_ids)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_qk_norm_rmsnorm(self):
        """Test forward pass with qk_norm='norm' and RMSNorm."""
        attn = ChainAwareAttention(
            d_model=64, n_heads=4, head_dim=16,
            qk_norm="norm", norm_type="rmsnorm"
        )

        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.cat(
            [torch.zeros(batch, seq_len // 2), torch.ones(batch, seq_len // 2)],
            dim=1,
        ).long()

        out = attn(x, chain_ids)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_qk_norm_learned_scale(self):
        """Test forward pass with qk_norm='learned_scale'."""
        attn = ChainAwareAttention(
            d_model=64, n_heads=4, head_dim=16,
            qk_norm="learned_scale"
        )
        assert isinstance(attn.qk_norm_self, LearnedQKScale)
        assert isinstance(attn.qk_norm_cross, LearnedQKScale)

        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.cat(
            [torch.zeros(batch, seq_len // 2), torch.ones(batch, seq_len // 2)],
            dim=1,
        ).long()

        out = attn(x, chain_ids)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_qk_norm_with_attention_weights(self):
        """Test that QK norm works with attention weight output."""
        attn = ChainAwareAttention(
            d_model=64, n_heads=4, head_dim=16,
            qk_norm="norm", norm_type="layernorm"
        )

        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.cat(
            [torch.zeros(batch, seq_len // 2), torch.ones(batch, seq_len // 2)],
            dim=1,
        ).long()

        attn.eval()
        out, attn_weights = attn(x, chain_ids, need_weights=True)

        assert out.shape == x.shape
        assert attn_weights.shape == (batch, 4, seq_len, seq_len)
        # Weights should still sum to 1
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
