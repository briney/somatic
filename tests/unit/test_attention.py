"""Tests for attention modules."""

import pytest
import torch

from somatic.model.attention import (
    ChainAwareAttention,
    MultiHeadAttention,
    SharedQKVChainAwareAttention,
)
from somatic.model.layers import TransformerBlock
from somatic.model.normalization import LearnedQKScale, QKNormModule, QKVNormModule
from somatic.model.transformer import SomaticConfig, SomaticModel


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


class TestMultiHeadAttentionHybridNorm:
    """Tests for MultiHeadAttention with HybridNorm (QKV-norm in attention)."""

    def test_hybrid_norm_disables_qk_norm(self):
        attn = MultiHeadAttention(
            d_model=64, n_heads=4, head_dim=16,
            qk_norm="norm", hybrid_norm=True,
        )
        assert attn.qk_norm_module is None
        assert isinstance(attn.qkv_norm, QKVNormModule)

    @pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
    def test_hybrid_norm_forward(self, norm_type):
        attn = MultiHeadAttention(
            d_model=64, n_heads=4, head_dim=16,
            hybrid_norm=True, norm_type=norm_type,
        )
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        out = attn(x, chain_ids)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_hybrid_norm_with_attention_weights(self):
        attn = MultiHeadAttention(
            d_model=64, n_heads=4, head_dim=16, hybrid_norm=True,
        )
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = torch.zeros(batch, seq_len).long()

        attn.eval()
        out, attn_weights = attn(x, chain_ids, need_weights=True)

        assert out.shape == x.shape
        assert attn_weights.shape == (batch, 4, seq_len, seq_len)
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestChainAwareAttentionHybridNorm:
    """Tests for ChainAwareAttention with HybridNorm (separate QKV-norm per path)."""

    def test_hybrid_norm_creates_separate_qkv_modules(self):
        attn = ChainAwareAttention(
            d_model=64, n_heads=4, head_dim=16, hybrid_norm=True,
        )
        # No QK-norm modules in hybrid mode
        assert attn.qk_norm_self is None
        assert attn.qk_norm_cross is None
        # Two separate QKV-norm modules (6 underlying norm layers total)
        assert isinstance(attn.qkv_norm_self, QKVNormModule)
        assert isinstance(attn.qkv_norm_cross, QKVNormModule)
        assert attn.qkv_norm_self is not attn.qkv_norm_cross

    @pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
    def test_hybrid_norm_forward(self, norm_type):
        attn = ChainAwareAttention(
            d_model=64, n_heads=4, head_dim=16,
            hybrid_norm=True, norm_type=norm_type,
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

    def test_hybrid_norm_with_attention_weights(self):
        attn = ChainAwareAttention(
            d_model=64, n_heads=4, head_dim=16, hybrid_norm=True,
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
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestRopeFractionPropagation:
    def test_rope_fraction_reaches_every_block(self):
        """SomaticConfig.rope_fraction must propagate to every block's RoPE."""
        config = SomaticConfig(
            d_model=32, n_layers=2, n_heads=2, max_seq_len=16, rope_fraction=0.5
        )
        model = SomaticModel(config)
        for block in model.encoder.layers:
            assert block.attention.rope.fraction == 0.5
            # head_dim = 32 / 2 = 16, fraction=0.5 -> rotated_dim=8
            assert block.attention.rope.rotated_dim == 8

    def test_default_rope_fraction_is_full_rope(self):
        """Default rope_fraction=1.0 -> rotated_dim == head_dim on every block."""
        config = SomaticConfig(d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
        model = SomaticModel(config)
        for block in model.encoder.layers:
            assert block.attention.rope.fraction == 1.0
            assert block.attention.rope.rotated_dim == 16

    def test_partial_rope_forward_backward_chain_aware(self):
        """Tiny ChainAware model with rope_fraction=0.5 runs forward+backward."""
        config = SomaticConfig(
            d_model=32, n_layers=2, n_heads=2, max_seq_len=16, rope_fraction=0.5
        )
        model = SomaticModel(config)
        token_ids = torch.randint(0, config.vocab_size, (2, 8))
        chain_ids = torch.zeros(2, 8, dtype=torch.long)
        out = model(token_ids, chain_ids)
        out["logits"].sum().backward()

    def test_nope_forward_backward_multihead(self):
        """Standard MultiHeadAttention with rope_fraction=0.0 (NoPE) runs."""
        config = SomaticConfig(
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=16,
            rope_fraction=0.0,
            use_chain_aware_attention=False,
        )
        model = SomaticModel(config)
        for block in model.encoder.layers:
            assert block.attention.rope.rotated_dim == 0
        token_ids = torch.randint(0, config.vocab_size, (2, 8))
        chain_ids = torch.zeros(2, 8, dtype=torch.long)
        out = model(token_ids, chain_ids)
        out["logits"].sum().backward()


class TestSharedQKVChainAwareAttention:
    """Forward shape, masking, and weight tests for SharedQKVChainAwareAttention."""

    @pytest.fixture
    def attention(self):
        return SharedQKVChainAwareAttention(
            d_model=64, n_heads=4, head_dim=16, dropout=0.0, max_seq_len=128
        )

    def _two_chain_ids(self, batch: int, seq_len: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros(batch, seq_len // 2),
                torch.ones(batch, seq_len // 2),
            ],
            dim=1,
        ).long()

    def test_forward_shape(self, attention):
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = self._two_chain_ids(batch, seq_len)

        out = attention(x, chain_ids)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_padding_mask_no_nans(self, attention):
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = self._two_chain_ids(batch, seq_len)
        attention_mask = torch.ones(batch, seq_len)
        attention_mask[:, -5:] = 0

        out = attention(x, chain_ids, attention_mask=attention_mask)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_three_chains(self, attention):
        batch, seq_len, d_model = 2, 30, 64
        x = torch.randn(batch, seq_len, d_model)
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

    def test_attention_weights_shape_and_rows(self, attention):
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = self._two_chain_ids(batch, seq_len)
        attention.eval()
        out, attn_weights = attention(x, chain_ids, need_weights=True)
        assert out.shape == x.shape
        assert attn_weights.shape == (batch, 4, seq_len, seq_len)
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_deterministic_without_dropout(self, attention):
        batch, seq_len, d_model = 2, 32, 64
        x = torch.randn(batch, seq_len, d_model)
        chain_ids = self._two_chain_ids(batch, seq_len)
        attention.eval()
        out1 = attention(x, chain_ids)
        out2 = attention(x, chain_ids)
        assert torch.allclose(out1, out2)

    def test_hybrid_norm_rejected(self):
        with pytest.raises(ValueError, match="HybridNorm"):
            SharedQKVChainAwareAttention(
                d_model=64, n_heads=4, head_dim=16, hybrid_norm=True,
            )


class TestSharedQKVEquivalence:
    """Weight-tying equivalence between MHA and SharedQKVChainAwareAttention."""

    def _copy_weights(
        self, src: MultiHeadAttention, dst: SharedQKVChainAwareAttention
    ) -> None:
        dst.q_proj.weight.data.copy_(src.q_proj.weight.data)
        dst.k_proj.weight.data.copy_(src.k_proj.weight.data)
        dst.v_proj.weight.data.copy_(src.v_proj.weight.data)
        dst.out_proj.weight.data.copy_(src.out_proj.weight.data)

    def test_same_chain_equivalence(self):
        """All-same-chain inputs should match standard MHA exactly."""
        kw = dict(d_model=64, n_heads=4, head_dim=16, dropout=0.0, max_seq_len=128)
        mha = MultiHeadAttention(**kw)
        shared = SharedQKVChainAwareAttention(**kw)
        self._copy_weights(mha, shared)

        mha.eval()
        shared.eval()
        torch.manual_seed(0)
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, kw["d_model"])
        chain_ids = torch.zeros(batch, seq_len, dtype=torch.long)
        # Use need_weights=True for both to keep manual-vs-SDPA consistency
        out_mha, _ = mha(x, chain_ids, need_weights=True)
        out_shared, _ = shared(x, chain_ids, need_weights=True)
        assert torch.allclose(out_mha, out_shared, atol=1e-5)

    def test_nope_equivalence_with_mixed_chains(self):
        """rope_fraction=0.0 makes intra/inter scores identical."""
        kw = dict(
            d_model=64, n_heads=4, head_dim=16,
            dropout=0.0, max_seq_len=128, rope_fraction=0.0,
        )
        mha = MultiHeadAttention(**kw)
        shared = SharedQKVChainAwareAttention(**kw)
        self._copy_weights(mha, shared)

        mha.eval()
        shared.eval()
        torch.manual_seed(0)
        batch, seq_len = 2, 32
        x = torch.randn(batch, seq_len, kw["d_model"])
        chain_ids = torch.cat(
            [torch.zeros(batch, seq_len // 2), torch.ones(batch, seq_len // 2)],
            dim=1,
        ).long()
        out_mha, _ = mha(x, chain_ids, need_weights=True)
        out_shared, _ = shared(x, chain_ids, need_weights=True)
        assert torch.allclose(out_mha, out_shared, atol=1e-5)


class TestAttentionDispatch:
    """TransformerBlock dispatches the right attention class based on config."""

    def test_default_is_separate_chain_aware(self):
        block = TransformerBlock(d_model=32, n_heads=2, head_dim=16, d_ffn=64)
        assert isinstance(block.attention, ChainAwareAttention)

    def test_use_chain_aware_false_is_mha(self):
        block = TransformerBlock(
            d_model=32,
            n_heads=2,
            head_dim=16,
            d_ffn=64,
            use_chain_aware_attention=False,
        )
        assert isinstance(block.attention, MultiHeadAttention)

    def test_shared_mode_dispatch(self):
        block = TransformerBlock(
            d_model=32,
            n_heads=2,
            head_dim=16,
            d_ffn=64,
            use_chain_aware_attention=True,
            chain_aware_projection_mode="shared",
        )
        assert isinstance(block.attention, SharedQKVChainAwareAttention)

    def test_invalid_projection_mode_raises_at_block(self):
        with pytest.raises(ValueError, match="chain_aware_projection_mode"):
            TransformerBlock(
                d_model=32,
                n_heads=2,
                head_dim=16,
                d_ffn=64,
                use_chain_aware_attention=True,
                chain_aware_projection_mode="bogus",
            )


class TestSharedQKVConfig:
    """SomaticConfig validation and end-to-end dispatch via SomaticModel."""

    def test_invalid_projection_mode_raises(self):
        with pytest.raises(ValueError, match="chain_aware_projection_mode"):
            SomaticConfig(chain_aware_projection_mode="bogus")

    def test_shared_with_hybrid_norm_raises(self):
        with pytest.raises(ValueError, match="shared"):
            SomaticConfig(
                chain_aware_projection_mode="shared", hybrid_norm="standard"
            )

    def test_shared_without_chain_aware_does_not_raise(self):
        # Mode is ignored when chain-aware attention is disabled.
        config = SomaticConfig(
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=16,
            use_chain_aware_attention=False,
            chain_aware_projection_mode="shared",
        )
        model = SomaticModel(config)
        for block in model.encoder.layers:
            assert isinstance(block.attention, MultiHeadAttention)

    def test_shared_model_blocks_are_shared(self):
        config = SomaticConfig(
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=16,
            chain_aware_projection_mode="shared",
        )
        model = SomaticModel(config)
        for block in model.encoder.layers:
            assert isinstance(block.attention, SharedQKVChainAwareAttention)

    def test_legacy_config_dict_missing_projection_mode(self):
        # Old serialized configs without the new field default to "separate".
        d = {"vocab_size": 32, "d_model": 64, "n_layers": 2, "n_heads": 2}
        config = SomaticConfig(**d)
        assert config.chain_aware_projection_mode == "separate"

    def test_save_load_roundtrip_shared(self, tmp_path):
        config = SomaticConfig(
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=16,
            dropout=0.0,
            attention_dropout=0.0,
            embedding_dropout=0.0,
            chain_aware_projection_mode="shared",
        )
        model = SomaticModel(config)
        save_path = tmp_path / "shared_model.pt"
        model.save_pretrained(str(save_path))

        loaded = SomaticModel.from_pretrained(str(save_path))
        assert loaded.config.chain_aware_projection_mode == "shared"
        for block in loaded.encoder.layers:
            assert isinstance(block.attention, SharedQKVChainAwareAttention)

        model.eval()
        loaded.eval()
        token_ids = torch.randint(0, config.vocab_size, (2, 8))
        chain_ids = torch.zeros(2, 8, dtype=torch.long)
        with torch.no_grad():
            out_a = model(token_ids, chain_ids)["logits"]
            out_b = loaded(token_ids, chain_ids)["logits"]
        assert torch.allclose(out_a, out_b)


class TestAttentionTrainingSmoke:
    """Tiny forward+backward+step on each attention mode."""

    @pytest.mark.parametrize(
        "config_kwargs",
        [
            dict(use_chain_aware_attention=False),
            dict(
                use_chain_aware_attention=True,
                chain_aware_projection_mode="separate",
            ),
            dict(
                use_chain_aware_attention=True,
                chain_aware_projection_mode="shared",
            ),
        ],
    )
    def test_forward_backward_step(self, config_kwargs):
        config = SomaticConfig(
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=16,
            dropout=0.0,
            attention_dropout=0.0,
            embedding_dropout=0.0,
            **config_kwargs,
        )
        model = SomaticModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        token_ids = torch.randint(0, config.vocab_size, (2, 8))
        chain_ids = torch.cat(
            [torch.zeros(2, 4), torch.ones(2, 4)], dim=1
        ).long()
        targets = torch.randint(0, config.vocab_size, (2, 8))

        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            logits = model(token_ids, chain_ids)["logits"]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, config.vocab_size), targets.reshape(-1)
            )
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()

