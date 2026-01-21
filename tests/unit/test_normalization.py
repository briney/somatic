"""Tests for normalization modules."""

import pytest
import torch

from somatic.model.normalization import (
    LearnedQKScale,
    QKNormModule,
    RMSNorm,
    create_norm_layer,
    create_qk_norm,
)


class TestRMSNorm:
    def test_forward_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 32, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_forward_single_batch(self):
        norm = RMSNorm(128)
        x = torch.randn(1, 16, 128)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization_effect(self):
        """RMSNorm should produce output with roughly unit RMS."""
        norm = RMSNorm(64)
        x = torch.randn(2, 32, 64) * 10  # Large scale input
        out = norm(x)
        # RMS of output should be close to 1 (due to learned weight initialized to 1)
        rms = torch.sqrt(torch.mean(out**2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.5)

    def test_gradient_flow(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 32, 64, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert norm.weight.grad is not None

    def test_weight_initialization(self):
        norm = RMSNorm(64)
        assert torch.allclose(norm.weight, torch.ones(64))

    def test_custom_eps(self):
        norm = RMSNorm(64, eps=1e-5)
        assert norm.eps == 1e-5


class TestLearnedQKScale:
    def test_forward_shape(self):
        scale = LearnedQKScale(n_heads=4, head_dim=16)
        q = torch.randn(2, 4, 32, 16)
        k = torch.randn(2, 4, 32, 16)
        q_out, k_out = scale(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_initial_identity(self):
        """Initially should be identity (scales = 1)."""
        scale = LearnedQKScale(n_heads=4, head_dim=16)
        q = torch.randn(2, 4, 32, 16)
        k = torch.randn(2, 4, 32, 16)
        q_out, k_out = scale(q, k)
        assert torch.allclose(q_out, q)
        assert torch.allclose(k_out, k)

    def test_gradient_flow(self):
        scale = LearnedQKScale(n_heads=4, head_dim=16)
        q = torch.randn(2, 4, 32, 16, requires_grad=True)
        k = torch.randn(2, 4, 32, 16, requires_grad=True)
        q_out, k_out = scale(q, k)
        loss = (q_out.sum() + k_out.sum())
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None
        assert scale.q_scale.grad is not None
        assert scale.k_scale.grad is not None

    def test_per_head_scaling(self):
        """Test that scales are applied per head."""
        scale = LearnedQKScale(n_heads=4, head_dim=16)
        # Modify scale for first head
        with torch.no_grad():
            scale.q_scale[0] = 2.0
            scale.k_scale[0] = 0.5

        q = torch.ones(1, 4, 8, 16)
        k = torch.ones(1, 4, 8, 16)
        q_out, k_out = scale(q, k)

        # First head should be scaled
        assert torch.allclose(q_out[0, 0], torch.ones(8, 16) * 2.0)
        assert torch.allclose(k_out[0, 0], torch.ones(8, 16) * 0.5)
        # Other heads should be unchanged
        assert torch.allclose(q_out[0, 1], torch.ones(8, 16))


class TestQKNormModule:
    def test_forward_shape_layernorm(self):
        qk_norm = QKNormModule("layernorm", head_dim=16)
        q = torch.randn(2, 4, 32, 16)
        k = torch.randn(2, 4, 32, 16)
        q_out, k_out = qk_norm(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_forward_shape_rmsnorm(self):
        qk_norm = QKNormModule("rmsnorm", head_dim=16)
        q = torch.randn(2, 4, 32, 16)
        k = torch.randn(2, 4, 32, 16)
        q_out, k_out = qk_norm(q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_separate_norms(self):
        """Test that Q and K have separate norm layers."""
        qk_norm = QKNormModule("layernorm", head_dim=16)
        assert qk_norm.q_norm is not qk_norm.k_norm
        # Check they have separate parameters
        assert qk_norm.q_norm.weight is not qk_norm.k_norm.weight


class TestCreateNormLayer:
    def test_layernorm(self):
        norm = create_norm_layer("layernorm", 64)
        assert isinstance(norm, torch.nn.LayerNorm)
        assert norm.normalized_shape == (64,)

    def test_rmsnorm(self):
        norm = create_norm_layer("rmsnorm", 64)
        assert isinstance(norm, RMSNorm)

    def test_custom_eps(self):
        norm = create_norm_layer("layernorm", 64, eps=1e-5)
        assert norm.eps == 1e-5

        norm = create_norm_layer("rmsnorm", 64, eps=1e-5)
        assert norm.eps == 1e-5

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown norm_type"):
            create_norm_layer("invalid", 64)


class TestCreateQKNorm:
    def test_none(self):
        qk_norm = create_qk_norm("none", "layernorm", 4, 16)
        assert qk_norm is None

    def test_norm_layernorm(self):
        qk_norm = create_qk_norm("norm", "layernorm", 4, 16)
        assert isinstance(qk_norm, QKNormModule)
        assert isinstance(qk_norm.q_norm, torch.nn.LayerNorm)
        assert isinstance(qk_norm.k_norm, torch.nn.LayerNorm)

    def test_norm_rmsnorm(self):
        qk_norm = create_qk_norm("norm", "rmsnorm", 4, 16)
        assert isinstance(qk_norm, QKNormModule)
        assert isinstance(qk_norm.q_norm, RMSNorm)
        assert isinstance(qk_norm.k_norm, RMSNorm)

    def test_learned_scale(self):
        qk_norm = create_qk_norm("learned_scale", "layernorm", 4, 16)
        assert isinstance(qk_norm, LearnedQKScale)
        assert qk_norm.n_heads == 4
        assert qk_norm.head_dim == 16

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown qk_norm_type"):
            create_qk_norm("invalid", "layernorm", 4, 16)
