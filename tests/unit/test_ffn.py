"""Tests for SwiGLU Feed-Forward Networks."""

import pytest
import torch

from somatic.model.ffn import FusedSwiGLUFFN


class TestFusedSwiGLUFFN:
    @pytest.fixture
    def ffn(self):
        return FusedSwiGLUFFN(d_model=64, d_ffn=128, dropout=0.0)

    def test_forward_shape(self, ffn):
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_d_ffn_required(self):
        """d_ffn is required."""
        with pytest.raises(TypeError):
            FusedSwiGLUFFN(d_model=64)

    def test_custom_d_ffn(self):
        ffn = FusedSwiGLUFFN(d_model=64, d_ffn=256)
        assert ffn.d_ffn == 256

    def test_gradient_flow(self):
        """Test that gradients flow through the FFN."""
        ffn = FusedSwiGLUFFN(d_model=64, d_ffn=128, dropout=0.0)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
