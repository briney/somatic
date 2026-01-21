"""Tests for embedding layers."""

import math

import pytest
import torch

from somatic.model.embeddings import SomaticEmbedding, TokenEmbedding


class TestTokenEmbedding:
    @pytest.fixture
    def embedding(self):
        return TokenEmbedding(vocab_size=32, d_model=64, padding_idx=1)

    def test_forward_shape(self, embedding):
        token_ids = torch.randint(0, 32, (2, 10))
        out = embedding(token_ids)
        assert out.shape == (2, 10, 64)

    def test_scaling(self):
        emb_scaled = TokenEmbedding(vocab_size=32, d_model=64, scale=True)
        emb_unscaled = TokenEmbedding(vocab_size=32, d_model=64, scale=False)

        # Make them share the same weights
        emb_unscaled.embedding.weight.data = emb_scaled.embedding.weight.data.clone()

        # Use non-padding tokens to avoid zeros
        token_ids = torch.randint(2, 30, (2, 10))
        out_scaled = emb_scaled(token_ids)
        out_unscaled = emb_unscaled(token_ids)

        # Check that scaled output is sqrt(d_model) times the unscaled output
        expected_ratio = math.sqrt(64)
        assert torch.allclose(out_scaled, out_unscaled * expected_ratio)

    def test_padding_idx(self, embedding):
        token_ids = torch.tensor([[1, 1, 1]])  # All padding
        out = embedding(token_ids)
        assert torch.allclose(out, torch.zeros_like(out))


class TestSomaticEmbedding:
    def test_basic_forward(self):
        embedding = SomaticEmbedding(
            vocab_size=32,
            d_model=64,
        )
        token_ids = torch.randint(0, 32, (2, 10))
        out = embedding(token_ids)
        assert out.shape == (2, 10, 64)

    def test_with_dropout(self):
        embedding = SomaticEmbedding(
            vocab_size=32,
            d_model=64,
            dropout=0.1,
        )
        token_ids = torch.randint(0, 32, (2, 10))
        out = embedding(token_ids)
        assert out.shape == (2, 10, 64)

    def test_padding_idx(self):
        embedding = SomaticEmbedding(
            vocab_size=32,
            d_model=64,
            padding_idx=1,
        )
        token_ids = torch.tensor([[1, 1, 1]])  # All padding
        out = embedding(token_ids)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_d_model_attribute(self):
        embedding = SomaticEmbedding(
            vocab_size=32,
            d_model=128,
        )
        assert embedding.d_model == 128
