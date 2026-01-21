"""Tests for pooling strategies."""

import pytest
import torch

from somatic.encoding import (
    CLSPooling,
    MaxPooling,
    MeanMaxPooling,
    MeanPooling,
    PoolingType,
    create_pooling,
)


class TestPoolingStrategies:
    @pytest.fixture
    def hidden_states(self):
        """Create sample hidden states: (batch=2, seq=10, hidden=64)."""
        return torch.randn(2, 10, 64)

    @pytest.fixture
    def attention_mask(self):
        """Create sample attention mask with variable lengths."""
        mask = torch.ones(2, 10)
        mask[0, 7:] = 0  # First sequence has length 7
        mask[1, 5:] = 0  # Second sequence has length 5
        return mask

    def test_mean_pooling_no_mask(self, hidden_states):
        pooler = MeanPooling()
        output = pooler(hidden_states)

        assert output.shape == (2, 64)
        # Should be the mean over sequence dimension
        expected = hidden_states.mean(dim=1)
        assert torch.allclose(output, expected)

    def test_mean_pooling_with_mask(self, hidden_states, attention_mask):
        pooler = MeanPooling()
        output = pooler(hidden_states, attention_mask)

        assert output.shape == (2, 64)
        # Verify first sample uses only first 7 positions
        expected_0 = hidden_states[0, :7, :].mean(dim=0)
        assert torch.allclose(output[0], expected_0, atol=1e-5)

    def test_cls_pooling(self, hidden_states, attention_mask):
        pooler = CLSPooling()
        output = pooler(hidden_states, attention_mask)

        assert output.shape == (2, 64)
        # Should be the first position
        assert torch.allclose(output, hidden_states[:, 0, :])

    def test_max_pooling_no_mask(self, hidden_states):
        pooler = MaxPooling()
        output = pooler(hidden_states)

        assert output.shape == (2, 64)
        expected = hidden_states.max(dim=1).values
        assert torch.allclose(output, expected)

    def test_max_pooling_with_mask(self, hidden_states, attention_mask):
        pooler = MaxPooling()
        output = pooler(hidden_states, attention_mask)

        assert output.shape == (2, 64)
        # Verify masked positions don't contribute
        # For first sample, max should only consider first 7 positions
        expected_0 = hidden_states[0, :7, :].max(dim=0).values
        assert torch.allclose(output[0], expected_0)

    def test_mean_max_pooling(self, hidden_states, attention_mask):
        pooler = MeanMaxPooling()
        output = pooler(hidden_states, attention_mask)

        # Output should be concatenation of mean and max
        assert output.shape == (2, 128)

        # Verify it's the concatenation
        mean_output = MeanPooling()(hidden_states, attention_mask)
        max_output = MaxPooling()(hidden_states, attention_mask)
        expected = torch.cat([mean_output, max_output], dim=-1)
        assert torch.allclose(output, expected)


class TestCreatePooling:
    def test_create_mean(self):
        pooler = create_pooling("mean")
        assert isinstance(pooler, MeanPooling)

    def test_create_cls(self):
        pooler = create_pooling("cls")
        assert isinstance(pooler, CLSPooling)

    def test_create_max(self):
        pooler = create_pooling("max")
        assert isinstance(pooler, MaxPooling)

    def test_create_mean_max(self):
        pooler = create_pooling("mean_max")
        assert isinstance(pooler, MeanMaxPooling)

    def test_create_from_enum(self):
        pooler = create_pooling(PoolingType.MEAN)
        assert isinstance(pooler, MeanPooling)

    def test_create_case_insensitive(self):
        pooler = create_pooling("MEAN")
        assert isinstance(pooler, MeanPooling)

    def test_create_invalid_raises(self):
        with pytest.raises(ValueError):
            create_pooling("invalid")
