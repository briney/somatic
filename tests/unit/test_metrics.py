"""Tests for training metrics."""

import pytest
import torch

from somatic.training import (
    MLMMetrics,
    MetricAccumulator,
    compute_accuracy,
    compute_mlm_metrics,
    compute_masked_cross_entropy,
    compute_perplexity,
)


class TestMetricAccumulator:
    def test_update_and_compute(self):
        acc = MetricAccumulator()
        acc.update("loss", 1.0)
        acc.update("loss", 2.0)
        acc.update("loss", 3.0)

        result = acc.compute("loss")
        assert result == pytest.approx(2.0)  # (1+2+3)/3

    def test_update_with_count(self):
        acc = MetricAccumulator()
        acc.update("loss", 1.0, count=2)
        acc.update("loss", 4.0, count=1)

        result = acc.compute("loss")
        assert result == pytest.approx(2.0)  # (1*2 + 4*1) / 3 = 6/3

    def test_compute_nonexistent(self):
        acc = MetricAccumulator()
        assert acc.compute("nonexistent") is None

    def test_compute_all(self):
        acc = MetricAccumulator()
        acc.update("loss", 1.0)
        acc.update("acc", 0.5)

        result = acc.compute_all()
        assert "loss" in result
        assert "acc" in result
        assert result["loss"] == 1.0
        assert result["acc"] == 0.5

    def test_reset(self):
        acc = MetricAccumulator()
        acc.update("loss", 1.0)
        acc.reset()

        assert acc.compute("loss") is None
        assert acc.compute_all() == {}


class TestMaskedCrossEntropy:
    def test_basic_loss(self):
        batch_size, seq_len, vocab_size = 2, 10, 32
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask_labels = torch.ones(batch_size, seq_len)

        loss = compute_masked_cross_entropy(logits, targets, mask_labels)
        assert loss.ndim == 0
        assert loss > 0

    def test_partial_masking(self):
        batch_size, seq_len, vocab_size = 2, 10, 32
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Only mask first half
        mask_labels = torch.zeros(batch_size, seq_len)
        mask_labels[:, :5] = 1

        loss = compute_masked_cross_entropy(logits, targets, mask_labels)
        assert loss.ndim == 0
        assert loss > 0

    def test_no_mask_returns_zero(self):
        batch_size, seq_len, vocab_size = 2, 10, 32
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask_labels = torch.zeros(batch_size, seq_len)

        loss = compute_masked_cross_entropy(logits, targets, mask_labels)
        assert loss == 0

    def test_reduction_none(self):
        batch_size, seq_len, vocab_size = 2, 10, 32
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask_labels = torch.ones(batch_size, seq_len)

        loss = compute_masked_cross_entropy(
            logits, targets, mask_labels, reduction="none"
        )
        assert loss.shape == (batch_size, seq_len)

    def test_reduction_sum(self):
        batch_size, seq_len, vocab_size = 2, 10, 32
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask_labels = torch.ones(batch_size, seq_len)

        loss = compute_masked_cross_entropy(
            logits, targets, mask_labels, reduction="sum"
        )
        assert loss.ndim == 0


class TestAccuracy:
    def test_perfect_accuracy(self):
        vocab_size = 32
        # Create logits where the correct answer has highest probability
        logits = torch.zeros(2, 10, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 10))

        for b in range(2):
            for s in range(10):
                logits[b, s, targets[b, s]] = 10.0

        mask_labels = torch.ones(2, 10)
        accuracy, num_total = compute_accuracy(logits, targets, mask_labels)

        assert accuracy == 1.0
        assert num_total == 20

    def test_zero_accuracy(self):
        vocab_size = 32
        # Create logits where wrong answer has highest probability
        logits = torch.zeros(2, 10, vocab_size)
        targets = torch.zeros(2, 10, dtype=torch.long)
        logits[:, :, 1] = 10.0  # Predict 1 everywhere

        mask_labels = torch.ones(2, 10)
        accuracy, num_total = compute_accuracy(logits, targets, mask_labels)

        assert accuracy == 0.0
        assert num_total == 20

    def test_no_mask(self):
        vocab_size = 32
        logits = torch.randn(2, 10, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 10))
        mask_labels = torch.zeros(2, 10)

        accuracy, num_total = compute_accuracy(logits, targets, mask_labels)
        assert accuracy == 0.0
        assert num_total == 0


class TestPerplexity:
    def test_basic_perplexity(self):
        loss = torch.tensor(2.0)
        perplexity = compute_perplexity(loss)
        assert perplexity == pytest.approx(torch.exp(torch.tensor(2.0)).item())

    def test_zero_loss(self):
        loss = torch.tensor(0.0)
        perplexity = compute_perplexity(loss)
        assert perplexity == 1.0


class TestMLMMetrics:
    def test_compute_mlm_metrics(self):
        batch_size, seq_len, vocab_size = 2, 10, 32
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask_labels = torch.ones(batch_size, seq_len)
        attention_mask = torch.ones(batch_size, seq_len)

        metrics = compute_mlm_metrics(
            logits, targets, mask_labels, attention_mask
        )

        assert isinstance(metrics, MLMMetrics)
        assert metrics.loss > 0
        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.perplexity >= 1.0
        assert metrics.num_masked_tokens == batch_size * seq_len
        assert metrics.mask_rate == 1.0

    def test_to_dict(self):
        metrics = MLMMetrics(
            loss=1.5,
            accuracy=0.8,
            perplexity=4.5,
            num_masked_tokens=100,
            mask_rate=0.5,
        )

        d = metrics.to_dict()
        assert d["loss"] == 1.5
        assert d["accuracy"] == 0.8
        assert d["perplexity"] == 4.5
        assert d["num_masked_tokens"] == 100
        assert d["mask_rate"] == 0.5
