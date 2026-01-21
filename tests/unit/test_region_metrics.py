"""Tests for per-region evaluation metrics."""

import pytest
import torch

from somatic.eval.metrics.region import (
    RegionAccuracyMetric,
    RegionLossMetric,
    RegionPerplexityMetric,
)


class TestRegionAccuracyMetric:
    """Tests for RegionAccuracyMetric."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch with CDR annotations."""
        batch_size = 2
        seq_len = 20

        # Token IDs
        token_ids = torch.randint(4, 24, (batch_size, seq_len))

        # Chain IDs: first half heavy, second half light
        chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        chain_ids[:, 10:] = 1

        # Attention mask (all valid)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        # Special tokens mask
        special_tokens_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        special_tokens_mask[:, 0] = True
        special_tokens_mask[:, -1] = True

        # CDR mask: positions 2-4, 6-8 in heavy, 12-14, 16-18 in light
        cdr_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        cdr_mask[:, 2:5] = 1   # HCDR1
        cdr_mask[:, 6:9] = 1   # HCDR2 (position 9 won't exist, but 6-8)
        cdr_mask[:, 12:15] = 1  # LCDR1
        cdr_mask[:, 16:19] = 1  # LCDR2

        return {
            "token_ids": token_ids,
            "chain_ids": chain_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
            "cdr_mask": cdr_mask,
        }

    @pytest.fixture
    def sample_outputs(self, sample_batch):
        """Create sample model outputs."""
        batch_size = sample_batch["token_ids"].shape[0]
        seq_len = sample_batch["token_ids"].shape[1]
        vocab_size = 32

        # Create logits that predict the correct token
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s, sample_batch["token_ids"][b, s]] = 10.0

        return {"logits": logits}

    @pytest.fixture
    def sample_mask_labels(self, sample_batch):
        """Create mask labels."""
        batch_size, seq_len = sample_batch["token_ids"].shape
        mask_labels = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        # Mask some positions
        mask_labels[:, 3] = True  # In HCDR1
        mask_labels[:, 5] = True  # In FWR (between HCDR1 and HCDR2)
        mask_labels[:, 13] = True  # In LCDR1
        return mask_labels

    def test_metric_creation(self):
        """Test metric can be created with default settings."""
        metric = RegionAccuracyMetric()
        assert metric.aggregate_by == "all"
        assert len(metric.regions) == 14  # All regions

    def test_metric_with_specific_regions(self):
        """Test metric with specific regions."""
        metric = RegionAccuracyMetric(regions=["hcdr1", "lcdr3"])
        assert len(metric.regions) == 2

    def test_metric_update_and_compute(self, sample_batch, sample_outputs, sample_mask_labels):
        """Test update and compute cycle."""
        metric = RegionAccuracyMetric(aggregate_by="cdr")
        metric.reset()

        metric.update(sample_outputs, sample_batch, sample_mask_labels)
        results = metric.compute()

        assert "cdr/acc" in results
        assert "fwr/acc" in results
        assert 0.0 <= results["cdr/acc"] <= 1.0
        assert 0.0 <= results["fwr/acc"] <= 1.0

    def test_perfect_predictions(self, sample_batch, sample_outputs, sample_mask_labels):
        """Test with perfect predictions."""
        metric = RegionAccuracyMetric(aggregate_by="cdr")
        metric.reset()

        metric.update(sample_outputs, sample_batch, sample_mask_labels)
        results = metric.compute()

        # Since outputs predict correct tokens, accuracy should be 1.0
        assert results["cdr/acc"] == 1.0

    def test_reset_clears_state(self, sample_batch, sample_outputs, sample_mask_labels):
        """Test that reset clears accumulated state."""
        metric = RegionAccuracyMetric(aggregate_by="cdr")

        metric.update(sample_outputs, sample_batch, sample_mask_labels)
        metric.reset()

        results = metric.compute()
        # After reset, should be 0.0 (no data)
        assert results["cdr/acc"] == 0.0

    def test_aggregate_by_chain(self, sample_batch, sample_outputs, sample_mask_labels):
        """Test aggregation by chain."""
        metric = RegionAccuracyMetric(aggregate_by="chain")
        metric.reset()

        metric.update(sample_outputs, sample_batch, sample_mask_labels)
        results = metric.compute()

        assert "heavy/acc" in results
        assert "light/acc" in results

    def test_state_tensors_for_distributed(self, sample_batch, sample_outputs, sample_mask_labels):
        """Test state_tensors and load_state_tensors for distributed training."""
        metric = RegionAccuracyMetric(aggregate_by="cdr")
        metric.reset()

        metric.update(sample_outputs, sample_batch, sample_mask_labels)

        # Get state
        state = metric.state_tensors()
        assert len(state) == 1
        assert isinstance(state[0], torch.Tensor)

        # Reset and reload
        metric.reset()
        metric.load_state_tensors(state)

        # Should recover same results
        results = metric.compute()
        assert "cdr/acc" in results


class TestRegionPerplexityMetric:
    """Tests for RegionPerplexityMetric."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch."""
        batch_size = 2
        seq_len = 20

        return {
            "token_ids": torch.randint(4, 24, (batch_size, seq_len)),
            "chain_ids": torch.cat([
                torch.zeros(batch_size, 10, dtype=torch.long),
                torch.ones(batch_size, 10, dtype=torch.long),
            ], dim=1),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "special_tokens_mask": torch.zeros(batch_size, seq_len, dtype=torch.bool),
            "cdr_mask": torch.cat([
                torch.zeros(batch_size, 2, dtype=torch.long),
                torch.ones(batch_size, 3, dtype=torch.long),  # CDR1
                torch.zeros(batch_size, 2, dtype=torch.long),
                torch.ones(batch_size, 3, dtype=torch.long),  # CDR2
                torch.zeros(batch_size, 10, dtype=torch.long),
            ], dim=1),
        }

    @pytest.fixture
    def sample_outputs(self, sample_batch):
        """Create sample model outputs."""
        batch_size, seq_len = sample_batch["token_ids"].shape
        vocab_size = 32
        logits = torch.randn(batch_size, seq_len, vocab_size)
        return {"logits": logits}

    @pytest.fixture
    def sample_mask_labels(self, sample_batch):
        """Create mask labels."""
        batch_size, seq_len = sample_batch["token_ids"].shape
        mask_labels = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_labels[:, 3] = True  # In CDR
        mask_labels[:, 5] = True  # In FW
        return mask_labels

    def test_perplexity_computation(self, sample_batch, sample_outputs, sample_mask_labels):
        """Test perplexity is computed correctly."""
        metric = RegionPerplexityMetric(aggregate_by="cdr")
        metric.reset()

        metric.update(sample_outputs, sample_batch, sample_mask_labels)
        results = metric.compute()

        assert "cdr/ppl" in results
        assert "fwr/ppl" in results
        # Perplexity should be positive
        assert results["cdr/ppl"] > 0
        assert results["fwr/ppl"] > 0


class TestRegionLossMetric:
    """Tests for RegionLossMetric."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch."""
        batch_size = 2
        seq_len = 20

        return {
            "token_ids": torch.randint(4, 24, (batch_size, seq_len)),
            "chain_ids": torch.cat([
                torch.zeros(batch_size, 10, dtype=torch.long),
                torch.ones(batch_size, 10, dtype=torch.long),
            ], dim=1),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "special_tokens_mask": torch.zeros(batch_size, seq_len, dtype=torch.bool),
            "cdr_mask": torch.cat([
                torch.zeros(batch_size, 2, dtype=torch.long),
                torch.ones(batch_size, 3, dtype=torch.long),
                torch.zeros(batch_size, 2, dtype=torch.long),
                torch.ones(batch_size, 3, dtype=torch.long),
                torch.zeros(batch_size, 10, dtype=torch.long),
            ], dim=1),
        }

    @pytest.fixture
    def sample_outputs(self, sample_batch):
        """Create sample model outputs."""
        batch_size, seq_len = sample_batch["token_ids"].shape
        vocab_size = 32
        logits = torch.randn(batch_size, seq_len, vocab_size)
        return {"logits": logits}

    @pytest.fixture
    def sample_mask_labels(self, sample_batch):
        """Create mask labels."""
        batch_size, seq_len = sample_batch["token_ids"].shape
        mask_labels = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask_labels[:, 3] = True
        mask_labels[:, 5] = True
        return mask_labels

    def test_loss_computation(self, sample_batch, sample_outputs, sample_mask_labels):
        """Test loss is computed correctly."""
        metric = RegionLossMetric(aggregate_by="cdr")
        metric.reset()

        metric.update(sample_outputs, sample_batch, sample_mask_labels)
        results = metric.compute()

        assert "cdr/loss" in results
        assert "fwr/loss" in results
        # Loss should be positive (cross-entropy is always positive)
        assert results["cdr/loss"] > 0
        assert results["fwr/loss"] > 0
