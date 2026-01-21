"""Tests for masking frequency tracking."""

import pytest
import torch

from somatic.training.masking_frequency import (
    MaskingFrequencyConfig,
    MaskingFrequencyTracker,
)


class TestMaskingFrequencyConfig:
    """Tests for MaskingFrequencyConfig dataclass."""

    def test_default_disabled(self):
        """Test that all options are disabled by default."""
        config = MaskingFrequencyConfig()
        assert not config.enabled
        assert not config.hcdr1
        assert not config.all_cdr
        assert not config.overall

    def test_enable_individual_regions(self):
        """Test enabling individual regions."""
        config = MaskingFrequencyConfig(
            enabled=True,
            hcdr1=True,
            hcdr3=True,
            lcdr2=True,
        )
        tracker = MaskingFrequencyTracker(config)
        enabled = tracker.get_enabled_regions()
        assert enabled == {"hcdr1", "hcdr3", "lcdr2"}

    def test_enable_aggregates(self):
        """Test enabling aggregate groups."""
        config = MaskingFrequencyConfig(
            enabled=True,
            all_cdr=True,
            overall=True,
        )
        tracker = MaskingFrequencyTracker(config)
        aggregates = tracker.get_enabled_aggregates()
        assert aggregates == {"all_cdr", "overall"}

    def test_enable_all_cdrs(self):
        """Test enabling all CDR regions individually."""
        config = MaskingFrequencyConfig(
            enabled=True,
            hcdr1=True,
            hcdr2=True,
            hcdr3=True,
            lcdr1=True,
            lcdr2=True,
            lcdr3=True,
        )
        tracker = MaskingFrequencyTracker(config)
        enabled = tracker.get_enabled_regions()
        assert len(enabled) == 6
        assert all(r.endswith("cdr1") or r.endswith("cdr2") or r.endswith("cdr3") for r in enabled)


class TestMaskingFrequencyTracker:
    """Tests for MaskingFrequencyTracker class."""

    @pytest.fixture
    def sample_batch_with_cdr_mask(self):
        """Create a batch with CDR annotations.

        Sequence structure (length 40):
        - Position 0: CLS (special)
        - Positions 1-19: Heavy chain
            - 1-3: HFWR1 (mask=0)
            - 4-6: HCDR1 (mask=1)
            - 7-9: HFWR2 (mask=0)
            - 10-12: HCDR2 (mask=2)
            - 13-15: HFWR3 (mask=0)
            - 16-18: HCDR3 (mask=3)
            - 19: HFWR4 (mask=0)
        - Positions 20-38: Light chain
            - 20-22: LFWR1 (mask=0)
            - 23-25: LCDR1 (mask=1)
            - 26-28: LFWR2 (mask=0)
            - 29-31: LCDR2 (mask=2)
            - 32-34: LFWR3 (mask=0)
            - 35-37: LCDR3 (mask=3)
            - 38: LFWR4 (mask=0)
        - Position 39: EOS (special)
        """
        batch_size, seq_len = 2, 40

        token_ids = torch.randint(4, 24, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        special_tokens_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        special_tokens_mask[:, 0] = True   # CLS
        special_tokens_mask[:, 39] = True  # EOS

        chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        chain_ids[:, 20:] = 1  # Light chain

        # CDR mask: 0=FW, 1=CDR1, 2=CDR2, 3=CDR3
        cdr_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        # Heavy CDRs
        cdr_mask[:, 4:7] = 1    # HCDR1
        cdr_mask[:, 10:13] = 2  # HCDR2
        cdr_mask[:, 16:19] = 3  # HCDR3
        # Light CDRs
        cdr_mask[:, 23:26] = 1  # LCDR1
        cdr_mask[:, 29:32] = 2  # LCDR2
        cdr_mask[:, 35:38] = 3  # LCDR3

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
            "chain_ids": chain_ids,
            "cdr_mask": cdr_mask,
        }

    def test_disabled_tracker_no_op(self, sample_batch_with_cdr_mask):
        """Test that disabled tracker produces no metrics."""
        config = MaskingFrequencyConfig(enabled=False, hcdr1=True)
        tracker = MaskingFrequencyTracker(config)

        mask_labels = torch.ones(2, 40, dtype=torch.bool)
        tracker.update(mask_labels, sample_batch_with_cdr_mask)

        assert tracker.compute() == {}

    def test_missing_cdr_mask_graceful(self):
        """Test graceful degradation when cdr_mask is missing."""
        config = MaskingFrequencyConfig(enabled=True, hcdr1=True)
        tracker = MaskingFrequencyTracker(config)

        batch = {
            "token_ids": torch.zeros(2, 10),
            "attention_mask": torch.ones(2, 10),
            "chain_ids": torch.zeros(2, 10),
        }
        mask_labels = torch.ones(2, 10, dtype=torch.bool)

        # Should not raise, just skip tracking
        tracker.update(mask_labels, batch)
        assert tracker.compute() == {}

    def test_individual_region_tracking(self, sample_batch_with_cdr_mask):
        """Test tracking a single individual region."""
        config = MaskingFrequencyConfig(enabled=True, hcdr1=True)
        tracker = MaskingFrequencyTracker(config)

        # Mask all HCDR1 positions (4-6)
        mask_labels = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels[:, 4:7] = True  # Only mask HCDR1

        tracker.update(mask_labels, sample_batch_with_cdr_mask)
        results = tracker.compute()

        assert "hcdr1/fraction_masked" in results
        assert results["hcdr1/fraction_masked"] == 1.0  # All HCDR1 masked
        assert "hcdr1/share_of_total" in results
        assert results["hcdr1/share_of_total"] == 1.0  # All masked tokens are in HCDR1

    def test_partial_region_masking(self, sample_batch_with_cdr_mask):
        """Test partial masking of a region."""
        config = MaskingFrequencyConfig(enabled=True, hcdr1=True)
        tracker = MaskingFrequencyTracker(config)

        # Mask 1 of 3 HCDR1 positions
        mask_labels = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels[:, 4] = True  # Only first HCDR1 position

        tracker.update(mask_labels, sample_batch_with_cdr_mask)
        results = tracker.compute()

        # 1 out of 3 positions per sequence, 2 sequences = 2/6 = 1/3
        assert results["hcdr1/fraction_masked"] == pytest.approx(1 / 3)

    def test_aggregate_all_cdr_tracking(self, sample_batch_with_cdr_mask):
        """Test tracking all_cdr aggregate."""
        config = MaskingFrequencyConfig(enabled=True, all_cdr=True)
        tracker = MaskingFrequencyTracker(config)

        # Mask some CDR positions
        mask_labels = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels[:, 4:7] = True   # HCDR1 (3 positions per seq)
        mask_labels[:, 35:38] = True  # LCDR3 (3 positions per seq)

        tracker.update(mask_labels, sample_batch_with_cdr_mask)
        results = tracker.compute()

        assert "all_cdr/fraction_masked" in results
        # 6 CDRs total, each with 3 positions = 18 positions per seq
        # Masked: HCDR1 (3) + LCDR3 (3) = 6 per seq
        # 2 seqs: 12 masked / 36 total = 1/3
        expected_fraction = 6 / 18
        assert results["all_cdr/fraction_masked"] == pytest.approx(expected_fraction)

    def test_aggregate_overall_tracking(self, sample_batch_with_cdr_mask):
        """Test tracking overall statistics."""
        config = MaskingFrequencyConfig(enabled=True, overall=True)
        tracker = MaskingFrequencyTracker(config)

        # Mask specific positions
        mask_labels = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels[:, 4:7] = True  # 3 positions per seq

        tracker.update(mask_labels, sample_batch_with_cdr_mask)
        results = tracker.compute()

        assert "overall/fraction_masked" in results
        # Total valid positions = 40 - 2 special = 38 per seq
        # 2 seqs: 6 masked / 76 total
        expected = 6 / 76
        assert results["overall/fraction_masked"] == pytest.approx(expected)

    def test_multiple_regions_and_aggregates(self, sample_batch_with_cdr_mask):
        """Test tracking multiple regions and aggregates simultaneously."""
        config = MaskingFrequencyConfig(
            enabled=True,
            hcdr1=True,
            hcdr3=True,
            all_cdr=True,
            overall=True,
        )
        tracker = MaskingFrequencyTracker(config)

        mask_labels = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels[:, 4:7] = True   # HCDR1
        mask_labels[:, 16:19] = True  # HCDR3

        tracker.update(mask_labels, sample_batch_with_cdr_mask)
        results = tracker.compute()

        # Should have metrics for all enabled items
        assert "hcdr1/fraction_masked" in results
        assert "hcdr3/fraction_masked" in results
        assert "all_cdr/fraction_masked" in results
        assert "overall/fraction_masked" in results

        # Both individual and share metrics
        assert "hcdr1/share_of_total" in results
        assert "overall/share_of_total" in results

    def test_reset(self, sample_batch_with_cdr_mask):
        """Test that reset clears accumulators."""
        config = MaskingFrequencyConfig(enabled=True, hcdr1=True)
        tracker = MaskingFrequencyTracker(config)

        mask_labels = torch.ones(2, 40, dtype=torch.bool)
        tracker.update(mask_labels, sample_batch_with_cdr_mask)

        # Verify we have data
        assert tracker.compute() != {}

        # Reset and verify empty
        tracker.reset()
        assert tracker.compute() == {}

    def test_accumulation_over_batches(self, sample_batch_with_cdr_mask):
        """Test that metrics accumulate correctly over multiple batches."""
        config = MaskingFrequencyConfig(enabled=True, hcdr1=True)
        tracker = MaskingFrequencyTracker(config)

        # First batch: mask 1 of 3 HCDR1 positions
        mask_labels1 = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels1[:, 4] = True  # 1 position
        tracker.update(mask_labels1, sample_batch_with_cdr_mask)

        # Second batch: mask 2 of 3 HCDR1 positions
        mask_labels2 = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels2[:, 5:7] = True  # 2 positions
        tracker.update(mask_labels2, sample_batch_with_cdr_mask)

        results = tracker.compute()

        # Total: (2+4) masked out of (6+6) HCDR1 positions = 6/12 = 0.5
        assert results["hcdr1/fraction_masked"] == pytest.approx(0.5)

    def test_share_of_total_calculation(self, sample_batch_with_cdr_mask):
        """Test share_of_total metric calculation."""
        config = MaskingFrequencyConfig(enabled=True, hcdr1=True, hcdr3=True)
        tracker = MaskingFrequencyTracker(config)

        # Mask equal amounts from HCDR1 and HCDR3
        mask_labels = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels[:, 4:7] = True   # HCDR1: 3 per seq = 6 total
        mask_labels[:, 16:19] = True  # HCDR3: 3 per seq = 6 total

        tracker.update(mask_labels, sample_batch_with_cdr_mask)
        results = tracker.compute()

        # Each should be 50% of total masked
        assert results["hcdr1/share_of_total"] == pytest.approx(0.5)
        assert results["hcdr3/share_of_total"] == pytest.approx(0.5)

    def test_fwr_aggregate_tracking(self, sample_batch_with_cdr_mask):
        """Test all_fwr aggregate tracking."""
        config = MaskingFrequencyConfig(enabled=True, all_fwr=True)
        tracker = MaskingFrequencyTracker(config)

        # Mask framework positions
        mask_labels = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels[:, 1:4] = True  # HFWR1 positions

        tracker.update(mask_labels, sample_batch_with_cdr_mask)
        results = tracker.compute()

        assert "all_fwr/fraction_masked" in results
        assert "all_fwr/share_of_total" in results

    def test_chain_aggregate_tracking(self, sample_batch_with_cdr_mask):
        """Test heavy and light chain aggregate tracking."""
        config = MaskingFrequencyConfig(enabled=True, heavy=True, light=True)
        tracker = MaskingFrequencyTracker(config)

        # Mask only heavy chain positions
        mask_labels = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels[:, 4:7] = True  # HCDR1

        tracker.update(mask_labels, sample_batch_with_cdr_mask)
        results = tracker.compute()

        assert "heavy/fraction_masked" in results
        assert "light/fraction_masked" in results
        assert "heavy/share_of_total" in results

        # All masked tokens from heavy chain
        assert results["heavy/share_of_total"] == 1.0
        assert results["light/share_of_total"] == 0.0

    def test_no_masked_tokens(self, sample_batch_with_cdr_mask):
        """Test handling when no tokens are masked."""
        config = MaskingFrequencyConfig(enabled=True, hcdr1=True, overall=True)
        tracker = MaskingFrequencyTracker(config)

        # No masking
        mask_labels = torch.zeros(2, 40, dtype=torch.bool)

        tracker.update(mask_labels, sample_batch_with_cdr_mask)
        results = tracker.compute()

        # Fraction masked should be 0
        assert results["hcdr1/fraction_masked"] == 0.0
        assert results["overall/fraction_masked"] == 0.0

        # share_of_total not included when total_masked is 0
        assert "hcdr1/share_of_total" not in results

    def test_enabled_regions_helper(self):
        """Test get_enabled_regions helper method."""
        config = MaskingFrequencyConfig(
            enabled=True,
            hcdr1=True,
            lcdr3=True,
            hfwr2=True,
        )
        tracker = MaskingFrequencyTracker(config)

        regions = tracker.get_enabled_regions()
        assert regions == {"hcdr1", "lcdr3", "hfwr2"}

    def test_enabled_aggregates_helper(self):
        """Test get_enabled_aggregates helper method."""
        config = MaskingFrequencyConfig(
            enabled=True,
            all_cdr=True,
            heavy=True,
        )
        tracker = MaskingFrequencyTracker(config)

        aggregates = tracker.get_enabled_aggregates()
        assert aggregates == {"all_cdr", "heavy"}

    @pytest.fixture
    def sample_batch_with_nongermline_mask(self, sample_batch_with_cdr_mask):
        """Add non_templated_mask to the sample batch.

        Non-germline positions (value=1) are set at CDR positions,
        germline positions (value=0) at framework positions.
        """
        batch = sample_batch_with_cdr_mask.copy()
        batch_size, seq_len = 2, 40

        # Create non_templated_mask: 0=germline, 1=nongermline
        # Set CDR positions as nongermline, FWR as germline
        non_templated_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        # Heavy CDRs - nongermline
        non_templated_mask[:, 4:7] = 1    # HCDR1
        non_templated_mask[:, 10:13] = 1  # HCDR2
        non_templated_mask[:, 16:19] = 1  # HCDR3
        # Light CDRs - nongermline
        non_templated_mask[:, 23:26] = 1  # LCDR1
        non_templated_mask[:, 29:32] = 1  # LCDR2
        non_templated_mask[:, 35:38] = 1  # LCDR3

        batch["non_templated_mask"] = non_templated_mask
        return batch

    def test_germline_tracking(self, sample_batch_with_nongermline_mask):
        """Test germline aggregate tracking."""
        config = MaskingFrequencyConfig(enabled=True, germline=True)
        tracker = MaskingFrequencyTracker(config)

        # Mask some germline (framework) positions
        mask_labels = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels[:, 1:4] = True  # HFWR1 positions (germline)

        tracker.update(mask_labels, sample_batch_with_nongermline_mask)
        results = tracker.compute()

        assert "germline/fraction_masked" in results
        assert "germline/share_of_total" in results
        # All masked tokens are germline
        assert results["germline/share_of_total"] == 1.0

    def test_nongermline_tracking(self, sample_batch_with_nongermline_mask):
        """Test nongermline aggregate tracking."""
        config = MaskingFrequencyConfig(enabled=True, nongermline=True)
        tracker = MaskingFrequencyTracker(config)

        # Mask some nongermline (CDR) positions
        mask_labels = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels[:, 4:7] = True  # HCDR1 positions (nongermline)

        tracker.update(mask_labels, sample_batch_with_nongermline_mask)
        results = tracker.compute()

        assert "nongermline/fraction_masked" in results
        assert "nongermline/share_of_total" in results
        # All masked tokens are nongermline
        assert results["nongermline/share_of_total"] == 1.0

    def test_germline_nongermline_split(self, sample_batch_with_nongermline_mask):
        """Test germline and nongermline tracking together."""
        config = MaskingFrequencyConfig(
            enabled=True,
            germline=True,
            nongermline=True,
        )
        tracker = MaskingFrequencyTracker(config)

        # Mask equal amounts from germline and nongermline
        mask_labels = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels[:, 1:4] = True   # HFWR1 (3 germline positions)
        mask_labels[:, 4:7] = True   # HCDR1 (3 nongermline positions)

        tracker.update(mask_labels, sample_batch_with_nongermline_mask)
        results = tracker.compute()

        # Each should be 50% of total masked
        assert results["germline/share_of_total"] == pytest.approx(0.5)
        assert results["nongermline/share_of_total"] == pytest.approx(0.5)

    def test_missing_nongermline_mask_skips_silently(self, sample_batch_with_cdr_mask):
        """Test that missing non_templated_mask skips germline/nongermline tracking."""
        config = MaskingFrequencyConfig(
            enabled=True,
            germline=True,
            nongermline=True,
            hcdr1=True,  # This should still work
        )
        tracker = MaskingFrequencyTracker(config)

        # Mask some positions
        mask_labels = torch.zeros(2, 40, dtype=torch.bool)
        mask_labels[:, 4:7] = True  # HCDR1

        # batch doesn't have non_templated_mask
        tracker.update(mask_labels, sample_batch_with_cdr_mask)
        results = tracker.compute()

        # hcdr1 tracking should work
        assert "hcdr1/fraction_masked" in results
        # germline/nongermline should not be present (skipped silently)
        assert "germline/fraction_masked" not in results
        assert "nongermline/fraction_masked" not in results
