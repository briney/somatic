"""Tests for antibody region extraction."""

import pytest
import torch

from somatic.eval.regions import (
    AntibodyRegion,
    CDR_REGIONS,
    FWR_REGIONS,
    HEAVY_REGIONS,
    LIGHT_REGIONS,
    aggregate_region_masks,
    extract_region_masks,
)


class TestAntibodyRegion:
    """Tests for AntibodyRegion enum."""

    def test_all_regions_exist(self):
        """Test that all expected regions are defined."""
        expected = [
            "hcdr1", "hcdr2", "hcdr3",
            "lcdr1", "lcdr2", "lcdr3",
            "hfwr1", "hfwr2", "hfwr3", "hfwr4",
            "lfwr1", "lfwr2", "lfwr3", "lfwr4",
        ]
        actual = [r.value for r in AntibodyRegion]
        assert set(expected) == set(actual)

    def test_cdr_regions_grouping(self):
        """Test CDR_REGIONS contains only CDR regions."""
        for region in CDR_REGIONS:
            assert "cdr" in region.value

    def test_fwr_regions_grouping(self):
        """Test FWR_REGIONS contains only framework regions."""
        for region in FWR_REGIONS:
            assert "fwr" in region.value

    def test_heavy_regions_grouping(self):
        """Test HEAVY_REGIONS contains only heavy chain regions."""
        for region in HEAVY_REGIONS:
            assert region.value.startswith("h")

    def test_light_regions_grouping(self):
        """Test LIGHT_REGIONS contains only light chain regions."""
        for region in LIGHT_REGIONS:
            assert region.value.startswith("l")


class TestExtractRegionMasks:
    """Tests for extract_region_masks function."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch with detailed CDR annotations.

        Uses detailed CDR mask format: 0=FW, 1=CDR1, 2=CDR2, 3=CDR3

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
        batch_size = 2
        seq_len = 40

        # Token IDs (just placeholders)
        token_ids = torch.randint(4, 24, (batch_size, seq_len))

        # Attention mask (all valid)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        # Special tokens mask (CLS at 0, EOS at 39)
        special_tokens_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        special_tokens_mask[:, 0] = True
        special_tokens_mask[:, 39] = True

        # Chain IDs: 0 for CLS + heavy, 1 for light + EOS
        chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        chain_ids[:, 20:] = 1

        # Detailed CDR mask: 0=FWR, 1=CDR1, 2=CDR2, 3=CDR3
        cdr_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        # Heavy chain CDRs
        cdr_mask[:, 4:7] = 1    # HCDR1 (positions 4-6)
        cdr_mask[:, 10:13] = 2  # HCDR2 (positions 10-12)
        cdr_mask[:, 16:19] = 3  # HCDR3 (positions 16-18)
        # Light chain CDRs
        cdr_mask[:, 23:26] = 1  # LCDR1 (positions 23-25)
        cdr_mask[:, 29:32] = 2  # LCDR2 (positions 29-31)
        cdr_mask[:, 35:38] = 3  # LCDR3 (positions 35-37)

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
            "chain_ids": chain_ids,
            "cdr_mask": cdr_mask,
        }

    def test_extract_all_regions(self, sample_batch):
        """Test extracting all region masks."""
        region_masks = extract_region_masks(sample_batch)

        # Should have all 14 regions
        assert len(region_masks) == 14

        # Check HCDR1 positions
        hcdr1_mask = region_masks[AntibodyRegion.HCDR1]
        assert hcdr1_mask.shape == (2, 40)
        assert hcdr1_mask[0, 4:7].all()  # Positions 4-6 should be True
        assert not hcdr1_mask[0, :4].any()  # Positions before should be False
        assert not hcdr1_mask[0, 7:10].any()  # Positions after should be False

    def test_extract_specific_regions(self, sample_batch):
        """Test extracting only specific regions."""
        regions = {AntibodyRegion.HCDR1, AntibodyRegion.LCDR3}
        region_masks = extract_region_masks(sample_batch, regions)

        assert len(region_masks) == 2
        assert AntibodyRegion.HCDR1 in region_masks
        assert AntibodyRegion.LCDR3 in region_masks

    def test_cdr_positions_correct(self, sample_batch):
        """Test that CDR positions are correctly identified using detailed mask values."""
        region_masks = extract_region_masks(sample_batch)

        # HCDR1: positions 4-6 (mask value = 1)
        hcdr1 = region_masks[AntibodyRegion.HCDR1][0]
        assert hcdr1[4:7].all()
        assert hcdr1.sum() == 3

        # HCDR2: positions 10-12 (mask value = 2)
        hcdr2 = region_masks[AntibodyRegion.HCDR2][0]
        assert hcdr2[10:13].all()
        assert hcdr2.sum() == 3

        # HCDR3: positions 16-18 (mask value = 3)
        hcdr3 = region_masks[AntibodyRegion.HCDR3][0]
        assert hcdr3[16:19].all()
        assert hcdr3.sum() == 3

        # LCDR3: positions 35-37 (mask value = 3)
        lcdr3 = region_masks[AntibodyRegion.LCDR3][0]
        assert lcdr3[35:38].all()
        assert lcdr3.sum() == 3

    def test_cdrs_non_overlapping(self, sample_batch):
        """Test that CDR regions identified by different mask values don't overlap."""
        region_masks = extract_region_masks(sample_batch)

        hcdr1 = region_masks[AntibodyRegion.HCDR1][0]
        hcdr2 = region_masks[AntibodyRegion.HCDR2][0]
        hcdr3 = region_masks[AntibodyRegion.HCDR3][0]

        # No overlap between CDR regions
        assert not (hcdr1 & hcdr2).any()
        assert not (hcdr2 & hcdr3).any()
        assert not (hcdr1 & hcdr3).any()

    def test_framework_positions_correct(self, sample_batch):
        """Test that framework positions are correctly inferred."""
        region_masks = extract_region_masks(sample_batch)

        # HFWR1: positions 1-3 (between CLS and HCDR1)
        hfwr1 = region_masks[AntibodyRegion.HFWR1][0]
        assert hfwr1[1:4].all()
        assert hfwr1.sum() == 3

        # HFWR2: positions 7-9 (between HCDR1 and HCDR2)
        hfwr2 = region_masks[AntibodyRegion.HFWR2][0]
        assert hfwr2[7:10].all()
        assert hfwr2.sum() == 3

    def test_no_cdr_mask_raises_error(self):
        """Test that missing cdr_mask raises ValueError."""
        batch = {
            "token_ids": torch.zeros(2, 10, dtype=torch.long),
            "chain_ids": torch.zeros(2, 10, dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }

        with pytest.raises(ValueError, match="cdr_mask is required"):
            extract_region_masks(batch)

    def test_batch_processing(self, sample_batch):
        """Test that batch processing works correctly."""
        region_masks = extract_region_masks(sample_batch)

        # All sequences in batch should have same regions
        for region, mask in region_masks.items():
            assert mask.shape[0] == 2  # batch_size
            # Both sequences should have same pattern
            assert torch.equal(mask[0], mask[1])


class TestAggregateRegionMasks:
    """Tests for aggregate_region_masks function."""

    @pytest.fixture
    def sample_region_masks(self):
        """Create sample region masks for testing aggregation."""
        batch_size = 2
        seq_len = 20

        masks = {}
        for region in AntibodyRegion:
            masks[region] = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        # Set some positions
        masks[AntibodyRegion.HCDR1][:, 2:4] = True
        masks[AntibodyRegion.HCDR2][:, 5:7] = True
        masks[AntibodyRegion.HCDR3][:, 8:10] = True
        masks[AntibodyRegion.HFWR1][:, 0:2] = True
        masks[AntibodyRegion.HFWR2][:, 4:5] = True

        return masks

    def test_aggregate_all(self, sample_region_masks):
        """Test 'all' aggregation returns individual regions."""
        result = aggregate_region_masks(sample_region_masks, "all")

        assert "hcdr1" in result
        assert "hcdr2" in result
        assert "hfwr1" in result
        assert len(result) == len(sample_region_masks)

    def test_aggregate_cdr(self, sample_region_masks):
        """Test 'cdr' aggregation groups CDRs and frameworks."""
        result = aggregate_region_masks(sample_region_masks, "cdr")

        assert "cdr" in result
        assert "fwr" in result
        assert len(result) == 2

        # CDR mask should include all CDR positions
        cdr_mask = result["cdr"]
        assert cdr_mask[:, 2:4].all()  # HCDR1
        assert cdr_mask[:, 5:7].all()  # HCDR2
        assert cdr_mask[:, 8:10].all()  # HCDR3

    def test_aggregate_chain(self, sample_region_masks):
        """Test 'chain' aggregation groups by chain."""
        result = aggregate_region_masks(sample_region_masks, "chain")

        assert "heavy" in result
        assert "light" in result
        assert len(result) == 2

    def test_aggregate_region_type(self, sample_region_masks):
        """Test 'region_type' aggregation groups by CDR/FWR number."""
        result = aggregate_region_masks(sample_region_masks, "region_type")

        assert "cdr1" in result
        assert "cdr2" in result
        assert "cdr3" in result
        assert "fwr1" in result
        assert "fwr2" in result

    def test_invalid_aggregate_by(self, sample_region_masks):
        """Test that invalid aggregate_by raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregate_by"):
            aggregate_region_masks(sample_region_masks, "invalid")
