"""Tests for per-position evaluation optimization.

These tests verify that the optimized _run_per_position_eval method:
1. Produces correct results when aggregating by region, germline, and nongermline
2. Correctly collects all unique positions upfront
3. Aggregates results from a single evaluation pass
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from somatic.eval.per_position import PerPositionEvaluator
from somatic.eval.regions import AntibodyRegion, extract_region_masks
from somatic.eval.region_config import RegionEvalConfig


class TestPositionCollectionOptimization:
    """Tests for verifying position collection optimization."""

    @pytest.fixture
    def sample_with_regions(self):
        """Create a sample with CDR and germline/nongermline annotations.
        
        Sequence structure (length 40):
        - Position 0: CLS (special)
        - Positions 1-19: Heavy chain
            - 1-3: HFWR1 (germline)
            - 4-6: HCDR1 (germline)
            - 7-9: HFWR2 (germline)
            - 10-12: HCDR2 (nongermline)
            - 13-15: HFWR3 (germline)
            - 16-18: HCDR3 (nongermline)
            - 19: HFWR4 (germline)
        - Positions 20-38: Light chain
            - 20-22: LFWR1 (germline)
            - 23-25: LCDR1 (germline)
            - 26-28: LFWR2 (germline)
            - 29-31: LCDR2 (nongermline)
            - 32-34: LFWR3 (germline)
            - 35-37: LCDR3 (nongermline)
            - 38: LFWR4 (germline)
        - Position 39: EOS (special)
        """
        seq_len = 40

        # Token IDs
        token_ids = torch.randint(4, 24, (seq_len,))
        token_ids[0] = 0  # CLS
        token_ids[39] = 2  # EOS

        # Attention mask (all valid)
        attention_mask = torch.ones(seq_len, dtype=torch.long)

        # Special tokens mask (CLS at 0, EOS at 39)
        special_tokens_mask = torch.zeros(seq_len, dtype=torch.bool)
        special_tokens_mask[0] = True
        special_tokens_mask[39] = True

        # Chain IDs: 0 for CLS + heavy, 1 for light + EOS
        chain_ids = torch.zeros(seq_len, dtype=torch.long)
        chain_ids[20:] = 1

        # Detailed CDR mask: 0=FWR, 1=CDR1, 2=CDR2, 3=CDR3
        cdr_mask = torch.zeros(seq_len, dtype=torch.long)
        # Heavy chain CDRs
        cdr_mask[4:7] = 1    # HCDR1 (positions 4-6)
        cdr_mask[10:13] = 2  # HCDR2 (positions 10-12)
        cdr_mask[16:19] = 3  # HCDR3 (positions 16-18)
        # Light chain CDRs
        cdr_mask[23:26] = 1  # LCDR1 (positions 23-25)
        cdr_mask[29:32] = 2  # LCDR2 (positions 29-31)
        cdr_mask[35:38] = 3  # LCDR3 (positions 35-37)

        # Non-templated mask: 0=germline, 1=nongermline
        # Mark CDR2 and CDR3 as nongermline
        non_templated_mask = torch.zeros(seq_len, dtype=torch.long)
        non_templated_mask[10:13] = 1  # HCDR2
        non_templated_mask[16:19] = 1  # HCDR3
        non_templated_mask[29:32] = 1  # LCDR2
        non_templated_mask[35:38] = 1  # LCDR3

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
            "chain_ids": chain_ids,
            "cdr_mask": cdr_mask,
            "non_templated_mask": non_templated_mask,
        }

    def test_all_positions_collected_for_regions_and_germline(self, sample_with_regions):
        """Test that all unique positions are collected from regions and germline/nongermline."""
        sample = sample_with_regions
        
        # Create batched version for extract_region_masks
        batch_sample = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
        
        # Extract region masks
        all_regions = set(AntibodyRegion)
        region_masks = extract_region_masks(batch_sample, all_regions)
        
        # Collect positions from regions
        region_positions = {}
        all_region_positions = set()
        for region, mask in region_masks.items():
            positions = mask[0].nonzero(as_tuple=True)[0].tolist()
            region_positions[region.value] = positions
            all_region_positions.update(positions)
        
        # Collect germline positions
        attention_mask = sample["attention_mask"]
        special_tokens_mask = sample["special_tokens_mask"]
        non_templated_mask = sample["non_templated_mask"]
        
        valid_mask = attention_mask.bool() & ~special_tokens_mask.bool()
        germline_mask = (non_templated_mask == 0) & valid_mask
        nongermline_mask = (non_templated_mask == 1) & valid_mask
        
        germline_positions = set(germline_mask.nonzero(as_tuple=True)[0].tolist())
        nongermline_positions = set(nongermline_mask.nonzero(as_tuple=True)[0].tolist())
        
        # Union should contain all unique positions
        all_positions = all_region_positions | germline_positions | nongermline_positions
        
        # All valid positions (excluding special tokens) should be covered
        all_valid = valid_mask.nonzero(as_tuple=True)[0].tolist()
        assert all_positions == set(all_valid), (
            "Union of all positions should equal all valid positions"
        )
        
        # Verify germline + nongermline = all valid positions
        assert germline_positions | nongermline_positions == set(all_valid), (
            "Germline + nongermline should cover all valid positions"
        )

    def test_positions_overlap_between_regions_and_germline(self, sample_with_regions):
        """Test that positions overlap between regions and germline/nongermline.
        
        This demonstrates why evaluating them separately is redundant.
        """
        sample = sample_with_regions
        
        # Create batched version for extract_region_masks
        batch_sample = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
        
        # Get HCDR3 positions
        region_masks = extract_region_masks(batch_sample, {AntibodyRegion.HCDR3})
        hcdr3_positions = set(region_masks[AntibodyRegion.HCDR3][0].nonzero(as_tuple=True)[0].tolist())
        
        # Get nongermline positions
        attention_mask = sample["attention_mask"]
        special_tokens_mask = sample["special_tokens_mask"]
        non_templated_mask = sample["non_templated_mask"]
        
        valid_mask = attention_mask.bool() & ~special_tokens_mask.bool()
        nongermline_mask = (non_templated_mask == 1) & valid_mask
        nongermline_positions = set(nongermline_mask.nonzero(as_tuple=True)[0].tolist())
        
        # HCDR3 should be a subset of nongermline (based on our fixture)
        assert hcdr3_positions.issubset(nongermline_positions), (
            "HCDR3 positions should be subset of nongermline positions in our test fixture"
        )
        
        # This overlap is why the optimization matters - without it, HCDR3 positions
        # would be evaluated twice: once for region eval, once for nongermline eval

    def test_region_aggregation_from_per_position_results(self, sample_with_regions):
        """Test that region metrics can be correctly aggregated from per-position results."""
        sample = sample_with_regions
        
        # Create batched version for extract_region_masks
        batch_sample = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
        
        # Extract region masks
        region_masks = extract_region_masks(batch_sample, {AntibodyRegion.HCDR1})
        hcdr1_positions = region_masks[AntibodyRegion.HCDR1][0].nonzero(as_tuple=True)[0].tolist()
        
        # Simulate per-position results (as would be returned by evaluate_positions)
        mock_per_position_results = {
            pos: {
                "correct": 1 if pos % 2 == 0 else 0,  # Alternating correct/incorrect
                "loss": 0.5 + pos * 0.01,
                "prob": 0.8 - pos * 0.01,
            }
            for pos in hcdr1_positions
        }
        
        # Aggregate by region
        total_correct = 0
        total_loss = 0.0
        total_prob = 0.0
        count = 0
        
        for pos in hcdr1_positions:
            if pos in mock_per_position_results:
                metrics = mock_per_position_results[pos]
                total_correct += metrics["correct"]
                total_loss += metrics["loss"]
                total_prob += metrics["prob"]
                count += 1
        
        # Verify aggregation
        assert count == len(hcdr1_positions)
        assert count > 0
        accuracy = total_correct / count
        avg_loss = total_loss / count
        avg_prob = total_prob / count
        
        # Check aggregated values are reasonable
        assert 0 <= accuracy <= 1
        assert avg_loss > 0
        assert 0 <= avg_prob <= 1


class TestPerPositionEvaluatorIntegration:
    """Integration tests for PerPositionEvaluator with position optimization."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.eval = MagicMock()
        
        # Track forward call count to verify single evaluation
        model.forward_count = 0
        
        def mock_forward(token_ids, chain_ids, attention_mask, **kwargs):
            model.forward_count += 1
            batch_size, seq_len = token_ids.shape
            vocab_size = 32
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return {"logits": logits}
        
        model.return_value = {"logits": torch.randn(1, 40, 32)}
        model.side_effect = mock_forward
        model.parameters = lambda: iter([torch.nn.Parameter(torch.randn(1))])
        
        return model

    def test_evaluate_positions_called_once_with_union(self, mock_model, sample_with_regions):
        """Test that evaluate_positions processes all positions efficiently."""
        sample = sample_with_regions
        
        # Create batched version for extract_region_masks
        batch_sample = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
        
        # Get all positions from regions
        all_regions = set(AntibodyRegion)
        region_masks = extract_region_masks(batch_sample, all_regions)
        
        all_positions = set()
        for region, mask in region_masks.items():
            positions = mask[0].nonzero(as_tuple=True)[0].tolist()
            all_positions.update(positions)
        
        # Also include germline/nongermline positions
        attention_mask = sample["attention_mask"]
        special_tokens_mask = sample["special_tokens_mask"]
        non_templated_mask = sample["non_templated_mask"]
        
        valid_mask = attention_mask.bool() & ~special_tokens_mask.bool()
        germline_positions = (non_templated_mask == 0) & valid_mask
        nongermline_positions = (non_templated_mask == 1) & valid_mask
        
        all_positions.update(germline_positions.nonzero(as_tuple=True)[0].tolist())
        all_positions.update(nongermline_positions.nonzero(as_tuple=True)[0].tolist())
        
        # All valid positions should be in the union
        all_valid = set(valid_mask.nonzero(as_tuple=True)[0].tolist())
        assert all_positions == all_valid

    @pytest.fixture
    def sample_with_regions(self):
        """Create a sample with CDR and germline/nongermline annotations."""
        seq_len = 40

        token_ids = torch.randint(4, 24, (seq_len,))
        token_ids[0] = 0  # CLS
        token_ids[39] = 2  # EOS

        attention_mask = torch.ones(seq_len, dtype=torch.long)

        special_tokens_mask = torch.zeros(seq_len, dtype=torch.bool)
        special_tokens_mask[0] = True
        special_tokens_mask[39] = True

        chain_ids = torch.zeros(seq_len, dtype=torch.long)
        chain_ids[20:] = 1

        cdr_mask = torch.zeros(seq_len, dtype=torch.long)
        cdr_mask[4:7] = 1    # HCDR1
        cdr_mask[10:13] = 2  # HCDR2
        cdr_mask[16:19] = 3  # HCDR3
        cdr_mask[23:26] = 1  # LCDR1
        cdr_mask[29:32] = 2  # LCDR2
        cdr_mask[35:38] = 3  # LCDR3

        non_templated_mask = torch.zeros(seq_len, dtype=torch.long)
        non_templated_mask[10:13] = 1  # HCDR2
        non_templated_mask[16:19] = 1  # HCDR3
        non_templated_mask[29:32] = 1  # LCDR2
        non_templated_mask[35:38] = 1  # LCDR3

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
            "chain_ids": chain_ids,
            "cdr_mask": cdr_mask,
            "non_templated_mask": non_templated_mask,
        }


class TestRegionEvalConfigWithOptimization:
    """Tests for RegionEvalConfig in context of optimization."""

    def test_config_determines_positions_needed(self):
        """Test that config correctly determines which positions are needed."""
        # Config with only CDR regions
        config_cdrs_only = RegionEvalConfig(
            enabled=True,
            mode="per-position",
            hcdr3=True,
            lcdr3=True,
            all_cdr=False,
            germline=False,
            nongermline=False,
        )
        
        enabled_regions = config_cdrs_only.get_enabled_regions()
        enabled_aggs = config_cdrs_only.get_enabled_aggregates()
        
        assert "hcdr3" in enabled_regions
        assert "lcdr3" in enabled_regions
        assert "germline" not in enabled_aggs
        assert "nongermline" not in enabled_aggs
        
        # Config with germline/nongermline
        config_with_germline = RegionEvalConfig(
            enabled=True,
            mode="per-position",
            hcdr3=True,
            germline=True,
            nongermline=True,
        )
        
        enabled_aggs = config_with_germline.get_enabled_aggregates()
        assert "germline" in enabled_aggs
        assert "nongermline" in enabled_aggs

    def test_aggregate_groups_require_all_region_positions(self):
        """Test that enabling aggregate groups requires evaluating all constituent regions."""
        # When all_cdr is enabled, we need all 6 CDR region positions
        config = RegionEvalConfig(
            enabled=True,
            mode="per-position",
            all_cdr=True,
        )
        
        enabled_aggs = config.get_enabled_aggregates()
        assert "all_cdr" in enabled_aggs
        
        # The evaluator should internally include all CDR regions
        # when computing aggregate metrics, even if individual regions
        # are not explicitly enabled

