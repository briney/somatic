"""Tests for masking utilities."""

import pytest
import torch

from somatic.masking import InformationWeightedMasker, UniformMasker
from somatic.tokenizer import tokenizer


class TestUniformMasker:
    @pytest.fixture
    def masker(self):
        return UniformMasker(mask_rate=0.5)

    def test_apply_mask_shape(self, masker):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        masked_ids, mask_labels = masker.apply_mask(token_ids, attention_mask)

        assert masked_ids.shape == token_ids.shape
        assert mask_labels.shape == token_ids.shape

    def test_masked_positions_have_mask_token(self, masker):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        masked_ids, mask_labels = masker.apply_mask(token_ids, attention_mask)

        # Where mask_labels is True, masked_ids should be MASK_IDX
        assert (masked_ids[mask_labels] == tokenizer.mask_token_id).all()

    def test_respects_attention_mask(self, masker):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -10:] = 0  # Last 10 positions are padding

        _, mask_labels = masker.apply_mask(token_ids, attention_mask)

        # Padding positions should not be masked
        assert not mask_labels[:, -10:].any()

    def test_respects_special_tokens_mask(self, masker):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        token_ids[:, 0] = tokenizer.cls_token_id
        token_ids[:, -1] = tokenizer.eos_token_id
        attention_mask = torch.ones(batch_size, seq_len)
        special_tokens_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        special_tokens_mask[:, 0] = True
        special_tokens_mask[:, -1] = True

        _, mask_labels = masker.apply_mask(
            token_ids, attention_mask, special_tokens_mask=special_tokens_mask
        )

        # Special token positions should not be masked
        assert not mask_labels[:, 0].any()
        assert not mask_labels[:, -1].any()

    def test_mask_rate_validation(self):
        with pytest.raises(ValueError):
            UniformMasker(mask_rate=0.0)
        with pytest.raises(ValueError):
            UniformMasker(mask_rate=1.0)
        with pytest.raises(ValueError):
            UniformMasker(mask_rate=-0.1)
        with pytest.raises(ValueError):
            UniformMasker(mask_rate=1.5)


class TestInformationWeightedMasker:
    @pytest.fixture
    def masker(self):
        return InformationWeightedMasker(
            mask_rate=0.5,
            cdr_weight_multiplier=1.0,
            nongermline_weight_multiplier=1.0,
        )

    def test_apply_mask_shape(self, masker):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        masked_ids, mask_labels = masker.apply_mask(token_ids, attention_mask)

        assert masked_ids.shape == token_ids.shape
        assert mask_labels.shape == token_ids.shape

    def test_compute_weights_uniform(self, masker):
        batch_size, seq_len = 2, 10
        attention_mask = torch.ones(batch_size, seq_len)

        weights = masker.compute_weights(
            cdr_mask=None, non_templated_mask=None, attention_mask=attention_mask
        )

        # Without CDR/NT masks, weights should be uniform
        expected = torch.ones(batch_size, seq_len) / seq_len
        assert torch.allclose(weights, expected)

    def test_compute_weights_with_cdr(self, masker):
        batch_size, seq_len = 1, 10
        attention_mask = torch.ones(batch_size, seq_len)
        cdr_mask = torch.zeros(batch_size, seq_len)
        cdr_mask[:, 5:8] = 1  # Positions 5-7 are CDR

        weights = masker.compute_weights(
            cdr_mask=cdr_mask, non_templated_mask=None, attention_mask=attention_mask
        )

        # CDR positions should have higher weights
        assert weights[0, 5] > weights[0, 0]
        assert weights[0, 6] > weights[0, 0]
        assert weights[0, 7] > weights[0, 0]

    def test_compute_weights_with_detailed_cdr(self, masker):
        """Test weight computation with detailed CDR mask (0=FW, 1=CDR1, 2=CDR2, 3=CDR3)."""
        batch_size, seq_len = 1, 12
        attention_mask = torch.ones(batch_size, seq_len)
        # Detailed CDR mask: 0=FW, 1=CDR1, 2=CDR2, 3=CDR3
        cdr_mask = torch.tensor([[0, 0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 0]])

        weights = masker.compute_weights(
            cdr_mask=cdr_mask, non_templated_mask=None, attention_mask=attention_mask
        )

        # All CDR positions (values 1, 2, or 3) should have higher weights than FW (0)
        fw_weight = weights[0, 0].item()
        assert weights[0, 2].item() > fw_weight  # CDR1
        assert weights[0, 3].item() > fw_weight  # CDR1
        assert weights[0, 5].item() > fw_weight  # CDR2
        assert weights[0, 6].item() > fw_weight  # CDR2
        assert weights[0, 8].item() > fw_weight  # CDR3
        assert weights[0, 9].item() > fw_weight  # CDR3
        # All CDR types should have same weight boost
        assert torch.isclose(weights[0, 2], weights[0, 5])
        assert torch.isclose(weights[0, 5], weights[0, 8])

    def test_separate_cdr_and_nongermline_multipliers(self):
        """Test that CDR and nongermline multipliers are applied independently."""
        # Create masker with higher CDR weight
        masker_high_cdr = InformationWeightedMasker(
            mask_rate=0.5,
            cdr_weight_multiplier=2.0,
            nongermline_weight_multiplier=0.5,
        )

        batch_size, seq_len = 1, 10
        attention_mask = torch.ones(batch_size, seq_len)
        cdr_mask = torch.zeros(batch_size, seq_len)
        cdr_mask[:, 2:4] = 1  # CDR at positions 2-3
        nt_mask = torch.zeros(batch_size, seq_len)
        nt_mask[:, 6:8] = 1  # Nongermline at positions 6-7

        weights = masker_high_cdr.compute_weights(
            cdr_mask=cdr_mask, non_templated_mask=nt_mask, attention_mask=attention_mask
        )

        # CDR should have weight = 1 + 2.0 = 3.0 (unnormalized)
        # NT should have weight = 1 + 0.5 = 1.5 (unnormalized)
        # FW should have weight = 1.0 (unnormalized)
        # CDR should have higher weight than NT
        assert weights[0, 2] > weights[0, 6]
        assert weights[0, 3] > weights[0, 7]
        # NT should have higher weight than FW
        assert weights[0, 6] > weights[0, 0]

    def test_zero_cdr_multiplier(self):
        """Test that zero CDR multiplier gives CDR same weight as framework."""
        masker = InformationWeightedMasker(
            mask_rate=0.5,
            cdr_weight_multiplier=0.0,
            nongermline_weight_multiplier=1.0,
        )

        batch_size, seq_len = 1, 10
        attention_mask = torch.ones(batch_size, seq_len)
        cdr_mask = torch.zeros(batch_size, seq_len)
        cdr_mask[:, 5:8] = 1

        weights = masker.compute_weights(
            cdr_mask=cdr_mask, non_templated_mask=None, attention_mask=attention_mask
        )

        # With zero multiplier, all weights should be equal (uniform)
        expected = torch.ones(batch_size, seq_len) / seq_len
        assert torch.allclose(weights, expected)

    def test_high_nongermline_multiplier(self):
        """Test that high nongermline multiplier increases nongermline weight."""
        masker = InformationWeightedMasker(
            mask_rate=0.5,
            cdr_weight_multiplier=1.0,
            nongermline_weight_multiplier=5.0,
        )

        batch_size, seq_len = 1, 10
        attention_mask = torch.ones(batch_size, seq_len)
        cdr_mask = torch.zeros(batch_size, seq_len)
        cdr_mask[:, 2:4] = 1
        nt_mask = torch.zeros(batch_size, seq_len)
        nt_mask[:, 6:8] = 1

        weights = masker.compute_weights(
            cdr_mask=cdr_mask, non_templated_mask=nt_mask, attention_mask=attention_mask
        )

        # NT should now have higher weight than CDR (1 + 5.0 vs 1 + 1.0)
        assert weights[0, 6] > weights[0, 2]
        assert weights[0, 7] > weights[0, 3]

    def test_respects_special_tokens_mask(self, masker):
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        special_tokens_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        special_tokens_mask[:, 0] = True

        _, mask_labels = masker.apply_mask(
            token_ids, attention_mask, special_tokens_mask=special_tokens_mask
        )

        assert not mask_labels[:, 0].any()

    def test_mask_count_matches_rate(self):
        masker = InformationWeightedMasker(mask_rate=0.5)
        batch_size, seq_len = 4, 100
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        _, mask_labels = masker.apply_mask(token_ids, attention_mask)

        # Should mask approximately 50 tokens per sequence
        mask_counts = mask_labels.sum(dim=-1)
        assert (mask_counts >= 40).all()  # Allow some tolerance
        assert (mask_counts <= 60).all()

    def test_mask_rate_validation(self):
        with pytest.raises(ValueError):
            InformationWeightedMasker(mask_rate=0.0)
        with pytest.raises(ValueError):
            InformationWeightedMasker(mask_rate=1.0)


class TestGumbelSampling:
    """Tests for Gumbel-top-k selection method."""

    def test_sampled_is_stochastic(self):
        """Test that sampled selection produces different outputs across runs."""
        masker = InformationWeightedMasker(
            mask_rate=0.3,
            cdr_weight_multiplier=2.0,
            nongermline_weight_multiplier=1.0,
            selection_method="sampled",
        )

        batch_size, seq_len = 2, 50
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        cdr_mask = torch.zeros(batch_size, seq_len)
        cdr_mask[:, 10:20] = 1  # CDR positions

        # Run multiple times and check for variation
        results = []
        for _ in range(10):
            _, mask_labels = masker.apply_mask(
                token_ids, attention_mask, cdr_mask=cdr_mask
            )
            results.append(mask_labels.clone())

        # Not all results should be identical (stochastic)
        all_same = all(torch.equal(results[0], r) for r in results[1:])
        assert not all_same, "Sampled masking should produce different results across runs"

    def test_ranked_is_deterministic(self):
        """Test that ranked selection produces consistent high-weight position masking."""
        masker = InformationWeightedMasker(
            mask_rate=0.1,  # Low mask rate
            cdr_weight_multiplier=10.0,  # Very high weight to ensure CDR always selected first
            nongermline_weight_multiplier=1.0,
            selection_method="ranked",
        )

        batch_size, seq_len = 2, 50
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        cdr_mask = torch.zeros(batch_size, seq_len)
        cdr_mask[:, 10:20] = 1  # 10 CDR positions (more than we'll mask)

        # Run multiple times - with ranked selection and high CDR weight,
        # the masked positions should always be within CDR region
        for _ in range(5):
            _, mask_labels = masker.apply_mask(
                token_ids, attention_mask, cdr_mask=cdr_mask
            )
            # All masked positions should be CDR positions (10-19)
            cdr_region = mask_labels[:, 10:20]

            # With ranked selection and high weight, all masks should be in CDR
            total_masked = mask_labels.sum()
            cdr_masked = cdr_region.sum()
            assert cdr_masked == total_masked, (
                "Ranked selection with high CDR weight should only mask CDR positions"
            )

    def test_sampled_respects_weights(self):
        """Test that higher-weight positions are masked more often on average."""
        masker = InformationWeightedMasker(
            mask_rate=0.15,
            cdr_weight_multiplier=3.0,
            nongermline_weight_multiplier=1.0,
            selection_method="sampled",
        )

        batch_size, seq_len = 4, 100
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # CDR at positions 10-20 (10 positions)
        cdr_mask = torch.zeros(batch_size, seq_len)
        cdr_mask[:, 10:20] = 1

        # Run multiple trials
        cdr_mask_counts = []
        fw_mask_counts = []
        num_trials = 50

        for _ in range(num_trials):
            _, mask_labels = masker.apply_mask(
                token_ids, attention_mask, cdr_mask=cdr_mask
            )
            cdr_masked = mask_labels[:, 10:20].sum().item()
            fw_masked = mask_labels[:, :10].sum().item() + mask_labels[:, 20:].sum().item()
            cdr_mask_counts.append(cdr_masked)
            fw_mask_counts.append(fw_masked)

        # CDR has 10 positions with weight 4 (1+3), FW has 90 positions with weight 1
        # Expected ratio should favor CDR positions
        avg_cdr = sum(cdr_mask_counts) / num_trials / (10 * batch_size)  # Per position
        avg_fw = sum(fw_mask_counts) / num_trials / (90 * batch_size)  # Per position

        # CDR should have higher mask rate per position
        assert avg_cdr > avg_fw, f"CDR rate ({avg_cdr:.3f}) should exceed FW rate ({avg_fw:.3f})"

    def test_sampled_allows_low_weight_positions(self):
        """Test that framework positions can still be masked with sampled selection."""
        masker = InformationWeightedMasker(
            mask_rate=0.2,
            cdr_weight_multiplier=2.0,
            nongermline_weight_multiplier=1.0,
            selection_method="sampled",
        )

        batch_size, seq_len = 4, 100
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Small CDR region (only 5 positions)
        cdr_mask = torch.zeros(batch_size, seq_len)
        cdr_mask[:, 10:15] = 1

        # With 20% masking of 100 positions = 20 masked positions
        # Only 5 are CDR, so FW must be masked too
        fw_masked_ever = False
        for _ in range(20):
            _, mask_labels = masker.apply_mask(
                token_ids, attention_mask, cdr_mask=cdr_mask
            )
            # Check if any FW positions (not 10-14) were masked
            fw_positions = torch.cat([mask_labels[:, :10], mask_labels[:, 15:]], dim=1)
            if fw_positions.any():
                fw_masked_ever = True
                break

        assert fw_masked_ever, "Sampled selection should allow framework positions to be masked"

    def test_selection_method_validation(self):
        """Test that invalid selection_method raises ValueError."""
        with pytest.raises(ValueError, match="selection_method must be"):
            InformationWeightedMasker(
                mask_rate=0.15,
                cdr_weight_multiplier=1.0,
                nongermline_weight_multiplier=1.0,
                selection_method="invalid",
            )

    def test_default_selection_method_is_sampled(self):
        """Test that default selection_method is 'sampled'."""
        masker = InformationWeightedMasker(mask_rate=0.15)
        assert masker.selection_method == "sampled"
