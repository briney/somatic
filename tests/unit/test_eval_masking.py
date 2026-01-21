"""Tests for EvalMasker controlled masking."""

import pytest
import torch
from omegaconf import OmegaConf

from somatic.eval.masking import EvalMasker, create_eval_masker
from somatic.tokenizer import tokenizer


class TestEvalMasker:
    """Tests for EvalMasker class."""

    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        batch_size = 4
        seq_len = 20

        # Create token IDs (avoid special token IDs 0-3, 31)
        token_ids = torch.randint(4, 24, (batch_size, seq_len))

        # Set CLS and EOS tokens
        token_ids[:, 0] = tokenizer.cls_token_id
        token_ids[:, -1] = tokenizer.eos_token_id

        # Create attention mask (all valid)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        # Create special tokens mask
        special_tokens_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        special_tokens_mask[:, 0] = True
        special_tokens_mask[:, -1] = True

        # Create chain IDs (first half heavy, second half light)
        chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        chain_ids[:, seq_len // 2 :] = 1

        # Create CDR mask (positions 3-5 and 13-15 are CDRs)
        cdr_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        cdr_mask[:, 3:6] = 1
        cdr_mask[:, 13:16] = 1

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
            "chain_ids": chain_ids,
            "cdr_mask": cdr_mask,
        }

    def test_uniform_masker_creation(self):
        """Test creating a uniform masker."""
        masker = EvalMasker(
            masker_type="uniform",
            mask_rate=0.15,
            seed=42,
        )
        assert masker.masker_type == "uniform"
        assert masker.mask_rate == 0.15
        assert masker.seed == 42

    def test_information_weighted_masker_creation(self):
        """Test creating an information-weighted masker."""
        masker = EvalMasker(
            masker_type="information_weighted",
            mask_rate=0.15,
            cdr_weight_multiplier=1.5,
            nongermline_weight_multiplier=2.0,
            seed=42,
        )
        assert masker.masker_type == "information_weighted"
        assert masker.cdr_weight_multiplier == 1.5
        assert masker.nongermline_weight_multiplier == 2.0

    def test_reproducibility_with_seed(self, sample_batch):
        """Test that same seed produces same masks."""
        masker = EvalMasker(masker_type="uniform", mask_rate=0.15, seed=42)

        device = sample_batch["token_ids"].device

        # First run
        gen1 = masker.get_generator(device)
        masked_ids_1, mask_labels_1 = masker.apply_mask(sample_batch, generator=gen1)

        # Second run with fresh generator (same seed)
        gen2 = masker.get_generator(device)
        masked_ids_2, mask_labels_2 = masker.apply_mask(sample_batch, generator=gen2)

        assert torch.equal(mask_labels_1, mask_labels_2)
        assert torch.equal(masked_ids_1, masked_ids_2)

    def test_different_seeds_different_masks(self, sample_batch):
        """Test that different seeds produce different masks."""
        masker1 = EvalMasker(masker_type="uniform", mask_rate=0.15, seed=42)
        masker2 = EvalMasker(masker_type="uniform", mask_rate=0.15, seed=123)

        device = sample_batch["token_ids"].device

        gen1 = masker1.get_generator(device)
        _, mask_labels_1 = masker1.apply_mask(sample_batch, generator=gen1)

        gen2 = masker2.get_generator(device)
        _, mask_labels_2 = masker2.apply_mask(sample_batch, generator=gen2)

        # Should be different (with very high probability)
        assert not torch.equal(mask_labels_1, mask_labels_2)

    def test_mask_rate_respected(self, sample_batch):
        """Test that mask_rate is approximately respected."""
        masker = EvalMasker(
            masker_type="uniform",
            mask_rate=0.3,  # 30%
            seed=42,
        )

        device = sample_batch["token_ids"].device
        gen = masker.get_generator(device)
        _, mask_labels = masker.apply_mask(sample_batch, generator=gen)

        # Count masked positions (excluding special tokens)
        special_mask = sample_batch["special_tokens_mask"]
        attention_mask = sample_batch["attention_mask"]
        valid_positions = attention_mask & ~special_mask

        total_valid = valid_positions.sum().item()
        total_masked = mask_labels.sum().item()

        # Should be approximately 30% (with some tolerance)
        actual_rate = total_masked / total_valid
        assert 0.15 < actual_rate < 0.45  # Wide tolerance for small batch

    def test_special_tokens_not_masked(self, sample_batch):
        """Test that special tokens (CLS, EOS) are never masked."""
        masker = EvalMasker(
            masker_type="uniform",
            mask_rate=0.5,  # High rate to ensure masking
            seed=42,
        )

        device = sample_batch["token_ids"].device
        gen = masker.get_generator(device)
        masked_ids, mask_labels = masker.apply_mask(sample_batch, generator=gen)

        # CLS and EOS should not be masked
        assert torch.all(~mask_labels[:, 0])  # CLS
        assert torch.all(~mask_labels[:, -1])  # EOS

        # CLS and EOS tokens should be unchanged
        assert torch.all(masked_ids[:, 0] == tokenizer.cls_token_id)
        assert torch.all(masked_ids[:, -1] == tokenizer.eos_token_id)

    def test_create_eval_masker_from_config(self):
        """Test creating EvalMasker from OmegaConf config."""
        cfg = OmegaConf.create(
            {
                "type": "uniform",
                "mask_rate": 0.2,
                "seed": 123,
            }
        )

        masker = create_eval_masker(cfg)
        assert masker.masker_type == "uniform"
        assert masker.mask_rate == 0.2
        assert masker.seed == 123

    def test_create_eval_masker_information_weighted(self):
        """Test creating information-weighted EvalMasker from config."""
        cfg = OmegaConf.create(
            {
                "type": "information_weighted",
                "mask_rate": 0.15,
                "cdr_weight_multiplier": 2.0,
                "nongermline_weight_multiplier": 1.5,
                "selection_method": "sampled",
                "seed": 42,
            }
        )

        masker = create_eval_masker(cfg)
        assert masker.masker_type == "information_weighted"
        assert masker.mask_rate == 0.15
        assert masker.cdr_weight_multiplier == 2.0
        assert masker.nongermline_weight_multiplier == 1.5
        assert masker.selection_method == "sampled"

    def test_information_weighted_biases_cdr(self, sample_batch):
        """Test that information-weighted masker biases toward CDR positions."""
        # Use high mask rate to see clear bias
        masker = EvalMasker(
            masker_type="information_weighted",
            mask_rate=0.2,
            cdr_weight_multiplier=2.0,  # Strong CDR bias
            nongermline_weight_multiplier=1.0,
            seed=42,
        )

        device = sample_batch["token_ids"].device

        # Run multiple times to accumulate statistics
        cdr_masked_total = 0
        cdr_total = 0
        non_cdr_masked_total = 0
        non_cdr_total = 0

        for seed in range(42, 52):
            masker.seed = seed
            gen = masker.get_generator(device)
            _, mask_labels = masker.apply_mask(sample_batch, generator=gen)

            cdr_mask = sample_batch["cdr_mask"].bool()
            special_mask = sample_batch["special_tokens_mask"]
            valid_mask = sample_batch["attention_mask"].bool() & ~special_mask

            cdr_valid = cdr_mask & valid_mask
            non_cdr_valid = ~cdr_mask & valid_mask

            cdr_masked_total += (mask_labels & cdr_valid).sum().item()
            cdr_total += cdr_valid.sum().item()
            non_cdr_masked_total += (mask_labels & non_cdr_valid).sum().item()
            non_cdr_total += non_cdr_valid.sum().item()

        cdr_rate = cdr_masked_total / cdr_total if cdr_total > 0 else 0
        non_cdr_rate = non_cdr_masked_total / non_cdr_total if non_cdr_total > 0 else 0

        # CDR positions should be masked more frequently
        assert cdr_rate > non_cdr_rate

    def test_selection_methods(self, sample_batch):
        """Test both ranked and sampled selection methods work."""
        for selection_method in ["ranked", "sampled"]:
            masker = EvalMasker(
                masker_type="information_weighted",
                mask_rate=0.15,
                cdr_weight_multiplier=1.5,
                nongermline_weight_multiplier=1.0,
                selection_method=selection_method,
                seed=42,
            )

            device = sample_batch["token_ids"].device
            gen = masker.get_generator(device)
            masked_ids, mask_labels = masker.apply_mask(sample_batch, generator=gen)

            # Should produce valid output
            assert masked_ids.shape == sample_batch["token_ids"].shape
            assert mask_labels.shape == sample_batch["token_ids"].shape
            assert mask_labels.sum() > 0  # Should mask something
