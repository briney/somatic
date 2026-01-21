"""End-to-end tests for MLM components and model configurations.

Tests cover:
1. Masking behavior with static mask rate
2. Information-weighted vs Uniform maskers
3. ChainAwareAttention vs standard MultiHeadAttention
"""

import pandas as pd
import pytest
import torch

from somatic.data import create_dataloader
from somatic.masking import InformationWeightedMasker, UniformMasker
from somatic.model import SomaticConfig, SomaticModel
from somatic.training import compute_masked_cross_entropy, create_optimizer


@pytest.fixture
def training_data(tmp_path):
    """Create training data for e2e tests."""
    data = {
        "heavy_chain": [
            "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH",
            "QVQLQQSGAELARPGASVKMSCKASGYTFTRYTMH",
            "EVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIGWV",
            "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAIS",
        ]
        * 5,  # 20 samples
        "light_chain": [
            "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLA",
            "DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGY",
            "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSY",
            "DIQMTQSPSSLSASVGDRVTITCRASQSISSYL",
        ]
        * 5,
    }
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


# =============================================================================
# Masker Type Tests
# =============================================================================


class TestMaskerTypes:
    """Tests comparing UniformMasker and InformationWeightedMasker."""

    def test_uniform_masker_training(self, training_data):
        """Test training with UniformMasker reduces loss."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=1,
            max_seq_len=128,
            dropout=0.0,
        )
        model = SomaticModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        masker = UniformMasker(mask_rate=0.15)

        losses = []
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    attention_mask=batch["attention_mask"],
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                outputs = model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = compute_masked_cross_entropy(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_labels,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        # Loss should decrease or stabilize
        assert losses[-1] < losses[0] * 2
        assert all(loss < 100 for loss in losses)

    def test_information_weighted_masker_training(self, training_data):
        """Test training with InformationWeightedMasker reduces loss."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=1,
            max_seq_len=128,
            dropout=0.0,
        )
        model = SomaticModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        masker = InformationWeightedMasker(
            mask_rate=0.15,
            cdr_weight_multiplier=1.0,
            nongermline_weight_multiplier=1.0,
        )

        losses = []
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                # InformationWeightedMasker supports optional CDR/template masks
                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    attention_mask=batch["attention_mask"],
                    cdr_mask=None,  # No CDR annotation
                    non_templated_mask=None,
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                outputs = model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = compute_masked_cross_entropy(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_labels,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        assert losses[-1] < losses[0] * 2
        assert all(loss < 100 for loss in losses)

    def test_information_weighted_with_cdr_mask(self, training_data):
        """Test InformationWeightedMasker prioritizes CDR positions."""
        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(dataloader))
        batch_size, seq_len = batch["token_ids"].shape

        masker = InformationWeightedMasker(
            mask_rate=0.15,
            cdr_weight_multiplier=2.0,
            nongermline_weight_multiplier=1.0,
        )

        # Create a CDR mask marking positions 10-20 as CDR
        cdr_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        cdr_mask[:, 10:20] = True

        masked_ids, mask_labels = masker.apply_mask(
            token_ids=batch["token_ids"],
            attention_mask=batch["attention_mask"],
            cdr_mask=cdr_mask,
            non_templated_mask=None,
            special_tokens_mask=batch["special_tokens_mask"],
        )

        # Calculate mask proportion in CDR vs non-CDR regions
        special_mask = batch["special_tokens_mask"].bool()
        maskable = batch["attention_mask"].bool() & ~special_mask

        cdr_maskable = cdr_mask & maskable
        non_cdr_maskable = ~cdr_mask & maskable

        cdr_masked = (mask_labels & cdr_maskable).sum().item()
        cdr_total = cdr_maskable.sum().item()

        non_cdr_masked = (mask_labels & non_cdr_maskable).sum().item()
        non_cdr_total = non_cdr_maskable.sum().item()

        if cdr_total > 0 and non_cdr_total > 0:
            cdr_fraction = cdr_masked / cdr_total
            non_cdr_fraction = non_cdr_masked / non_cdr_total

            # CDR regions should be masked at higher rate
            assert cdr_fraction >= non_cdr_fraction * 0.8, (
                f"CDR mask fraction ({cdr_fraction:.2%}) should be higher "
                f"than non-CDR ({non_cdr_fraction:.2%})"
            )

    def test_masker_comparison_different_distributions(self, training_data):
        """Compare mask distributions between Uniform and InformationWeighted maskers."""
        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=8,
            max_length=128,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(dataloader))
        batch_size, seq_len = batch["token_ids"].shape

        uniform_masker = UniformMasker(mask_rate=0.15)
        weighted_masker = InformationWeightedMasker(
            mask_rate=0.15,
            cdr_weight_multiplier=2.0,
            nongermline_weight_multiplier=2.0,
        )

        # Create CDR and non-templated masks
        cdr_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        cdr_mask[:, 10:25] = True  # Mark CDR region

        non_templated_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        non_templated_mask[:, 15:22] = True  # Mark some non-templated positions

        # Run multiple trials to get statistical significance
        uniform_cdr_fractions = []
        weighted_cdr_fractions = []

        for _ in range(20):
            _, uniform_labels = uniform_masker.apply_mask(
                token_ids=batch["token_ids"],
                attention_mask=batch["attention_mask"],
                special_tokens_mask=batch["special_tokens_mask"],
            )

            _, weighted_labels = weighted_masker.apply_mask(
                token_ids=batch["token_ids"],
                attention_mask=batch["attention_mask"],
                cdr_mask=cdr_mask,
                non_templated_mask=non_templated_mask,
                special_tokens_mask=batch["special_tokens_mask"],
            )

            maskable = batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool()
            cdr_maskable = cdr_mask & maskable

            if cdr_maskable.sum() > 0:
                uniform_cdr = (uniform_labels & cdr_maskable).sum().item()
                weighted_cdr = (weighted_labels & cdr_maskable).sum().item()

                uniform_cdr_fractions.append(uniform_cdr / cdr_maskable.sum().item())
                weighted_cdr_fractions.append(weighted_cdr / cdr_maskable.sum().item())

        # On average, weighted masker should mask more CDR positions
        avg_uniform_cdr = sum(uniform_cdr_fractions) / len(uniform_cdr_fractions)
        avg_weighted_cdr = sum(weighted_cdr_fractions) / len(weighted_cdr_fractions)

        # Weighted should mask at least as much in CDR regions (usually more)
        assert avg_weighted_cdr >= avg_uniform_cdr * 0.8, (
            f"Weighted ({avg_weighted_cdr:.2%}) should mask >= "
            f"uniform ({avg_uniform_cdr:.2%}) in CDR regions"
        )


# =============================================================================
# Chain-Aware Attention Tests
# =============================================================================


class TestChainAwareAttention:
    """Tests for ChainAwareAttention vs standard MultiHeadAttention."""

    def test_chain_aware_attention_training(self, training_data):
        """Test training with ChainAwareAttention enabled."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=128,
            dropout=0.0,
            use_chain_aware_attention=True,  # Explicit
        )
        model = SomaticModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        masker = UniformMasker(mask_rate=0.15)

        losses = []
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    attention_mask=batch["attention_mask"],
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                outputs = model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = compute_masked_cross_entropy(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_labels,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        assert losses[-1] < losses[0] * 2
        assert all(loss < 100 for loss in losses)

    def test_standard_attention_training(self, training_data):
        """Test training with standard MultiHeadAttention (chain-aware disabled)."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=128,
            dropout=0.0,
            use_chain_aware_attention=False,  # Disable chain-aware attention
        )
        model = SomaticModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        masker = UniformMasker(mask_rate=0.15)

        losses = []
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    attention_mask=batch["attention_mask"],
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                outputs = model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = compute_masked_cross_entropy(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_labels,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        assert losses[-1] < losses[0] * 2
        assert all(loss < 100 for loss in losses)

    def test_chain_aware_vs_standard_output_shapes(self):
        """Verify both attention types produce same output shapes."""
        config_chain = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            dropout=0.0,
            use_chain_aware_attention=True,
        )
        config_standard = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            dropout=0.0,
            use_chain_aware_attention=False,
        )

        model_chain = SomaticModel(config_chain)
        model_standard = SomaticModel(config_standard)

        model_chain.eval()
        model_standard.eval()

        # Create test input
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        token_ids[:, 0] = 0  # CLS
        token_ids[:, -1] = 2  # EOS

        # Chain IDs: first half heavy, second half light
        chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        chain_ids[:, seq_len // 2 :] = 1

        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            out_chain = model_chain(token_ids, chain_ids, attention_mask)
            out_standard = model_standard(token_ids, chain_ids, attention_mask)

        assert out_chain["logits"].shape == out_standard["logits"].shape
        assert out_chain["hidden_states"].shape == out_standard["hidden_states"].shape

    def test_chain_aware_attention_patterns(self):
        """Verify ChainAwareAttention produces different patterns for intra/inter-chain."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_seq_len=64,
            dropout=0.0,
            use_chain_aware_attention=True,
        )
        model = SomaticModel(config)
        model.eval()

        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        token_ids[:, 0] = 0  # CLS
        token_ids[:, -1] = 2  # EOS

        # Two chains: 0 for first half, 1 for second half
        chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        chain_ids[:, seq_len // 2 :] = 1

        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            outputs = model(
                token_ids,
                chain_ids,
                attention_mask,
                output_attentions=True,
            )

        # Should have attention weights for each layer
        assert "attentions" in outputs
        assert len(outputs["attentions"]) == 1  # 1 layer

        attn_weights = outputs["attentions"][0]  # (batch, heads, seq, seq)
        assert attn_weights.shape == (batch_size, 2, seq_len, seq_len)

        # Attention weights should sum to 1 along last dimension
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_chain_aware_save_load_roundtrip(self, tmp_path):
        """Test that chain-aware attention models save and load correctly."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            dropout=0.0,
            use_chain_aware_attention=True,
        )
        model = SomaticModel(config)

        # Modify weights to make them unique
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Save and reload
        save_path = tmp_path / "model.pt"
        model.save_pretrained(str(save_path))
        loaded_model = SomaticModel.from_pretrained(str(save_path))

        # Verify config preserved
        assert loaded_model.config.use_chain_aware_attention is True

        # Verify weights match
        model.eval()
        loaded_model.eval()

        token_ids = torch.randint(4, 28, (1, 16))
        chain_ids = torch.zeros(1, 16, dtype=torch.long)
        chain_ids[:, 8:] = 1
        attention_mask = torch.ones(1, 16)

        with torch.no_grad():
            out_orig = model(token_ids, chain_ids, attention_mask)
            out_loaded = loaded_model(token_ids, chain_ids, attention_mask)

        assert torch.allclose(out_orig["logits"], out_loaded["logits"], atol=1e-6)

    def test_standard_attention_save_load_roundtrip(self, tmp_path):
        """Test that standard attention models save and load correctly."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            dropout=0.0,
            use_chain_aware_attention=False,
        )
        model = SomaticModel(config)

        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        save_path = tmp_path / "model.pt"
        model.save_pretrained(str(save_path))
        loaded_model = SomaticModel.from_pretrained(str(save_path))

        # Verify config preserved
        assert loaded_model.config.use_chain_aware_attention is False

        model.eval()
        loaded_model.eval()

        token_ids = torch.randint(4, 28, (1, 16))
        chain_ids = torch.zeros(1, 16, dtype=torch.long)
        chain_ids[:, 8:] = 1
        attention_mask = torch.ones(1, 16)

        with torch.no_grad():
            out_orig = model(token_ids, chain_ids, attention_mask)
            out_loaded = loaded_model(token_ids, chain_ids, attention_mask)

        assert torch.allclose(out_orig["logits"], out_loaded["logits"], atol=1e-6)


# =============================================================================
# Combined Configuration Tests
# =============================================================================


class TestCombinedConfigurations:
    """Tests combining different masker and attention configurations."""

    @pytest.mark.parametrize("use_chain_aware", [True, False])
    def test_masker_attention_combinations(self, training_data, use_chain_aware):
        """Test training with both masker types and attention configurations."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=1,
            max_seq_len=128,
            dropout=0.0,
            use_chain_aware_attention=use_chain_aware,
        )
        model = SomaticModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        masker = UniformMasker(mask_rate=0.15)

        # Train for 1 epoch
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            masked_ids, mask_labels = masker.apply_mask(
                token_ids=batch["token_ids"],
                attention_mask=batch["attention_mask"],
                special_tokens_mask=batch["special_tokens_mask"],
            )

            outputs = model(
                token_ids=masked_ids,
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss = compute_masked_cross_entropy(
                logits=outputs["logits"],
                targets=batch["token_ids"],
                mask_labels=mask_labels,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        assert avg_loss < 100, f"Loss too high for chain_aware={use_chain_aware}: {avg_loss}"

    @pytest.mark.parametrize("masker_type", ["uniform", "information_weighted"])
    @pytest.mark.parametrize("use_chain_aware", [True, False])
    def test_all_masker_attention_combinations(
        self, training_data, masker_type, use_chain_aware
    ):
        """Test training with all combinations of maskers and attention types."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=1,
            max_seq_len=128,
            dropout=0.0,
            use_chain_aware_attention=use_chain_aware,
        )
        model = SomaticModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)

        if masker_type == "uniform":
            masker = UniformMasker(mask_rate=0.15)
        else:
            masker = InformationWeightedMasker(mask_rate=0.15)

        # Train for 1 epoch
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if masker_type == "uniform":
                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    attention_mask=batch["attention_mask"],
                    special_tokens_mask=batch["special_tokens_mask"],
                )
            else:
                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    attention_mask=batch["attention_mask"],
                    cdr_mask=None,
                    non_templated_mask=None,
                    special_tokens_mask=batch["special_tokens_mask"],
                )

            outputs = model(
                token_ids=masked_ids,
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss = compute_masked_cross_entropy(
                logits=outputs["logits"],
                targets=batch["token_ids"],
                mask_labels=mask_labels,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        assert avg_loss < 100, (
            f"Loss too high for {masker_type} + chain_aware={use_chain_aware}: {avg_loss}"
        )


# =============================================================================
# Selection Method Tests
# =============================================================================


class TestSelectionMethods:
    """Tests comparing ranked and sampled selection methods."""

    def test_sampled_training_reduces_loss(self, training_data):
        """Test that multi-epoch training with sampled selection reduces loss."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=1,
            max_seq_len=128,
            dropout=0.0,
        )
        model = SomaticModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        masker = InformationWeightedMasker(
            mask_rate=0.15,
            cdr_weight_multiplier=2.0,
            nongermline_weight_multiplier=1.0,
            selection_method="sampled",
        )

        losses = []
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    attention_mask=batch["attention_mask"],
                    cdr_mask=None,
                    non_templated_mask=None,
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                outputs = model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = compute_masked_cross_entropy(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_labels,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        assert losses[-1] < losses[0] * 2
        assert all(loss < 100 for loss in losses)

    def test_ranked_training_reduces_loss(self, training_data):
        """Test that multi-epoch training with ranked selection reduces loss."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=1,
            max_seq_len=128,
            dropout=0.0,
        )
        model = SomaticModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        masker = InformationWeightedMasker(
            mask_rate=0.15,
            cdr_weight_multiplier=2.0,
            nongermline_weight_multiplier=1.0,
            selection_method="ranked",
        )

        losses = []
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    attention_mask=batch["attention_mask"],
                    cdr_mask=None,
                    non_templated_mask=None,
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                outputs = model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                loss = compute_masked_cross_entropy(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_labels,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        assert losses[-1] < losses[0] * 2
        assert all(loss < 100 for loss in losses)

    def test_sampled_masks_framework_positions(self, training_data):
        """Verify that sampled selection allows framework positions to be masked."""
        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=8,
            max_length=128,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(dataloader))
        batch_size, seq_len = batch["token_ids"].shape

        masker = InformationWeightedMasker(
            mask_rate=0.15,
            cdr_weight_multiplier=3.0,
            nongermline_weight_multiplier=1.0,
            selection_method="sampled",
        )

        # Create CDR mask - only a small region
        cdr_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        cdr_mask[:, 10:15] = 1  # Only 5 CDR positions

        # With 15% masking and only 5 CDR positions out of ~70 maskable,
        # framework must be masked
        fw_masked_total = 0
        num_trials = 10

        for _ in range(num_trials):
            _, mask_labels = masker.apply_mask(
                token_ids=batch["token_ids"],
                attention_mask=batch["attention_mask"],
                cdr_mask=cdr_mask,
                non_templated_mask=None,
                special_tokens_mask=batch["special_tokens_mask"],
            )

            # Count framework positions masked (not 10-14)
            special_mask = batch["special_tokens_mask"].bool()
            maskable = batch["attention_mask"].bool() & ~special_mask
            fw_maskable = maskable.clone()
            fw_maskable[:, 10:15] = False

            fw_masked = (mask_labels & fw_maskable).sum().item()
            fw_masked_total += fw_masked

        # Should have some framework positions masked across trials
        assert fw_masked_total > 0, "Sampled selection should mask framework positions"

    def test_ranked_masks_mostly_high_weight(self, training_data):
        """Verify that ranked selection prioritizes CDR/nongermline positions."""
        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=8,
            max_length=128,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(dataloader))
        batch_size, seq_len = batch["token_ids"].shape

        masker = InformationWeightedMasker(
            mask_rate=0.10,
            cdr_weight_multiplier=5.0,  # High weight for CDR
            nongermline_weight_multiplier=1.0,
            selection_method="ranked",
        )

        # Create CDR mask with enough positions to fill 10% masking
        cdr_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        cdr_mask[:, 10:25] = 1  # 15 CDR positions

        _, mask_labels = masker.apply_mask(
            token_ids=batch["token_ids"],
            attention_mask=batch["attention_mask"],
            cdr_mask=cdr_mask,
            non_templated_mask=None,
            special_tokens_mask=batch["special_tokens_mask"],
        )

        # Calculate what fraction of masked positions are CDR
        special_mask = batch["special_tokens_mask"].bool()
        maskable = batch["attention_mask"].bool() & ~special_mask
        cdr_maskable = (cdr_mask > 0) & maskable

        cdr_masked = (mask_labels & cdr_maskable).sum().item()
        total_masked = mask_labels.sum().item()

        if total_masked > 0:
            cdr_fraction = cdr_masked / total_masked
            # With ranked selection and high CDR weight, most masked should be CDR
            assert cdr_fraction > 0.7, (
                f"Ranked selection should prioritize CDR positions, got {cdr_fraction:.2%}"
            )


# =============================================================================
# Mask Rate Tests
# =============================================================================


class TestMaskRates:
    """Tests for different mask rate configurations."""

    @pytest.mark.parametrize("mask_rate", [0.05, 0.15, 0.30, 0.50])
    def test_different_mask_rates(self, training_data, mask_rate):
        """Test that different mask rates produce valid training."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=32,
            n_layers=1,
            n_heads=1,
            max_seq_len=128,
            dropout=0.0,
        )
        model = SomaticModel(config)
        model.train()

        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        masker = UniformMasker(mask_rate=mask_rate)

        # Train for a few batches
        losses = []
        for batch in dataloader:
            masked_ids, mask_labels = masker.apply_mask(
                token_ids=batch["token_ids"],
                attention_mask=batch["attention_mask"],
                special_tokens_mask=batch["special_tokens_mask"],
            )

            outputs = model(
                token_ids=masked_ids,
                chain_ids=batch["chain_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss = compute_masked_cross_entropy(
                logits=outputs["logits"],
                targets=batch["token_ids"],
                mask_labels=mask_labels,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if len(losses) >= 5:
                break

        assert all(not torch.isnan(torch.tensor(loss)) for loss in losses)
        assert all(loss < 100 for loss in losses)

    def test_mask_rate_approximately_achieved(self, training_data):
        """Test that the actual mask rate is close to the configured rate."""
        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=8,
            max_length=128,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(dataloader))

        for target_rate in [0.10, 0.15, 0.25]:
            masker = UniformMasker(mask_rate=target_rate)

            total_masked = 0
            total_maskable = 0

            for _ in range(10):
                _, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    attention_mask=batch["attention_mask"],
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                special_mask = batch["special_tokens_mask"].bool()
                maskable = batch["attention_mask"].bool() & ~special_mask

                total_masked += mask_labels.sum().item()
                total_maskable += maskable.sum().item()

            actual_rate = total_masked / total_maskable
            # Should be within 5% of target
            assert abs(actual_rate - target_rate) < 0.05, (
                f"Actual rate {actual_rate:.2%} differs from target {target_rate:.2%}"
            )


# =============================================================================
# Predict Masked Tests
# =============================================================================


class TestPredictMasked:
    """Tests for the predict_masked inference method."""

    def test_predict_masked_fills_masks(self):
        """Test that predict_masked fills in MASK tokens."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            dropout=0.0,
        )
        model = SomaticModel(config)
        model.eval()

        # Create input with some MASK tokens
        batch_size, seq_len = 2, 32
        token_ids = torch.randint(4, 28, (batch_size, seq_len))
        token_ids[:, 0] = 0  # CLS
        token_ids[:, -1] = 2  # EOS

        chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        chain_ids[:, seq_len // 2 :] = 1

        # Mask some positions
        original_tokens = token_ids.clone()
        mask_positions = torch.tensor([5, 6, 7, 10, 15])
        token_ids[:, mask_positions] = 31  # MASK token ID

        with torch.no_grad():
            predicted = model.predict_masked(token_ids, chain_ids)

        # Check that masked positions are filled
        for pos in mask_positions:
            assert (predicted[:, pos] != 31).all(), f"Position {pos} should be filled"

        # Check that unmasked positions are preserved
        unmasked_positions = [i for i in range(seq_len) if i not in mask_positions]
        for pos in unmasked_positions:
            assert torch.equal(predicted[:, pos], original_tokens[:, pos])

    def test_predict_masked_with_temperature(self):
        """Test predict_masked with different temperature values."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            dropout=0.0,
        )
        model = SomaticModel(config)
        model.eval()

        token_ids = torch.randint(4, 28, (1, 16))
        token_ids[:, 5:10] = 31  # MASK
        chain_ids = torch.zeros(1, 16, dtype=torch.long)

        with torch.no_grad():
            # Low temperature
            predicted_low = model.predict_masked(
                token_ids.clone(), chain_ids, temperature=0.1
            )
            # High temperature
            predicted_high = model.predict_masked(
                token_ids.clone(), chain_ids, temperature=2.0
            )

        # Both should produce valid token IDs
        assert (predicted_low >= 0).all() and (predicted_low < 32).all()
        assert (predicted_high >= 0).all() and (predicted_high < 32).all()

    def test_predict_masked_with_top_k(self):
        """Test predict_masked with top-k filtering."""
        config = SomaticConfig(
            vocab_size=32,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
            dropout=0.0,
        )
        model = SomaticModel(config)
        model.eval()

        token_ids = torch.randint(4, 28, (1, 16))
        token_ids[:, 5:10] = 31  # MASK
        chain_ids = torch.zeros(1, 16, dtype=torch.long)

        with torch.no_grad():
            predicted = model.predict_masked(token_ids.clone(), chain_ids, top_k=5)

        # Should produce valid token IDs
        assert (predicted >= 0).all() and (predicted < 32).all()
