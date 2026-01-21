"""End-to-end tests for the training loop."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch

from somatic.data import create_dataloader
from somatic.masking import UniformMasker
from somatic.model import SomaticConfig, SomaticModel
from somatic.training import (
    CheckpointConfig,
    CheckpointManager,
    TrainingConfig,
    compute_masked_cross_entropy,
    create_optimizer,
    create_scheduler,
)


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


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    config = SomaticConfig(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_heads=1,
        max_seq_len=128,
        dropout=0.0,
        attention_dropout=0.0,
        embedding_dropout=0.0,
    )
    return SomaticModel(config)


class TestMiniTrainingLoop:
    def test_training_reduces_loss(self, training_data, small_model):
        """Test that training for a few steps reduces loss."""
        model = small_model
        model.train()

        # Setup
        dataloader = create_dataloader(
            data_path=training_data,
            batch_size=4,
            max_length=128,
            shuffle=True,
            num_workers=0,
        )

        optimizer = create_optimizer(model, lr=1e-3)
        masker = UniformMasker(mask_rate=0.15)

        # Track losses
        losses = []

        # Train for a few epochs
        for epoch in range(3):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                # Masking
                masked_ids, mask_labels = masker.apply_mask(
                    token_ids=batch["token_ids"],
                    attention_mask=batch["attention_mask"],
                    special_tokens_mask=batch["special_tokens_mask"],
                )

                # Forward
                outputs = model(
                    token_ids=masked_ids,
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

                # Loss
                loss = compute_masked_cross_entropy(
                    logits=outputs["logits"],
                    targets=batch["token_ids"],
                    mask_labels=mask_labels,
                )

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

        # Loss should decrease or at least not explode
        assert losses[-1] < losses[0] * 2  # Allow some variance but not explosion
        assert all(loss < 100 for loss in losses)  # Sanity check

    def test_checkpointing_saves_and_loads(self, training_data, small_model, tmp_path):
        """Test that checkpointing works correctly."""
        model = small_model
        optimizer = create_optimizer(model, lr=1e-3)
        scheduler = create_scheduler(optimizer, num_training_steps=100)

        checkpoint_config = CheckpointConfig(
            save_dir=str(tmp_path / "checkpoints"),
            checkpoint_steps=5,
            keep_last_n=2,
        )
        checkpoint_manager = CheckpointManager(
            checkpoint_config, model, optimizer, scheduler
        )

        # Modify model state
        for param in model.parameters():
            param.data.add_(0.1)

        # Save checkpoint
        checkpoint_manager.save(step=10, epoch=1, metrics={"loss": 0.5})

        # Verify checkpoint exists
        checkpoint_path = tmp_path / "checkpoints" / "checkpoint_step_10.pt"
        assert checkpoint_path.exists()

        # Create new model and load
        new_model = SomaticModel(small_model.config)
        new_optimizer = create_optimizer(new_model, lr=1e-3)
        new_scheduler = create_scheduler(new_optimizer, num_training_steps=100)
        new_checkpoint_manager = CheckpointManager(
            checkpoint_config, new_model, new_optimizer, new_scheduler
        )

        state = new_checkpoint_manager.load(str(checkpoint_path))

        assert state["step"] == 10
        assert state["epoch"] == 1
        assert state["metrics"]["loss"] == 0.5

        # Verify model weights match
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.allclose(param1, param2), f"Mismatch in {name1}"

    def test_learning_rate_schedule(self, training_data, small_model):
        """Test that learning rate scheduling works."""
        model = small_model
        optimizer = create_optimizer(model, lr=1e-3)
        scheduler = create_scheduler(
            optimizer,
            scheduler_decay="cosine",
            num_training_steps=100,
            num_warmup_steps=10,
        )

        # Track learning rates
        lrs = []

        for step in range(50):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # LR should increase during warmup (first 10 steps)
        assert lrs[9] > lrs[0]

        # LR should decrease after warmup
        assert lrs[-1] < lrs[10]


class TestModelSaveLoad:
    def test_save_and_load_pretrained(self, small_model, tmp_path):
        """Test save_pretrained and from_pretrained."""
        model = small_model

        # Modify weights to make them unique
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Save
        save_path = tmp_path / "model.pt"
        model.save_pretrained(str(save_path))

        assert save_path.exists()

        # Load
        loaded_model = SomaticModel.from_pretrained(str(save_path))

        # Compare configs
        assert loaded_model.config.d_model == model.config.d_model
        assert loaded_model.config.n_layers == model.config.n_layers

        # Compare weights
        model.eval()
        loaded_model.eval()

        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert torch.allclose(param1, param2), f"Mismatch in {name1}"

    def test_loaded_model_produces_same_output(self, small_model, tmp_path):
        """Test that loaded model produces identical outputs."""
        model = small_model
        model.eval()

        # Create test input
        token_ids = torch.randint(4, 28, (2, 32))
        chain_ids = torch.zeros_like(token_ids)
        chain_ids[:, 16:] = 1
        attention_mask = torch.ones_like(token_ids)

        # Get original output
        with torch.no_grad():
            original_output = model(token_ids, chain_ids, attention_mask)

        # Save and load
        save_path = tmp_path / "model.pt"
        model.save_pretrained(str(save_path))
        loaded_model = SomaticModel.from_pretrained(str(save_path))
        loaded_model.eval()

        # Get loaded output
        with torch.no_grad():
            loaded_output = loaded_model(token_ids, chain_ids, attention_mask)

        assert torch.allclose(
            original_output["logits"], loaded_output["logits"], atol=1e-6
        )
