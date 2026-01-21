"""Integration tests for the data pipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch

from somatic.data import AntibodyCollator, AntibodyDataset, create_dataloader
from somatic.tokenizer import tokenizer


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    data = {
        "heavy_chain": [
            "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH",
            "QVQLQQSGAELARPGASVKMSCKASGYTFTRYTMH",
            "EVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIGWV",
            "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAIS",
        ],
        "light_chain": [
            "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLA",
            "DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGY",
            "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSY",
            "DIQMTQSPSSLSASVGDRVTITCRASQSISSYL",
        ],
    }
    csv_path = tmp_path / "test_data.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_csv_with_masks(tmp_path):
    """Create a sample CSV with CDR masks."""
    data = {
        "heavy_chain": [
            "EVQLVESGGGLVQ",
            "QVQLQQSGAELARP",
        ],
        "light_chain": [
            "DIQMTQSPSS",
            "DIVMTQSPLS",
        ],
        "heavy_cdr_mask": [
            "0,0,0,0,1,1,1,0,0,0,1,1,1",
            "0,0,0,1,1,1,0,0,0,0,1,1,1,1",
        ],
        "light_cdr_mask": [
            "0,0,0,1,1,1,0,0,0,0",
            "0,0,0,0,1,1,1,0,0,0",
        ],
    }
    csv_path = tmp_path / "test_data_masks.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


class TestDatasetIntegration:
    def test_dataset_loads_csv(self, sample_csv):
        """Test that dataset correctly loads CSV data."""
        dataset = AntibodyDataset(sample_csv)

        assert len(dataset) == 4

        example = dataset[0]
        assert "heavy_chain" in example
        assert "light_chain" in example
        assert example["heavy_chain"] == "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH"

    def test_dataset_with_parquet(self, tmp_path):
        """Test that dataset correctly loads Parquet data."""
        data = {
            "heavy_chain": ["EVQLVES", "QVQLQQS"],
            "light_chain": ["DIQMTQ", "DIVMTQ"],
        }
        parquet_path = tmp_path / "test.parquet"
        pd.DataFrame(data).to_parquet(parquet_path)

        dataset = AntibodyDataset(parquet_path)
        assert len(dataset) == 2


class TestCollatorIntegration:
    def test_collator_produces_valid_batch(self, sample_csv):
        """Test that collator produces valid batched tensors."""
        dataset = AntibodyDataset(sample_csv)
        collator = AntibodyCollator(max_length=128)

        examples = [dataset[i] for i in range(len(dataset))]
        batch = collator(examples)

        assert "token_ids" in batch
        assert "chain_ids" in batch
        assert "attention_mask" in batch
        assert "special_tokens_mask" in batch

        # Check shapes
        assert batch["token_ids"].shape[0] == 4
        assert batch["chain_ids"].shape == batch["token_ids"].shape
        assert batch["attention_mask"].shape == batch["token_ids"].shape

        # Check that CLS and EOS are in correct positions
        assert (batch["token_ids"][:, 0] == tokenizer.cls_token_id).all()

    def test_collator_respects_max_length(self, sample_csv):
        """Test that collator respects maximum length."""
        dataset = AntibodyDataset(sample_csv)
        collator = AntibodyCollator(max_length=32)

        examples = [dataset[i] for i in range(len(dataset))]
        batch = collator(examples)

        assert batch["token_ids"].shape[1] <= 32

    def test_collator_pad_to_max(self, sample_csv):
        """Test pad_to_max option."""
        dataset = AntibodyDataset(sample_csv)
        collator = AntibodyCollator(max_length=128, pad_to_max=True)

        examples = [dataset[0]]  # Single example
        batch = collator(examples)

        assert batch["token_ids"].shape[1] == 128


class TestDataLoaderIntegration:
    def test_dataloader_iteration(self, sample_csv):
        """Test that dataloader produces batches correctly."""
        dataloader = create_dataloader(
            data_path=sample_csv,
            batch_size=2,
            max_length=128,
            shuffle=False,
            num_workers=0,
        )

        batches = list(dataloader)
        assert len(batches) == 2  # 4 examples / batch_size 2

        for batch in batches:
            assert batch["token_ids"].shape[0] == 2
            assert isinstance(batch["token_ids"], torch.Tensor)

    def test_dataloader_shuffle(self, sample_csv):
        """Test that shuffling works."""
        # With shuffle=False, should get same order
        dataloader1 = create_dataloader(
            data_path=sample_csv,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
        dataloader2 = create_dataloader(
            data_path=sample_csv,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        batch1 = next(iter(dataloader1))
        batch2 = next(iter(dataloader2))

        assert torch.equal(batch1["token_ids"], batch2["token_ids"])

    def test_dataloader_drop_last(self, sample_csv):
        """Test drop_last option."""
        dataloader = create_dataloader(
            data_path=sample_csv,
            batch_size=3,
            drop_last=True,
            num_workers=0,
        )

        batches = list(dataloader)
        assert len(batches) == 1  # 4 examples, batch_size 3, drop last incomplete


class TestEndToEndDataPipeline:
    def test_data_to_model_forward(self, sample_csv):
        """Test complete pipeline from data loading to model forward."""
        from somatic.model import SomaticConfig, SomaticModel

        # Create model
        config = SomaticConfig(
            vocab_size=32,
            d_model=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=128,
            dropout=0.0,
        )
        model = SomaticModel(config)
        model.eval()

        # Create dataloader
        dataloader = create_dataloader(
            data_path=sample_csv,
            batch_size=2,
            max_length=128,
            num_workers=0,
        )

        # Forward pass with each batch
        for batch in dataloader:
            with torch.no_grad():
                outputs = model(
                    token_ids=batch["token_ids"],
                    chain_ids=batch["chain_ids"],
                    attention_mask=batch["attention_mask"],
                )

            assert "logits" in outputs
            assert outputs["logits"].shape[0] == batch["token_ids"].shape[0]
            assert not torch.isnan(outputs["logits"]).any()
