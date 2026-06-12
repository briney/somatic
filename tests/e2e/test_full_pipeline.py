"""End-to-end tests for the complete Somatic pipeline."""

import numpy as np
import pandas as pd
import pytest
import torch

from somatic import SomaticConfig, SomaticEncoder, SomaticForMaskedLM
from somatic.data import create_dataloader
from somatic.masking import UniformMasker
from somatic.model.tokenization_somatic import tokenizer
from somatic.training import create_optimizer


@pytest.fixture
def sample_data(tmp_path):
    """Create sample data files."""
    train_data = {
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

    train_path = tmp_path / "train.csv"
    pd.DataFrame(train_data).to_csv(train_path, index=False)

    return {"train": train_path}


@pytest.fixture
def trained_model(sample_data, tmp_path):
    """Create and train a small model."""
    # Create model
    config = SomaticConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        max_position_embeddings=128,
        hidden_dropout=0.0,
    )
    model = SomaticForMaskedLM(config)
    model.train()

    # Setup training
    dataloader = create_dataloader(
        data_path=sample_data["train"],
        batch_size=2,
        max_length=128,
        num_workers=0,
    )
    optimizer = create_optimizer(model, lr=1e-3)
    masker = UniformMasker(mask_rate=0.15)

    # Train for a few steps
    for _epoch in range(2):
        for batch in dataloader:
            masked_ids, labels = masker.apply_mask(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                special_tokens_mask=batch["special_tokens_mask"],
            )

            outputs = model(
                input_ids=masked_ids,
                token_type_ids=batch["token_type_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Save model (HuggingFace directory format)
    model_path = tmp_path / "trained_model"
    model.save_pretrained(str(model_path))

    return model_path


class TestTrainEncodeGenerate:
    def test_train_save_load_encode(self, trained_model):
        """Test the full train -> save -> load -> encode pipeline."""
        # Load the trained model
        encoder = SomaticEncoder.from_pretrained(trained_model, pooling="mean")

        # Encode some sequences
        heavy = "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH"
        light = "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLA"

        embedding = encoder.encode(heavy, light)

        assert embedding.shape == (32,)  # d_model
        assert not torch.isnan(embedding).any()

    def test_batch_encoding_after_training(self, trained_model):
        """Test batch encoding with a trained model."""
        encoder = SomaticEncoder.from_pretrained(trained_model, pooling="mean")

        heavy_chains = [
            "EVQLVESGGGLVQ",
            "QVQLQQSGAELARP",
            "EVQLVQSGAEVKKP",
        ]
        light_chains = [
            "DIQMTQSPSS",
            "DIVMTQSPLS",
            "EIVLTQSPGT",
        ]

        embeddings = encoder.encode_batch(heavy_chains, light_chains, batch_size=2)

        assert embeddings.shape == (3, 32)
        assert not torch.isnan(embeddings).any()

    def test_predict_masked_with_trained_model(self, trained_model):
        """Test masked prediction with a trained model."""
        model = SomaticForMaskedLM.from_pretrained(str(trained_model))
        model.eval()

        # Start with a partially masked sequence
        heavy = "EVQLVESGGGLVQ"
        light = "DIQMTQSPSS"

        heavy_ids = tokenizer.encode(heavy, add_special_tokens=False)
        light_ids = tokenizer.encode(light, add_special_tokens=False)

        tokens = [tokenizer.cls_token_id] + heavy_ids + light_ids + [tokenizer.eos_token_id]
        chains = [0] * (1 + len(heavy_ids)) + [1] * (len(light_ids) + 1)

        token_ids = torch.tensor([tokens])
        token_type_ids = torch.tensor([chains])

        # Mask CDR-like region (positions 5-10)
        masked_ids = token_ids.clone()
        masked_ids[0, 5:10] = tokenizer.mask_token_id

        # Predict masked positions
        with torch.no_grad():
            predicted = model.predict_masked(
                input_ids=masked_ids,
                token_type_ids=token_type_ids,
            )

        # Check that we got valid amino acids
        assert predicted.shape == token_ids.shape
        assert (predicted >= 0).all() and (predicted < 32).all()

        # Masked positions should be filled (not all mask tokens)
        mask_positions = masked_ids == tokenizer.mask_token_id
        predicted_at_mask = predicted[mask_positions]
        # At least some predictions should not be mask token
        assert (predicted_at_mask != tokenizer.mask_token_id).any()

        # Decode and verify it's valid
        decoded = tokenizer.decode(predicted[0].tolist())
        assert len(decoded) > 0


class TestEncoderOutputFormats:
    def test_numpy_output(self, trained_model):
        """Test numpy output format."""
        encoder = SomaticEncoder.from_pretrained(trained_model, pooling="mean")

        embedding = encoder.encode("EVQLVES", "DIQMTQ", return_numpy=True)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (32,)

    def test_batch_numpy_output(self, trained_model):
        """Test batch numpy output format."""
        encoder = SomaticEncoder.from_pretrained(trained_model, pooling="mean")

        embeddings = encoder.encode_batch(
            ["EVQLVES", "QVQLQQS"],
            ["DIQMTQ", "DIVMTQ"],
            return_numpy=True,
        )

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 32)


class TestDifferentPoolingStrategies:
    def test_all_pooling_strategies(self, trained_model):
        """Test all pooling strategies produce valid outputs."""
        for pooling in ["mean", "cls", "max", "mean_max"]:
            encoder = SomaticEncoder.from_pretrained(trained_model, pooling=pooling)

            embedding = encoder.encode("EVQLVESGGGLVQ", "DIQMTQSPSS")

            expected_dim = 64 if pooling == "mean_max" else 32
            assert embedding.shape == (expected_dim,), f"Failed for {pooling}"
            assert not torch.isnan(embedding).any(), f"NaN in {pooling}"

    def test_no_pooling_returns_sequence(self, trained_model):
        """Test that no pooling returns full sequence embeddings."""
        encoder = SomaticEncoder.from_pretrained(trained_model, pooling=None)

        heavy = "EVQLVES"
        light = "DIQMTQ"

        embedding = encoder.encode(heavy, light)

        # Should be (seq_len, d_model)
        expected_len = 1 + len(heavy) + len(light) + 1  # CLS + heavy + light + EOS
        assert embedding.shape == (expected_len, 32)


class TestTokenizerRoundtrip:
    def test_encode_decode_roundtrip(self):
        """Test that sequences survive encode/decode roundtrip."""
        sequences = [
            "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH",
            "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLA",
            "ACDEFGHIKLMNPQRSTVWY",  # All standard amino acids
        ]

        for seq in sequences:
            encoded = tokenizer.encode(seq, add_special_tokens=False)
            # HF tokenizers add spaces between tokens, so remove them
            decoded = tokenizer.decode(encoded, skip_special_tokens=True).replace(" ", "")
            assert decoded == seq, f"Roundtrip failed for {seq}"

    def test_special_tokens_handling(self):
        """Test special token handling in encode/decode."""
        seq = "EVQLVES"

        # With special tokens
        with_special = tokenizer.encode(seq, add_special_tokens=True)
        assert with_special[0] == tokenizer.cls_token_id
        assert with_special[-1] == tokenizer.eos_token_id

        # Decode should strip special tokens
        decoded = tokenizer.decode(with_special, skip_special_tokens=True).replace(" ", "")
        assert decoded == seq
