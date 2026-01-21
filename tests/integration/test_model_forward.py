"""Integration tests for model forward pass."""

import pytest
import torch

from somatic.model import SomaticConfig, SomaticModel
from somatic.tokenizer import tokenizer


@pytest.fixture
def model():
    """Create a small model for testing."""
    config = SomaticConfig(
        vocab_size=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=128,
        dropout=0.0,
        attention_dropout=0.0,
        embedding_dropout=0.0,
    )
    return SomaticModel(config)


class TestModelForwardIntegration:
    def test_forward_with_real_sequences(self, model):
        """Test forward pass with real antibody-like sequences."""
        heavy = "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMHWVRQAPGKGLEWVS"
        light = "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLAWYQQKPGKAPKLLIY"

        # Encode sequences
        heavy_ids = tokenizer.encode(heavy, add_special_tokens=False)
        light_ids = tokenizer.encode(light, add_special_tokens=False)

        # Build full sequence: [CLS] heavy light [EOS]
        token_ids = [tokenizer.cls_token_id] + heavy_ids + light_ids + [tokenizer.eos_token_id]
        chain_ids = [0] * (1 + len(heavy_ids)) + [1] * (len(light_ids) + 1)

        # Convert to tensors
        token_ids = torch.tensor([token_ids])
        chain_ids = torch.tensor([chain_ids])
        attention_mask = torch.ones_like(token_ids)

        # Forward pass
        outputs = model(token_ids, chain_ids, attention_mask)

        assert "logits" in outputs
        assert "hidden_states" in outputs
        assert outputs["logits"].shape == (1, len(token_ids[0]), 32)

    def test_forward_batch_different_lengths(self, model):
        """Test forward with batch of different length sequences."""
        sequences = [
            ("EVQLVESGGGLVQ", "DIQMTQSPSS"),
            ("QVQLQQSGAELARPGASVKMSCKASGYTFTRYTMHWVK", "DIVMTQSPLSLPVTPGEPAS"),
        ]

        max_len = 0
        batch_tokens = []
        batch_chains = []

        for heavy, light in sequences:
            heavy_ids = tokenizer.encode(heavy, add_special_tokens=False)
            light_ids = tokenizer.encode(light, add_special_tokens=False)
            tokens = [tokenizer.cls_token_id] + heavy_ids + light_ids + [tokenizer.eos_token_id]
            chains = [0] * (1 + len(heavy_ids)) + [1] * (len(light_ids) + 1)
            batch_tokens.append(tokens)
            batch_chains.append(chains)
            max_len = max(max_len, len(tokens))

        # Pad sequences
        attention_mask = []
        for i in range(len(batch_tokens)):
            seq_len = len(batch_tokens[i])
            attention_mask.append([1] * seq_len + [0] * (max_len - seq_len))
            batch_tokens[i] += [tokenizer.pad_token_id] * (max_len - seq_len)
            batch_chains[i] += [0] * (max_len - seq_len)

        token_ids = torch.tensor(batch_tokens)
        chain_ids = torch.tensor(batch_chains)
        attention_mask = torch.tensor(attention_mask)

        outputs = model(token_ids, chain_ids, attention_mask)

        assert outputs["logits"].shape == (2, max_len, 32)

    def test_forward_with_mask_tokens(self, model):
        """Test forward with MASK tokens in input."""
        heavy = "EVQLVESGGGLVQ"
        light = "DIQMTQSPSS"

        heavy_ids = tokenizer.encode(heavy, add_special_tokens=False)
        light_ids = tokenizer.encode(light, add_special_tokens=False)

        # Replace some tokens with MASK
        heavy_ids[3] = tokenizer.mask_token_id
        heavy_ids[7] = tokenizer.mask_token_id
        light_ids[2] = tokenizer.mask_token_id

        tokens = [tokenizer.cls_token_id] + heavy_ids + light_ids + [tokenizer.eos_token_id]
        chains = [0] * (1 + len(heavy_ids)) + [1] * (len(light_ids) + 1)

        token_ids = torch.tensor([tokens])
        chain_ids = torch.tensor([chains])
        attention_mask = torch.ones_like(token_ids)

        outputs = model(token_ids, chain_ids, attention_mask)

        # Model should still produce valid outputs
        assert outputs["logits"].shape == (1, len(tokens), 32)
        assert not torch.isnan(outputs["logits"]).any()

    def test_model_deterministic_eval_mode(self, model):
        """Test that model is deterministic in eval mode."""
        model.eval()

        token_ids = torch.randint(4, 28, (2, 32))
        token_ids[:, 0] = tokenizer.cls_token_id
        token_ids[:, -1] = tokenizer.eos_token_id
        chain_ids = torch.zeros_like(token_ids)
        chain_ids[:, 16:] = 1
        attention_mask = torch.ones_like(token_ids)

        with torch.no_grad():
            out1 = model(token_ids, chain_ids, attention_mask)
            out2 = model(token_ids, chain_ids, attention_mask)

        assert torch.allclose(out1["logits"], out2["logits"])

    def test_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        model.train()

        token_ids = torch.randint(4, 28, (2, 32))
        token_ids[:, 0] = tokenizer.cls_token_id
        token_ids[:, -1] = tokenizer.eos_token_id
        chain_ids = torch.zeros_like(token_ids)
        chain_ids[:, 16:] = 1
        attention_mask = torch.ones_like(token_ids)

        outputs = model(token_ids, chain_ids, attention_mask)
        loss = outputs["logits"].sum()
        loss.backward()

        # Check that some parameters have gradients
        has_grads = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break

        assert has_grads
