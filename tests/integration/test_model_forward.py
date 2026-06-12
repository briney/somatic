"""Integration tests for model forward pass."""

import pytest
import torch

from somatic.model import SomaticConfig, SomaticForMaskedLM
from somatic.model.tokenization_somatic import tokenizer


@pytest.fixture
def model():
    """Create a small masked-LM model for testing (asserts on logits)."""
    config = SomaticConfig(
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    return SomaticForMaskedLM(config)


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
        token_type_ids = [0] * (1 + len(heavy_ids)) + [1] * (len(light_ids) + 1)

        # Convert to tensors
        input_ids = torch.tensor([token_ids])
        token_type_ids = torch.tensor([token_type_ids])
        attention_mask = torch.ones_like(input_ids)

        # Forward pass
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        assert outputs.logits is not None
        assert outputs.hidden_states is not None
        assert outputs.logits.shape == (1, input_ids.shape[1], 32)

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

        input_ids = torch.tensor(batch_tokens)
        token_type_ids = torch.tensor(batch_chains)
        attention_mask = torch.tensor(attention_mask)

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        assert outputs.logits.shape == (2, max_len, 32)

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

        input_ids = torch.tensor([tokens])
        token_type_ids = torch.tensor([chains])
        attention_mask = torch.ones_like(input_ids)

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Model should still produce valid outputs
        assert outputs.logits.shape == (1, len(tokens), 32)
        assert not torch.isnan(outputs.logits).any()

    def test_model_deterministic_eval_mode(self, model):
        """Test that model is deterministic in eval mode."""
        model.eval()

        input_ids = torch.randint(4, 28, (2, 32))
        input_ids[:, 0] = tokenizer.cls_token_id
        input_ids[:, -1] = tokenizer.eos_token_id
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[:, 16:] = 1
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out1 = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            out2 = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        assert torch.allclose(out1.logits, out2.logits)

    def test_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        model.train()

        input_ids = torch.randint(4, 28, (2, 32))
        input_ids[:, 0] = tokenizer.cls_token_id
        input_ids[:, -1] = tokenizer.eos_token_id
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[:, 16:] = 1
        attention_mask = torch.ones_like(input_ids)

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = outputs.logits.sum()
        loss.backward()

        # Check that some parameters have gradients
        has_grads = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break

        assert has_grads
