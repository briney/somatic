"""Tests for the main Somatic transformer model."""

import pytest
import torch

from somatic.model import SomaticConfig, SomaticForMaskedLM, SomaticModel
from somatic.model.normalization import QKVNormModule, RMSNorm


class TestSomaticConfig:
    def test_default_values(self):
        config = SomaticConfig()
        assert config.vocab_size == 32
        assert config.hidden_size == 256
        assert config.num_hidden_layers == 16

    def test_intermediate_size_auto_computed(self):
        config = SomaticConfig(hidden_size=64)
        assert config.intermediate_size is not None
        assert config.intermediate_size > config.hidden_size
        assert config.intermediate_size % 64 == 0

    def test_custom_intermediate_size(self):
        config = SomaticConfig(hidden_size=64, intermediate_size=256)
        assert config.intermediate_size == 256

    def test_default_norm_values(self):
        config = SomaticConfig()
        assert config.norm_type == "layernorm"
        assert config.pre_norm is True
        assert config.post_norm is False
        assert config.qk_norm == "none"
        assert config.norm_eps == 1e-6

    def test_invalid_norm_type(self):
        with pytest.raises(ValueError, match="norm_type"):
            SomaticConfig(norm_type="invalid")

    def test_invalid_qk_norm(self):
        with pytest.raises(ValueError, match="qk_norm"):
            SomaticConfig(qk_norm="invalid")

    def test_both_norm_false_raises(self):
        with pytest.raises(ValueError, match="pre_norm or post_norm"):
            SomaticConfig(pre_norm=False, post_norm=False)

    def test_valid_norm_configurations(self):
        # Pre-norm only (default)
        config = SomaticConfig(pre_norm=True, post_norm=False)
        assert config.pre_norm is True
        assert config.post_norm is False

        # Post-norm only
        config = SomaticConfig(pre_norm=False, post_norm=True)
        assert config.pre_norm is False
        assert config.post_norm is True

        # Both pre and post norm
        config = SomaticConfig(pre_norm=True, post_norm=True)
        assert config.pre_norm is True
        assert config.post_norm is True

    def test_rmsnorm_config(self):
        config = SomaticConfig(norm_type="rmsnorm")
        assert config.norm_type == "rmsnorm"

    def test_qk_norm_options(self):
        config = SomaticConfig(qk_norm="norm")
        assert config.qk_norm == "norm"

        config = SomaticConfig(qk_norm="learned_scale")
        assert config.qk_norm == "learned_scale"

    def test_hybrid_norm_default_none(self):
        config = SomaticConfig()
        assert config.hybrid_norm == "none"

    def test_hybrid_norm_invalid_value_raises(self):
        with pytest.raises(ValueError, match="hybrid_norm"):
            SomaticConfig(hybrid_norm="bogus")

    @pytest.mark.parametrize("variant", ["standard", "star"])
    def test_hybrid_norm_skips_pre_post_validation(self, variant):
        # Both pre/post False would normally raise; hybrid_norm makes them inert
        config = SomaticConfig(pre_norm=False, post_norm=False, hybrid_norm=variant)
        assert config.hybrid_norm == variant

    @pytest.mark.parametrize("variant", ["standard", "star"])
    def test_hybrid_norm_skips_qk_norm_validation(self, variant):
        # An invalid qk_norm value would normally raise; hybrid_norm makes it inert
        config = SomaticConfig(qk_norm="bogus", hybrid_norm=variant)
        assert config.hybrid_norm == variant

    def test_hybrid_norm_still_validates_norm_type(self):
        with pytest.raises(ValueError, match="norm_type"):
            SomaticConfig(norm_type="invalid", hybrid_norm="standard")

    def test_config_dict_missing_hybrid_norm(self):
        # Configs that omit hybrid_norm default cleanly to "none".
        d = {
            "vocab_size": 32,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
        }
        config = SomaticConfig(**d)
        assert config.hybrid_norm == "none"

    def test_config_dict_missing_rope_fraction(self):
        # Configs that omit rope_fraction default to full RoPE.
        d = {
            "vocab_size": 32,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
        }
        config = SomaticConfig(**d)
        assert config.rope_fraction == 1.0

    def test_default_rope_fraction(self):
        config = SomaticConfig()
        assert config.rope_fraction == 1.0

    @pytest.mark.parametrize("bad_value", [-0.1, 1.1, 2.0])
    def test_invalid_rope_fraction_raises(self, bad_value):
        with pytest.raises(ValueError, match="rope_fraction"):
            SomaticConfig(rope_fraction=bad_value)


class TestSomaticModel:
    @pytest.fixture
    def config(self):
        return SomaticConfig(
            vocab_size=32,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=64,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )

    @pytest.fixture
    def model(self, config):
        return SomaticModel(config)

    def test_forward_basic(self, model, config):
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        token_type_ids = torch.zeros(batch_size, seq_len).long()

        outputs = model(input_ids, token_type_ids=token_type_ids)

        assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)

    def test_forward_with_attention_mask(self, model, config):
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        token_type_ids = torch.zeros(batch_size, seq_len).long()
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)

    def test_forward_output_hidden_states(self, model, config):
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        token_type_ids = torch.zeros(batch_size, seq_len).long()

        outputs = model(input_ids, token_type_ids=token_type_ids, output_hidden_states=True)

        assert outputs.hidden_states is not None
        # Should be num_hidden_layers + 1 (embedding + each layer output)
        assert len(outputs.hidden_states) == config.num_hidden_layers + 1

        # Check shapes
        for hidden_state in outputs.hidden_states:
            assert hidden_state.shape == (batch_size, seq_len, config.hidden_size)

    def test_forward_output_attentions(self, model, config):
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        token_type_ids = torch.zeros(batch_size, seq_len).long()

        outputs = model(input_ids, token_type_ids=token_type_ids, output_attentions=True)

        assert outputs.attentions is not None
        # Should be num_hidden_layers attention weight tensors
        assert len(outputs.attentions) == config.num_hidden_layers

        # Each attention output is a single merged attention weight tensor
        for attn_weights in outputs.attentions:
            assert attn_weights.shape == (
                batch_size,
                config.num_attention_heads,
                seq_len,
                seq_len,
            )

    def test_forward_output_both(self, model, config):
        """Test returning both hidden states and attentions."""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        token_type_ids = torch.zeros(batch_size, seq_len).long()

        outputs = model(
            input_ids,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=True,
        )

        assert outputs.hidden_states is not None
        assert outputs.attentions is not None
        assert len(outputs.hidden_states) == config.num_hidden_layers + 1
        assert len(outputs.attentions) == config.num_hidden_layers

    def test_forward_with_multiple_chains(self, model, config):
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        token_type_ids = torch.cat(
            [
                torch.zeros(batch_size, seq_len // 2),
                torch.ones(batch_size, seq_len // 2),
            ],
            dim=1,
        ).long()

        outputs = model(input_ids, token_type_ids=token_type_ids)

        assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)

    def test_get_num_params(self, model):
        n_params = model.get_num_params(non_embedding=True)
        assert n_params > 0

        n_params_with_emb = model.get_num_params(non_embedding=False)
        assert n_params_with_emb > n_params

    def test_save_and_load(self, model, config, tmp_path):
        save_path = tmp_path / "model"
        model.save_pretrained(str(save_path))

        loaded_model = SomaticModel.from_pretrained(str(save_path))

        # Check configs match
        assert loaded_model.config.hidden_size == model.config.hidden_size
        assert loaded_model.config.num_hidden_layers == model.config.num_hidden_layers

        # Check outputs match
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        token_type_ids = torch.zeros(batch_size, seq_len).long()

        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            out1 = model(input_ids, token_type_ids=token_type_ids)
            out2 = loaded_model(input_ids, token_type_ids=token_type_ids)

        assert torch.allclose(out1.last_hidden_state, out2.last_hidden_state)

    def test_weight_tying(self, config):
        """The MLM head weight is tied to the input embedding."""
        mlm = SomaticForMaskedLM(config)
        assert mlm.lm_head.weight is mlm.somatic.embeddings.token_embedding.embedding.weight

    def test_standard_attention_mode(self):
        """Test model with standard MultiHeadAttention instead of ChainAwareAttention."""
        config = SomaticConfig(
            vocab_size=32,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=64,
            hidden_dropout=0.0,
            use_chain_aware_attention=False,  # Use standard attention
        )
        model = SomaticModel(config)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        token_type_ids = torch.zeros(batch_size, seq_len).long()

        outputs = model(input_ids, token_type_ids=token_type_ids)

        assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)

    def test_attention_mode_comparison(self):
        """Test that both attention modes produce valid outputs with same config."""
        base_config = dict(
            vocab_size=32,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=64,
            hidden_dropout=0.0,
        )

        # Create both models
        chain_aware_model = SomaticModel(
            SomaticConfig(**base_config, use_chain_aware_attention=True)
        )
        standard_model = SomaticModel(SomaticConfig(**base_config, use_chain_aware_attention=False))

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        token_type_ids = torch.zeros(batch_size, seq_len).long()

        chain_aware_model.eval()
        standard_model.eval()

        with torch.no_grad():
            out1 = chain_aware_model(input_ids, token_type_ids=token_type_ids)
            out2 = standard_model(input_ids, token_type_ids=token_type_ids)

        # Both should produce valid outputs with same shape
        assert out1.last_hidden_state.shape == out2.last_hidden_state.shape
        assert not torch.isnan(out1.last_hidden_state).any()
        assert not torch.isnan(out2.last_hidden_state).any()


class TestSomaticModelNormalization:
    """Tests for SomaticModel with different normalization configurations."""

    @pytest.fixture
    def base_config_kwargs(self):
        return dict(
            vocab_size=32,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=64,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )

    @pytest.fixture
    def sample_inputs(self):
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 32, (batch_size, seq_len))
        token_type_ids = torch.zeros(batch_size, seq_len).long()
        return input_ids, token_type_ids

    def test_rmsnorm_forward(self, base_config_kwargs, sample_inputs):
        """Test model with RMSNorm produces valid outputs."""
        config = SomaticConfig(**base_config_kwargs, norm_type="rmsnorm")
        model = SomaticModel(config)

        input_ids, token_type_ids = sample_inputs
        outputs = model(input_ids, token_type_ids=token_type_ids)

        assert not torch.isnan(outputs.last_hidden_state).any()
        assert outputs.last_hidden_state.shape == (2, 32, config.hidden_size)

    def test_post_norm_forward(self, base_config_kwargs, sample_inputs):
        """Test model with post-norm produces valid outputs."""
        config = SomaticConfig(**base_config_kwargs, pre_norm=False, post_norm=True)
        model = SomaticModel(config)

        input_ids, token_type_ids = sample_inputs
        outputs = model(input_ids, token_type_ids=token_type_ids)

        assert not torch.isnan(outputs.last_hidden_state).any()

    def test_both_norm_forward(self, base_config_kwargs, sample_inputs):
        """Test model with both pre-norm and post-norm produces valid outputs."""
        config = SomaticConfig(**base_config_kwargs, pre_norm=True, post_norm=True)
        model = SomaticModel(config)

        input_ids, token_type_ids = sample_inputs
        outputs = model(input_ids, token_type_ids=token_type_ids)

        assert not torch.isnan(outputs.last_hidden_state).any()

    def test_qk_norm_forward(self, base_config_kwargs, sample_inputs):
        """Test model with QK normalization produces valid outputs."""
        config = SomaticConfig(**base_config_kwargs, qk_norm="norm")
        model = SomaticModel(config)

        input_ids, token_type_ids = sample_inputs
        outputs = model(input_ids, token_type_ids=token_type_ids)

        assert not torch.isnan(outputs.last_hidden_state).any()

    def test_qk_learned_scale_forward(self, base_config_kwargs, sample_inputs):
        """Test model with learned QK scaling produces valid outputs."""
        config = SomaticConfig(**base_config_kwargs, qk_norm="learned_scale")
        model = SomaticModel(config)

        input_ids, token_type_ids = sample_inputs
        outputs = model(input_ids, token_type_ids=token_type_ids)

        assert not torch.isnan(outputs.last_hidden_state).any()

    def test_combined_options(self, base_config_kwargs, sample_inputs):
        """Test model with multiple normalization options combined."""
        config = SomaticConfig(
            **base_config_kwargs,
            norm_type="rmsnorm",
            pre_norm=False,
            post_norm=True,
            qk_norm="norm",
        )
        model = SomaticModel(config)

        input_ids, token_type_ids = sample_inputs
        outputs = model(input_ids, token_type_ids=token_type_ids)

        assert not torch.isnan(outputs.last_hidden_state).any()

    def test_rmsnorm_layer_check(self, base_config_kwargs):
        """Test that RMSNorm layers are actually used when configured."""
        config = SomaticConfig(**base_config_kwargs, norm_type="rmsnorm")
        model = SomaticModel(config)

        # Check that final_norm is RMSNorm
        assert isinstance(model.encoder.final_norm, RMSNorm)

        # Check that block norms are RMSNorm
        block = model.encoder.layers[0]
        assert isinstance(block.attention_pre_norm, RMSNorm)
        assert isinstance(block.ffn_pre_norm, RMSNorm)

    def test_standard_attention_with_qk_norm(self, base_config_kwargs, sample_inputs):
        """Test standard attention (not chain-aware) with QK normalization."""
        config = SomaticConfig(
            **base_config_kwargs,
            use_chain_aware_attention=False,
            qk_norm="norm",
        )
        model = SomaticModel(config)

        input_ids, token_type_ids = sample_inputs
        outputs = model(input_ids, token_type_ids=token_type_ids)

        assert not torch.isnan(outputs.last_hidden_state).any()

    def test_save_load_with_normalization(self, base_config_kwargs, sample_inputs, tmp_path):
        """Test that models with normalization options save and load correctly."""
        config = SomaticConfig(
            **base_config_kwargs,
            norm_type="rmsnorm",
            pre_norm=True,
            post_norm=True,
            qk_norm="learned_scale",
        )
        model = SomaticModel(config)

        save_path = tmp_path / "model_norm"
        model.save_pretrained(str(save_path))

        loaded_model = SomaticModel.from_pretrained(str(save_path))

        # Check config matches
        assert loaded_model.config.norm_type == "rmsnorm"
        assert loaded_model.config.pre_norm is True
        assert loaded_model.config.post_norm is True
        assert loaded_model.config.qk_norm == "learned_scale"

        # Check outputs match
        input_ids, token_type_ids = sample_inputs
        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            out1 = model(input_ids, token_type_ids=token_type_ids)
            out2 = loaded_model(input_ids, token_type_ids=token_type_ids)

        assert torch.allclose(out1.last_hidden_state, out2.last_hidden_state)

    @pytest.mark.parametrize("use_chain_aware", [True, False])
    @pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
    @pytest.mark.parametrize("variant", ["standard", "star"])
    def test_hybrid_norm_forward(
        self, base_config_kwargs, sample_inputs, use_chain_aware, norm_type, variant
    ):
        """HybridNorm variants produce valid outputs for both attention modes and norm types."""
        config = SomaticConfig(
            **base_config_kwargs,
            hybrid_norm=variant,
            norm_type=norm_type,
            use_chain_aware_attention=use_chain_aware,
        )
        model = SomaticModel(config)

        input_ids, token_type_ids = sample_inputs
        outputs = model(input_ids, token_type_ids=token_type_ids)

        assert not torch.isnan(outputs.last_hidden_state).any()
        assert outputs.last_hidden_state.shape == (2, 32, config.hidden_size)

    def test_hybrid_norm_standard_layer_check(self, base_config_kwargs):
        """Standard HybridNorm blocks have only ffn_norm + attention QKV-norm."""
        config = SomaticConfig(**base_config_kwargs, hybrid_norm="standard")
        model = SomaticModel(config)

        for block in model.encoder.layers:
            assert block.hybrid_norm is True
            assert block.hybrid_first_layer is False
            assert block.attention_pre_norm is None
            assert block.ffn_pre_norm is None
            assert block.attention_post_norm is None
            assert block.ffn_post_norm is None
            assert block.ffn_norm is not None
            # Chain-aware attention by default → both QKV norms exist, no QK norms
            assert isinstance(block.attention.qkv_norm_self, QKVNormModule)
            assert isinstance(block.attention.qkv_norm_cross, QKVNormModule)
            assert block.attention.qk_norm_self is None
            assert block.attention.qk_norm_cross is None

    def test_hybrid_norm_star_layer_check(self, base_config_kwargs):
        """HybridNorm* layer 0 is Pre-Norm with QKV-norm; later layers are standard hybrid."""
        kwargs = {**base_config_kwargs, "num_hidden_layers": 3}
        config = SomaticConfig(**kwargs, hybrid_norm="star")
        model = SomaticModel(config)

        # Layer 0: Pre-Norm wiring (attention + FFN pre-norms; no ffn_norm)
        first = model.encoder.layers[0]
        assert first.hybrid_norm is True
        assert first.hybrid_first_layer is True
        assert first.attention_pre_norm is not None
        assert first.ffn_pre_norm is not None
        assert first.attention_post_norm is None
        assert first.ffn_post_norm is None
        assert first.ffn_norm is None
        # QKV-norm is still active at layer 0
        assert isinstance(first.attention.qkv_norm_self, QKVNormModule)
        assert isinstance(first.attention.qkv_norm_cross, QKVNormModule)

        # Subsequent layers: standard HybridNorm
        for block in model.encoder.layers[1:]:
            assert block.hybrid_norm is True
            assert block.hybrid_first_layer is False
            assert block.attention_pre_norm is None
            assert block.ffn_pre_norm is None
            assert block.ffn_norm is not None

    @pytest.mark.parametrize("variant", ["standard", "star"])
    def test_hybrid_norm_save_load(self, base_config_kwargs, sample_inputs, tmp_path, variant):
        """HybridNorm config round-trips through save_pretrained/from_pretrained."""
        config = SomaticConfig(
            **base_config_kwargs,
            hybrid_norm=variant,
            norm_type="rmsnorm",
        )
        model = SomaticModel(config)

        save_path = tmp_path / f"hybrid_norm_{variant}_model"
        model.save_pretrained(str(save_path))

        loaded_model = SomaticModel.from_pretrained(str(save_path))
        assert loaded_model.config.hybrid_norm == variant
        assert loaded_model.config.norm_type == "rmsnorm"

        input_ids, token_type_ids = sample_inputs
        model.eval()
        loaded_model.eval()
        with torch.no_grad():
            out1 = model(input_ids, token_type_ids=token_type_ids)
            out2 = loaded_model(input_ids, token_type_ids=token_type_ids)
        assert torch.allclose(out1.last_hidden_state, out2.last_hidden_state)

    def test_hybrid_norm_overrides_pre_post_qk(self, base_config_kwargs, sample_inputs):
        """When hybrid_norm=standard, pre_norm/post_norm/qk_norm flags are ignored."""
        config = SomaticConfig(
            **base_config_kwargs,
            hybrid_norm="standard",
            pre_norm=True,
            post_norm=True,
            qk_norm="norm",
        )
        model = SomaticModel(config)

        block = model.encoder.layers[0]
        # Pre/post norm layers must not be created in standard hybrid mode
        assert block.attention_pre_norm is None
        assert block.ffn_post_norm is None
        # Attention should not have a qk_norm_module
        assert block.attention.qk_norm_self is None
        assert block.attention.qk_norm_cross is None

        input_ids, token_type_ids = sample_inputs
        outputs = model(input_ids, token_type_ids=token_type_ids)
        assert not torch.isnan(outputs.last_hidden_state).any()
