"""Tests for the main Somatic transformer model."""

import pytest
import torch

from somatic.model import SomaticConfig, SomaticForMaskedLM, SomaticModel
from somatic.model.layers import TransformerBlock
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
        assert config.norm_strategy == "pre"
        assert config.qk_norm == "none"
        assert config.norm_eps == 1e-6

    def test_invalid_norm_type(self):
        with pytest.raises(ValueError, match="norm_type"):
            SomaticConfig(norm_type="invalid")

    def test_invalid_qk_norm(self):
        with pytest.raises(ValueError, match="qk_norm"):
            SomaticConfig(qk_norm="invalid")

    def test_invalid_norm_strategy_raises(self):
        with pytest.raises(ValueError, match="norm_strategy"):
            SomaticConfig(norm_strategy="bogus")

    @pytest.mark.parametrize("strategy", ["pre", "hybrid", "sandwich"])
    def test_valid_norm_strategies(self, strategy):
        config = SomaticConfig(norm_strategy=strategy)
        assert config.norm_strategy == strategy

    def test_rmsnorm_config(self):
        config = SomaticConfig(norm_type="rmsnorm")
        assert config.norm_type == "rmsnorm"

    def test_qk_norm_options(self):
        config = SomaticConfig(qk_norm="norm")
        assert config.qk_norm == "norm"

        config = SomaticConfig(qk_norm="learned_scale")
        assert config.qk_norm == "learned_scale"

    def test_qk_norm_validated_under_hybrid(self):
        # qk_norm is validated unconditionally, even though it is ignored at
        # runtime under the hybrid strategy (QKV-norm replaces it).
        with pytest.raises(ValueError, match="qk_norm"):
            SomaticConfig(norm_strategy="hybrid", qk_norm="bogus")

    def test_config_dict_missing_norm_strategy(self):
        # Configs that omit norm_strategy default cleanly to "pre".
        d = {
            "vocab_size": 32,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
        }
        config = SomaticConfig(**d)
        assert config.norm_strategy == "pre"

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

    @pytest.mark.parametrize("use_chain_aware", [True, False])
    @pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
    @pytest.mark.parametrize("strategy", ["pre", "hybrid", "sandwich"])
    def test_norm_strategy_forward(
        self, base_config_kwargs, sample_inputs, strategy, norm_type, use_chain_aware
    ):
        """Each norm strategy produces valid outputs across norm types and attention modes."""
        config = SomaticConfig(
            **base_config_kwargs,
            norm_strategy=strategy,
            norm_type=norm_type,
            use_chain_aware_attention=use_chain_aware,
        )
        model = SomaticModel(config)

        input_ids, token_type_ids = sample_inputs
        outputs = model(input_ids, token_type_ids=token_type_ids)

        assert not torch.isnan(outputs.last_hidden_state).any()
        assert outputs.last_hidden_state.shape == (2, 32, config.hidden_size)

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
            norm_strategy="sandwich",
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

        # Check that block pre-norms are RMSNorm
        block = model.encoder.layers[0]
        assert isinstance(block.attn_norm, RMSNorm)
        assert isinstance(block.ffn_norm, RMSNorm)

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

    @pytest.mark.parametrize("strategy", ["pre", "hybrid", "sandwich"])
    def test_block_norm_submodules(self, base_config_kwargs, strategy):
        """Each strategy instantiates exactly the norm submodules it needs."""
        config = SomaticConfig(**base_config_kwargs, norm_strategy=strategy)
        model = SomaticModel(config)

        for block in model.encoder.layers:
            assert block.norm_strategy == strategy
            assert block.ffn_norm is not None  # FFN pre-norm is always present
            if strategy == "pre":
                assert block.attn_norm is not None
                assert block.attn_post_norm is None
                assert block.ffn_post_norm is None
            elif strategy == "sandwich":
                assert block.attn_norm is not None
                assert block.attn_post_norm is not None
                assert block.ffn_post_norm is not None
            else:  # hybrid: no outer attention pre-norm, no post-norms; QKV-norm in attention
                assert block.attn_norm is None
                assert block.attn_post_norm is None
                assert block.ffn_post_norm is None
                # Chain-aware attention by default → QKV norms exist, QK norms absent
                assert isinstance(block.attention.qkv_norm_self, QKVNormModule)
                assert isinstance(block.attention.qkv_norm_cross, QKVNormModule)
                assert block.attention.qk_norm_self is None
                assert block.attention.qk_norm_cross is None

    def test_sandwich_norm_residual_formula(self):
        """Sandwich-LN normalizes each sublayer output OUTSIDE the residual stream.

        Confirms x_out = h + Norm(FFN(Norm(h))) with h = x + Norm(Attn(Norm(x))) —
        i.e. the post-norm acts on the sublayer output before the residual add, not
        on the residual sum (the latter was the old, incorrect "both pre+post" path).
        """
        torch.manual_seed(0)
        block = TransformerBlock(
            d_model=64,
            n_heads=2,
            head_dim=32,
            d_ffn=128,
            dropout=0.0,
            attention_dropout=0.0,
            max_seq_len=64,
            norm_type="layernorm",
            norm_strategy="sandwich",
        )
        block.eval()

        x = torch.randn(2, 16, 64)
        token_type_ids = torch.zeros(2, 16).long()

        with torch.no_grad():
            out = block(x, token_type_ids)

            attn_out = block.attention(block.attn_norm(x), token_type_ids, need_weights=False)
            h = x + block.attn_post_norm(attn_out)
            ffn_out = block.ffn(block.ffn_norm(h))
            expected = h + block.ffn_post_norm(ffn_out)

        assert torch.allclose(out, expected, atol=1e-5)

    def test_save_load_with_normalization(self, base_config_kwargs, sample_inputs, tmp_path):
        """Test that models with normalization options save and load correctly."""
        config = SomaticConfig(
            **base_config_kwargs,
            norm_type="rmsnorm",
            norm_strategy="sandwich",
            qk_norm="learned_scale",
        )
        model = SomaticModel(config)

        save_path = tmp_path / "model_norm"
        model.save_pretrained(str(save_path))

        loaded_model = SomaticModel.from_pretrained(str(save_path))

        # Check config matches
        assert loaded_model.config.norm_type == "rmsnorm"
        assert loaded_model.config.norm_strategy == "sandwich"
        assert loaded_model.config.qk_norm == "learned_scale"

        # Check outputs match
        input_ids, token_type_ids = sample_inputs
        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            out1 = model(input_ids, token_type_ids=token_type_ids)
            out2 = loaded_model(input_ids, token_type_ids=token_type_ids)

        assert torch.allclose(out1.last_hidden_state, out2.last_hidden_state)

    def test_hybrid_norm_save_load(self, base_config_kwargs, sample_inputs, tmp_path):
        """HybridNorm config round-trips through save_pretrained/from_pretrained."""
        config = SomaticConfig(
            **base_config_kwargs,
            norm_strategy="hybrid",
            norm_type="rmsnorm",
        )
        model = SomaticModel(config)

        save_path = tmp_path / "hybrid_norm_model"
        model.save_pretrained(str(save_path))

        loaded_model = SomaticModel.from_pretrained(str(save_path))
        assert loaded_model.config.norm_strategy == "hybrid"
        assert loaded_model.config.norm_type == "rmsnorm"

        input_ids, token_type_ids = sample_inputs
        model.eval()
        loaded_model.eval()
        with torch.no_grad():
            out1 = model(input_ids, token_type_ids=token_type_ids)
            out2 = loaded_model(input_ids, token_type_ids=token_type_ids)
        assert torch.allclose(out1.last_hidden_state, out2.last_hidden_state)

        input_ids, token_type_ids = sample_inputs
        outputs = model(input_ids, token_type_ids=token_type_ids)
        assert not torch.isnan(outputs.last_hidden_state).any()
