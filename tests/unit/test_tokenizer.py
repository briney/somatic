"""Tests for tokenizer."""

from somatic.model.tokenization_somatic import (
    AA_END_IDX,
    AA_START_IDX,
    DEFAULT_VOCAB,
    SomaticTokenizerFast,
    tokenizer,
)


class TestTokenizer:
    def test_vocab_size(self):
        assert len(tokenizer) == 32
        assert tokenizer.vocab_size == 32

    def test_special_token_ids(self):
        assert tokenizer.cls_token_id == 0
        assert tokenizer.pad_token_id == 1
        assert tokenizer.eos_token_id == 2
        assert tokenizer.unk_token_id == 3
        assert tokenizer.mask_token_id == 31

    def test_convert_tokens_to_ids(self):
        assert tokenizer.convert_tokens_to_ids("<cls>") == 0
        assert tokenizer.convert_tokens_to_ids("L") == 4
        assert tokenizer.convert_tokens_to_ids("<mask>") == 31

    def test_unknown_token(self):
        assert tokenizer.convert_tokens_to_ids("?") == tokenizer.unk_token_id

    def test_encode_simple(self):
        sequence = "LAG"
        encoded = tokenizer.encode(sequence, add_special_tokens=False)
        assert encoded == [4, 5, 6]

    def test_encode_with_special_tokens(self):
        sequence = "LA"
        encoded = tokenizer.encode(sequence, add_special_tokens=True)
        assert encoded[0] == tokenizer.cls_token_id
        assert encoded[-1] == tokenizer.eos_token_id
        assert len(encoded) == 4

    def test_roundtrip(self):
        sequence = "EVQLVESGGGLVQ"
        encoded = tokenizer.encode(sequence, add_special_tokens=False)
        # HF tokenizers add spaces between tokens, so remove them
        decoded = tokenizer.decode(encoded, skip_special_tokens=True).replace(" ", "")
        assert decoded == sequence

    def test_aa_index_constants(self):
        assert AA_START_IDX == 4
        assert AA_END_IDX == 30

    def test_default_vocab_length(self):
        assert len(DEFAULT_VOCAB) == 32

    def test_tokenizer_instance(self):
        # Test that creating a new tokenizer works
        tok = SomaticTokenizerFast()
        assert len(tok) == 32
        assert tok.cls_token_id == 0


class TestEncodePaired:
    """Tests for encode_paired method."""

    def test_encode_paired(self):
        """Test paired encoding emits input_ids/token_type_ids/attention_mask."""
        result = tokenizer.encode_paired("AC", "DE")

        # Check input_ids: <cls> A C D E <eos>
        assert result["input_ids"][0] == tokenizer.cls_token_id
        assert result["input_ids"][-1] == tokenizer.eos_token_id
        assert len(result["input_ids"]) == 6  # CLS + 2 + 2 + EOS

        # Check token_type_ids: [0, 0, 0, 1, 1, 1] (CLS+heavy=0, light+EOS=1)
        assert result["token_type_ids"] == [0, 0, 0, 1, 1, 1]

        # Check attention_mask
        assert result["attention_mask"] == [1, 1, 1, 1, 1, 1]

    def test_encode_paired_return_tensors(self):
        """Test paired encoding with PyTorch tensor output."""
        import torch

        result = tokenizer.encode_paired("AC", "DE", return_tensors="pt")

        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["token_type_ids"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)

        # Should have batch dimension
        assert result["input_ids"].shape == (1, 6)
        assert result["token_type_ids"].shape == (1, 6)

    def test_encode_paired_longer_sequences(self):
        """Test paired encoding with longer realistic sequences."""
        heavy = "EVQLVESGGGLVQ"
        light = "DIQMTQSPSS"
        result = tokenizer.encode_paired(heavy, light)

        expected_len = 1 + len(heavy) + len(light) + 1  # CLS + heavy + light + EOS
        assert len(result["input_ids"]) == expected_len
        assert len(result["token_type_ids"]) == expected_len

        # Heavy chain tokens should have token_type_id 0
        heavy_type_ids = result["token_type_ids"][: 1 + len(heavy)]
        assert all(c == 0 for c in heavy_type_ids)

        # Light chain tokens should have token_type_id 1
        light_type_ids = result["token_type_ids"][1 + len(heavy) :]
        assert all(c == 1 for c in light_type_ids)
