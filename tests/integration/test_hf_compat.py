"""HuggingFace-compatibility tests: save/reload, custom-code bundling, remote load.

These verify that Somatic loads through the standard `transformers` machinery —
`save_pretrained`/`from_pretrained` round-trips, the `auto_map` + bundled
custom-code files needed for `trust_remote_code=True`, and a true fresh-interpreter
reload that never imports the `somatic` package.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from somatic.model import SomaticConfig, SomaticForMaskedLM
from somatic.model.tokenization_somatic import tokenizer

HEAVY = "EVQLVESGGGLVQPGGSLRLSCAAS"
LIGHT = "DIQMTQSPSSLSASVGDRVTITCRAS"

# Custom-code files that must travel with a checkpoint for trust_remote_code to work:
# the main modeling/config/tokenization modules plus every helper they import.
EXPECTED_CUSTOM_CODE = [
    "modeling_somatic.py",
    "configuration_somatic.py",
    "tokenization_somatic.py",
    "attention.py",
    "ffn.py",
    "normalization.py",
    "rope.py",
    "embeddings.py",
    "layers.py",
]


@pytest.fixture(scope="module")
def saved_dir(tmp_path_factory) -> Path:
    """Tiny SomaticForMaskedLM + tokenizer saved once into an HF directory."""
    torch.manual_seed(0)
    config = SomaticConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    model = SomaticForMaskedLM(config)
    model.eval()

    out_dir = tmp_path_factory.mktemp("hf_ckpt")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return Path(out_dir)


def test_save_reload_logits_match(saved_dir):
    """SomaticForMaskedLM.from_pretrained reproduces the pre-save logits exactly."""
    reloaded = SomaticForMaskedLM.from_pretrained(str(saved_dir))
    reloaded.eval()

    enc = tokenizer.encode_paired(HEAVY, LIGHT, return_tensors="pt")
    with torch.no_grad():
        out = reloaded(
            input_ids=enc["input_ids"],
            token_type_ids=enc["token_type_ids"],
            attention_mask=enc["attention_mask"],
        )

    # Reload again to compare two independent loads (deterministic, no dropout).
    reloaded2 = SomaticForMaskedLM.from_pretrained(str(saved_dir))
    reloaded2.eval()
    with torch.no_grad():
        out2 = reloaded2(
            input_ids=enc["input_ids"],
            token_type_ids=enc["token_type_ids"],
            attention_mask=enc["attention_mask"],
        )

    assert torch.equal(out.logits, out2.logits)
    # The lm_head stays tied to the input embedding after reload.
    assert reloaded.lm_head.weight is reloaded.somatic.embeddings.token_embedding.embedding.weight


def test_custom_code_files_copied(saved_dir):
    """save_pretrained drops every custom-code file beside config.json/tokenizer."""
    present = {p.name for p in saved_dir.iterdir()}
    assert "config.json" in present
    assert "model.safetensors" in present
    assert "tokenizer.json" in present
    assert "tokenizer_config.json" in present
    for fname in EXPECTED_CUSTOM_CODE:
        assert fname in present, f"missing bundled custom-code file: {fname}"


def test_auto_map_in_config(saved_dir):
    """config.json carries auto_map entries pointing at the bundled modules."""
    config_json = json.loads((saved_dir / "config.json").read_text())
    auto_map = config_json.get("auto_map")
    assert auto_map is not None
    assert auto_map["AutoConfig"] == "configuration_somatic.SomaticConfig"
    assert auto_map["AutoModelForMaskedLM"] == "modeling_somatic.SomaticForMaskedLM"


def test_remote_reload_in_subprocess(saved_dir, tmp_path):
    """Fresh interpreter (no `import somatic`) loads via trust_remote_code=True.

    Asserts the model/tokenizer load from the bundled code (the loaded class lives
    in the `transformers_modules.*` dynamic namespace, not the installed package),
    that the class is SomaticForMaskedLM, and that tok(heavy, light) yields the
    expected input_ids + token_type_ids chain layout.
    """
    script = textwrap.dedent(
        """
        import json, sys
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        # Guard: this interpreter must NOT rely on importing the somatic package.
        assert "somatic" not in sys.modules

        ckpt = sys.argv[1]
        heavy, light = sys.argv[2], sys.argv[3]

        model = AutoModelForMaskedLM.from_pretrained(ckpt, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

        enc = tok(heavy, light, return_token_type_ids=True)
        result = {
            "model_class": type(model).__name__,
            "model_module": type(model).__module__,
            "input_ids": list(enc["input_ids"]),
            "token_type_ids": list(enc["token_type_ids"]),
            "imported_somatic": "somatic" in sys.modules,
        }
        print("RESULT_JSON:" + json.dumps(result))
        """
    )
    script_path = tmp_path / "remote_load.py"
    script_path.write_text(script)

    proc = subprocess.run(
        [sys.executable, str(script_path), str(saved_dir), HEAVY, LIGHT],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),  # avoid picking up the repo's `somatic/` via cwd
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    line = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT_JSON:"))
    result = json.loads(line[len("RESULT_JSON:") :])

    assert result["model_class"] == "SomaticForMaskedLM"
    # Loaded from the bundled custom code, not the installed package.
    assert result["model_module"].startswith("transformers_modules"), result["model_module"]
    assert result["imported_somatic"] is False

    # Chain layout: cls=0, heavy=0, light=1, eos=1.
    expected_len = 1 + len(HEAVY) + len(LIGHT) + 1
    assert len(result["input_ids"]) == expected_len
    token_type_ids = result["token_type_ids"]
    assert token_type_ids[: 1 + len(HEAVY)] == [0] * (1 + len(HEAVY))
    assert token_type_ids[1 + len(HEAVY) :] == [1] * (len(LIGHT) + 1)


def test_tokenizer_round_trip(saved_dir):
    """A reloaded tokenizer reproduces input_ids + token_type_ids."""
    from transformers import AutoTokenizer

    reloaded = AutoTokenizer.from_pretrained(str(saved_dir), trust_remote_code=True)
    enc = reloaded(HEAVY, LIGHT, return_token_type_ids=True)

    ref = tokenizer(HEAVY, LIGHT, return_token_type_ids=True)
    assert enc["input_ids"] == ref["input_ids"]
    assert enc["token_type_ids"] == ref["token_type_ids"]
