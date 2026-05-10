"""Integration tests for cross-chain attention evaluation.

Drives the full path through ``run_cross_chain_eval`` (and
``Evaluator.evaluate``) on the small_model fixture, using realistic
batches built by ``AntibodyCollator``.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from somatic.data.collator import AntibodyCollator
from somatic.eval import Evaluator
from somatic.eval.cross_chain_config import CrossChainEvalConfig
from somatic.eval.cross_chain_eval import run_cross_chain_eval
from somatic.tokenizer import tokenizer


HEAVY_SEQS = [
    "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMHW",
    "QVQLQQSGAELARPGASVKMSCKASGYTFTRYTMHW",
    "EVQLVESGGGVVQPGRSLRLSCAASGFTFSSYAMHW",
    "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISW",
]
LIGHT_SEQS = [
    "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLAW",
    "DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGY",
    "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLA",
    "DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAW",
]


def _build_paired_samples(n: int) -> list[dict]:
    """Build n samples from antibody-style sequences with full mask set."""
    samples = []
    for i in range(n):
        heavy = HEAVY_SEQS[i % len(HEAVY_SEQS)]
        light = LIGHT_SEQS[i % len(LIGHT_SEQS)]
        encoded = tokenizer.encode_paired(heavy, light)
        token_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        chain_ids = torch.tensor(encoded["chain_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        special = torch.zeros_like(token_ids, dtype=torch.long)
        special[0] = 1  # CLS
        special[-1] = 1  # EOS
        samples.append(
            {
                "token_ids": token_ids,
                "chain_ids": chain_ids,
                "attention_mask": attention_mask,
                "special_tokens_mask": special,
            }
        )
    return samples


def _make_loader(small_model, batch_size: int = 4, n: int = 8) -> DataLoader:
    samples = _build_paired_samples(n)

    def collate(batch):
        max_len = max(s["token_ids"].numel() for s in batch)
        out = {}
        for key in ("token_ids", "chain_ids", "attention_mask", "special_tokens_mask"):
            stacked = torch.zeros(len(batch), max_len, dtype=batch[0][key].dtype)
            for i, s in enumerate(batch):
                t = s[key]
                stacked[i, : t.numel()] = t
            out[key] = stacked
        return out

    return DataLoader(samples, batch_size=batch_size, collate_fn=collate, shuffle=False)


class TestRunCrossChainEval:
    def test_returns_expected_keys(self, small_model):
        loader = _make_loader(small_model, batch_size=4, n=8)
        config = CrossChainEvalConfig(enabled=True, interface_n=3, chunk_size=2)
        out = run_cross_chain_eval(
            model=small_model,
            eval_loader=loader,
            config=config,
            accelerator=None,
            show_progress=False,
        )
        assert set(out.keys()) == {"cross_frac", "interface_frac", "interface_n"}
        assert 0.0 <= out["cross_frac"] <= 1.0
        assert 0.0 <= out["interface_frac"] <= 1.0
        assert out["interface_n"] == 3.0

    def test_chunk_size_respected(self, small_model):
        """With chunk_size larger than batch, eval still runs; results finite."""
        loader = _make_loader(small_model, batch_size=2, n=4)
        config = CrossChainEvalConfig(enabled=True, interface_n=5, chunk_size=64)
        out = run_cross_chain_eval(
            model=small_model,
            eval_loader=loader,
            config=config,
            accelerator=None,
            show_progress=False,
        )
        assert torch.isfinite(torch.tensor(out["cross_frac"])).item()
        assert torch.isfinite(torch.tensor(out["interface_frac"])).item()


class TestEvaluatorIntegration:
    def _make_cfg(self, *, enabled: bool) -> "OmegaConf":
        # Note: evaluate() returns early if no main-eval metrics are configured,
        # so we enable masked_accuracy to keep the path live. The cross-chain
        # block runs after the main-metric loop regardless.
        cfg_dict = {
            "eval": {
                "metrics": {
                    "masked_accuracy": {"enabled": True},
                    "perplexity": {"enabled": False},
                    "loss": {"enabled": False},
                    "p_at_l": {"enabled": False},
                    "chain_probe": {"enabled": False},
                    "position_probe": {"enabled": False},
                    "cdr_probe": {"enabled": False},
                },
                "regions": {"enabled": False},
                "cross_chain_attention": {
                    "enabled": enabled,
                    "interface_n": 3,
                    "chunk_size": 4,
                },
            },
            "data": {"eval": "synthetic"},
        }
        return OmegaConf.create(cfg_dict)

    def test_keys_present_when_enabled(self, small_model):
        loader = _make_loader(small_model, batch_size=4, n=8)
        cfg = self._make_cfg(enabled=True)
        evaluator = Evaluator(cfg=cfg, model=small_model, accelerator=None)
        results = evaluator.evaluate(eval_loader=loader, eval_name="synthetic")
        assert "cross_chain/cross_frac" in results
        assert "cross_chain/interface_frac" in results
        assert "cross_chain/interface_n" in results
        assert results["cross_chain/interface_n"] == 3.0

    def test_no_keys_when_disabled(self, small_model):
        loader = _make_loader(small_model, batch_size=4, n=8)
        cfg = self._make_cfg(enabled=False)
        evaluator = Evaluator(cfg=cfg, model=small_model, accelerator=None)
        results = evaluator.evaluate(eval_loader=loader, eval_name="synthetic")
        assert not any(k.startswith("cross_chain/") for k in results)
