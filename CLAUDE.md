# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Somatic is an antibody language model that overcomes germline bias by learning somatic hypermutation patterns from paired heavy/light chain sequences. It uses information-weighted masked language modeling with chain-aware attention (MINT-style hybrid self/cross attention).

## Installation

```bash
pip install -e .   # development install
pip install somatic-lm  # from PyPI
```

## Common Commands

### Training
```bash
somatic train data.train=/path/to/train.csv
somatic train data.train=/path/to/data.csv model=small train.batch_size=64
accelerate launch -m somatic.train data.train=/path/to/sequences.csv  # multi-GPU
```

### Encoding/Inference
```bash
somatic encode -c checkpoints/best.pt -i data/seqs.csv -o embeddings.pt --pooling mean
somatic model-size model=large
```

### Testing
```bash
pytest
pytest tests/unit/test_transformer.py
pytest tests/unit/test_transformer.py::test_forward_pass -v
```

## Architecture

### Forward Pass Data Flow

```
token_ids, chain_ids
    → SomaticEmbedding (TokenEmbedding scaled by sqrt(d_model) + dropout)
    → TransformerEncoder (n_layers × TransformerBlock)
        Each block: PreNorm → ChainAwareAttention → residual → PreNorm → SwiGLU FFN → residual
        Final RMSNorm after all layers
    → lm_head (Linear, weight-tied to TokenEmbedding)
    → {logits, hidden_states}
```

### Chain-Aware Attention (`model/attention.py`)

The core innovation: 6 separate projections (q/k/v for both self and cross paths) instead of 3. Self-path gets RoPE; cross-path has no positional encoding (position is meaningful within a chain but not across chains). A single merged softmax lets all positions compete, then attention weights are split by chain membership to route through separate value matrices. Standard `MultiHeadAttention` accepts `chain_ids` but ignores it, making the two attention classes interchangeable.

### Tokenizer (`tokenizer.py`)

Fixed 32-token vocabulary (multiple of 8 for GPU efficiency). `encode_paired(heavy, light)` produces `[CLS] heavy_tokens light_tokens [EOS]` with chain IDs 0 (CLS + heavy) and 1 (light + EOS). A module-level singleton `tokenizer = Tokenizer()` is used everywhere.

### Data Pipeline (`data/`)

`AntibodyCollator` is the hub — assembles `token_ids`, `chain_ids`, `attention_mask`, `special_tokens_mask`, and optional `cdr_mask` (0=FW, 1=CDR1, 2=CDR2, 3=CDR3), `non_templated_mask`, `coords`. Default CSV columns follow AIRR format: `sequence_aa:0` (heavy), `sequence_aa:1` (light), with optional `cdr_mask_aa:0/1` and `nongermline_mask_aa:0/1`. Long sequences are silently truncated to `max_length`.

### Information-Weighted Masking (`masking/masking.py`)

`InformationWeightedMasker` (default) computes per-position weights: `w = 1 + cdr_binary * cdr_multiplier + nt_binary * nt_multiplier`. With default multipliers, CDR+non-templated positions get weight 3, either alone gets 2, framework templated gets 1. Uses the **Gumbel-top-k trick** for weighted sampling without replacement. Falls back to `UniformMasker` when CDR/NT masks are unavailable.

### Eval System (`eval/`)

**Metric pattern:** `Metric` is a `@runtime_checkable Protocol` defining the interface. `MetricBase` is an ABC providing default `reset()`, `state_tensors()` (returns `[tensor([total, count])]`), and `load_state_tensors()`. Metrics that need variable-length state (e.g., logistic regression features for probes) override `state_objects()` instead, which triggers `accelerator.gather_for_metrics(use_gather_object=True)`.

**Registry:** `@register_metric("name")` decorator populates `METRIC_REGISTRY`. `build_metrics()` triggers the import to ensure decorators fire. Region metrics (`RegionAccuracyMetric`, `RegionPerplexityMetric`, `RegionLossMetric`) are NOT registered — they're invoked separately through `_evaluate_regions`.

**Classification metric hierarchy:**
- `_MaskedCrossEntropyMetric` → `PerplexityMetric`, `LossMetric`
- `_RegionCrossEntropyMetric` → `RegionPerplexityMetric`, `RegionLossMetric`

### Configuration System

Hydra configs in `src/somatic/configs/`. Composition order: `model/base.yaml` → `train/default.yaml` → `data/default.yaml` → `masking/default.yaml` → `log/default.yaml` → `eval/default.yaml` → `_self_`.

Model variants: `small` (24 layers, d=256), `base` (56 layers, d=384), `large` (128 layers, d=512).

CLI entry: `somatic` command → `cli.py:main()`. For `accelerate launch`, uses `train.py:main()` (argparse-based for accelerate compatibility). Config resolution first checks for a local `configs/` dir, then falls back to bundled package configs.

### Key Implementation Details

- `lm_head` weight is tied to embedding weight; `get_num_params()` subtracts embeddings to avoid double-counting
- `d_ffn` defaults to `int(d_model * 8/3)` rounded up to nearest multiple of 64 (hardware alignment)
- The scheduler is intentionally NOT wrapped by `accelerator.prepare()` — `AcceleratedScheduler` causes 8x step rate in multi-GPU DDP
- RoPE precomputes sin/cos cache and auto-extends for longer sequences at inference
- `need_weights=False` uses `F.scaled_dot_product_attention` (enables Flash Attention); `True` falls back to manual matmul
- `from_pretrained` pops legacy fields `max_timesteps` and `use_timestep_embedding` from config

### Checkpoint Format

Training: `{step, epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, metrics, config}`
Inference: `{config, model_state_dict}` (via `SomaticModel.save_pretrained/from_pretrained`)

### Test Fixtures (`tests/conftest.py`)

- `small_config`: `SomaticConfig(d_model=64, n_layers=2, n_heads=2, max_seq_len=64, dropout=0.0)`
- `small_model`: `SomaticModel(small_config)`
- `sample_batch`: batch_size=2, seq_len=32, token range 4–27 (amino acids only, excludes special tokens and mask). Does not include `cdr_mask`, `non_templated_mask`, or `special_tokens_mask` — tests needing those create them inline.
