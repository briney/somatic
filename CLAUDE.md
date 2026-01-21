# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Somatic is an antibody language model designed to overcome the germline bias problem by learning patterns of somatic hypermutation from paired heavy/light chain sequences. It uses an information-weighted masked language modeling (MLM) objective with a modern transformer architecture featuring chain-aware attention.

## Installation
```bash
pip install somatic-lm
```
  
## Common Commands

### Training
```bash
# Single GPU training via CLI
somatic train data.train=/path/to/train.csv

# Multi-GPU training with accelerate
accelerate launch -m somatic.train data.train=/path/to/train.csv

# With config overrides
somatic train -c my_config.yaml data.train=/path/to/data.csv model=small
```

### Encoding/Inference
```bash
# Encode sequences to embeddings
somatic encode -c checkpoints/best.pt -i data/seqs.csv -o embeddings.pt --pooling mean

# Check model size
somatic model-size model=large
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_transformer.py

# Run specific test
pytest tests/unit/test_transformer.py::test_forward_pass -v
```




## Architecture

### Core Components

**Model (`src/somatic/model/`):**
- `SomaticModel` - Main transformer with MLM head (weight-tied to embedding)
- `SomaticConfig` - Dataclass for model hyperparameters
- `ChainAwareAttention` - Hybrid self/cross attention for paired heavy/light chains (MINT-style)
- Pre-norm transformer with RoPE and SwiGLU FFN

**Tokenizer (`src/somatic/tokenizer.py`):**
- Fixed 32-token vocabulary (amino acids + special tokens)
- `tokenizer.encode_paired(heavy, light)` returns `{input_ids, chain_ids, attention_mask}`
- Chain IDs: 0 for heavy chain positions, 1 for light chain positions
- Format: `<cls> heavy_sequence light_sequence <eos>`

**Encoding (`src/somatic/encoding/`):**
- `SomaticEncoder` - High-level API for inference
- Pooling strategies: mean, cls, max, mean_max, or none (full sequence)

**Data (`src/somatic/data/`):**
- `AntibodyCollator` - Handles batching with variable-length sequences
- Expects CSV/parquet with `heavy_chain` and `light_chain` columns
- Optional CDR masks and non-templated region masks for weighted masking

**Training (`src/somatic/training/`):**
- `Trainer` - Main training loop with accelerate for distributed training
- `TrainingConfig` - Training hyperparameters
- Checkpoint management, gradient accumulation, mixed precision

**Evaluation (`src/somatic/eval/`):**
- Region-based evaluation (CDR1/2/3, framework regions)
- Per-position metrics, masking frequency tracking

### Configuration System

Uses Hydra for hierarchical configs in `src/somatic/configs/`:
- `config.yaml` - Main config composing defaults
- `model/` - Model configs (base, small, large)
- `train/` - Training configs
- `data/` - Data loading configs
- `masking/` - Masking strategy configs
- `eval/` - Evaluation configs

Override with CLI: `somatic train model=small train.batch_size=64`

### Test Structure

- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for data/model pipelines
- `tests/e2e/` - End-to-end training loop tests
- `tests/conftest.py` - Shared fixtures (`small_config`, `small_model`, `sample_batch`)
