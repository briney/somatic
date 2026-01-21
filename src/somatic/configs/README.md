# Somatic Configuration Guide

This directory contains Hydra configuration files for training, evaluating, and configuring Somatic models. Hydra enables hierarchical configuration composition with command-line overrides.

## Usage

### Custom Config Files

Use the `-c` flag to provide a custom YAML configuration file that extends or overrides defaults:

```bash
somatic train -c my_config.yaml data.train=/path/to/data.csv
```

Example custom config (`my_config.yaml`):
```yaml
defaults:
  - model: small

train:
  batch_size: 64
  max_steps: 50000

masking:
  cdr_weight_multiplier: 2.0
```

### CLI Overrides

Override any configuration option directly from the command line:

```bash
# Basic overrides
somatic train train.batch_size=64 train.max_steps=50000

# Nested overrides
somatic train train.optimizer.learning_rate=1e-4

# Config group selection
somatic train model=small train=debug

# Adding new keys (use + prefix)
somatic train +experiment=baseline

# Multiple overrides
somatic train model=large train.batch_size=16 masking.mask_rate=0.20
```

---

## Global Configuration (`config.yaml`)

The main configuration file that composes all sub-configs.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | string | `somatic_experiment` | Experiment name for logging and outputs |
| `seed` | int | `42` | Random seed for reproducibility |
| `output_dir` | string | `outputs/${name}` | Output directory for checkpoints and logs |

### Composition

The main config composes defaults from:
```yaml
defaults:
  - model: base
  - train: default
  - data: default
  - masking: default
  - log: default
  - eval: default
```

---

## Model Configuration (`model/`)

Three model size variants are available: `small`, `base` (default), and `large`.

### Model Sizes

| Variant | `d_model` | `n_layers` | `n_heads` | Parameters |
|---------|-----------|------------|-----------|------------|
| `small` | 256 | 24 | 4 | ~19M (24M with chain-aware attention) |
| `base` | 384 | 56 | 6 | ~99M (124M with chain-aware attention) |
| `large` | 512 | 128 | 8 | ~411M (512M with chain-aware attention) |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `vocab_size` | int | `32` | Token vocabulary size |
| `padding_idx` | int | `1` | Padding token index |
| `d_model` | int | varies | Embedding/hidden dimension |
| `n_layers` | int | varies | Number of transformer layers |
| `n_heads` | int | varies | Number of attention heads |
| `d_ffn` | int | `null` | FFN intermediate dimension (auto-computed if null) |
| `ffn_multiplier` | float | `null` | FFN size multiplier (alternative to `d_ffn`) |
| `max_seq_len` | int | `320` | Maximum sequence length |
| `dropout` | float | `0.1` | General dropout rate |
| `attention_dropout` | float | `0.1` | Attention-specific dropout |
| `embedding_dropout` | float | `0.1` | Embedding layer dropout |
| `use_chain_aware_attention` | bool | `true` | Enable chain-aware (MINT-style) attention |
| `norm_type` | string | `layernorm` | Normalization type: `layernorm` \| `rmsnorm` |
| `pre_norm` | bool | `true` | Use pre-normalization (recommended) |
| `post_norm` | bool | `false` | Use post-normalization |
| `qk_norm` | string | `none` | QK normalization: `none` \| `norm` \| `learned_scale` |
| `layer_norm_eps` | float | `1e-6` | LayerNorm epsilon value |

### Usage Examples

```bash
# Use small model
somatic train model=small data.train=/path/to/data.csv

# Override model parameters
somatic train model.d_model=512 model.n_layers=32

# Disable chain-aware attention
somatic train model.use_chain_aware_attention=false
```

---

## Training Configuration (`train/`)

Two variants: `default` and `debug`.

### Options

| Option | Type | Default | Debug | Description |
|--------|------|---------|-------|-------------|
| `max_steps` | int | `100000` | `100` | Maximum training steps |
| `max_epochs` | int | `null` | `null` | Maximum epochs (null = unlimited) |
| `batch_size` | int | `32` | `4` | Training batch size |
| `gradient_accumulation_steps` | int | `1` | `1` | Gradient accumulation steps |
| `max_grad_norm` | float | `1.0` | `1.0` | Gradient clipping threshold |
| `mixed_precision` | string | `"no"` | `"no"` | Mixed precision mode |

#### Optimizer (`train.optimizer`)

| Option | Type | Default | Debug | Description |
|--------|------|---------|-------|-------------|
| `name` | string | `adamw` | `adamw` | Optimizer name |
| `learning_rate` | float | `3e-4` | `1e-4` | Learning rate |
| `weight_decay` | float | `0.01` | `0.01` | Weight decay |
| `betas` | list | `[0.9, 0.999]` | `[0.9, 0.999]` | Adam beta parameters |

#### Scheduler (`train.scheduler`)

| Option | Type | Default | Debug | Description |
|--------|------|---------|-------|-------------|
| `decay` | string | `linear` | `constant` | LR decay: `linear` \| `cosine` \| `constant` |
| `warmup_steps` | int | `10000` | `10` | Warmup steps |
| `min_lr_ratio` | float | `0.0` | `0.1` | Minimum LR as ratio of initial LR |

#### Checkpointing

| Option | Type | Default | Debug | Description |
|--------|------|---------|-------|-------------|
| `log_steps` | int | `10` | `1` | Logging frequency (steps) |
| `eval_steps` | int | `5000` | `50` | Evaluation frequency (steps) |
| `checkpoint_steps` | int | `50000` | `50` | Checkpoint frequency (steps) |
| `checkpoint_dir` | string | `${output_dir}/checkpoints` | same | Checkpoint directory |
| `keep_last_n_checkpoints` | int | `5` | `2` | Number of checkpoints to keep |
| `save_best` | bool | `true` | `false` | Save best model by validation loss |

### FLOPs Tracking (`train/flops/`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `flops.enabled` | bool | `true` | Enable FLOPs and MFU tracking |

### Masking Frequency Tracking (`train/masking_frequency/`)

Two variants: `default` and `detailed`.

| Option | Type | Default | Detailed | Description |
|--------|------|---------|----------|-------------|
| `masking_frequency.enabled` | bool | `true` | `true` | Master switch |
| `masking_frequency.hcdr1` | bool | `false` | `true` | Track heavy CDR1 |
| `masking_frequency.hcdr2` | bool | `false` | `true` | Track heavy CDR2 |
| `masking_frequency.hcdr3` | bool | `true` | `true` | Track heavy CDR3 |
| `masking_frequency.lcdr1` | bool | `false` | `true` | Track light CDR1 |
| `masking_frequency.lcdr2` | bool | `false` | `true` | Track light CDR2 |
| `masking_frequency.lcdr3` | bool | `true` | `true` | Track light CDR3 |
| `masking_frequency.hfwr1-4` | bool | `false` | `false` | Track heavy FWRs |
| `masking_frequency.lfwr1-4` | bool | `false` | `false` | Track light FWRs |
| `masking_frequency.all_cdr` | bool | `true` | `true` | Combined CDR stats |
| `masking_frequency.all_fwr` | bool | `true` | `true` | Combined FWR stats |
| `masking_frequency.heavy` | bool | `true` | `false` | All heavy chain regions |
| `masking_frequency.light` | bool | `true` | `false` | All light chain regions |
| `masking_frequency.overall` | bool | `true` | `true` | Total masking statistics |
| `masking_frequency.germline` | bool | `true` | `true` | Germline positions |
| `masking_frequency.nongermline` | bool | `true` | `true` | Non-germline positions |

### Usage Examples

```bash
# Use debug configuration
somatic train train=debug data.train=/path/to/data.csv

# Custom training settings
somatic train train.batch_size=64 train.optimizer.learning_rate=1e-4

# Enable detailed masking frequency tracking
somatic train train/masking_frequency=detailed
```

---

## Data Configuration (`data/`)

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `train` | string/dict | `null` | Training data path(s) - **required** |
| `eval` | string/dict | `null` | Evaluation data path(s) - optional |
| `max_length` | int | `320` | Maximum sequence length |
| `num_workers` | int | `4` | DataLoader worker processes |
| `pin_memory` | bool | `true` | Pin memory for faster GPU transfer |
| `drop_last` | bool | `true` | Drop incomplete final batch |
| `pad_to_max` | bool | `false` | Pad all sequences to max_length |

#### Column Names

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `heavy_col` | string | `sequence_aa:0` | Heavy chain sequence column |
| `light_col` | string | `sequence_aa:1` | Light chain sequence column |
| `heavy_cdr_col` | string | `cdr_mask_aa:0` | Heavy chain CDR mask column |
| `light_cdr_col` | string | `cdr_mask_aa:1` | Light chain CDR mask column |
| `heavy_nongermline_col` | string | `nongermline_mask_aa:0` | Heavy non-germline mask column |
| `light_nongermline_col` | string | `nongermline_mask_aa:1` | Light non-germline mask column |

#### Coordinate Loading

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `load_coords` | bool | `false` | Load structure coordinates |
| `heavy_coords_col` | string | `heavy_coords` | Heavy chain coordinates column |
| `light_coords_col` | string | `light_coords` | Light chain coordinates column |

### Dataset Formats

**Single dataset:**
```bash
somatic train data.train=/path/to/train.parquet
```

**Multiple datasets with weighted sampling:**
```yaml
# In config file
data:
  train:
    dataset_a:
      path: /path/to/dataset_a.parquet
      fraction: 0.6
    dataset_b:
      path: /path/to/dataset_b.parquet
      fraction: 0.4
```

```bash
# Via CLI
somatic train \
  +data.train.dataset_a.path=/path/to/a.parquet \
  +data.train.dataset_a.fraction=0.6 \
  +data.train.dataset_b.path=/path/to/b.parquet \
  +data.train.dataset_b.fraction=0.4
```

**Multiple evaluation datasets:**
```yaml
data:
  eval:
    validation: /path/to/val.parquet
    test: /path/to/test.parquet
```

**Per-dataset configuration:**
```yaml
data:
  eval:
    seq_val:
      path: /path/to/seq_val.parquet
      load_coords: false
      metrics:
        only: [masked_accuracy, perplexity]
    struct_val:
      path: /path/to/struct_val.parquet
      load_coords: true
      metrics:
        only: [masked_accuracy, p_at_l]
```

---

## Masking Configuration (`masking/`)

Controls MLM masking behavior during training.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mask_rate` | float | `0.15` | Fraction of tokens to mask |
| `use_information_weighted_masking` | bool | `true` | Enable weighted masking |
| `cdr_weight_multiplier` | float | `1.0` | Weight boost for CDR positions |
| `nongermline_weight_multiplier` | float | `1.0` | Weight boost for non-germline positions |
| `masking_selection` | string | `sampled` | Selection method: `sampled` \| `ranked` |

### Selection Methods

- **`sampled`** (recommended): Probabilistic selection using Gumbel-top-k sampling
- **`ranked`**: Deterministic selection of top-K highest-weighted positions

### Usage Examples

```bash
# Increase CDR masking probability
somatic train masking.cdr_weight_multiplier=2.0

# Use ranked selection
somatic train masking.masking_selection=ranked

# Higher mask rate
somatic train masking.mask_rate=0.20
```

---

## Evaluation Configuration (`eval/`)

### Metrics (`eval/default.yaml`)

#### Core Metrics

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `metrics.masked_accuracy.enabled` | bool | `true` | Compute masked token accuracy |
| `metrics.perplexity.enabled` | bool | `true` | Compute perplexity |
| `metrics.loss.enabled` | bool | `true` | Compute loss |

#### Contact Prediction (`metrics.p_at_l`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable P@L metric |
| `contact_threshold` | float | `8.0` | Contact distance threshold (Å) |
| `min_seq_sep` | int | `6` | Minimum sequence separation |
| `use_attention` | bool | `true` | Use attention weights |
| `attention_layer` | string | `"last"` | Layer selection: `"last"` \| `"mean"` \| layer index |
| `head_aggregation` | string | `"mean"` | Head aggregation: `"mean"` \| `"max"` |
| `num_layers` | int | `null` | Layers to use (null = 10% of total) |
| `use_logistic_regression` | bool | `false` | Train LogReg on attention |
| `logreg_n_train` | int | `20` | Sequences for LogReg training |
| `logreg_lambda` | float | `0.15` | L2 regularization |

#### Probe Metrics

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chain_probe.enabled` | bool | `false` | Chain classification probe |
| `chain_probe.pool_strategy` | string | `"mean"` | Pooling strategy |
| `chain_probe.n_train` | int | `100` | Training samples |
| `chain_probe.regularization` | float | `0.1` | Regularization strength |
| `position_probe.enabled` | bool | `false` | Position prediction probe |
| `position_probe.n_bins` | int | `3` | Position bins |
| `position_probe.sample_per_seq` | int | `10` | Samples per sequence |
| `cdr_probe.enabled` | bool | `false` | CDR classification probe |
| `cdr_probe.sample_per_seq` | int | `20` | Samples per sequence |

### Masking (`eval/masking/`)

Controls masking during evaluation for reproducible comparisons.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `masking.type` | string | `uniform` | Masking type: `uniform` \| `information_weighted` |
| `masking.mask_rate` | float | `0.15` | Evaluation mask rate |
| `masking.seed` | int | `42` | Random seed for reproducibility |
| `masking.cdr_weight_multiplier` | float | `1.0` | CDR weight (if information_weighted) |
| `masking.nongermline_weight_multiplier` | float | `1.0` | Non-germline weight |
| `masking.selection_method` | string | `sampled` | Selection method |

### Regions (`eval/regions/`)

Two variants: `default` and `detailed`.

| Option | Type | Default | Detailed | Description |
|--------|------|---------|----------|-------------|
| `regions.enabled` | bool | `true` | `true` | Enable region evaluation |
| `regions.mode` | string | `per-position` | `per-position` | Evaluation mode |
| `regions.position_batch_size` | int | `32` | `32` | Batch size for per-position eval |

#### Evaluation Modes

- **`standard`**: Uses existing eval masking, computes metrics on masked positions (fast)
- **`per-position`**: Mask one position at a time, aggregate by region (most detailed, slow)
- **`region-level`**: Mask entire regions at once (tests region reconstruction)

#### Region Toggles

| Region | Default | Detailed | Description |
|--------|---------|----------|-------------|
| `hcdr1` | `false` | `true` | Heavy CDR1 |
| `hcdr2` | `false` | `true` | Heavy CDR2 |
| `hcdr3` | `true` | `true` | Heavy CDR3 |
| `lcdr1` | `false` | `true` | Light CDR1 |
| `lcdr2` | `false` | `true` | Light CDR2 |
| `lcdr3` | `true` | `true` | Light CDR3 |
| `hfwr1-4` | `false` | `false` | Heavy framework regions |
| `lfwr1-4` | `false` | `false` | Light framework regions |
| `all_cdr` | `true` | `true` | Combined CDR stats |
| `all_fwr` | `true` | `true` | Combined FWR stats |
| `heavy` | `true` | `false` | All heavy chain |
| `light` | `true` | `false` | All light chain |
| `overall` | `false` | `true` | Total across all regions |
| `germline` | `true` | `true` | Germline positions |
| `nongermline` | `true` | `true` | Non-germline positions |

### Usage Examples

```bash
# Use detailed region evaluation
somatic train eval/regions=detailed

# Enable chain probe
somatic train eval.metrics.chain_probe.enabled=true

# Custom contact threshold
somatic train eval.metrics.p_at_l.contact_threshold=6.0
```

---

## Logging Configuration (`log/`)

Controls Weights & Biases logging.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `wandb.enabled` | bool | `true` | Enable W&B logging |
| `wandb.project` | string | `somatic` | W&B project name |
| `wandb.entity` | string | `null` | W&B team/entity |
| `wandb.name` | string | `${name}` | Run name (uses experiment name) |
| `wandb.tags` | list | `[]` | Run tags |
| `wandb.notes` | string | `null` | Run notes/description |

### Usage Examples

```bash
# Disable W&B logging
somatic train log.wandb.enabled=false

# Custom project and entity
somatic train log.wandb.project=my_project log.wandb.entity=my_team

# Add tags
somatic train 'log.wandb.tags=["baseline", "v1"]'
```

---

## Complete Example

```bash
# Full training command with multiple overrides
somatic train \
  model=small \
  train=default \
  train.batch_size=64 \
  train.optimizer.learning_rate=1e-4 \
  train.max_steps=200000 \
  masking.cdr_weight_multiplier=2.0 \
  masking.nongermline_weight_multiplier=1.5 \
  eval/regions=detailed \
  log.wandb.project=antibody_lm \
  data.train=/path/to/train.parquet \
  data.eval=/path/to/eval.parquet \
  name=cdr_weighted_run \
  seed=123
```

```bash
# Multi-GPU training with accelerate
accelerate launch -m somatic.train \
  model=large \
  train.batch_size=8 \
  train.gradient_accumulation_steps=4 \
  data.train=/path/to/train.parquet
```
