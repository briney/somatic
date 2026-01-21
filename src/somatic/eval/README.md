# Evaluation Harness

This module provides a flexible, extensible system for computing validation metrics during training. It supports multiple evaluation datasets, distributed training, region-based antibody analysis, and custom metric implementations.

## Overview

Key features:
- **Multiple evaluation datasets** with independent metric configurations
- **Distributed training support** via Accelerate with automatic tensor/object gathering
- **Region-based evaluation** for antibody CDR/framework analysis
- **Extensible metric system** with decorator-based registration
- **Controlled evaluation masking** for reproducible comparisons

## Module Structure

```
src/somatic/eval/
├── __init__.py           # Public API exports
├── base.py               # Metric protocol and MetricBase class
├── evaluator.py          # Main Evaluator orchestration class
├── registry.py           # Metric registration and factory functions
├── regions.py            # Antibody region definitions and extraction
├── region_config.py      # Region evaluation configuration
├── masking.py            # Evaluation masking utilities
├── per_position.py       # Per-position and region-level evaluators
└── metrics/              # Built-in metric implementations
    ├── __init__.py
    ├── classification.py # Accuracy, perplexity, loss metrics
    ├── contact.py        # Contact prediction (Precision@L)
    ├── probes.py         # Linear probe metrics (chain, position, CDR)
    └── region.py         # Region-specific accuracy/loss metrics
```

## Evaluator Class

The `Evaluator` class orchestrates metric computation across evaluation datasets.

### Initialization

```python
from somatic.eval import Evaluator

evaluator = Evaluator(
    cfg=config,           # Full Hydra configuration
    model=model,          # SomaticModel instance
    accelerator=accel,    # Optional Accelerator for distributed training
)
```

### Key Methods

#### `evaluate(eval_loader, eval_name, masker=None)`

Evaluates a single dataset and returns metric results.

```python
results = evaluator.evaluate(
    eval_loader=val_dataloader,
    eval_name="validation",
    masker=uniform_masker,  # Optional, for legacy masking
)
# Returns: {"mask_acc": 0.85, "ppl": 12.3, ...}
```

**Masking priority order:**
1. `self.eval_masker` if configured (controlled, reproducible evaluation)
2. Passed `masker` parameter (legacy behavior)
3. Default 15% random masking fallback

#### `evaluate_all(eval_loaders, masker=None)`

Evaluates all configured datasets.

```python
all_results = evaluator.evaluate_all(
    eval_loaders={"validation": val_loader, "test": test_loader},
)
# Returns: {"validation": {...}, "test": {...}}
```

## Evaluation Workflow

When called by the `Trainer`, evaluation follows this workflow:

1. **Trainer triggers evaluation** at configured intervals:
   - Every `eval_steps` training steps
   - At checkpoint saves
   - After final training step

2. **Evaluator.evaluate_all()** is called with all eval dataloaders

3. **For each dataset:**
   - Build/retrieve cached metrics via `build_metrics()`
   - Reset all metric accumulators
   - Iterate through batches:
     - Apply masking (eval_masker, passed masker, or 15% random)
     - Forward pass through model
     - Update each metric with outputs and batch
   - Gather states across distributed processes
   - Compute final metric values
   - Optionally run region-based evaluation

4. **Results are used for:**
   - Logging to wandb/tensorboard
   - Best model checkpoint selection
   - Training monitoring

```python
# In Trainer.train():
if should_eval:
    all_eval_metrics = self.evaluate_all()
    if self.logger:
        self.logger.log_eval_all(all_eval_metrics, step=self.global_step)
```

## Metric Protocol

All metrics must implement the `Metric` protocol:

```python
from somatic.eval import Metric

class Metric(Protocol):
    # Class attributes
    name: ClassVar[str]              # Unique identifier for logging
    requires_coords: ClassVar[bool]  # True if metric needs 3D coordinates
    needs_attentions: ClassVar[bool] # True if metric needs attention weights

    def update(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate values from a batch."""
        ...

    def compute(self) -> dict[str, float]:
        """Compute final metric values from accumulated state."""
        ...

    def reset(self) -> None:
        """Reset state for new evaluation run."""
        ...

    def state_tensors(self) -> list[Tensor]:
        """Return state as tensors for distributed aggregation."""
        ...

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Load state from gathered tensors."""
        ...
```

### Method Details

| Method | Purpose |
|--------|---------|
| `update()` | Called once per batch during evaluation. Accumulates statistics (e.g., correct predictions, total loss). |
| `compute()` | Called after all batches. Returns final metric values as `{metric_name: value}`. |
| `reset()` | Called before evaluation starts. Clears accumulated state. |
| `state_tensors()` | For distributed training. Returns internal state as tensors that can be gathered across processes. |
| `load_state_tensors()` | Receives gathered tensors and updates internal state. |

## Creating Custom Metrics

### Step 1: Define the Metric Class

```python
from typing import ClassVar
import torch
from torch import Tensor

from somatic.eval.base import MetricBase
from somatic.eval.registry import register_metric


@register_metric("my_metric")  # Register with unique name
class MyMetric(MetricBase):
    """Description of what this metric measures."""

    name: ClassVar[str] = "my_metric"       # Used in result keys
    requires_coords: ClassVar[bool] = False  # Set True if needs coords
    needs_attentions: ClassVar[bool] = False # Set True if needs attention

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        super().__init__()
        self.threshold = threshold
        # Initialize accumulators
        self._correct: int = 0
        self._total: int = 0

    def update(
        self,
        outputs: dict[str, Tensor | tuple[Tensor, ...]],
        batch: dict[str, Tensor | None],
        mask_labels: Tensor,
    ) -> None:
        """Accumulate metric from batch."""
        logits = outputs["logits"]
        targets = batch["token_ids"]
        mask = mask_labels.bool()

        predictions = logits.argmax(dim=-1)
        correct = (predictions == targets) & mask

        self._correct += correct.sum().item()
        self._total += mask.sum().item()

    def compute(self) -> dict[str, float]:
        """Compute final metric value."""
        if self._total == 0:
            return {self.name: 0.0}
        return {self.name: self._correct / self._total}

    def reset(self) -> None:
        """Reset accumulators."""
        self._correct = 0
        self._total = 0

    def state_tensors(self) -> list[Tensor]:
        """Return state for distributed gathering."""
        return [torch.tensor([float(self._correct), float(self._total)])]

    def load_state_tensors(self, tensors: list[Tensor]) -> None:
        """Load gathered state."""
        if tensors:
            state = tensors[0]
            self._correct = int(state[0].item())
            self._total = int(state[1].item())
```

### Step 2: Import in metrics/__init__.py

```python
# In src/somatic/eval/metrics/__init__.py
from .my_module import MyMetric

__all__ = [
    # ... existing exports
    "MyMetric",
]
```

### Step 3: Configure in Hydra

```yaml
# In configs/eval.yaml or dataset-specific config
eval:
  metrics:
    my_metric:
      enabled: true
      threshold: 0.7  # Custom parameter
```

### Best Practices

1. **Use `MetricBase`** for simpler implementation with default `state_tensors()`
2. **Accept `**kwargs`** in `__init__` to handle extra config parameters gracefully
3. **Handle edge cases** (empty batches, missing data) without raising exceptions
4. **Return multiple metrics** by including multiple keys in `compute()` return dict
5. **Use descriptive names** that clearly indicate what's being measured

## Available Metrics Reference

| Metric Name | Class | Description | Requires |
|-------------|-------|-------------|----------|
| `masked_accuracy` | `MaskedAccuracyMetric` | Accuracy on masked positions | - |
| `perplexity` | `PerplexityMetric` | exp(cross-entropy loss) | - |
| `loss` | `LossMetric` | Average cross-entropy loss | - |
| `p_at_l` | `PrecisionAtLMetric` | Contact prediction precision@L | coords, attentions |
| `chain_probe` | `ChainProbeMetric` | Chain identity classification | - |
| `position_probe` | `PositionProbeMetric` | Relative position prediction | - |
| `cdr_probe` | `CDRProbeMetric` | CDR membership classification | - |
| `region_accuracy` | `RegionAccuracyMetric` | Per-region masked accuracy | - |
| `region_perplexity` | `RegionPerplexityMetric` | Per-region perplexity | - |
| `region_loss` | `RegionLossMetric` | Per-region loss | - |

## Region-Based Evaluation

The harness supports detailed evaluation of antibody structural regions.

### Antibody Regions

Regions follow standard antibody structure:

| Region Type | Heavy Chain | Light Chain |
|-------------|-------------|-------------|
| CDR1 | `hcdr1` | `lcdr1` |
| CDR2 | `hcdr2` | `lcdr2` |
| CDR3 | `hcdr3` | `lcdr3` |
| Framework 1 | `hfwr1` | `lfwr1` |
| Framework 2 | `hfwr2` | `lfwr2` |
| Framework 3 | `hfwr3` | `lfwr3` |
| Framework 4 | `hfwr4` | `lfwr4` |

### Aggregate Groups

| Aggregate | Description |
|-----------|-------------|
| `all_cdr` | All CDR regions combined |
| `all_fwr` | All framework regions combined |
| `heavy` | All heavy chain regions |
| `light` | All light chain regions |
| `overall` | All regions combined |
| `germline` | Positions matching germline (non_templated_mask == 0) |
| `nongermline` | Mutated positions (non_templated_mask == 1) |

### Evaluation Modes

1. **Standard** (`mode: "standard"`): Computes metrics on masked positions within each region
2. **Per-position** (`mode: "per-position"`): Masks and evaluates each position individually
3. **Region-level** (`mode: "region-level"`): Masks entire regions at once

### Configuration

```yaml
eval:
  regions:
    enabled: true
    mode: per-position
    position_batch_size: 32

    # Individual regions (all default to false)
    hcdr3: true
    lcdr3: true

    # Aggregate groups
    all_cdr: true
    germline: true
    nongermline: true
```

### Programmatic Access

```python
from somatic.eval.regions import (
    AntibodyRegion,
    CDR_REGIONS,
    FWR_REGIONS,
    extract_region_masks,
)

# Extract region masks from batch
region_masks = extract_region_masks(
    batch,
    regions={AntibodyRegion.HCDR3, AntibodyRegion.LCDR3},
)

# region_masks: {AntibodyRegion.HCDR3: Tensor, AntibodyRegion.LCDR3: Tensor}
```

## Distributed Training Support

The evaluator handles distributed training automatically through two mechanisms:

### Tensor-Based Gathering

For metrics with fixed-size state (counters, sums):

```python
def state_tensors(self) -> list[Tensor]:
    return [torch.tensor([self._total, float(self._count)])]

def load_state_tensors(self, tensors: list[Tensor]) -> None:
    state = tensors[0]
    self._total = state[0].item()
    self._count = int(state[1].item())
```

The evaluator uses `accelerator.gather()` and sums across processes.

### Object-Based Gathering

For metrics with variable-length state (lists of features):

```python
def state_objects(self) -> list[Any] | None:
    return {"features": self._features, "targets": self._targets}

def load_state_objects(self, gathered: list[Any]) -> None:
    all_features = []
    for item in gathered:
        if item is not None:
            all_features.extend(item.get("features", []))
    self._features = all_features[:self.n_train]
```

The evaluator uses `accelerator.gather_for_metrics(use_gather_object=True)`.

**When to use each:**
- **Tensor-based**: Simple counters, sums, fixed-size accumulators
- **Object-based**: Variable-length lists, complex Python objects, probe training data

## Configuration Reference

### Global Metric Configuration

```yaml
# configs/eval.yaml
eval:
  metrics:
    masked_accuracy:
      enabled: true
    perplexity:
      enabled: true
    p_at_l:
      enabled: true
      contact_threshold: 8.0
      min_seq_sep: 6
      num_layers: null  # Auto: 10% of encoder layers
```

### Per-Dataset Configuration

```yaml
# configs/data.yaml
data:
  eval:
    validation:
      path: /path/to/val.csv
      metrics:
        only: [masked_accuracy, perplexity]  # Whitelist metrics
    test:
      path: /path/to/test.csv
      load_coords: true  # Enable for this dataset
      metrics:
        p_at_l:
          enabled: true
          contact_threshold: 6.0  # Override global setting
```

### Metric Whitelisting

Use `only` to restrict which metrics run on a dataset:

```yaml
data:
  eval:
    quick_val:
      path: /path/to/quick.csv
      metrics:
        only: [masked_accuracy]  # Only run accuracy
```

### Evaluation Masking

For reproducible evaluation with controlled masking:

```yaml
eval:
  masking:
    mask_rate: 0.15
    seed: 42  # Reproducible masking
```
