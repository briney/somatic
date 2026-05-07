# Attention Optimization And Ablation Implementation Plan

## Purpose

Somatic currently uses chain-aware attention to treat paired antibody chains differently from
ordinary single-sequence transformer inputs. The current mechanism has useful inductive bias,
but it is also more expensive than standard multi-head attention because it maintains separate
self and cross Q/K/V projections and computes dense self and cross attention paths over the full
concatenated sequence.

This document is the implementation plan for an ablation-first optimization effort. The first
goal is not to rewrite the current attention kernel. The first goal is to make the architecture
question measurable:

- Does separate-QKV chain-aware attention provide enough modeling benefit to justify its extra
  parameters and compute?
- Can a simpler shared-QKV chain-aware attention variant retain most or all of the benefit?
- Does a parameter-matched standard attention model match or beat both chain-aware variants?

Only after this ablation should the project invest in a more complex exact blockwise
implementation of the current separate-QKV mechanism.

## Current Behavior

The current implementation lives in `src/somatic/model/attention.py`.

`MultiHeadAttention` uses one Q/K/V projection path, applies RoPE to Q and K, then calls
PyTorch scaled dot-product attention when attention weights are not requested.

`ChainAwareAttention` implements the current chain-aware mechanism:

1. Project the input into separate self-attention tensors:
   `q_self`, `k_self`, `v_self`.
2. Project the input into separate cross-attention tensors:
   `q_cross`, `k_cross`, `v_cross`.
3. Apply RoPE only to `q_self` and `k_self`.
4. Compute dense self scores for every query/key pair.
5. Compute dense cross scores for every query/key pair.
6. Select self scores for intra-chain residue pairs and cross scores for inter-chain residue pairs.
7. Apply one global softmax over the merged score matrix.
8. Route attention probabilities through `v_self` for intra-chain pairs and `v_cross` for
   inter-chain pairs.

Mathematically, for query position `i` and key/value position `j`:

```text
score_ij = dot(q_self_i, k_self_j) / sqrt(d_head)   if chain_i == chain_j
score_ij = dot(q_cross_i, k_cross_j) / sqrt(d_head) if chain_i != chain_j

p_i = softmax(score_i + padding_mask)

out_i = sum_j p_ij * v_self_j  if chain_i == chain_j
      + sum_j p_ij * v_cross_j if chain_i != chain_j
```

The single global softmax is an important semantic detail. Same-chain and partner-chain residues
compete for one attention budget. A naive implementation that runs separate intra-chain and
inter-chain attentions and then adds the outputs would change the model because it would normalize
those groups separately.

## Goals

- Add a shared-QKV chain-aware attention variant that preserves the current chain-aware global
  softmax structure but removes the separate self/cross projection paths.
- Expose the new variant through config while keeping current behavior as the default.
- Add tests that prove shape, masking, dispatch, and equivalence properties.
- Add FLOPs accounting that distinguishes standard attention, separate-QKV chain-aware attention,
  and shared-QKV chain-aware attention.
- Add a documented pilot training protocol with comparison commands and decision criteria.
- Avoid implementation changes that would invalidate existing checkpoints by default.

## Non-Goals

- Do not make shared-QKV attention the default in this implementation.
- Do not add custom CUDA kernels, FlashAttention extensions, block-sparse kernels, or other
  hard-to-maintain backend dependencies.
- Do not migrate existing checkpoints.
- Do not implement the exact blockwise optimization yet.
- Do not add HybridNorm support for shared-QKV attention in the first pass unless explicitly
  requested later.

## Proposed Public Interface

### Model Configuration

Add a new field to `SomaticConfig` in `src/somatic/model/transformer.py`:

```python
chain_aware_projection_mode: str = "separate"
```

Valid values:

- `"separate"`: current separate self/cross Q/K/V chain-aware attention.
- `"shared"`: new shared-QKV chain-aware attention.

Validation rules:

- Validate the field even when `use_chain_aware_attention` is false so typos fail early.
- Keep `"separate"` as the default so older saved configs and existing training behavior remain
  compatible.
- If `use_chain_aware_attention=false`, ignore `chain_aware_projection_mode` during dispatch.
- If `chain_aware_projection_mode="shared"` and `hybrid_norm != "none"`, raise a clear
  `ValueError` explaining that shared-QKV chain-aware attention does not support HybridNorm yet.

Checkpoint compatibility:

- Older checkpoints that do not include `chain_aware_projection_mode` should load as `"separate"`
  because the dataclass default supplies the field.
- New checkpoints should save the field through the existing `asdict(self.config)` path.
- No state dict migration is required because the default still instantiates the existing
  `ChainAwareAttention` module.

### Attention Dispatch

Update `TransformerBlock` in `src/somatic/model/layers.py` to accept and dispatch on
`chain_aware_projection_mode`.

Dispatch rules:

```text
use_chain_aware_attention=false
    -> MultiHeadAttention

use_chain_aware_attention=true, chain_aware_projection_mode="separate"
    -> ChainAwareAttention

use_chain_aware_attention=true, chain_aware_projection_mode="shared"
    -> SharedQKVChainAwareAttention
```

Thread the new parameter through:

- `SomaticModel.__init__`
- `TransformerEncoder.__init__`
- `TransformerBlock.__init__`

Keep the existing attention forward interface unchanged:

```python
forward(
    x: Tensor,
    chain_ids: Tensor,
    attention_mask: Tensor | None = None,
    need_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]
```

## Shared-QKV Chain-Aware Attention Design

Add a new module class in `src/somatic/model/attention.py`:

```python
class SharedQKVChainAwareAttention(BaseAttention):
    """Chain-aware attention with shared Q/K/V projections."""
```

The class should reuse existing base utilities:

- `BaseAttention._create_padding_mask`
- `BaseAttention._create_chain_mask` logic can either be factored from
  `ChainAwareAttention` or duplicated as a small private helper.
- `BaseAttention.rope`
- `BaseAttention.dropout`
- `BaseAttention.out_proj`
- existing QK norm modules where appropriate.

### Forward Algorithm

Inputs:

- `x`: `(batch, seq_len, d_model)`
- `chain_ids`: `(batch, seq_len)`
- `attention_mask`: optional `(batch, seq_len)`
- `need_weights`: whether to return `(batch, n_heads, seq_len, seq_len)` attention weights.

Steps:

1. Project once:

   ```text
   q = q_proj(x)
   k = k_proj(x)
   v = v_proj(x)
   ```

   Reshape each projection to `(batch, n_heads, seq_len, head_dim)`.

2. Create RoPE versions for intra-chain scores:

   ```text
   q_rope, k_rope = rope(q, k)
   ```

3. Keep unrotated versions for inter-chain scores:

   ```text
   q_nope = q
   k_nope = k
   ```

4. Apply QK normalization if configured:

   - Use one shared QK norm module, not separate self/cross norm modules.
   - Apply that same module to the RoPE pair for intra-chain scores.
   - Apply that same module to the unrotated pair for inter-chain scores.
   - This keeps the projection path shared while preserving the existing ordering where RoPE is
     applied before QK norm.

5. Compute score matrices:

   ```text
   scores_intra = matmul(q_rope, k_rope.T) * scale
   scores_inter = matmul(q_nope, k_nope.T) * scale
   ```

6. Build the intra-chain mask:

   ```text
   intra_mask[b, 1, i, j] = chain_ids[b, i] == chain_ids[b, j]
   ```

7. Merge scores before softmax:

   ```text
   merged_scores = where(intra_mask, scores_intra, scores_inter)
   ```

8. Add padding mask if provided. Padding should mask key/value positions, matching existing
   attention behavior.

9. Apply one global softmax along the key dimension.

10. Apply `torch.nan_to_num(..., nan=0.0)` to preserve current all-masked-row behavior.

11. Apply attention dropout.

12. Use a single value path:

   ```text
   output = matmul(attn_weights, v)
   ```

13. Reshape to `(batch, seq_len, inner_dim)` and apply `out_proj`.

14. Return either `output` or `(output, attn_weights)` depending on `need_weights`.

### Why This Variant Is Useful

Shared-QKV chain-aware attention keeps two core pieces of the current inductive bias:

- intra-chain pairs see RoPE-aware scores;
- inter-chain pairs see no-position scores;
- both pair types compete in one global softmax.

It removes a different piece of the current design:

- separate self/cross Q/K/V projection subspaces.

That makes the ablation scientifically meaningful. It tests whether relation-specific projections
are necessary, rather than only testing whether chain-aware positional handling is useful.

### Expected Efficiency Difference

Compared with current separate-QKV chain-aware attention, the first shared-QKV implementation
should reduce:

- attention projection parameters from six projection matrices to three;
- attention projection FLOPs from six linear projections to three;
- value matmul work from two masked value matmuls to one value matmul.

It will still compute two dense score matrices:

- one RoPE score matrix for intra-chain candidates;
- one no-position score matrix for inter-chain candidates.

Therefore, this is a parameter and projection/value-path simplification, not a full quadratic
attention optimization. If the ablation shows separate-QKV attention is worth keeping, a later
blockwise exact implementation can target the remaining wasted score computation.

## File-Level Implementation Plan

### `src/somatic/model/attention.py`

- Add `SharedQKVChainAwareAttention`.
- Add Q/K/V projections named consistently with `MultiHeadAttention`:
  `q_proj`, `k_proj`, `v_proj`.
- Use a single QK norm module for shared mode.
- Do not implement HybridNorm in the first pass.
- Keep dropout behavior identical to existing attention modules.
- Keep attention weight return semantics identical to `ChainAwareAttention`.
- Consider factoring `_create_chain_mask` into `BaseAttention` only if it keeps the code clearer.
  This helper is small enough that duplication is acceptable if it avoids unnecessary refactoring.

### `src/somatic/model/layers.py`

- Import `SharedQKVChainAwareAttention`.
- Add `chain_aware_projection_mode` to `TransformerBlock.__init__`.
- Replace the current two-class dispatch with explicit branch logic.
- Pass all existing constructor arguments through unchanged.
- Thread `chain_aware_projection_mode` through `TransformerEncoder.__init__` and each
  `TransformerBlock`.

### `src/somatic/model/transformer.py`

- Add `chain_aware_projection_mode` to `SomaticConfig`.
- Validate allowed values in `__post_init__`.
- Raise for unsupported `shared` plus HybridNorm combinations.
- Pass the field into `TransformerEncoder`.
- Ensure `from_pretrained` continues to work for older checkpoints through the dataclass default.

### `src/somatic/model/__init__.py`

- Export `SharedQKVChainAwareAttention`.

### Hydra Configs

Update bundled model configs:

- `src/somatic/configs/model/small.yaml`
- `src/somatic/configs/model/base.yaml`
- `src/somatic/configs/model/large.yaml`

Add:

```yaml
chain_aware_projection_mode: separate
```

Place it next to `use_chain_aware_attention` so the relationship is obvious.

### Config Documentation

Update `src/somatic/configs/README.md`:

- Document `chain_aware_projection_mode`.
- Add example commands:

```bash
somatic train model=small model.chain_aware_projection_mode=shared
somatic train model=small model.use_chain_aware_attention=false
```

Clarify that the projection mode is ignored when `use_chain_aware_attention=false`.

### FLOPs Reporting

Update `src/somatic/training/flops.py`.

Current FLOPs accounting uses a single `qkv_factor` and does not distinguish all attention
paths. Replace that logic with explicit factors:

```text
standard MHA:
    projection_factor = 3
    score_factor = 1
    value_factor = 1

separate-QKV chain-aware:
    projection_factor = 6
    score_factor = 2
    value_factor = 2

shared-QKV chain-aware:
    projection_factor = 3
    score_factor = 2
    value_factor = 1
```

Use these factors in the per-layer estimate:

```text
projection_flops = projection_factor * 2 * d_model * d_model
score_flops = score_factor * 2 * seq_len * d_model
value_flops = value_factor * 2 * seq_len * d_model
out_projection_flops = 2 * d_model * d_model
ffn_flops = 6 * d_model * d_ffn
```

Keep the existing public function names unless a broader reporting refactor is requested later:

- `compute_model_flops_per_token`
- `compute_training_flops_per_token`

### `docs/ATTN_OPTIMIZATION.md`

This file is the implementation plan and should remain as the standalone reference for the
ablation. After implementation, update it with actual pilot results or create a separate results
document and link to it.

## Test Plan

### Unit Tests For Shared-QKV Attention

Add tests in `tests/unit/test_attention.py`.

Required cases:

- Forward shape:
  - input `(batch=2, seq_len=32, d_model=64)`;
  - two chains split halfway;
  - output shape equals input shape.
- Padding mask:
  - mask the last positions;
  - output has correct shape and no NaNs.
- Multiple chains:
  - chain IDs `[0, 1, 2]` in contiguous blocks;
  - output has correct shape.
- Attention weights:
  - `need_weights=True`;
  - weights shape is `(batch, n_heads, seq_len, seq_len)`;
  - rows sum to one for valid, unmasked rows.
- Determinism:
  - set dropout to zero and module to eval mode;
  - repeated calls produce identical outputs.

### Equivalence Tests

Add tests that copy weights from `MultiHeadAttention` into `SharedQKVChainAwareAttention`.

Same-chain equivalence:

- Set all `chain_ids` to zero.
- Use the same Q/K/V/out projection weights.
- Use `dropout=0.0`.
- Use the same `rope_fraction`.
- Assert outputs match within `atol=1e-5`.

NoPE equivalence:

- Set `rope_fraction=0.0`.
- Use arbitrary mixed chain IDs.
- Copy Q/K/V/out projection weights from `MultiHeadAttention`.
- Assert outputs match within `atol=1e-5`.

These tests prove:

- shared-QKV chain-aware attention collapses to standard attention when every pair is intra-chain;
- chain routing does not change scores when RoPE is disabled.

### Config And Dispatch Tests

Add tests in `tests/unit/test_transformer.py` or `tests/unit/test_attention.py`.

Required cases:

- Default config instantiates current `ChainAwareAttention`.
- `use_chain_aware_attention=false` instantiates `MultiHeadAttention`.
- `use_chain_aware_attention=true` and `chain_aware_projection_mode="shared"` instantiates
  `SharedQKVChainAwareAttention`.
- Invalid projection mode raises `ValueError`.
- `chain_aware_projection_mode="shared"` with `hybrid_norm != "none"` raises `ValueError`.
- Save/load roundtrip preserves `chain_aware_projection_mode="shared"`.

### Training Smoke Test

Add or update a tiny training smoke test covering:

- standard MHA;
- separate-QKV chain-aware attention;
- shared-QKV chain-aware attention.

The smoke test should use very small dimensions and only a few steps. It should validate that:

- forward pass succeeds;
- loss is finite;
- backward pass succeeds;
- optimizer step succeeds.

Mark as slow only if runtime becomes noticeable.

### Minimum Validation Commands

Run:

```bash
pytest tests/unit/test_attention.py tests/unit/test_transformer.py
```

If the smoke test lives outside those files, also run the specific smoke test file.

Before merging, run the full test suite when practical:

```bash
pytest
```

## Pilot Ablation Protocol

The pilot should use the same real train and validation data for every run. Use existing data
configuration paths and override only model and training settings.

Default pilot settings:

```text
train.max_steps = 10000
train.scheduler.warmup_steps = 1000
train.eval_steps = 1000
```

Keep constant across runs:

- seed;
- train and validation CSVs;
- tokenizer and collator settings;
- masking config;
- optimizer config;
- batch size and gradient accumulation;
- precision;
- hardware;
- number of workers;
- evaluation metrics.

### Variant 1: Current Separate-QKV Chain-Aware Baseline

```bash
somatic train \
  model=small \
  model.use_chain_aware_attention=true \
  model.chain_aware_projection_mode=separate \
  train.max_steps=10000 \
  train.scheduler.warmup_steps=1000 \
  train.eval_steps=1000 \
  data.train=/path/to/train.csv \
  output_dir=outputs/attn_ablation/separate_chain_aware
```

### Variant 2: Shared-QKV Chain-Aware

```bash
somatic train \
  model=small \
  model.use_chain_aware_attention=true \
  model.chain_aware_projection_mode=shared \
  train.max_steps=10000 \
  train.scheduler.warmup_steps=1000 \
  train.eval_steps=1000 \
  data.train=/path/to/train.csv \
  output_dir=outputs/attn_ablation/shared_chain_aware
```

### Variant 3: Same-Size Standard MHA

```bash
somatic train \
  model=small \
  model.use_chain_aware_attention=false \
  train.max_steps=10000 \
  train.scheduler.warmup_steps=1000 \
  train.eval_steps=1000 \
  data.train=/path/to/train.csv \
  output_dir=outputs/attn_ablation/standard_small
```

### Variant 4: Parameter-Matched Standard MHA

Use approximately:

```text
d_model = 288
n_heads = 4
n_layers = 24
```

This keeps `d_model` divisible by `n_heads` and makes the non-chain-aware model roughly match
the parameter count of the current small separate-QKV chain-aware model.

```bash
somatic train \
  model=small \
  model.use_chain_aware_attention=false \
  model.d_model=288 \
  model.n_heads=4 \
  model.n_layers=24 \
  train.max_steps=10000 \
  train.scheduler.warmup_steps=1000 \
  train.eval_steps=1000 \
  data.train=/path/to/train.csv \
  output_dir=outputs/attn_ablation/standard_param_matched
```

Before running the full pilot, verify parameter counts with:

```bash
somatic model-size model=small
somatic model-size model=small model.chain_aware_projection_mode=shared
somatic model-size model=small model.use_chain_aware_attention=false
somatic model-size model=small model.use_chain_aware_attention=false model.d_model=288 model.n_heads=4 model.n_layers=24
```

## Metrics To Record

Record these for each run:

- non-embedding parameter count;
- estimated FLOPs/token;
- training loss over time;
- validation loss;
- validation perplexity;
- validation masked accuracy;
- tokens/sec;
- wall-clock time to each eval checkpoint;
- peak GPU memory;
- any instability, NaNs, divergence, or OOM events.

If contact prediction data is available, record contact metrics as exploratory secondary metrics.
Do not make the primary architecture decision solely on contact metrics unless the training loss
and masked LM metrics are inconclusive.

## Decision Criteria

Prefer shared-QKV chain-aware attention if:

- validation loss or perplexity is within about 1-2% of separate-QKV chain-aware attention;
- it is faster, smaller, or materially easier to maintain;
- it beats or matches same-size standard MHA.

Keep separate-QKV chain-aware attention if:

- it clearly beats shared-QKV on validation loss and masked accuracy at the same pilot budget;
- the win is not explained away by the parameter-matched standard MHA baseline;
- the additional parameters and compute are acceptable for the target model sizes.

Reconsider chain-aware attention as the default if:

- parameter-matched standard MHA matches or beats both chain-aware variants;
- shared-QKV does not beat same-size standard MHA;
- the chain-aware variants are slower without a clear validation benefit.

Plan the exact blockwise optimization only if:

- separate-QKV chain-aware attention remains the preferred architecture after the pilot;
- profiling shows attention score/value work is a meaningful bottleneck;
- the implementation can preserve the current global-softmax semantics without custom kernels.

## Risks And Mitigations

### Risk: Shared-QKV Still Computes Two Score Matrices

The first shared-QKV implementation still computes RoPE and no-position score matrices densely.
It is simpler and smaller, but not a full quadratic compute optimization.

Mitigation:

- Make this explicit in FLOPs reporting.
- Use the pilot to decide whether a more complex exact optimization is worth pursuing.

### Risk: HybridNorm Compatibility Is Ambiguous

HybridNorm currently normalizes Q/K/V inside attention. Shared-QKV attention would need a clear
policy for whether QKV norm is shared before relation-specific score construction or applied
separately to RoPE/no-position views.

Mitigation:

- Reject `chain_aware_projection_mode="shared"` with `hybrid_norm != "none"` in the first pass.
- Add HybridNorm support only after the base ablation has value.

### Risk: Parameter-Matched Baseline Changes Head Dimension

The proposed parameter-matched standard baseline uses `d_model=288` and `n_heads=4`, giving
`head_dim=72`. This differs from the default small model's `head_dim=64`.

Mitigation:

- Record the exact configuration and parameter count.
- Treat this as a practical baseline, not a perfect controlled architecture match.
- If results are close, run a second-stage comparison with alternative matched shapes.

### Risk: Pilot Results Are Noisy

A 10k-step pilot may not fully predict final model quality.

Mitigation:

- Use the pilot for architecture triage, not final publication claims.
- If results are close, rerun the top variants with multiple seeds or longer schedules.

### Risk: Attention Weight Consumers Assume Current Semantics

Evaluation metrics can request attention weights, especially contact metrics. Shared-QKV attention
will return the merged global-softmax attention matrix, matching the current output shape and
normalization semantics, but not the same learned projections.

Mitigation:

- Preserve the public attention weight shape.
- Include attention-weight tests.
- Treat contact metrics as secondary until the base MLM behavior is understood.

## Implementation Checklist

- [ ] Add `SharedQKVChainAwareAttention`.
- [ ] Add `chain_aware_projection_mode` to `SomaticConfig`.
- [ ] Validate config values and unsupported HybridNorm combinations.
- [ ] Thread the config field through model, encoder, and block constructors.
- [ ] Dispatch to the correct attention class.
- [ ] Export the new attention class.
- [ ] Update Hydra model configs.
- [ ] Update config README examples.
- [ ] Update FLOPs accounting.
- [ ] Add unit tests for shared-QKV behavior.
- [ ] Add equivalence tests against `MultiHeadAttention`.
- [ ] Add config dispatch and save/load tests.
- [ ] Add a tiny training smoke test for all attention modes.
- [ ] Run targeted tests.
- [ ] Run full tests when practical.
- [ ] Run the pilot ablation matrix.
- [ ] Update this document or add a results document with pilot outcomes.

## Acceptance Criteria

The implementation is ready for pilot runs when:

- existing default model configs still instantiate separate-QKV chain-aware attention;
- existing checkpoints without `chain_aware_projection_mode` still load;
- shared-QKV mode can be selected with a Hydra override;
- shared-QKV forward passes return correct logits and optional attention weights;
- equivalence tests with standard MHA pass in the same-chain and NoPE cases;
- FLOPs reporting changes when switching among standard, separate-QKV, and shared-QKV modes;
- targeted tests pass.

## Pilot Outcomes (Summary)

The 4-variant 10k-step pilot has been executed; full data and trajectories
live in `docs/ATTN_OPTIMIZATION_LOG.md`. Headline result at `model=small`,
4× A6000, bf16, batch=128 effective, seed=42:

| Variant | Non-emb params | eval loss | eval masked acc | wall-clock |
|---|---|---|---|---|
| separate-QKV chain-aware | 24.0M | 0.2912 | 92.71% | 38.9 min |
| **shared-QKV chain-aware** | **19.3M** | **0.2922** | **92.60%** | **32.8 min** |
| standard MHA (same size) | 19.3M | 0.2933 | 92.57% | 28.9 min |
| param-matched MHA (d=288) | 23.9M | 0.2909 | 92.53% | 29.8 min |

Shared-QKV chain-aware attention satisfies all three "prefer shared-QKV"
criteria (within 1–2% of separate on val loss, faster/smaller, beats
same-size standard MHA), so it is the recommended chain-aware variant for
this size class. Whether chain-aware attention should remain the
project-wide default versus parameter-matched standard MHA is ambiguous at
this budget — eval loss favors MHA by 0.1%, eval masked accuracy favors
chain-aware by 0.18 pp. A 50k–100k-step, multi-seed stage-2 with
contact-prediction on coords-bearing eval data is needed before flipping
the project-wide default. The `separate` dataclass default is kept for
checkpoint compatibility; flipping the bundled Hydra defaults to `shared`
should wait for stage-2 confirmation.

