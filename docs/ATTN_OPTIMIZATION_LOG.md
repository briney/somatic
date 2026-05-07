# Attention Optimization Pilot — Running Log

This file tracks implementation steps, observations, decisions, and pilot
results for the chain-aware attention ablation described in
`docs/ATTN_OPTIMIZATION.md`.

## Environment

- Hardware: 4× NVIDIA A6000 (local).
- `accelerate` configured.
- W&B logged in; project name: `somatic-attn-optimization`.
- Train data: `/home/jovyan/work/ablm_training-data/v2026-05-03/train-test-eval_splits/train-clust95_testeval-clust80/minimal/train.parquet`
- Eval data: `/home/jovyan/work/ablm_training-data/v2026-05-03/train-test-eval_splits/train-clust95_testeval-clust80/minimal/L2677-eval_unclustered.parquet`

## Implementation log

### Implementation summary

- Added `SharedQKVChainAwareAttention` (`src/somatic/model/attention.py`) with
  one shared Q/K/V projection. Intra-chain pairs use RoPE-rotated scores,
  inter-chain pairs use unrotated (no-position) scores; the merged matrix
  goes through one global softmax and a single value matmul.
- Factored `_create_chain_mask` into a module-level helper reused by both the
  separate-QKV and shared-QKV chain-aware modules.
- Added `chain_aware_projection_mode` to `SomaticConfig` with values
  `"separate"` (default) and `"shared"`. Validation is unconditional, and the
  combination `shared` + `hybrid_norm != "none"` is rejected.
- Threaded the new config through `SomaticModel`, `TransformerEncoder`,
  `TransformerBlock`, the `somatic` CLI (`model-size`), and the train CLI.
  Older checkpoints/configs without the field default to `"separate"` via the
  dataclass default.
- Exported `SharedQKVChainAwareAttention` from `somatic.model`.
- Updated `small.yaml` / `base.yaml` / `large.yaml` and `configs/README.md`
  with `chain_aware_projection_mode: separate` and example overrides.
- Replaced the single `qkv_factor` heuristic in
  `src/somatic/training/flops.py` with explicit `(projection, score, value)`
  factors for each of the three attention modes.

### Tests

Added new test classes in `tests/unit/test_attention.py`:

- `TestSharedQKVChainAwareAttention` — shape, padding-mask, multi-chain,
  attention-weight rows, deterministic eval, hybrid-norm rejection.
- `TestSharedQKVEquivalence` — same-chain equivalence with `MultiHeadAttention`
  (`atol=1e-5`); NoPE (`rope_fraction=0.0`) equivalence with mixed chains.
- `TestAttentionDispatch` — `TransformerBlock` dispatches to the right class
  for `separate`, `shared`, and `use_chain_aware_attention=false`; invalid
  mode raises.
- `TestSharedQKVConfig` — `SomaticConfig` validation, hybrid-norm rejection,
  legacy-dict default, save/load roundtrip preserving `shared`.
- `TestAttentionTrainingSmoke` — parameterized 2-step train loop on each of
  the three attention modes confirming finite loss and a successful step.

Full suite: **544 passed** (`pytest tests/`).

### Parameter counts (model=small, max_seq_len=320)

| Variant | d_model | n_layers | n_heads | use_chain_aware | mode | Non-emb params |
|---|---|---|---|---|---|---|
| separate (current default) | 256 | 24 | 4 | true | separate | 24,011,264 |
| shared chain-aware | 256 | 24 | 4 | true | shared | 19,292,672 |
| standard small MHA | 256 | 24 | 4 | false | (n/a) | 19,292,672 |
| param-matched standard MHA | 288 | 24 | 4 | false | (n/a) | 23,916,096 |

The shared chain-aware variant has the same parameter count as standard
small MHA. That confirms that "separate vs shared" is a clean Q/K/V
parameter-budget ablation — the chain-routing logic adds zero learnable
parameters by itself. The param-matched standard MHA at d_model=288 lands
within ~0.4% of the separate-QKV chain-aware param count, which is the
intended regime for the parameter-matched comparison.

### Pilot kickoff

Pilot driver: `run_attn_pilot.sh` (sequential, 4× A6000 each variant).

Common settings for every run (locked in `run_attn_pilot.sh`):

- `train.max_steps=10000`, `train.scheduler.warmup_steps=1000`,
  `train.eval_steps=1000`, `train.checkpoint_steps=10000`,
  `train.save_best=false`.
- `train.mixed_precision=bf16` (matches `accelerate` default).
- `eval.regions.enabled=false` — per-position region eval added ~2 minutes per
  eval step in the smoke test, which would have dominated 10k-step pilot
  wall-clock without changing the metric we triage on. Regions can still be
  evaluated post-hoc on the saved checkpoints if needed.
- W&B project: `somatic-attn-optimization`. Each variant logs as its own run
  named after the variant.
- Single seed (42); same train/eval data; default batch (32 per GPU × 4 = 128
  effective).

Launch order (each starts only after the previous finishes):

1. `separate_chain_aware` — `model=small`, `chain_aware_projection_mode=separate`.
2. `shared_chain_aware` — `model=small`, `chain_aware_projection_mode=shared`.
3. `standard_small` — `use_chain_aware_attention=false`.
4. `standard_param_matched` — `use_chain_aware_attention=false`,
   `d_model=288`, `n_heads=4`, `n_layers=24`.

#### Notes on configuration choices made during kickoff

- **Output dir wiring**: passing `--output-dir` on the argparse layer
  in `train.py` overrides any Hydra `output_dir=` override (see the
  `_load_config` priority order). The pilot script therefore only sets
  `output_dir` via Hydra so each variant writes to its own subdirectory.
- **Per-position region eval**: the default `eval/regions/default.yaml` runs
  in `per-position` mode, which masked one position at a time across 71
  eval batches. Smoke testing showed ~2 min per eval. With 10 eval cycles
  per pilot run, that would have added ~20 min of overhead per variant for
  metrics that are not load-bearing for the architecture decision. Disabled
  for the pilot.
- **`save_best=false`**: pilot triages the loss curves, not produced
  checkpoints, so the per-checkpoint metric-tracking and disk write cost is
  unnecessary.

## Pilot results

Pilot ran sequentially, 4× A6000, bf16, batch=32 per GPU (effective 128),
seed=42, 10k steps each, eval every 1000 steps.

W&B project: `thebrineylab/somatic-attn-optimization`.

### Final-step summary

| Variant | Non-emb params | train loss | eval loss | eval ppl | eval masked acc | wall-clock | tokens/sec¹ | cumulative train FLOPs |
|---|---|---|---|---|---|---|---|---|
| separate-QKV chain-aware | 24,011,264 | 0.4920 | 0.2912 | 1.3380 | 92.71% | 38.9 min | ~176k | 5.95 × 10¹⁶ |
| shared-QKV chain-aware | 19,292,672 | 0.5017 | 0.2922 | 1.3394 | 92.60% | 32.8 min | ~208k | 4.70 × 10¹⁶ |
| standard MHA (same size) | 19,292,672 | 0.4881 | 0.2933 | 1.3408 | 92.57% | 28.9 min | ~236k | 4.34 × 10¹⁶ |
| param-matched MHA (d=288) | 23,916,096 | 0.5112 | 0.2909 | 1.3376 | 92.53% | 29.8 min | ~229k | 5.29 × 10¹⁶ |

¹ Approx tokens/sec = (max_steps × batch × seq_len) / runtime, with batch=128
effective and max_seq_len=320; pad-to-max is off so true sequence-token
throughput is somewhat lower in steady state but the comparison across
variants holds.

### Eval-loss trajectories (every 1000 steps)

```
step  separate  shared   standard  pm-matched
1000  0.3831    0.4046   0.3896    0.3826
2000  0.3327    0.3400   0.3360    0.3345
3000  0.3164    0.3226   0.3192    0.3165
4000  0.3077    0.3143   0.3107    0.3091
5000  0.3051    0.3076   0.3053    0.3070
6000  0.3002    0.3023   0.2996    0.3002
7000  0.2979    0.3019   0.2986    0.2968
8000  0.2951    0.2959   0.2962    0.2965
9000  0.2923    0.2938   0.2946    0.2925
10000 0.2912    0.2922   0.2933    0.2909
```

### Eval masked-accuracy trajectories (every 1000 steps)

```
step  separate  shared   standard  pm-matched
1000  0.9040    0.8989   0.9027    0.9059
2000  0.9171    0.9133   0.9163    0.9157
3000  0.9208    0.9175   0.9201    0.9210
4000  0.9225    0.9210   0.9208    0.9224
5000  0.9233    0.9220   0.9235    0.9235
6000  0.9239    0.9227   0.9246    0.9247
7000  0.9261    0.9219   0.9250    0.9240
8000  0.9263    0.9247   0.9250    0.9254
9000  0.9270    0.9254   0.9250    0.9256
10000 0.9271    0.9260   0.9257    0.9253
```

### Stability / instability notes

All four runs trained cleanly. No NaN losses, no divergence, no OOM events,
no checkpoint failures. The evaluation metric `p_at_l` was disabled
implicitly because the eval data does not include backbone coordinates;
basic LM metrics (loss, perplexity, masked accuracy) computed every 1000
steps. Per-position region eval was disabled as discussed above.

### Comparisons against the decision criteria

The plan's decision criteria for shared-QKV chain-aware attention:

> Prefer shared-QKV chain-aware attention if:
> - validation loss or perplexity is within about 1-2% of separate-QKV
>   chain-aware attention;
> - it is faster, smaller, or materially easier to maintain;
> - it beats or matches same-size standard MHA.

- **Within 1-2% of separate**: shared eval_loss 0.2922 vs separate 0.2912
  → +0.34%. Shared eval_ppl 1.3394 vs separate 1.3380 → +0.10%. Both well
  within the 1-2% threshold.
- **Faster / smaller / easier to maintain**: shared has 19.3M vs separate's
  24.0M non-embedding params (-19.6%), runs ~16% faster wall-clock at the
  same token budget, and is meaningfully simpler to read (one Q/K/V path,
  one value matmul) and maintain.
- **Beats or matches same-size standard MHA**: shared eval_loss 0.2922 vs
  standard_small 0.2933 (better by 0.37%); eval masked accuracy 92.60%
  vs 92.57% (better by 0.03 pp).

All three criteria are satisfied — shared-QKV chain-aware attention is the
preferred chain-aware variant under this pilot.

> Reconsider chain-aware attention as the default if:
> - parameter-matched standard MHA matches or beats both chain-aware variants;
> - shared-QKV does not beat same-size standard MHA;
> - the chain-aware variants are slower without a clear validation benefit.

- **Param-matched MHA vs both chain-aware**: param-matched eval_loss 0.2909
  vs separate 0.2912 (better by 0.10%) vs shared 0.2922 (better by 0.45%).
  Eval ppl: 1.3376 vs 1.3380 / 1.3394 — same direction. Eval masked
  accuracy goes the other way: 92.53% (pm-matched) < 92.60% (shared) <
  92.71% (separate). So param-matched MHA matches or beats both chain-aware
  variants on eval loss/perplexity but lags on masked accuracy.
- **Shared vs same-size standard MHA**: shared beats standard_small on
  eval loss, perplexity, and masked accuracy.
- **Speed without benefit**: shared is ~16% faster than separate at
  effectively-equivalent quality; standard is ~26% faster than separate at
  slightly-worse quality. So chain-aware attention is the slower option for
  no measurable masked-LM gain at this budget except on masked accuracy.

The "reconsider as default" trigger is partially fired: eval loss favors
param-matched MHA, but eval masked accuracy favors chain-aware. This is a
genuine ambiguity at the 10k-step budget. The pilot is plausibly noise-
limited at the 0.1-0.5% gap level (single seed, 10k steps, ~1.3M sequences
sampled) — see the workplan's "Pilot Results Are Noisy" risk.

### Recommendation

1. **Adopt shared-QKV chain-aware attention as the new default chain-aware
   variant in this size class.** It loses essentially nothing relative to
   separate-QKV chain-aware attention at this budget, runs faster, and
   removes ~5M parameters in the small size — and that gap will widen at
   `base` (3 saved projections × 56 layers × 384²) and `large` scales.
2. **Run a stage-2 comparison before changing the project-wide default
   between chain-aware and standard MHA.** The masked-accuracy advantage
   for chain-aware (~0.18 pp over param-matched MHA at 10k steps) and the
   eval-loss advantage for param-matched MHA (~0.1%) point in opposite
   directions at the noise floor. Suggested stage-2: re-run separate /
   shared / param-matched MHA for 50k–100k steps with ≥2 seeds each, and
   add the contact-prediction (`p_at_l`) eval on a coords-bearing subset
   to exercise the chain-aware attention's structural inductive bias.
3. **Defer the exact blockwise optimization** of separate-QKV chain-aware
   attention. Its remaining advantage is too small to justify the
   maintenance burden of a custom kernel, and the cleaner answer (drop the
   second projection matrix entirely) achieves most of the available wins.
4. Keep `chain_aware_projection_mode: separate` as the dataclass default
   for now to preserve checkpoint compatibility; flip the bundled Hydra
   defaults to `shared` only after a stage-2 confirmation.

### Surprises / unexpected findings

- **Param-matched MHA tied separate-QKV on eval loss.** I was expecting
  the chain-aware inductive bias to be load-bearing on a paired-chain
  task, but at this budget eval_loss is essentially indistinguishable
  (0.2909 vs 0.2912). The masked-accuracy gap (92.53% vs 92.71%) is the
  one place where chain-aware attention still leads.
- **Shared-QKV matches separate-QKV cleanly.** The 5M-parameter gap was
  large enough that I expected a noticeable ppl gap; instead it landed
  inside 0.1% on perplexity. That suggests that the modeling power of
  chain-aware attention is mostly in the routing logic (RoPE for
  intra-chain, no-position for inter-chain, single global softmax), not
  in the second set of projections.
- **Standard MHA (same size) trains slightly faster on training loss
  (0.488 vs 0.492 for separate) but ends up with worse eval metrics.**
  Likely a generalization story: chain-aware attention seems to overfit
  less per step. Worth confirming with longer runs.
- **No instabilities at all.** Even the largest model finished without a
  single NaN or grad-norm spike across 10k steps in bf16. Useful baseline
  for future stability comparisons.

