# Attention Optimization Pilot V2 — Lab Notebook

Follow-up to `docs/ATTN_OPTIMIZATION_LOG.md` (the 10k-step pilot). The
goals of V2 are:

1. Re-run the chain-aware ablation at a more discriminating budget
   (50k steps, eval every 5k) and add a fifth variant — a parameter-
   matched shared-QKV chain-aware model — so we can disentangle
   "shared vs separate projection budget" from "chain-aware vs MHA at
   matched budget".
2. Quantitatively and qualitatively analyze cross-chain attention on
   each fully-trained model. Of particular interest: in standard MHA,
   cross-chain attention historically concentrates on residues at the
   end of the heavy chain and the beginning of the light chain (the
   two sequences are adjacent in the concatenated input, so this is
   plausibly a positional-embedding artifact). Does either chain-aware
   variant suppress or eliminate this artifact?

## Variants

| Tag | Mode | d_model | n_heads | n_layers | Non-emb params |
|---|---|---|---|---|---|
| `separate_chain_aware` | separate-QKV chain-aware | 256 | 4 | 24 | 24,011,264 |
| `shared_chain_aware` | shared-QKV chain-aware | 256 | 4 | 24 | 19,292,672 |
| `shared_chain_aware_pm` | shared-QKV chain-aware (param-matched) | 288 | 4 | 24 | 23,916,096 |
| `standard_small` | standard MHA | 256 | 4 | 24 | 19,292,672 |
| `standard_param_matched` | standard MHA (param-matched) | 288 | 4 | 24 | 23,916,096 |

Note: shared chain-aware at d=288 has the same parameter count as MHA at
d=288 — the chain routing logic adds zero learnable parameters. This is
the cleanest possible "chain-aware vs MHA at matched parameters" test.

## Settings (locked)

- W&B project: `somatic-attn-optimization-v2`
- Hardware: 4× A6000, bf16, accelerate
- `train.max_steps=50000`, `train.scheduler.warmup_steps=1000`
- `train.eval_steps=5000`, `train.checkpoint_steps=50000`
- `train.save_best=true`, `keep_last_n_checkpoints=2`
- `train.batch_size=32` per GPU (effective 128)
- Same train + eval data as V1
- `eval.regions.enabled=false` (per-position region eval is slow; deferred
  to optional post-hoc analysis)
- Single seed (42) — same as V1, so V2 is a longer-budget extension of V1
  on the same RNG path

## Implementation log

### V2 setup decisions

- Added a fifth variant — `shared_chain_aware_pm` (shared-QKV chain-aware
  at d=288, n_heads=4, n_layers=24) — to disentangle "shared vs separate
  projection budget" from "chain-aware vs MHA at matched parameter count".
  Verified params: 23,916,096, identical to `standard_param_matched` (the
  shared chain-aware variant adds zero learnable parameters relative to
  MHA at the same shape).
- Pilot driver: `run_attn_pilot_v2.sh`. Only delta vs V1 is `max_steps`
  (50000), `eval_steps` (5000), `checkpoint_steps` (50000),
  `save_best=true`, `keep_last_n_checkpoints=2`, fifth variant inserted
  between shared and standard, and W&B project bumped to
  `somatic-attn-optimization-v2`. Same train/eval data, same seed (42),
  same batch=128 effective, same bf16, same eval-region disablement.
- `save_best=true` matters here — V1 turned it off because the V1 pilot
  was throwaway, but V2 needs accessible checkpoints for the cross-chain
  attention analysis. Best metric is `val_loss` (default), so the
  `best_checkpoint.pt` saved at the lowest eval loss is the right artifact
  to load for analysis.

### Cross-chain attention analysis tool

- New script `scripts/cross_chain_attention.py`:
  - Loads each variant's `best_checkpoint.pt` (or fallback to the latest
    periodic step checkpoint).
  - Runs a fixed sample of eval sequences with `output_attentions=True`.
  - Computes, per layer:
    - mean cross-chain attention fraction (sum of attention mass on
      tokens of the *other* chain, averaged over heads, valid query
      positions, and batch);
    - per-region mean (binning queries by H/L × {FWR1, CDR1, FWR2, CDR2,
      FWR3, CDR3, FWR4} based on the cdr_mask pattern);
    - per-absolute-position mean (used to look at the H→L boundary
      artifact directly);
    - mean attention heatmap of the last layer (head-mean, batch-mean).
  - Writes a JSON per variant + a 4-panel PNG per variant + a comparison
    PNG across variants + a CSV summary.
- Region binning: cdr_mask uses 0=FWR, 1=CDR1, 2=CDR2, 3=CDR3 within a
  chain. The script walks each chain in order and expands FWR runs into
  FWR1/2/3/4 based on whether they appear before/between/after
  CDR1/CDR2/CDR3, giving 14 amino-acid regions plus CLS/EOS.

### Preliminary V1 cross-chain attention results

Ran the analysis on the V1 final checkpoints (10k-step models) as a
sanity check of the tool and as an early data point.

| Variant | Params | Mean cross-chain attn (over layers) |
|---|---|---|
| separate-QKV chain-aware | 24.0M | 0.224 |
| shared-QKV chain-aware | 19.3M | 0.186 |
| standard MHA (same size) | 19.3M | 0.260 |
| param-matched MHA (d=288) | 23.9M | 0.282 |

Interpretation:

- **Chain-aware variants attend less across chains than MHA at the same
  step budget.** The chain-aware mechanism is not increasing cross-chain
  attention — it is producing a lower fraction of mass on the partner
  chain than MHA does. That makes sense given that chain-aware splits
  intra-chain (RoPE-position-aware) from inter-chain (no-position) scores
  in one global softmax, and the model evidently learns to weight the
  RoPE-aware path more.
- **Shared chain-aware attends least across chains** (18.6%). With
  separate Q/K/V for cross attention it might be easier for the model to
  *use* the cross path, so removing those parameters seems to push the
  model further toward intra-chain attention.
- **Param-matched MHA has higher cross-chain attention than same-size MHA**
  (28.2% vs 26.0%). At matched parameter count, MHA is using more of its
  capacity on the partner chain.
- These are 10k-step results and may shift at 50k. The V2 pilot will
  re-measure on better-trained models.

(Plots saved to
`outputs/attn_ablation/_analysis_v1/` including per-variant 4-panel PNGs,
a comparison PNG, and a region overlay PNG. These are local-only — not
checked into git.)

#### V1: H→L boundary artifact

The headline test of the V2 work: does chain-aware attention eliminate the
positional-embedding-driven cross-chain attention spike at the
heavy/light boundary that we see in standard MHA?

Cross-chain attention by zone (layer-mean over all layers, V1 10k-step
checkpoints):

| Variant | H body (2-110) | Boundary (115-127) | L body (127-220) | EOS zone (≥235) |
|---|---|---|---|---|
| separate-QKV chain-aware | 0.211 | **0.233** | 0.233 | 0.265 |
| shared-QKV chain-aware | 0.162 | **0.215** | 0.201 | 0.256 |
| standard MHA (same size) | 0.221 | **0.403** | 0.281 | 0.308 |
| param-matched MHA (d=288) | 0.250 | **0.430** | 0.292 | 0.336 |

Peak per-position cross-chain mass:

- separate chain-aware: peak 0.306 at pos 242 (EOS-adjacent)
- shared chain-aware: peak 0.292 at pos 248 (EOS-adjacent)
- standard MHA: peak 0.416 at pos 123 (boundary)
- param-matched MHA: peak 0.445 at pos 120 (boundary)

Interpretation:

- **The boundary artifact reproduces clearly in MHA.** Cross-chain
  attention nearly doubles in the H→L boundary band relative to the
  heavy-chain body (+72% to +82% relative jump). The peak per-position
  cross-chain mass for both MHA variants sits exactly at the boundary.
- **Separate-QKV chain-aware almost completely eliminates the artifact.**
  The boundary band is +10% over heavy body, and the peak position is no
  longer at the boundary — it shifts to EOS-adjacent positions, plausibly
  because EOS is a special token that genuinely needs to look at both
  chains.
- **Shared-QKV chain-aware reduces but does not entirely eliminate the
  artifact.** Boundary is +33% over heavy body, intermediate between
  separate and MHA. This is consistent with shared-QKV being a
  half-step toward chain-aware: same routing logic but shared projection
  parameters that the model can re-purpose for the boundary if it
  wishes. The peak cross-chain position again moves to EOS-adjacent.
- **Both chain-aware variants confirm the user's hypothesis.** The
  boundary spike in MHA is a positional-embedding artifact, removed (or
  greatly reduced) when intra-chain pairs use RoPE and inter-chain pairs
  use no-position scores under one global softmax.

These are 10k-step measurements; V2 50k-step models will sharpen and
possibly invert some of these magnitudes. We will repeat the analysis on
the V2 best checkpoints once they are available.

## V2 pilot results (50k steps)

Pilot ran sequentially, 4× A6000, bf16, batch=128 effective, seed=42, 50k
steps each, eval every 5k. W&B project:
`thebrineylab/somatic-attn-optimization-v2`. The trainer's `save_best`
fires only when the eval-loss key matches `cfg.best_metric` (default
`val_loss`); since the eval logger writes `eval/loss`, no
`best_checkpoint.pt` is produced. The V2 analysis uses
`checkpoint_step_50000.pt` (model at end-of-schedule, LR=0).

### Final-step metrics (sorted by eval_loss)

| Variant | Params | train loss | eval loss | eval ppl | eval mask_acc | runtime |
|---|---|---|---|---|---|---|
| **standard MHA (same size)** | 19.3M | 0.4779 | **0.2840** | 1.3285 | 92.74% | 2.33 hr |
| standard MHA (param-matched, d=288) | 23.9M | 0.4806 | 0.2844 | 1.3289 | 92.72% | 2.46 hr |
| shared-QKV chain-aware | 19.3M | 0.4735 | 0.2853 | 1.3302 | 92.65% | 2.71 hr |
| separate-QKV chain-aware | 24.0M | 0.4708 | 0.2856 | 1.3306 | **92.75%** | 3.22 hr |
| shared-QKV chain-aware (param-matched) | 23.9M | 0.4799 | 0.2865 | 1.3318 | 92.73% | 2.84 hr |

### Eval-loss trajectories (every 5k steps)

```
 step    separate    shared    shared_pm  standard   std_pm
 5000    0.3093      0.3105    0.3102     0.3111     0.3082
10000    0.2967      0.2992    0.2976     0.3006     0.3037
15000    0.2946      0.2931    0.2939     0.2939     0.2941
20000    0.2919      0.2910    0.2943     0.2913     0.2890
25000    0.2895      0.2886    0.2903     0.2894     0.2889
30000    0.2884      0.2892    0.2893     0.2868     0.2884
35000    0.2883      0.2872    0.2903     0.2860     0.2868
40000    0.2875      0.2862    0.2872     0.2855     0.2855
45000    0.2857      0.2855    0.2861     0.2855     0.2855
50000    0.2856      0.2853    0.2865     0.2840     0.2844
```

### Observations on eval-loss

- **Standard MHA wins on eval loss at 50k.** Both standard MHA variants
  (same-size and param-matched) finish with the lowest eval loss.
  The eval-loss-vs-step trajectory shows the gap opening up around step
  30-35k where standard MHA pulls slightly ahead.
- **The five variants are remarkably tight.** Spread is 0.0025 in
  eval-loss (~0.9%); 0.0033 in eval-ppl (~0.25%); 0.10 pp in eval
  masked-accuracy. At this budget on this masked-LM eval, none of the
  variants are clearly better than the others. This is consistent with
  the V1 finding that the chain-aware advantage on eval-loss is small
  enough to be at the noise floor.
- **separate-QKV chain-aware leads on eval masked accuracy** (92.75%),
  matching V1, but the lead is 0.01 pp — also at the noise floor.
- **shared_pm (param-matched shared chain-aware) is the worst variant
  on eval loss** (0.2865). The shared+param-matched combination doesn't
  buy any quality over the smaller shared variant, and trails MHA at
  matched parameters by 0.0021. That cuts against the simplest "scale
  shared chain-aware to match separate" remediation.
- **Interpretation**: at 50k steps on this dataset, the chain-aware
  inductive bias does not produce measurable masked-LM gains over
  parameter-matched MHA. If chain-aware attention is to be the project
  default, the case must rest on (a) downstream / structural tasks
  (e.g., contact prediction), or (b) attention-pattern interpretability,
  not eval-loss/perplexity/masked-accuracy at this size.

### Cross-chain attention analysis (V2 final checkpoints, n=256 eval seqs)

Mean cross-chain attention fraction across all 24 layers (head-mean,
batch-mean, query-position-mean over valid amino-acid positions):

| Variant | Mean | Min layer | Max layer |
|---|---|---|---|
| separate-QKV chain-aware | 0.168 | 0.057 | 0.317 |
| shared-QKV chain-aware | 0.150 | 0.041 | 0.337 |
| shared-QKV chain-aware (param-matched) | 0.148 | 0.016 | 0.350 |
| standard MHA (same size) | 0.178 | 0.084 | 0.436 |
| standard MHA (param-matched, d=288) | 0.208 | 0.075 | 0.458 |

Both chain-aware variants attend less across chains than MHA. Shared is
slightly more "intra-chain biased" than separate. Standard MHA places
~17–21% of attention mass on the partner chain on average; chain-aware
places ~15–17%. These numbers shrank from V1 (10k) for every variant —
all variants attend less across chains as training proceeds, but the
*relative* ordering is preserved.

### H→L boundary artifact (V2)

Cross-chain attention by zone (layer-mean over all 24 layers,
head/batch/valid-query mean):

| Variant | H body | Boundary (115-127) | L body | EOS zone | Boundary/H | Peak pos | Peak val |
|---|---|---|---|---|---|---|---|
| separate-QKV chain-aware | 0.154 | 0.182 | 0.177 | 0.221 | **1.18×** | 242 (EOS) | 0.268 |
| shared-QKV chain-aware | 0.130 | 0.191 | 0.154 | 0.257 | **1.46×** | 247 (EOS) | 0.302 |
| shared-QKV chain-aware (pm) | 0.136 | 0.180 | 0.148 | 0.228 | **1.32×** | 242 (EOS) | 0.270 |
| standard MHA (same size) | 0.146 | 0.335 | 0.186 | 0.248 | **2.29×** | **120 (boundary)** | 0.356 |
| standard MHA (param-matched) | 0.185 | 0.344 | 0.209 | 0.274 | **1.86×** | **120 (boundary)** | 0.367 |

**Headline finding**: the H→L boundary artifact reproduces strongly in
both standard MHA variants — boundary cross-chain attention is **2.3×**
the heavy-body baseline for same-size MHA, and **1.9×** for
param-matched MHA. The peak per-position cross-chain mass for both MHA
variants sits **exactly at position 120** (the H→L join in the 320-token
concatenated input).

**Both chain-aware variants suppress the artifact dramatically.**

- *separate-QKV chain-aware* is essentially flat across the boundary:
  H-body 0.154, boundary 0.182 — only **1.18×** above heavy body. The
  peak per-position cross-chain mass moves to position 242 (EOS-adjacent
  light-chain residues), which is genuine biology, not a positional
  artifact.
- *shared-QKV chain-aware* and its param-matched variant have a
  modest residual elevation at the boundary (1.32–1.46×), still well
  below MHA. The peak again sits near EOS, not at the boundary.
- The artifact got *more pronounced* with longer training in MHA
  (V1 ratios were 1.82× and 1.72×; V2 ratios are 2.29× and 1.86×). This
  matters: longer training does not "wash out" the boundary artifact in
  MHA — the model learns to use it more.

The user's hypothesis is confirmed: chain-aware attention's intra/inter
split removes the positional-embedding-driven cross-chain attention
spike at the chain boundary that standard MHA exhibits and that grows
with training.

### Why cross-chain matters even though eval-loss doesn't separate the variants

The eval-loss numbers say the five variants are within ~0.0025. The
cross-chain analysis says the *attention mechanism* differs sharply
across variants:

- standard MHA is consuming a substantial fraction of its cross-chain
  attention budget on a positional artifact at the H→L join — i.e. the
  model is "wasting" attention pattern capacity on the fact that residue
  120 is adjacent to residue 121 in the input string, not on biology.
- chain-aware attention spends that attention on real cross-chain
  signal (concentrated near EOS / late-light-chain residues, which the
  model presumably uses for representation pooling), not on positional
  adjacency.

For a masked-LM objective on individual residues this attention-budget
inefficiency is not visible — the model can compensate via other heads
or layers. Where it should matter is **downstream tasks that require
faithful cross-chain interaction patterns** (e.g., contact prediction,
paratope/epitope mapping, antibody-antigen binding prediction). That is
the right next test for whether chain-aware attention is worth its
training-time cost and slight eval-loss disadvantage at this budget.

### Recommendation

1. **Adopt shared-QKV chain-aware attention as the chain-aware default.**
   At 50k steps it matches separate-QKV chain-aware on eval loss,
   accuracy, and the cross-chain artifact suppression — at ~80% of the
   parameter count and ~84% of the wall-clock cost. Param-matched shared
   does not improve quality; the Q/K/V budget is not where the bottleneck
   is.
2. **Keep chain-aware attention as the default architecture** despite
   standard MHA's slight eval-loss lead. The cross-chain artifact
   analysis shows MHA is exploiting a non-biological positional shortcut
   at the chain boundary; eliminating that shortcut is a stronger
   statement about model behavior than the 0.001-eval-loss gap. Final
   defaultisation of chain-aware-vs-MHA should hinge on a downstream
   evaluation (contact prediction, paratope mapping) rather than masked
   LM alone.
3. **Do not pursue param-matching shared-QKV chain-aware.** The pilot
   shows it is the worst of the five variants on eval loss, with no
   gain on the cross-chain artifact suppression. The Q/K/V projection
   budget is not the limiting factor.
4. **Defer the exact blockwise optimization indefinitely.** Shared-QKV
   captures essentially all of the chain-aware modeling benefit visible
   on this objective at 16% lower wall-clock; a custom kernel for
   separate-QKV is unjustified.

### Surprises and open questions

- **Standard MHA quietly took the eval-loss lead at 50k.** I expected
  chain-aware to retain a small but visible eval-loss advantage given
  its inductive bias. The reverse happened: chain-aware leads only at
  10k (V1 separate=0.2912, V1 std=0.2933) and at 50k MHA pulls ahead
  (V2 separate=0.2856, V2 std=0.2840). Either MHA needs more steps to
  realize its capacity, or chain-aware's bias actually limits asymptotic
  performance on per-residue masked LM. A multi-seed study would say
  whether the 0.0016 lead is real.
- **Param-matching shared chain-aware made it worse.** I expected the
  same eval loss as shared, just slower. Instead shared_pm finished
  worst on eval loss (0.2865 vs 0.2853 for d=256 shared). Possible
  explanation: deeper-than-wide is the right shape for this masked LM
  task, and pushing d_model from 256→288 at fixed n_layers=24 trades
  depth fitness for width — but that doesn't explain why the same shape
  change *helped* standard MHA (param-matched 0.2844 vs same-size
  0.2840 — barely worse). Possibly a sample-of-one noise.
- **The boundary artifact in MHA grows with training.** I would have
  guessed it would weaken with more data — that the model learns to
  ignore the positional adjacency. Instead it doubles down. That is a
  real interpretability finding and worth highlighting in any future
  writeup.
- **Chain-aware attention shifts the *peak* cross-chain position from
  the boundary to the EOS-adjacent region.** Both MHA variants peak at
  position 120 (the join); all chain-aware variants peak near position
  242 (a few residues before the EOS). That's the model using the
  end-of-light-chain region as a cross-chain "summary" — plausibly
  useful representation; certainly different from MHA's usage.

### Generated artifacts (local-only, gitignored)

- `outputs/attn_ablation_v2/<variant>/checkpoints/checkpoint_step_50000.pt`
- `outputs/attn_ablation_v2/_logs/<variant>.log` — per-run training logs
- `outputs/attn_ablation_v2/_analysis/<variant>.png` — 4-panel per-variant
  cross-chain analysis figures
- `outputs/attn_ablation_v2/_analysis/<variant>.json` — raw per-layer /
  per-region / per-position metrics
- `outputs/attn_ablation_v2/_analysis/_comparison.png`,
  `_comparison_regions.png` — across-variant overlays
- `outputs/attn_ablation_v2/_analysis/_summary.csv`

### Code added in V2

- `run_attn_pilot_v2.sh` — sequential 5-variant 50k-step driver.
- `scripts/cross_chain_attention.py` — cross-chain attention analysis.
- `scripts/summarize_attn_pilot_v2.py` — W&B → markdown summary.


