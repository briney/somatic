# Workplan: HuggingFace `transformers` Compatibility Refactor

## Goal

Make Somatic loadable through the standard `transformers` Auto classes with
`from_pretrained(...)`, including remote loading via `trust_remote_code=True`:

```python
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
model = AutoModelForMaskedLM.from_pretrained("path/or/hub")          # local install
model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)  # remote
tok   = AutoTokenizer.from_pretrained(path)
```

This is a **clean break**: no backwards compatibility. Old `.pt` checkpoints
(the untracked `checkpoints/` dir) will not load and are discarded. The legacy
`from_pretrained` field-popping (`max_timesteps`, `use_timestep_embedding`,
bool `hybrid_norm`) is removed entirely.

The reference implementation is the sibling repo `../oplm` (a protein LM that
already has this compatibility). Throughout this plan, "ref:" points at files in
`/home/briney/git/oplm/src/oplm/`.

### Locked design decisions

1. **Config field names → HF canonical** (`hidden_size`, `num_hidden_layers`,
   `num_attention_heads`, `intermediate_size`, `max_position_embeddings`, ...).
   No `attribute_map` aliasing.
2. **Chain identity → `token_type_ids`** (HF's standard segment slot). The
   tokenizer emits it natively from paired input; the model forward accepts
   `token_type_ids`; chain-aware attention reads it as chain identity. This is
   a pure rename of the internal `chain_ids` concept — semantics unchanged.
3. **Four model classes**: `SomaticModel` (base), `SomaticForMaskedLM`,
   `SomaticForSequenceClassification`, `SomaticForTokenClassification`.
4. **Loss in model, masking retained**: the model computes loss internally from
   `labels` (`ignore_index=-100`). `InformationWeightedMasker` stays in the
   trainer/evaluator but now emits HF-style `labels`.

---

## Reference: field rename map (current → HF canonical)

Applied in `SomaticConfig`, all `configs/model/*.yaml`, `train.py` construction,
and everywhere a config attribute is read.

| Current (`SomaticConfig` dataclass) | New (HF canonical)         |
|-------------------------------------|----------------------------|
| `d_model`                           | `hidden_size`              |
| `n_layers`                          | `num_hidden_layers`        |
| `n_heads`                           | `num_attention_heads`      |
| `d_ffn`                             | `intermediate_size`        |
| `max_seq_len`                       | `max_position_embeddings`  |
| `dropout` + `embedding_dropout`     | `hidden_dropout`           |
| `attention_dropout`                 | `attention_dropout` (kept) |
| `layer_norm_eps`                    | `norm_eps`                 |
| `padding_idx`                       | use `pad_token_id`         |

Kept Somatic-specific fields (names unchanged): `use_chain_aware_attention`,
`chain_aware_projection_mode`, `rope_fraction`, `norm_type`, `pre_norm`,
`post_norm`, `qk_norm`, `hybrid_norm`, `ffn_multiplier`, `head_dim`,
`gradient_checkpointing`, `gradient_checkpointing_mode`.

New fields to add: `initializer_range` (default 0.02), special-token ids
(`pad_token_id=1, bos_token_id=0, eos_token_id=2, unk_token_id=3,
mask_token_id=31`), and classification-head fields (`num_labels`,
`classifier_dropout`, `classifier_pool`, `pre_head_norm`).

## Reference: batch / forward key rename map

| Current key/arg | New key/arg      |
|-----------------|------------------|
| `token_ids`     | `input_ids`      |
| `chain_ids`     | `token_type_ids` |
| (loss external) | `labels` (in-model loss, `-100` ignore) |

Unchanged batch keys: `attention_mask`, `special_tokens_mask`, `cdr_mask`,
`non_templated_mask`, `coords`.

## Reference: target package layout

```
src/somatic/
├── __init__.py                      # EDIT: Auto* registration + register_for_auto_class
├── model/
│   ├── __init__.py                  # EDIT: export config + 4 model classes + tokenizer
│   ├── configuration_somatic.py     # NEW (SomaticConfig moves out of transformer.py)
│   ├── modeling_somatic.py          # NEW (model classes move out of transformer.py)
│   ├── tokenization_somatic.py      # NEW (moved from src/somatic/tokenizer.py)
│   ├── attention.py                 # EDIT: rename chain_ids -> token_type_ids
│   ├── layers.py                    # EDIT: rename chain_ids -> token_type_ids
│   ├── ffn.py  normalization.py  rope.py  embeddings.py   # KEPT as-is (helpers)
│   └── transformer.py               # REMOVE after contents migrated
```

**`trust_remote_code` bundling rule (critical):** HF copies only the *direct*
relative imports (depth-1) of the main modeling file. So `modeling_somatic.py`
must directly `from .X import ...` every helper it transitively needs
(`attention`, `ffn`, `normalization`, `rope`, `embeddings`, `layers`,
`configuration_somatic`) and reference the otherwise-unused names in a
module-level `_REMOTE_CODE_DEPS` tuple. Pattern: `ref: model/modeling_oplm.py:24-49`.

---

## Phase 0 — Setup & branch

- [x] Create a feature branch: `git checkout -b feature/hf-compatibility`.
- [x] Confirm `transformers`, `tokenizers`, `safetensors` are declared in
      `pyproject.toml` dependencies; add/loosen version ranges if missing.
- [x] Note: `src/somatic/tokenizer.py` is currently a `PreTrainedTokenizerFast`
      already — good starting point. `src/somatic/model/transformer.py` holds
      BOTH `SomaticConfig` (dataclass) and `SomaticModel` (`nn.Module`); these
      get split across the two new files.

## Phase 1 — Configuration (`model/configuration_somatic.py`)

Pattern: `ref: model/configuration_oplm.py`.

- [x] Create `src/somatic/model/configuration_somatic.py` with
      `class SomaticConfig(PretrainedConfig)` and `model_type = "somatic"`.
- [x] Keyword-only `__init__(self, *, ...)` accepting all fields (renamed per the
      map above), special-token ids, `initializer_range`, and classification
      fields. End with:
      ```python
      self._resolve_derived_fields()
      self._validate()
      kwargs.setdefault("num_labels", int(num_labels))  # num_labels is a property; forward via kwargs
      super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id,
                       eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)
      ```
- [x] `_resolve_derived_fields()`: `head_dim = hidden_size // num_attention_heads`
      when None; `intermediate_size` from `ffn_multiplier` (default `8/3`)
      rounded up to a multiple of 64 — port the exact rounding from current
      `transformer.py`'s `d_ffn` logic.
- [x] `_validate()`: port all current `SomaticConfig.__post_init__` assertions
      (`hidden_size % num_attention_heads == 0`, `head_dim * heads == hidden_size`,
      `0 <= rope_fraction <= 1`, enum membership for `norm_type`, `hybrid_norm`,
      `qk_norm`, `gradient_checkpointing_mode`, `chain_aware_projection_mode`).
- [x] Do NOT assign `num_labels` as a direct attribute (it's a `PretrainedConfig`
      property backed by `id2label`/`label2id`).
- [x] Delete the dataclass `SomaticConfig` from `transformer.py`.

## Phase 2 — Modeling (`model/modeling_somatic.py`)

Pattern: `ref: model/modeling_oplm.py`. Keep the math identical to the current
implementation; this phase is repackaging + renaming + HF interfaces.

### 2a. Thread `token_type_ids` through the stack (internal rename)
- [x] In `model/attention.py`: rename the `chain_ids` parameter to
      `token_type_ids` in `ChainAwareAttention.forward`,
      `SharedQKVChainAwareAttention.forward`, `BaseAttention`, and
      `MultiHeadAttention.forward` (the last keeps accepting + ignoring it).
- [x] In `model/layers.py`: rename `chain_ids` → `token_type_ids` in
      `TransformerBlock.forward` and `TransformerEncoder.forward` and all
      internal passes to attention. Keep `set_gradient_checkpointing` as-is.

### 2b. Base classes
- [x] `class SomaticPreTrainedModel(PreTrainedModel)`:
      `config_class = SomaticConfig`, `base_model_prefix = "somatic"`,
      `main_input_name = "input_ids"`, `supports_gradient_checkpointing = True`,
      `_no_split_modules = ["TransformerBlock"]`, `_supports_sdpa = True`.
- [x] Port `_init_weights(self, module)` from current `transformer.py`, using
      `self.config.initializer_range`.
- [x] Add `_set_gradient_checkpointing(value, mode=None)`,
      `gradient_checkpointing_enable(gradient_checkpointing_kwargs=None)`,
      `gradient_checkpointing_disable()` that propagate to
      `TransformerEncoder.set_gradient_checkpointing` (ref: `modeling_oplm.py:119-148`).
      (Also preserved `get_num_params(non_embedding=...)` on the base class.)

### 2c. `SomaticModel` (base encoder → `BaseModelOutput`)
- [x] `__init__`: build `SomaticEmbedding` + `TransformerEncoder` (+ final norm
      already inside the encoder). Call `self.post_init()`.
- [x] `get_input_embeddings` / `set_input_embeddings` pointing at
      `embeddings.token_embedding.embedding`.
- [x] `forward(input_ids=None, attention_mask=None, token_type_ids=None,
      inputs_embeds=None, output_attentions=None, output_hidden_states=None,
      return_dict=None)`:
      - default `token_type_ids` to zeros (single chain) when None;
      - resolve `output_*`/`return_dict` from config when None;
      - return `BaseModelOutput(last_hidden_state=..., hidden_states=..., attentions=...)`.

### 2d. `SomaticForMaskedLM` (→ `MaskedLMOutput`)
- [x] `self.somatic = SomaticModel(config)`; keep Somatic's existing **bias-free
      tied Linear** `lm_head` (do NOT adopt oplm's dense+act+norm head — preserve
      the current architecture).
- [x] `_tied_weights_keys = {"lm_head.weight": "somatic.embeddings.token_embedding.embedding.weight"}`;
      rely on `config.tie_word_embeddings=True` + `post_init()` to tie.
- [x] `forward(..., token_type_ids=None, labels=None, ...)`: run base model,
      `logits = self.lm_head(last_hidden_state).float()`; if `labels is not None`,
      `loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)`;
      return `MaskedLMOutput(loss=loss, logits=logits, hidden_states=..., attentions=...)`.
- [x] Add `get_output_embeddings` / `set_output_embeddings` (the lm_head).
      (Also pulled the Phase 9 `predict_masked` helper onto `SomaticForMaskedLM`,
      reworked to use config token ids — keeps modeling self-contained for
      remote bundling.)

### 2e. Classification heads
- [x] `SomaticForSequenceClassification` (→ `SequenceClassifierOutput`) and
      `SomaticForTokenClassification` (→ `TokenClassifierOutput`): port directly
      from `ref: modeling_oplm.py:394-562`, threading `token_type_ids` into the
      inner `SomaticModel`. Use `config.classifier_pool` / `classifier_dropout` /
      `pre_head_norm`; seq-classification uses HF `problem_type` loss inference.
      (Added module-level `mean_pool`/`cls_pool` helpers in `modeling_somatic.py`.)

### 2f. Remote-code bundling
- [x] In `modeling_somatic.py`, directly import all helper modules and add:
      ```python
      _REMOTE_CODE_DEPS = (ChainAwareAttention, FusedSwiGLUFFN, RotaryPositionEmbedding, ...)
      ```
- [x] Remove the now-empty `model/transformer.py` (or reduce to re-exports if
      anything still imports it, then delete those imports).

### 2g. RoPE meta-device fix (required for HF `from_pretrained` round-trip)
- [x] `model/rope.py`: the non-persistent sin/cos buffers re-materialize as
      uninitialized memory under HF's meta-device fast init. Recompute `inv_freq`
      from `base`/`rotated_dim` and rebuild the cache on the first real forward
      (`_cache_initialized` flag). Rotation math is unchanged; verified
      save→reload logits match exactly (maxdiff 0.0).

## Phase 3 — Tokenizer (`model/tokenization_somatic.py`)

Pattern: `ref: model/tokenization_oplm.py`. Move `src/somatic/tokenizer.py` here.

- [x] Rename class `Tokenizer` → `SomaticTokenizerFast`. Keep `DEFAULT_VOCAB`,
      `AA_START_IDX`, `AA_END_IDX`, and a module-level singleton
      `tokenizer = SomaticTokenizerFast()`.
- [x] Set `vocab_files_names = {"tokenizer_file": "tokenizer.json"}` and
      `model_input_names = ["input_ids", "token_type_ids", "attention_mask"]`.
- [x] **Emit `token_type_ids` natively** via `TemplateProcessing` so segment ids
      reproduce the current chain layout (cls=0, heavy=0, light=1, eos=1, single
      trailing eos):
      ```python
      single = "<cls>:0 $A:0 <eos>:0"
      pair   = "<cls>:0 $A:0 $B:1 <eos>:1"
      ```
      so `tokenizer(heavy, light, return_token_type_ids=True)` yields the chain
      segmentation for free.
- [x] Replace `encode_paired` with a thin wrapper over `self(heavy, light, ...)`
      returning `input_ids` / `token_type_ids` / `attention_mask`. Drop the
      `add_chain_separator` variant (plain MHA ignores chain identity).
- [x] Verify `tokenizer.save_pretrained(dir)` writes `tokenizer.json` +
      `tokenizer_config.json`. (Round-trip via class + `AutoTokenizer` preserves
      `token_type_ids`; reload-kwarg conflicts on `unk_token`/`clean_up_*` fixed.)
- [x] Left `src/somatic/tokenizer.py` as a re-export shim (with a legacy
      `Tokenizer = SomaticTokenizerFast` alias) so existing importers keep working
      until Phases 5/8/9/11 migrate them; delete the shim afterward.

## Phase 4 — Package registration

Pattern: `ref: __init__.py:27-49`.

- [x] `src/somatic/model/__init__.py`: export `SomaticConfig`, `SomaticModel`,
      `SomaticForMaskedLM`, `SomaticForSequenceClassification`,
      `SomaticForTokenClassification`, `SomaticTokenizerFast`.
- [x] `src/somatic/__init__.py`: register and flag for auto-class file copy:
      ```python
      AutoConfig.register("somatic", SomaticConfig, exist_ok=True)
      AutoModel.register(SomaticConfig, SomaticModel, exist_ok=True)
      AutoModelForMaskedLM.register(SomaticConfig, SomaticForMaskedLM, exist_ok=True)
      AutoModelForSequenceClassification.register(SomaticConfig, SomaticForSequenceClassification, exist_ok=True)
      AutoModelForTokenClassification.register(SomaticConfig, SomaticForTokenClassification, exist_ok=True)
      AutoTokenizer.register(SomaticConfig, fast_tokenizer_class=SomaticTokenizerFast, exist_ok=True)

      SomaticConfig.register_for_auto_class("AutoConfig")
      SomaticModel.register_for_auto_class("AutoModel")
      SomaticForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")
      SomaticForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
      SomaticForTokenClassification.register_for_auto_class("AutoModelForTokenClassification")
      SomaticTokenizerFast.register_for_auto_class("AutoTokenizer")
      ```
- [x] Guard against import cycles (registration imports model classes, which
      import config/tokenizer). No cycle: tokenization/config/modeling only import
      transformers + tokenizers + torch, never back into the top package.

### 4a. Tied-weight load fix (required by `register_for_auto_class`)
- [x] Setting `register_for_auto_class` writes `auto_map` into `config.json`,
      which routes `from_pretrained` through the meta-device custom-code path. That
      path marks the tied target (`lm_head.weight`) as already-initialized, drops it
      from `missing_keys`, then declines to tie ("both present"), stranding it on
      the meta device. (oplm dodges this by defaulting `tie_word_embeddings=False`;
      Somatic requires the tie.) Fix: override `SomaticForMaskedLM.tie_weights` to
      perform the single `lm_head.weight <- input embedding` tie directly.
- [x] Verified end to end: in-process `AutoModelForMaskedLM`/`AutoModel`/
      `AutoTokenizer.from_pretrained` AND a subprocess `trust_remote_code=True`
      reload (no `import somatic`) both reproduce logits (maxdiff 0.0),
      `token_type_ids`, and the tie. All helper `.py` files + `auto_map` land
      beside `config.json` / `tokenizer_config.json`.

## Phase 5 — Collator (`data/collator.py`)

- [x] Update the singleton import to `from ..model.tokenization_somatic import tokenizer`.
- [x] Rename output dict keys: `token_ids` → `input_ids`, `chain_ids` →
      `token_type_ids` (keep the same 0/1 construction). Keep `attention_mask`,
      `special_tokens_mask`, `cdr_mask`, `non_templated_mask`, `coords`.
- [x] Update the internal `EncodedPair` TypedDict keys accordingly.
- [x] Keep the manual token assembly (it aligns coords/cdr/nt masks with the
      token layout — simpler than re-deriving offsets from the tokenizer).

## Phase 6 — Masking (`masking/masking.py`)

- [x] Rename `apply_mask` parameter `token_ids` → `input_ids` in
      `InformationWeightedMasker` and `UniformMasker`.
- [x] Change return contract to `(masked_input_ids, labels)` where
      `labels = input_ids.clone(); labels[~mask] = -100`. Internals keep the
      Gumbel-top-k weighted sampling; the boolean mask is now expressed
      downstream as `labels != -100`.

## Phase 7 — Trainer & checkpoints

Files: `training/trainer.py`, `training/checkpoint.py`, `training/metrics.py`,
`train.py`, `cli.py`.

- [x] `train.py`: build `SomaticForMaskedLM(SomaticConfig(**hf_named_fields))` via
      `_build_model_config(cfg.model)`. NOTE: that helper currently reads the
      *current* (pre-Phase-10) Hydra key names (`d_model`, `n_layers`, ...) and maps
      them onto the HF `SomaticConfig` fields, so `somatic train`/`model-size` keep
      working against today's YAMLs. Phase 10 renames the YAML keys → at that point
      simplify `_build_model_config` to read the HF names directly (also shared by
      `cli.py model-size`). `training/flops.py` updated to read `hidden_size` /
      `num_hidden_layers` / `intermediate_size` / `max_position_embeddings`.
- [x] Training step: get `labels` from the masker; call
      `model(input_ids=masked_input_ids, token_type_ids=batch["token_type_ids"],
      attention_mask=batch["attention_mask"], labels=labels)` and use
      `outputs.loss`. Kept `compute_masked_cross_entropy` (still used inside
      `compute_mlm_metrics` for train/eval logging — Phase 8 refactors metrics);
      only the trainer's direct loss call was removed. Also updated
      `eval/regions.py::extract_region_masks` to read `token_type_ids` (the
      training masking-frequency tracker depends on it).
- [x] **Checkpoint format (clean break)** — replaced the combined-`.pt` /
      `asdict(config)` scheme in `checkpoint.py`:
      - Publishable/best/final: `model.save_pretrained(dir)` (→ `config.json`
        with `auto_map` + `model.safetensors`) and `tokenizer.save_pretrained(dir)`
        (→ `tokenizer.json`, `tokenizer_config.json`). Verified the bundled
        custom-code `.py` files land in the dir.
      - Resume state: sibling `training_state.pt` =
        `{step, epoch, optimizer_state_dict, scheduler_state_dict, rng, metrics}`.
      - Resume path: `train.py` rebuilds the model via
        `SomaticForMaskedLM.from_pretrained(dir)`, then
        `CheckpointManager.load_training_state(dir)` restores optimizer/scheduler/
        RNG and sets `global_step`/`epoch`.
      - Legacy field-pop logic was already gone (removed with `transformer.py` in
        Phase 2); nothing to drop in `checkpoint.py`.
      - Hardened rotation: re-saving the same step (final checkpoint on a
        checkpoint_steps boundary) no longer deletes the just-written directory.
- [x] Preserved the existing constraint: the scheduler is NOT wrapped by
      `accelerator.prepare()` (avoids the 8x-step DDP bug). Unchanged.

## Phase 8 — Eval system (`eval/`, `training/metrics.py`)

- [x] All `model(...)` calls use `input_ids` / `token_type_ids` /
      `attention_mask`; read `outputs.logits` / `outputs.attentions` /
      `outputs.hidden_states` (attribute access). Replaced the dead
      `from ...model.transformer import ModelOutput` type import with
      `transformers.modeling_outputs.MaskedLMOutput` across base.py + all metric
      files. Added `assert logits is not None` where needed (stubs type
      `MaskedLMOutput.logits` as Optional).
- [x] Refactored every metric `update(outputs, batch, mask_labels)` to consume
      `labels`: `mask = labels != -100`, `targets = labels`. Region metrics still
      read `cdr_mask`/`token_type_ids` from the batch via `extract_region_masks`.
- [x] Updated `eval/region_eval.py`, `eval/per_position.py`,
      `eval/cross_chain_eval.py`, `eval/masking.py`, `eval/evaluator.py` forward
      calls + output access; threaded `token_type_ids`. `EvalMasker` now returns
      `(masked_input_ids, labels)`; dropped the dead diffusion `noise_schedule`
      branch in the evaluator.
- [x] Hidden-state metrics: `MaskedLMOutput.hidden_states` is the per-layer tuple
      (no `last_hidden_state` on the MLM head), so probes read `hidden_states[-1]`.
      Added a `needs_hidden_states` ClassVar (probes set it True) and wired the
      evaluator to pass `output_hidden_states` when any metric needs it (mirrors
      `needs_attentions`).
- [x] Verified end-to-end: `Evaluator.evaluate` (core MLM metrics + probes via the
      hidden-states path + standard region eval) and the full trainer→Evaluator
      integration both run with the new API.
- NOTE (Phase 10): `registry.build_metrics` still reads `cfg.model.n_layers` for
  the coords-only `p_at_l` default (skipped when no coords) — rename to
  `num_hidden_layers` when the YAMLs are renamed. `training/metrics.py` keeps its
  boolean-mask interface (`compute_mlm_metrics(..., mask_labels=...)`); the trainer
  feeds it `labels != -100`, so no change was needed there.

## Phase 9 — Encoding / inference & CLI (`encoding/encoder.py`, `cli.py`)

- [x] Use base `SomaticModel` for embeddings: `encode` / `encode_batch` now call
      `self.model.somatic(...)` and read `outputs.last_hidden_state`. The encoder
      holds a single `SomaticForMaskedLM`; its inner base encoder (`.somatic`)
      serves embeddings while the MLM head serves logits — one loaded artifact
      covers both paths (checkpoints are saved as `SomaticForMaskedLM`).
- [x] Use `SomaticForMaskedLM` for `get_logits` / `predict` / `log_likelihood` /
      `perplexity`: `self.model(...)` → `outputs.logits`.
- [x] Replaced `model(token_ids=, chain_ids=, attention_mask=)` with
      `model(input_ids=, token_type_ids=, attention_mask=)` everywhere in the
      encoder; renamed internal `chain_ids` locals → `token_type_ids`; fixed
      config attribute reads (`d_model` → `hidden_size`, `max_seq_len` →
      `max_position_embeddings`); migrated the tokenizer import to
      `..model.tokenization_somatic`.
- [x] `SomaticEncoder.from_pretrained` loads the HF dir format via
      `SomaticForMaskedLM.from_pretrained(dir)` (dropped the invalid HF
      `map_location` kwarg; the constructor moves the model to `device`).
      `predict_masked` already lives on `SomaticForMaskedLM` (Phase 2d) — no
      standalone helper remains.
- [x] `cli.py encode`: verified end-to-end against a saved HF checkpoint dir for
      both `--pooling none` (list of per-seq embeddings) and `--pooling mean`
      (stacked `(N, D)`). Updated `--checkpoint` help + examples to the dir format.
      ruff format/check + ty clean.

## Phase 10 — Hydra configs (`configs/model/*.yaml`)

- [x] Renamed keys to HF canonical names in `small.yaml`, `base.yaml`,
      `large.yaml`, `xlarge.yaml`: `d_model`→`hidden_size`,
      `n_layers`→`num_hidden_layers`, `n_heads`→`num_attention_heads`,
      `d_ffn`→`intermediate_size`, `max_seq_len`→`max_position_embeddings`,
      `layer_norm_eps`→`norm_eps`, `padding_idx`→`pad_token_id`. Consolidated
      `dropout` + `embedding_dropout` → a single `hidden_dropout` (the old
      `embedding_dropout` was already unread). Updated the commented
      alternative-shape blocks too. `configs/README.md` model tables + examples
      updated to match.
- [x] Hydra still orchestrates training (data/train/log/eval blocks unchanged);
      only the `model:` block field names changed.
- [x] Simplified `train.py::_build_model_config` to splat the resolved HF-named
      `model:` block straight onto `SomaticConfig` (shared by `cli.py model-size`).
      Updated `eval/registry.py` to read `num_hidden_layers` (was `n_layers`) for
      the `p_at_l` default, and the `cli.py model-size` docstring example.
- [x] Verified: `somatic model-size` for all four variants reports the expected
      param counts (24M / 141M / 648M / 1.2B); a 2-step `somatic train` smoke test
      against the toy CSV ran train + eval + region eval with no shape/config
      errors, wrote an HF checkpoint dir (config.json + safetensors + tokenizer +
      bundled custom-code), and that dir reloads via
      `SomaticForMaskedLM.from_pretrained`. ruff/ty clean.

## Phase 11 — Tests (`tests/`)

- [ ] `conftest.py`: update fixtures —
      `small_config = SomaticConfig(hidden_size=64, num_hidden_layers=2,
      num_attention_heads=2, max_position_embeddings=64, hidden_dropout=0.0)`;
      `small_model = SomaticForMaskedLM(small_config)`; add a base-`SomaticModel`
      fixture. `sample_batch` keys → `input_ids` / `token_type_ids`.
- [ ] Update every existing test: forward calls (`input_ids`/`token_type_ids`),
      output access (`outputs.logits` / `outputs.last_hidden_state`), collator key
      assertions, masker `(masked_input_ids, labels)` return, region/per-position
      eval.
- [ ] Add HF-compat tests (mirror `ref: tests/model/test_push_to_hub.py`,
      `tests/test_e2e_lifecycle.py`):
  - [ ] **save→reload**: tiny `SomaticForMaskedLM` → `save_pretrained(tmp)` →
        `SomaticForMaskedLM.from_pretrained(tmp)`; assert logits match.
  - [ ] **custom-code files copied**: after `save_pretrained` with
        `register_for_auto_class`, assert `modeling_somatic.py`,
        `configuration_somatic.py`, `tokenization_somatic.py`, and helper files
        land beside `config.json`, and `auto_map` entries exist in `config.json`.
  - [ ] **remote reload in subprocess**: fresh interpreter (no `import somatic`),
        `AutoModelForMaskedLM.from_pretrained(tmp, trust_remote_code=True)` and
        `AutoTokenizer.from_pretrained(tmp, trust_remote_code=True)`; assert class
        name and that `tok(heavy, light)` yields expected `input_ids` +
        `token_type_ids`.
  - [ ] Keep the "tiny model trains a few steps + one eval" end-to-end test
        (per global CLAUDE.md), updated to the new API.

## Phase 12 — Verification

- [ ] `ruff format` and `ruff check` clean.
- [ ] `ty` (type checker) clean.
- [ ] `pytest` green (unit + new HF-compat tests).
- [ ] Manual round-trip: build tiny `SomaticForMaskedLM`, `save_pretrained(tmp)`,
      `tokenizer.save_pretrained(tmp)`, reload via
      `AutoModelForMaskedLM.from_pretrained(tmp)` + `AutoTokenizer.from_pretrained(tmp)`;
      assert logits match pre-save and `tokenizer("EVQ...","DIQ...")` gives the
      expected `input_ids` + `token_type_ids` (cls/heavy=0, light/eos=1).
- [ ] `trust_remote_code` subprocess reload succeeds for both model and tokenizer
      with no `import somatic`.
- [ ] Train smoke test:
      `somatic train data.train=<tiny.csv> model=small train.batch_size=4`
      runs a few steps + one eval without shape errors; the resulting checkpoint
      dir loads via `AutoModelForMaskedLM.from_pretrained`.
- [ ] Encode smoke test:
      `somatic encode -c <dir> -i <seqs.csv> -o emb.pt --pooling mean`
      produces embeddings using `last_hidden_state`.

---

## Out of scope / consequences

- No backwards compatibility: old `.pt` checkpoints won't load and are discarded.
- Attention / RoPE / FFN / norm math is unchanged — repackaging + renaming + HF
  interfaces only. The single renamed internal concept is
  `chain_ids` → `token_type_ids`.
