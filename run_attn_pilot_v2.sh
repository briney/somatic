#!/usr/bin/env bash
# Sequential 5-variant pilot for chain-aware attention ablation V2.
# 50k steps each, eval every 5k, 4× A6000 with accelerate (bf16).
# Logs to W&B project: somatic-attn-optimization-v2.

set -euo pipefail

cd /home/jovyan/work/somatic/somatic

TRAIN=/home/jovyan/work/ablm_training-data/v2026-05-03/train-test-eval_splits/train-clust95_testeval-clust80/minimal/train.parquet
EVAL=/home/jovyan/work/ablm_training-data/v2026-05-03/train-test-eval_splits/train-clust95_testeval-clust80/minimal/L2677-eval_unclustered.parquet
OUT_BASE=/home/jovyan/work/somatic/somatic/outputs/attn_ablation_v2
LOG_DIR="${OUT_BASE}/_logs"
mkdir -p "${LOG_DIR}"

COMMON_OVERRIDES=(
  "data.train=${TRAIN}"
  "data.eval=${EVAL}"
  "train.max_steps=50000"
  "train.scheduler.warmup_steps=1000"
  "train.eval_steps=5000"
  "train.checkpoint_steps=50000"
  "train.save_best=true"
  "train.keep_last_n_checkpoints=2"
  "train.mixed_precision=bf16"
  "eval.regions.enabled=false"
  "log.wandb.project=somatic-attn-optimization-v2"
  "seed=42"
)

run_variant () {
  local name="$1"
  shift
  local out="${OUT_BASE}/${name}"
  local log="${LOG_DIR}/${name}.log"
  echo "===> [$(date -Is)] starting variant: ${name}" | tee -a "${LOG_DIR}/master.log"
  accelerate launch -m somatic.train \
    --name "${name}" \
    "${COMMON_OVERRIDES[@]}" \
    "output_dir=${out}" \
    "name=${name}" \
    "$@" \
    >> "${log}" 2>&1
  echo "===> [$(date -Is)] finished variant: ${name}" | tee -a "${LOG_DIR}/master.log"
}

# 1: separate-QKV chain-aware (current default)
run_variant "separate_chain_aware" \
  "model=small" \
  "model.use_chain_aware_attention=true" \
  "model.chain_aware_projection_mode=separate"

# 2: shared-QKV chain-aware (same size as standard small)
run_variant "shared_chain_aware" \
  "model=small" \
  "model.use_chain_aware_attention=true" \
  "model.chain_aware_projection_mode=shared"

# 3: shared-QKV chain-aware param-matched to separate (~24M)
run_variant "shared_chain_aware_pm" \
  "model=small" \
  "model.use_chain_aware_attention=true" \
  "model.chain_aware_projection_mode=shared" \
  "model.d_model=288" \
  "model.n_heads=4" \
  "model.n_layers=24"

# 4: standard MHA (same size as small chain-aware base width)
run_variant "standard_small" \
  "model=small" \
  "model.use_chain_aware_attention=false"

# 5: parameter-matched standard MHA (~24M, d=288)
run_variant "standard_param_matched" \
  "model=small" \
  "model.use_chain_aware_attention=false" \
  "model.d_model=288" \
  "model.n_heads=4" \
  "model.n_layers=24"

echo "===> [$(date -Is)] all five variants complete." | tee -a "${LOG_DIR}/master.log"
