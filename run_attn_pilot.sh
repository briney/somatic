#!/usr/bin/env bash
# Sequential 4-GPU pilot for chain-aware attention ablation (A6000 x4).
# Logs to W&B project: somatic-attn-optimization.

set -euo pipefail

cd /home/jovyan/work/somatic/somatic

TRAIN=/home/jovyan/work/ablm_training-data/v2026-05-03/train-test-eval_splits/train-clust95_testeval-clust80/minimal/train.parquet
EVAL=/home/jovyan/work/ablm_training-data/v2026-05-03/train-test-eval_splits/train-clust95_testeval-clust80/minimal/L2677-eval_unclustered.parquet
OUT_BASE=/home/jovyan/work/somatic/somatic/outputs/attn_ablation
LOG_DIR="${OUT_BASE}/_logs"
mkdir -p "${LOG_DIR}"

COMMON_OVERRIDES=(
  "data.train=${TRAIN}"
  "data.eval=${EVAL}"
  "train.max_steps=10000"
  "train.scheduler.warmup_steps=1000"
  "train.eval_steps=1000"
  "train.checkpoint_steps=10000"
  "train.save_best=false"
  "train.mixed_precision=bf16"
  "eval.regions.enabled=false"
  "log.wandb.project=somatic-attn-optimization"
  "seed=42"
)

run_variant () {
  local name="$1"
  shift
  local out="${OUT_BASE}/${name}"
  local log="${LOG_DIR}/${name}.log"
  echo "===> [$(date -Is)] starting variant: ${name}" | tee -a "${LOG_DIR}/master.log"
  # Note: --output-dir on the argparse layer would override the Hydra
  # output_dir= override below, so we only set it via Hydra.
  accelerate launch -m somatic.train \
    --name "${name}" \
    "${COMMON_OVERRIDES[@]}" \
    "output_dir=${out}" \
    "name=${name}" \
    "$@" \
    >> "${log}" 2>&1
  echo "===> [$(date -Is)] finished variant: ${name}" | tee -a "${LOG_DIR}/master.log"
}

# Variant 1: separate-QKV chain-aware (current default).
run_variant "separate_chain_aware" \
  "model=small" \
  "model.use_chain_aware_attention=true" \
  "model.chain_aware_projection_mode=separate"

# Variant 2: shared-QKV chain-aware.
run_variant "shared_chain_aware" \
  "model=small" \
  "model.use_chain_aware_attention=true" \
  "model.chain_aware_projection_mode=shared"

# Variant 3: same-size standard MHA.
run_variant "standard_small" \
  "model=small" \
  "model.use_chain_aware_attention=false"

# Variant 4: parameter-matched standard MHA.
run_variant "standard_param_matched" \
  "model=small" \
  "model.use_chain_aware_attention=false" \
  "model.d_model=288" \
  "model.n_heads=4" \
  "model.n_layers=24"

echo "===> [$(date -Is)] all four variants complete." | tee -a "${LOG_DIR}/master.log"
