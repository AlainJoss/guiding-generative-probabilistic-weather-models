#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Config
# -------------------------

MULTISTEP=2
GEN_MEMBERS=2
MAX_SAMPLES=2

PRED_DIR="data/multistep"
EVAL_DIR="data/eval"
PLOT_DIR="data/plots"

GROUNDTRUTH_PATH="data/era5"

MODEL="gen"
SAVE_NAME="gen_time"

GEN_PRED="${PRED_DIR}/${SAVE_NAME}_multistep=${MULTISTEP}_members=${GEN_MEMBERS}_0.nc"
GEN_EVAL="${EVAL_DIR}/${SAVE_NAME}_multistep=${MULTISTEP}"

GEN_ENSEMBLE_METRIC="${GEN_EVAL}/${SAVE_NAME}_multistep=${MULTISTEP}_members=${GEN_MEMBERS}_0-era5_ensemble_metrics.nc"
GEN_BRIER_METRIC="${GEN_EVAL}/${SAVE_NAME}_multistep=${MULTISTEP}_members=${GEN_MEMBERS}_0-era5_brier_skill_score.nc"


# -------------------------
# 1. Run generative ensemble rollouts
# -------------------------

python -m src.run_multistep \
  --model "${MODEL}" \
  --output-name "${SAVE_NAME}" \
  --multistep "${MULTISTEP}" \
  --num-members "${GEN_MEMBERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --force


# -------------------------
# 2. Evaluate generative ensemble
# -------------------------

python -m geoarches.evaluation.eval_multistep \
  --pred_path "${GEN_PRED}" \
  --output_dir "${GEN_EVAL}" \
  --groundtruth_path "${GROUNDTRUTH_PATH}" \
  --multistep "${MULTISTEP}" \
  --metrics era5_ensemble_metrics era5_brier_skill_score \
  --eval_batch_size 1 \
  --num_workers 0


# -------------------------
# 3. Plot ensemble metrics
# -------------------------

python -m geoarches.evaluation.plot \
  --output_dir "${PLOT_DIR}/${SAVE_NAME}_multistep=${MULTISTEP}_ensemble" \
  --metric_paths "${GEN_ENSEMBLE_METRIC}" \
  --model_names_for_legend "${SAVE_NAME}" \
  --model_colors orange \
  --metrics rmse crps fcrps spskr \
  --vars T2m SP Z500 T850 Q700 U850 V850 \
  --force


# -------------------------
# 4. Plot Brier skill score
# -------------------------

python -m geoarches.evaluation.plot \
  --output_dir "${PLOT_DIR}/${SAVE_NAME}_multistep=${MULTISTEP}_brier" \
  --metric_paths "${GEN_BRIER_METRIC}" \
  --model_names_for_legend "${SAVE_NAME}" \
  --model_colors orange \
  --metrics brierskillscore \
  --vars T2m SP Z500 T850 Q700 U850 V850 \
  --brier_quantile_levels high high high high high high high \
  --force


echo "Done. ${SAVE_NAME} rollouts, metrics, and plots are saved."