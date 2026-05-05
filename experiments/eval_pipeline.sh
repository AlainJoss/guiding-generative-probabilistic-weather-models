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

# Old models, kept for reference
# DET_MODEL="det_ens"
# GEN_MODEL="gen"

GEN_MODEL_TIME_CORRECT="gen_time_correct"
SAVE_NAME="gen_time"

GEN_TIME_PRED="${PRED_DIR}/${SAVE_NAME}_multistep=${MULTISTEP}_members=${GEN_MEMBERS}_0.nc"
GEN_TIME_EVAL="${EVAL_DIR}/${SAVE_NAME}_multistep=${MULTISTEP}"

GEN_TIME_ENSEMBLE_METRIC="${GEN_TIME_EVAL}/${SAVE_NAME}_multistep=${MULTISTEP}_members=${GEN_MEMBERS}_0-era5_ensemble_metrics.nc"
GEN_TIME_BRIER_METRIC="${GEN_TIME_EVAL}/${SAVE_NAME}_multistep=${MULTISTEP}_members=${GEN_MEMBERS}_0-era5_brier_skill_score.nc"


# -------------------------
# Old deterministic ensemble rollout
# -------------------------

# python -m src.run_multistep \
#   --model "${DET_MODEL}" \
#   --multistep "${MULTISTEP}" \
#   --max-samples "${MAX_SAMPLES}" \
#   --force


# -------------------------
# Old generative rollout
# -------------------------

# python -m src.run_multistep \
#   --model "${GEN_MODEL}" \
#   --multistep "${MULTISTEP}" \
#   --num-members "${GEN_MEMBERS}" \
#   --max-samples "${MAX_SAMPLES}" \
#   --force


# -------------------------
# 1. Run time-corrected generative rollout
# -------------------------

python -m src.run_multistep \
  --model "${GEN_MODEL_TIME_CORRECT}" \
  --output-name "${SAVE_NAME}" \
  --multistep "${MULTISTEP}" \
  --num-members "${GEN_MEMBERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --force


# -------------------------
# Old deterministic eval
# -------------------------

# python -m geoarches.evaluation.eval_multistep \
#   --pred_path "${DET_PRED}" \
#   --output_dir "${DET_EVAL}" \
#   --groundtruth_path "${GROUNDTRUTH_PATH}" \
#   --multistep "${MULTISTEP}" \
#   --metrics era5_ensemble_metrics era5_brier_skill_score \
#   --eval_batch_size 1 \
#   --num_workers 0


# -------------------------
# Old generative eval
# -------------------------

# python -m geoarches.evaluation.eval_multistep \
#   --pred_path "${GEN_PRED}" \
#   --output_dir "${GEN_EVAL}" \
#   --groundtruth_path "${GROUNDTRUTH_PATH}" \
#   --multistep "${MULTISTEP}" \
#   --metrics era5_ensemble_metrics era5_brier_skill_score \
#   --eval_batch_size 1 \
#   --num_workers 0


# -------------------------
# 2. Evaluate time-corrected generative rollout
# -------------------------

python -m geoarches.evaluation.eval_multistep \
  --pred_path "${GEN_TIME_PRED}" \
  --output_dir "${GEN_TIME_EVAL}" \
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
  --metric_paths "${GEN_TIME_ENSEMBLE_METRIC}" \
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
  --metric_paths "${GEN_TIME_BRIER_METRIC}" \
  --model_names_for_legend "${SAVE_NAME}" \
  --model_colors orange \
  --metrics brierskillscore \
  --vars T2m SP Z500 T850 Q700 U850 V850 \
  --brier_quantile_levels high high high high high high high \
  --force


echo "Done. ${SAVE_NAME} rollouts, metrics, and plots are saved."