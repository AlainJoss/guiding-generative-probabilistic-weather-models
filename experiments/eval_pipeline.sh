#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Config
# -------------------------

MULTISTEP=10
DET_MEMBERS=4
GEN_MEMBERS=4
MAX_SAMPLES=100

PRED_DIR="data/multistep"
EVAL_DIR="data/eval"
PLOT_DIR="data/plots"

GROUNDTRUTH_PATH="data/era5"

DET_MODEL="det_ens"
GEN_MODEL="gen"

DET_PRED="${PRED_DIR}/${DET_MODEL}_multistep=${MULTISTEP}_members=${DET_MEMBERS}_0.nc"
GEN_PRED="${PRED_DIR}/${GEN_MODEL}_multistep=${MULTISTEP}_members=${GEN_MEMBERS}_0.nc"

DET_EVAL="${EVAL_DIR}/${DET_MODEL}_multistep=${MULTISTEP}"
GEN_EVAL="${EVAL_DIR}/${GEN_MODEL}_multistep=${MULTISTEP}"

DET_ENSEMBLE_METRIC="${DET_EVAL}/${DET_MODEL}_multistep=${MULTISTEP}_members=${DET_MEMBERS}_0-era5_ensemble_metrics.nc"
GEN_ENSEMBLE_METRIC="${GEN_EVAL}/${GEN_MODEL}_multistep=${MULTISTEP}_members=${GEN_MEMBERS}_0-era5_ensemble_metrics.nc"

DET_BRIER_METRIC="${DET_EVAL}/${DET_MODEL}_multistep=${MULTISTEP}_members=${DET_MEMBERS}_0-era5_brier_skill_score.nc"
GEN_BRIER_METRIC="${GEN_EVAL}/${GEN_MODEL}_multistep=${MULTISTEP}_members=${GEN_MEMBERS}_0-era5_brier_skill_score.nc"


# -------------------------
# 1. Run deterministic ensemble rollouts
# -------------------------

python -m experiments.run_multistep \
  --model "${DET_MODEL}" \
  --multistep "${MULTISTEP}" \
  --max-samples "${MAX_SAMPLES}" \
  --force


# -------------------------
# 2. Run generative ensemble rollouts
# -------------------------

python -m experiments.run_multistep \
  --model "${GEN_MODEL}" \
  --multistep "${MULTISTEP}" \
  --num-members "${GEN_MEMBERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --force


# -------------------------
# 3. Evaluate deterministic ensemble
# -------------------------

python -m geoarches.evaluation.eval_multistep \
  --pred_path "${DET_PRED}" \
  --output_dir "${DET_EVAL}" \
  --groundtruth_path "${GROUNDTRUTH_PATH}" \
  --multistep "${MULTISTEP}" \
  --metrics era5_ensemble_metrics era5_brier_skill_score \
  --eval_batch_size 1 \
  --num_workers 0


# -------------------------
# 4. Evaluate generative ensemble
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
# 5. Plot ensemble metrics: det_ens vs gen
# -------------------------

python -m geoarches.evaluation.plot \
  --output_dir "${PLOT_DIR}/det_ens_vs_gen_multistep=${MULTISTEP}_ensemble" \
  --metric_paths \
    "${DET_ENSEMBLE_METRIC}" \
    "${GEN_ENSEMBLE_METRIC}" \
  --model_names_for_legend det_ens gen \
  --model_colors blue orange \
  --metrics rmse crps fcrps spskr \
  --vars T2m SP Z500 T850 Q700 U850 V850 \
  --force


# -------------------------
# 6. Plot Brier skill score: det_ens vs gen
# -------------------------

python -m geoarches.evaluation.plot \
  --output_dir "${PLOT_DIR}/det_ens_vs_gen_multistep=${MULTISTEP}_brier" \
  --metric_paths \
    "${DET_BRIER_METRIC}" \
    "${GEN_BRIER_METRIC}" \
  --model_names_for_legend det_ens gen \
  --model_colors blue orange \
  --metrics brierskillscore \
  --vars T2m SP Z500 T850 Q700 U850 V850 \
  --brier_quantile_levels high high high high high high high \
  --force


echo "Done. det_ens vs gen rollouts, metrics, and plots are saved."