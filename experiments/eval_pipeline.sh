#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Config
# -------------------------

MULTISTEP=2
DET_MEMBERS=4
GEN_MEMBERS=2
MAX_SAMPLES=2

PRED_DIR="data/multistep"
EVAL_DIR="data/eval"
PLOT_DIR="data/plots"

GROUNDTRUTH_PATH="data/era5"

DET_MODEL="det_ens"
GEN_MODEL="gen"
GEN_TIME_MODEL="gen_time_correct"

DET_SAVE="det_ens"
GEN_SAVE="gen"
GEN_TIME_SAVE="gen_time"

# New workflow: pred paths are directories
DET_PRED="${PRED_DIR}/${DET_SAVE}_multistep=${MULTISTEP}_members=${DET_MEMBERS}"
GEN_PRED="${PRED_DIR}/${GEN_SAVE}_multistep=${MULTISTEP}_members=${GEN_MEMBERS}"
GEN_TIME_PRED="${PRED_DIR}/${GEN_TIME_SAVE}_multistep=${MULTISTEP}_members=${GEN_MEMBERS}"

DET_EVAL="${EVAL_DIR}/${DET_SAVE}_multistep=${MULTISTEP}"
GEN_EVAL="${EVAL_DIR}/${GEN_SAVE}_multistep=${MULTISTEP}"
GEN_TIME_EVAL="${EVAL_DIR}/${GEN_TIME_SAVE}_multistep=${MULTISTEP}"

# New eval filenames because pred_path is a directory
DET_ENSEMBLE_METRIC="${DET_EVAL}/test-multistep=${MULTISTEP}-era5_ensemble_metrics.nc"
GEN_ENSEMBLE_METRIC="${GEN_EVAL}/test-multistep=${MULTISTEP}-era5_ensemble_metrics.nc"
GEN_TIME_ENSEMBLE_METRIC="${GEN_TIME_EVAL}/test-multistep=${MULTISTEP}-era5_ensemble_metrics.nc"

DET_BRIER_METRIC="${DET_EVAL}/test-multistep=${MULTISTEP}-era5_brier_skill_score.nc"
GEN_BRIER_METRIC="${GEN_EVAL}/test-multistep=${MULTISTEP}-era5_brier_skill_score.nc"
GEN_TIME_BRIER_METRIC="${GEN_TIME_EVAL}/test-multistep=${MULTISTEP}-era5_brier_skill_score.nc"


# -------------------------
# 1. Run rollouts
# -------------------------

python -m src.run_multistep \
  --model "${DET_MODEL}" \
  --output-name "${DET_SAVE}" \
  --multistep "${MULTISTEP}" \
  --max-samples "${MAX_SAMPLES}" \
  --force

python -m src.run_multistep \
  --model "${GEN_MODEL}" \
  --output-name "${GEN_SAVE}" \
  --multistep "${MULTISTEP}" \
  --num-members "${GEN_MEMBERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --force

python -m src.run_multistep \
  --model "${GEN_TIME_MODEL}" \
  --output-name "${GEN_TIME_SAVE}" \
  --multistep "${MULTISTEP}" \
  --num-members "${GEN_MEMBERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --force


# -------------------------
# 2. Evaluate rollouts
# -------------------------

python -m geoarches.evaluation.eval_multistep \
  --pred_path "${DET_PRED}" \
  --output_dir "${DET_EVAL}" \
  --groundtruth_path "${GROUNDTRUTH_PATH}" \
  --multistep "${MULTISTEP}" \
  --metrics era5_ensemble_metrics era5_brier_skill_score \
  --eval_batch_size 1 \
  --num_workers 0

python -m geoarches.evaluation.eval_multistep \
  --pred_path "${GEN_PRED}" \
  --output_dir "${GEN_EVAL}" \
  --groundtruth_path "${GROUNDTRUTH_PATH}" \
  --multistep "${MULTISTEP}" \
  --metrics era5_ensemble_metrics era5_brier_skill_score \
  --eval_batch_size 1 \
  --num_workers 0

python -m geoarches.evaluation.eval_multistep \
  --pred_path "${GEN_TIME_PRED}" \
  --output_dir "${GEN_TIME_EVAL}" \
  --groundtruth_path "${GROUNDTRUTH_PATH}" \
  --multistep "${MULTISTEP}" \
  --metrics era5_ensemble_metrics era5_brier_skill_score \
  --eval_batch_size 1 \
  --num_workers 0


# -------------------------
# 3. Plot comparison: det_ens vs gen vs gen_time
# -------------------------

python -m geoarches.evaluation.plot \
  --output_dir "${PLOT_DIR}/det_ens_vs_gen_vs_gen_time_multistep=${MULTISTEP}_ensemble" \
  --metric_paths \
    "${DET_ENSEMBLE_METRIC}" \
    "${GEN_ENSEMBLE_METRIC}" \
    "${GEN_TIME_ENSEMBLE_METRIC}" \
  --model_names_for_legend det_ens gen gen_time \
  --model_colors blue orange green \
  --metrics rmse crps fcrps spskr \
  --vars T2m SP Z500 T850 Q700 U850 V850 \
  --force

python -m geoarches.evaluation.plot \
  --output_dir "${PLOT_DIR}/det_ens_vs_gen_vs_gen_time_multistep=${MULTISTEP}_brier" \
  --metric_paths \
    "${DET_BRIER_METRIC}" \
    "${GEN_BRIER_METRIC}" \
    "${GEN_TIME_BRIER_METRIC}" \
  --model_names_for_legend det_ens gen gen_time \
  --model_colors blue orange green \
  --metrics brierskillscore \
  --vars T2m SP Z500 T850 Q700 U850 V850 \
  --brier_quantile_levels high high high high high high high \
  --force

echo "Done. det_ens, gen, gen_time rollouts, metrics, and plots are saved."