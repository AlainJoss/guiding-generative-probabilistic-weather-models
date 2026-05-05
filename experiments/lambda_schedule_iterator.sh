#!/usr/bin/env bash
set -euo pipefail

CONFIG_TYPE="guided"
TEST_FLAG=""

# Uncomment for test mode
# TEST_FLAG="--test"

GUIDANCE_MODES=(
  "manual_trajectory"
  "ground_truth"
  "lower_boundary"
  "upper_boundary"
)

# Only used for manual_trajectory
ALPHAS=(1.0 2.0 3.0)
WS=(1.0 2.0 3.0)

for guidance_mode in "${GUIDANCE_MODES[@]}"; do

  if [[ "${guidance_mode}" == "manual_trajectory" ]]; then
    for alpha in "${ALPHAS[@]}"; do
      for w in "${WS[@]}"; do
        echo "Running: mode=${guidance_mode}, alpha=${alpha}, w=${w}"

        python -m src.run_all_configs \
          --config-type "${CONFIG_TYPE}" \
          --guidance-mode "${guidance_mode}" \
          --alpha "${alpha}" \
          --w "${w}" \
          ${TEST_FLAG}
      done
    done

  else
    echo "Running: mode=${guidance_mode}"

    python -m src.run_all_configs \
      --config-type "${CONFIG_TYPE}" \
      --guidance-mode "${guidance_mode}" \
      ${TEST_FLAG}
  fi

done

echo "Done. Sweep finished."