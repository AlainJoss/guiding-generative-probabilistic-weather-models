"""
python -m src.run_all_configs \
  --config-type unguided 

python -m src.run_all_configs \
  --config-type guided

# Achtung: this will just overwrite the previous results every time. Need to make subdirs (new timestamps for guided experiments)
# unguided rollouts can stay as they are as can the ground truth --> impacts analysis notebook, need to write subfoldering logic with experiment type (like manual or other), and if manual slide experiment variables (w and alpha)
python -m src.run_all_configs \
  --config-type guided \
  --guidance-mode manual_trajectory \
  --alpha 1.0 \
  --w 2.0
"""

import argparse
from copy import deepcopy
from typing import Any

from src.paths import CONFIGS
from src.utils import read_json, list_tens_to_floats
from src.run_from_config import run_from_config
from src.funcs import T_schedule


GUIDANCE_MODES = [
    "manual_trajectory",
    "ground_truth",
    "lower_boundary",
    "upper_boundary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--config-type", choices=["guided", "unguided"], required=True)
    parser.add_argument("--guidance-mode", choices=GUIDANCE_MODES)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--w", type=float)
    return parser.parse_args()


def load_configs(config_type: str) -> list[tuple[str, dict[str, Any]]]:
    config_dir = CONFIGS / config_type
    config_paths = sorted(config_dir.glob("*.json"))

    configs = []
    for config_path in config_paths:
        config_id = config_path.stem
        config = read_json(config_dir, config_id)
        configs.append((config_id, config))

    return configs


def apply_overrides(
    config: dict[str, Any],
    guidance_mode: str | None = None,
    alpha: float | None = None,
    w: float | None = None,
) -> dict[str, Any]:
    config = deepcopy(config)

    if guidance_mode is not None:
        config["guidance_mode"] = guidance_mode

    if alpha is not None:
        config["alpha"] = alpha

    if w is not None:
        config["w"] = w

    if alpha is not None or w is not None:
        if "alpha" not in config:
            raise KeyError("Cannot recompute lambda_: config has no 'alpha'.")
        if "w" not in config:
            raise KeyError("Cannot recompute lambda_: config has no 'w'.")

        config["lambda_"] = list_tens_to_floats(
            T_schedule(config["alpha"], config["w"])
        )

    return config


def main():
    args = parse_args()

    configs = load_configs(args.config_type)
    print(f"found {len(configs)} configs")

    for idx, (config_id, config) in enumerate(configs, start=1):
        print(f"running config {idx}/{len(configs)}: {config_id}")

        config = apply_overrides(
            config,
            guidance_mode=args.guidance_mode,
            alpha=args.alpha,
            w=args.w,
        )

        rollout_dir = run_from_config(
            config,
            args.config_type,
            test=args.test,
        )

        print(f"saved rollout to: {rollout_dir}")


if __name__ == "__main__":
    main()