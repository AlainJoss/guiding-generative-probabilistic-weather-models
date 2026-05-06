"""
python -m src.run_from_config \
  --config-id 2026-05-05_17:37:52 \
  --config-type guided \
  --test
"""

import argparse
from pathlib import Path
from typing import Any

from src.paths import CONFIGS
from src.utils import (
    read_json,
    get_model,
    ensure_rollout_dir,
    save_to_json,
    get_device,
)
from src.rollout import rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", type=str, required=True)
    parser.add_argument("--config-type", choices=["guided", "unguided"], required=True)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def read_config(config_type: str, config_id: str) -> dict[str, Any]:
    return read_json(CONFIGS / config_type, config_id)


GUIDANCE_PARAM_KEYS = ["guidance_mode", "alpha", "w"]

def update_experiment_params(
    rollout_dir: Path,
    config: dict[str, Any],
) -> dict[str, list[Any]]:
    try:
        experiment_params = read_json(rollout_dir, "experiment_params")
    except FileNotFoundError:
        experiment_params = {k: [] for k in GUIDANCE_PARAM_KEYS}

    for k in GUIDANCE_PARAM_KEYS:
        experiment_params.setdefault(k, [])

        value = config[k]
        if value not in experiment_params[k]:
            experiment_params[k].append(value)

    save_to_json(experiment_params, rollout_dir, "experiment_params")
    return experiment_params


def run_from_config(
    config: dict[str, Any],
    test: bool = False,
) -> Path:
    device = get_device()
    flow_model = get_model(device)

    rollout_dir = ensure_rollout_dir(config["rollout_id"])
    
    if config["guidance_flag"]:
        update_experiment_params(rollout_dir, config)

    rollout(
        guidance_flag=config["guidance_flag"],
        rollout_dir=rollout_dir,
        flow_model=flow_model,
        timestamp=config["timestamp"],
        mask_corners=config["mask_corners"],
        init_mask_term=config["init_mask_term"],
        y=config.get("y"),
        lambda_=config.get("lambda_"),
        N=config["N"],
        partition=config["partition"],
        level_idx=config["level_idx"],
        var_idx=config["var_idx"],
        M=config["M"],
        test=test,
        config=config,
    )

    return rollout_dir


def main():
    print("running experiment")

    args = parse_args()
    config = read_config(args.config_type, args.config_id)

    rollout_dir = run_from_config(
        config,
        test=args.test,
    )

    print(f"saved rollout to: {rollout_dir}")


if __name__ == "__main__":
    main()