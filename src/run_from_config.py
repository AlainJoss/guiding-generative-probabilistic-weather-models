"""
python -m src.run_from_config \
  --config-id 2026-05-05_17:37:52 \
  --config-type unguided \
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


def run_from_config(
    config: dict[str, Any],
    config_type: str,
    test: bool = False,
) -> Path:
    device = get_device()
    flow_model = get_model(device)

    rollout_dir = ensure_rollout_dir(config["rollout_id"])

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
    )

    save_to_json(
        config,
        rollout_dir,
        "config",
    )

    return rollout_dir


def main():
    print("running experiment")

    args = parse_args()
    config = read_config(args.config_type, args.config_id)

    rollout_dir = run_from_config(
        config,
        args.config_type,
        test=args.test,
    )

    print(f"saved rollout to: {rollout_dir}")


if __name__ == "__main__":
    main()