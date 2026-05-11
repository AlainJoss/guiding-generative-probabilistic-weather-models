import argparse
from pathlib import Path
from typing import Any

from src.utils import (
    get_model,
    ensure_rollout_dir,
    get_device,
    update_experiment_params,
    read_config
)
from src.rollout import rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", type=str, required=True)
    parser.add_argument("--config-type", choices=["guided", "unguided"], required=True)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def rollout_from_config(
    flow_model,
    rollout_dir,
    config: dict[str, Any],
    test: bool = False,
) -> Path:
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

    device = get_device()
    flow_model = get_model(device)

    args = parse_args()
    config = read_config(args.config_type, args.config_id)

    rollout_dir = ensure_rollout_dir(config["rollout_id"])
    
    if config["guidance_flag"]:
        update_experiment_params(rollout_dir, config)

    rollout_dir = rollout_from_config(
        flow_model,
        rollout_dir,
        config,
        test=args.test,
    )

    print(f"saved rollout to: {rollout_dir}")


if __name__ == "__main__":
    main()