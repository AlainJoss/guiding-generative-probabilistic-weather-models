import argparse
from pathlib import Path
from typing import Any

from src.paths import CONFIGS
from src.utils import (
    read_json,
    get_dataset,
    get_model,
    ensure_rollout_dir,
    save_to_json,
    state_to_device,
    get_device,
)
from src.rollout import rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", type=str, required=True)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def read_config(config_id: str) -> dict[str, Any]:
    config_dir = (
        CONFIGS / "guided"
        if (CONFIGS / "guided" / f"{config_id}.json").exists()
        else CONFIGS / "unguided"
    )
    return read_json(config_dir, config_id)


def run_from_config(config: dict[str, Any], test: bool = False) -> Path:
    guidance_flag = config["guidance_flag"]
    rollout_type = "guided" if guidance_flag else "unguided"

    device = get_device()
    ds = get_dataset()
    model = get_model(device)

    x_start = ds[config["timestamp_idx"]]
    rollout_dir = ensure_rollout_dir(rollout_type, config["N"])

    for m in range(1, config["M"] + 1):
        print(f"m: {m}/{config['M']}")

        rollout(
            guidance_flag=config["guidance_flag"],
            rollout_dir=rollout_dir,
            ds=ds,
            x_start=state_to_device(x_start, device),
            gen_model=model,
            init_mask_term=config["init_mask_term"],
            mask_corners=config["mask_corners"],
            y=config["y"],
            lambda_=config["lambda_"],
            N=config["N"],
            partition=config["partition"],
            level_idx=config["level_idx"],
            var_idx=config["var_idx"],
            m=m,
            test=test,
        )

    save_to_json(
        {
            **config,
            "rollout_dir": str(rollout_dir)
        },
        rollout_dir,
        "config",
    )

    print(f"Saved rollout to: {rollout_dir}")
    return rollout_dir


def main() -> None:
    args = parse_args()
    config = read_config(args.config_id)
    run_from_config(config, test=args.test)


if __name__ == "__main__":
    main()