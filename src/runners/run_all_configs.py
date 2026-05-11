import argparse
from copy import deepcopy
from typing import Any

from src.paths import CONFIGS
from src.runners.run_from_config import rollout_from_config
from src.funcs import T_schedule
from src.constants import GUIDANCE_MODES
from src.utils import (
    get_model,
    ensure_rollout_dir,
    get_device,
    update_experiment_params,
    read_config, read_json, list_tens_to_floats
)


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
        config["lambda_"] = list_tens_to_floats(
            T_schedule(config["alpha"], config["w"])
        )

    return config


def main():
    device = get_device()
    flow_model = get_model(device)

    args = parse_args()

    configs = load_configs(args.config_type)
    print(f"found {len(configs)} configs")

    for idx, (config_id, config) in enumerate(configs, start=1):
        print(f"running config {idx}/{len(configs)}: {config_id}")

        rollout_dir = ensure_rollout_dir(config["rollout_id"])
    
        config = apply_overrides(
            config,
            guidance_mode=args.guidance_mode,
            alpha=args.alpha,
            w=args.w,
        )
    
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