import argparse
from typing import Any

from src.paths import CONFIGS
from src.utils import read_json
from src.run_from_config import run_from_config
from src.funcs import T_schedule
from src.utils import list_tens_to_floats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--config-type", choices=["guided", "unguided"], required=True)
    parser.add_argument("--guidance-mode", choices=[
        "manual_trajectory",
        "ground_truth",
        "lower_boundary",
        "upper_boundary",
    ])
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--w", type=float)
    return parser.parse_args()


def load_configs(config_type) -> list[tuple[str, dict[str, Any]]]:
    config_dir = CONFIGS / "to_run" / f"{config_type}"

    config_paths = sorted(config_dir.glob("*.json"))

    configs = []
    for config_path in config_paths:
        config_id = config_path.stem
        config = read_json(config_dir, config_id)
        configs.append((config_id, config))

    return configs

def main():
    args = parse_args()

    configs = load_configs(args.config_type)

    print(f"found {len(configs)} configs")

    for idx, (config_id, config) in enumerate(configs, start=1):
        print(f"running config {idx}/{len(configs)}: {config_id}")

        if args.guidance_mode is not None:
            config["guidance_mode"] = args.guidance_mode

        if args.alpha is not None:
            config["alpha"] = args.alpha

        if args.w is not None:
            config["w"] = args.w

        if args.alpha is not None or args.w is not None:
            alpha = config["alpha"]
            w = config["w"]
            config["lambda_"] = list_tens_to_floats(T_schedule(alpha, w))

        rollout_dir = run_from_config(config, args.config_type, test=args.test)

        print(f"saved rollout to: {rollout_dir}")

if __name__ == "__main__":
    main()