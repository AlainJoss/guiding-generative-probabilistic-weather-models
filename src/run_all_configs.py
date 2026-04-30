import argparse
from pathlib import Path
from typing import Any

from src.paths import CONFIGS
from src.utils import read_json
from run_from_cfg import run_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of configs to run.",
    )
    return parser.parse_args()


def load_unguided_configs() -> list[tuple[str, dict[str, Any]]]:
    config_dir = CONFIGS / "unguided"

    config_paths = sorted(config_dir.glob("*.json"))

    configs = []
    for config_path in config_paths:
        config_id = config_path.stem
        config = read_json(config_dir, config_id)
        configs.append((config_id, config))

    return configs

def main():
    args = parse_args()

    configs = load_unguided_configs()

    if args.limit is not None:
        configs = configs[: args.limit]

    print(f">>> found {len(configs)} unguided configs")

    for idx, (config_id, config) in enumerate(configs, start=1):
        print(f"\n>>> running config {idx}/{len(configs)}: {config_id}")

        rollout_dir = run_from_config(config, test=args.test)

        print(f">>> finished config {config_id}")
        print(f">>> saved rollout to: {rollout_dir}")


if __name__ == "__main__":
    main()