import argparse
from typing import Any

from src.paths import CONFIGS
from src.utils import read_json
from src.run_from_cfg import run_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--config-type", choices=["guided", "unguided"], required=True)
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

        rollout_dir = run_from_config(config, args.config_type, test=args.test)

        print(f"saved rollout to: {rollout_dir}")

    # move configs after running
    src_dir = CONFIGS / "to_run" / args.config_type
    dst_dir = CONFIGS / "archive" /args.config_type
    
    for file in src_dir.glob("*.json"):
        dst = dst_dir / file.name
        if dst.exists():
            raise FileExistsError(f"Archive file already exists: {dst}")
        file.rename(dst)
        print(f"moved {file.name} -> {dst_dir}")

if __name__ == "__main__":
    main()