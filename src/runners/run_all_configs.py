import argparse
from copy import deepcopy
from itertools import product
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
    read_json,
    list_tens_to_floats,
)

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"run_all_configs_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )


ALPHAS = [2.0]
WS = [5.0, 10.0, 15.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--rollout-type",
        choices=["guided", "unguided"],
        required=True,
    )
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


def main() -> None:
    setup_logging()

    args = parse_args()

    logging.info("Starting experiment run")
    logging.info(f"Log file: {LOG_FILE}")
    logging.info(f"rollout_type={args.rollout_type}, test={args.test}")

    device = get_device()
    flow_model = get_model(device)

    configs = load_configs(args.rollout_type)
    logging.info(f"Found {len(configs)} configs")

    for idx, (config_id, config) in enumerate(configs, start=1):
        logging.info(f"Running config {idx}/{len(configs)}: {config_id}")

        rollout_dir = ensure_rollout_dir(config["rollout_id"])

        if args.rollout_type == "guided":
            for guidance_mode, alpha, w in product(GUIDANCE_MODES, ALPHAS, WS):
                config_ = apply_overrides(
                    config,
                    guidance_mode=guidance_mode,
                    alpha=alpha,
                    w=w,
                )

                logging.info(
                    f"Running guided rollout | "
                    f"config_id={config_id}, "
                    f"mode={guidance_mode}, alpha={alpha}, w={w}"
                )

                try:
                    update_experiment_params(rollout_dir, config_)

                    rollout_dir = rollout_from_config(
                        flow_model,
                        rollout_dir,
                        config_,
                        test=args.test,
                    )

                    logging.info(
                        f"Saved rollout to: {rollout_dir} | "
                        f"mode={guidance_mode}, alpha={alpha}, w={w}"
                    )

                except Exception:
                    logging.exception(
                        f"FAILED guided rollout | "
                        f"config_id={config_id}, "
                        f"mode={guidance_mode}, alpha={alpha}, w={w}"
                    )
                    raise

        else:
            logging.info(f"Running unguided rollout | config_id={config_id}")

            try:
                rollout_dir = rollout_from_config(
                    flow_model,
                    rollout_dir,
                    config,
                    test=args.test,
                )

                logging.info(f"Saved rollout to: {rollout_dir}")

            except Exception:
                logging.exception(
                    f"FAILED unguided rollout | config_id={config_id}"
                )
                raise

    logging.info("Done. All experiments finished.")

    
if __name__ == "__main__":
    main()