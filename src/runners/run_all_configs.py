import argparse
from copy import deepcopy
from itertools import product
from typing import Any
import logging

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
    setup_logging
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
    args = parse_args()
    device = get_device()    
    print("Loading model")
    flow_model = get_model(device)
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting experiment run")
    logger.info(f"rollout_type={args.rollout_type}, test={args.test}")

    configs = load_configs(args.rollout_type)
    logger.info(f"Found {len(configs)} configs")

    for idx, (config_id, config) in enumerate(configs, start=1):
        logger.info(f"Running config {idx}/{len(configs)}: {config_id}")

        rollout_dir = ensure_rollout_dir(config["rollout_id"])

        if args.rollout_type == "guided":
            for guidance_mode, alpha, w in product(GUIDANCE_MODES, ALPHAS, WS):
                config_ = apply_overrides(
                    config,
                    guidance_mode=guidance_mode,
                    alpha=alpha,
                    w=w,
                )

                logger.info(
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

                    logger.info(
                        f"Saved rollout to: {rollout_dir} | "
                        f"mode={guidance_mode}, alpha={alpha}, w={w}"
                    )

                except Exception:
                    logger.exception(
                        f"FAILED guided rollout | "
                        f"config_id={config_id}, "
                        f"mode={guidance_mode}, alpha={alpha}, w={w}"
                    )
                    raise

        else:
            logger.info(f"Running unguided rollout | config_id={config_id}")

            try:
                rollout_dir = rollout_from_config(
                    flow_model,
                    rollout_dir,
                    config,
                    test=args.test,
                )

                logger.info(f"Saved rollout to: {rollout_dir}")

            except Exception:
                logger.exception(
                    f"FAILED unguided rollout | config_id={config_id}"
                )
                raise

    logger.info("Done. All experiments finished.")


if __name__ == "__main__":
    main()