import argparse
import logging
from copy import deepcopy 
from itertools import product
from typing import Any

from src.paths import RUN_CONFIGS
from funcs import T_schedule
from src.utils.read_write import (
    get_model,
    read_json,
)
from src.utils.setup import (
    ensure_rollout_dir,
    get_device,
    setup_logging,
)
from src.utils.converters import (
    list_tensors_to_floats,
)
from src.rollout import rollout 
from src.config import UnguidedConfig, GUIDANCE_REFERENCES


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
    config_dir = RUN_CONFIGS / config_type
    config_paths = sorted(config_dir.glob("*.json"))

    configs = []
    for config_path in config_paths:
        config_id = config_path.stem
        config = read_json(config_dir, config_id)
        configs.append((config_id, config))

    return configs


def apply_guidance_overrides(
    config: UnguidedConfig,
    guidance_mode: str,
    alpha: float,
    w: float,
) -> UnguidedConfig:
    config_ = deepcopy(config)
    config_.guidance_mode = guidance_mode
    config_.alpha = alpha
    config_.w = w
    config_.lambda_ = list_tensors_to_floats(T_schedule(alpha, w))
    return config_


def main() -> None:
    args = parse_args()

    print("Loading model")
    if args.test:
        flow_model = None 
    else:
        device = get_device()    
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
            for guidance_mode, alpha, w in product(GUIDANCE_REFERENCES, ALPHAS, WS):
                config_ = apply_guidance_overrides(
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
                    rollout_dir = rollout(
                        config_,
                        flow_model,
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
                rollout_dir = rollout(
                    config,
                    flow_model,
                    None,
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