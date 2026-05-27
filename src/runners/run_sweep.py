import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from src.utils.read_write import get_model, get_run_config, get_rollout_ids
from src.utils.setup import ensure_rollout_dir, get_device, setup_logging
from src.rollout import rollout
from src.rollout_config import RolloutConfig

"""
Change hash to:
experiment_params = {
    "guidance_reference": config.guidance_reference,
    "mask_mode": config.mask_mode,
    "mask_corners": config.mask_corners,
    "partition": config.partition,
    "var": config.var,
    "level": config.level,
    "alpha": config.alpha,
    "w": config.w,
}

guided_id = make_hash(experiment_params)

But then also change on id!
"""

logger = logging.getLogger(__name__)

SWEEP_PARAMS = ["guidance_reference", "mask_mode", "alpha", "w"]

GUIDANCE_REFERENCES = [
    "unguided_members",
    "ground_truth",
    # "lower_boundary",
    # "upper_boundary",
]
MASK_MODES = [
    "bbox", 
    "normal"
]
ALPHAS = [
    0.0, 2.0
]
WS = [
    10.0, 100
]

# TODO: check if specific rollout already exists and skip if true


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--rollout_id",
        required=True,
    )
    return parser.parse_args()


# def load_configs(rollout_type: str) -> list[RolloutConfig]:
#     """Load and parse every config JSON in the directory for `config_type`."""
#     run_ids = [id_[:-5] for id_ in get_rollout_ids(rollout_type, "run")]

#     configs = []
#     for run_id in run_ids:
#         config = get_run_config(rollout_type=rollout_type, rollout_id=run_id)
#         configs.append(config)
#     return configs


@dataclass(frozen=True)
class RolloutJob:
    config: RolloutConfig
    label: str


def build_jobs(config: RolloutConfig) -> list[RolloutJob]:
    jobs = []
    for guidance_ref, mask_mode, alpha, w in product(GUIDANCE_REFERENCES, MASK_MODES, ALPHAS, WS):
        swept = deepcopy(config)
        swept.guidance_reference = guidance_ref
        swept.mask_mode = mask_mode
        swept.alpha = alpha
        swept.w = w

        label = (
            f"config_id={config.rollout_id}, "
            f"guidance_reference={guidance_ref}, "
            f"mask_mode={mask_mode}, "
            f"alpha={alpha}, "
            f"w={w}"
        )
        jobs.append(RolloutJob(swept, label))
    print(f"created {len(jobs)} sweep jobs:")
    print(f"{jobs}")
    return jobs


def run_job(job: RolloutJob, flow_model, *, test: bool) -> Path:
    """Run one rollout. Raises on failure; caller decides what to do."""
    rollout_dir = rollout(job.config, flow_model, test=test)
    logger.info(f"Saved rollout to: {rollout_dir} | {job.label}")
    return rollout_dir


def main() -> None:
    args = parse_args()
    setup_logging()

    logger.info("Loading model")
    flow_model = None if args.test else get_model(get_device())
    
    config = get_run_config(rollout_type="guided_rollout", rollout_id=args.rollout_id)

    for job in build_jobs(config):
        print(f"Running {job.label}")
        run_job(job, flow_model, test=args.test)

    # for idx, config in enumerate(configs, start=1):
    #     logger.info(f"Running config {idx}/{len(configs)}: {config.rollout_id}")
    #     ensure_rollout_dir(config.rollout_id)

    #     for job in build_jobs(config, guided=guided):
    #         logger.info(f"Running {args.rollout_type} rollout | {job.label}")
    #         try:
    #             run_job(job, flow_model, test=args.test)
    #         except Exception:
    #             logger.exception(f"FAILED {args.rollout_type} rollout | {job.label}")
    #             failures.append(job.label)
    #             # To restore fail-fast behavior, replace the line above with: raise

    # if failures:
    #     logger.warning(f"Done with {len(failures)} failure(s):")
    #     for label in failures:
    #         logger.warning(f"  - {label}")
    # else:
    #     logger.info("Done. All experiments finished.")


if __name__ == "__main__":
    main()