import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from src.utils.read_write import get_model, get_run_config, get_rollout_ids
from src.utils.setup import ensure_rollout_dir, get_device, setup_logging
from src.rollout import rollout
from src.rollout_config import RolloutConfig, GUIDANCE_REFERENCES, MASK_MODES, ALPHAS, WS

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--rollout_id",
        required=True,
    )
    return parser.parse_args()


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
    return jobs


def run_job(job: RolloutJob, flow_model, *, test: bool) -> Path:
    rollout_dir = rollout(job.config, flow_model, test=test)
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


if __name__ == "__main__":
    main()