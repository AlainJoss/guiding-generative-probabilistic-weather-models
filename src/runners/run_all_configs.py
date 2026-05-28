import argparse
from dataclasses import dataclass
from pathlib import Path

from src.rollout import rollout
from src.rollout_config import RolloutConfig
from src.utils.read_write import get_model, get_run_config, get_rollout_ids
from src.utils.setup import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout_type",
        choices=["guided_rollout", "unguided_rollout"],
        required=True,
    )
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class RolloutJob:
    config: RolloutConfig
    label: str


def load_jobs(rollout_type: str) -> list[RolloutJob]:
    rollout_ids = [
        file_name.removesuffix(".json")
        for file_name in get_rollout_ids(rollout_type, "run")
    ]

    return [
        RolloutJob(
            config=get_run_config(rollout_type=rollout_type, rollout_id=rollout_id),
            label=f"rollout_type={rollout_type}, rollout_id={rollout_id}",
        )
        for rollout_id in rollout_ids
    ]


def run_job(job: RolloutJob, flow_model, *, test: bool) -> Path:
    print(f"Running {job.label}")
    rollout_dir = rollout(job.config, flow_model, test=test)
    print(f"Saved rollout to: {rollout_dir}")
    return rollout_dir


def main() -> None:
    args = parse_args()

    flow_model = None if args.test else get_model(get_device())

    jobs = load_jobs(args.rollout_type)
    print(f"Found {len(jobs)} config(s).")

    for job in jobs:
        run_job(job, flow_model, test=args.test)


if __name__ == "__main__":
    main()