import argparse
import logging
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from itertools import product


from src.rollout import rollout
from src.rollout_config import RolloutConfig
from src.utils import get_model, get_config, get_sweep_dict, get_rollout_dir, ensure_rollout_dir
from src.utils import get_device, setup_logging
from src.utils import create_full_zarr_container


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_type", choices=["gui", "ung"], required=True)
    parser.add_argument("--rollout_id", required=True)
    return parser.parse_args()


@dataclass(frozen=True)
class RolloutJob:
    config: RolloutConfig
    sweep: dict[str, any]
    label: str


def iter_sweeps(sweep_params: dict[str, list]) -> Iterator[dict]:
    keys = list(sweep_params)

    for values in product(*(sweep_params[k] for k in keys)):
        yield dict(zip(keys, values))


def build_jobs(
    base_config: RolloutConfig,
    sweep_params: dict[str, list],
) -> Iterator[RolloutJob]:
    for sweep in iter_sweeps(sweep_params):
        config = deepcopy(base_config)

        # set value in rollout config
        for key, value in sweep.items():
            setattr(config, key, value)

        label = ", ".join(
            [f"{k}={v}" for k, v in sweep.items()]
        )

        yield RolloutJob(config=config, sweep=sweep, label=label)


def create_zarr_containers(rollout_type, rollout_id, M, N, sweep_params):
    match rollout_type:
        case "ung":
            container_args = [("ung", False)]
        case "gui":
            container_args = [
                ("grads", True), 
                ("vfs", True), 
                # ("gui_vfs", True),
                ("clean_preds", True),
                ("gui", False), 
                ("ung_gui", False)
            ]
        case _:
            pass

    ensure_rollout_dir(rollout_id)

    for (container_type, t_dim_flag) in container_args:
        container_ds = create_full_zarr_container(M, N, t_dim_flag, sweep_params)

        save_path = get_rollout_dir(rollout_id) / f"{container_type}.zarr"
        # compute=False: write only metadata + coords (dask-backed NaN chunks are
        # never materialized); append_to_zarr fills real (m, n, sweep) chunks later.
        container_ds.to_zarr(save_path, mode="w", compute=False)

        
def main() -> None:
    args = parse_args()
    setup_logging()

    logger.info("Loading model")
    flow_model = get_model(get_device())

    config = get_config(args.rollout_id)
    rollout_dir = get_rollout_dir(args.rollout_id)

    match args.rollout_type:
        case "ung":
            guidance_flag=False
            sweep_params = {}
        case "gui":
            guidance_flag=True
            sweep_params = get_sweep_dict(args.rollout_id)
        case _:
            pass

    create_zarr_containers(args.rollout_type, args.rollout_id, config.M, config.N, sweep_params)

    for job in build_jobs(config, sweep_params):
        logger.info("Running %s", job.label)

        rollout(rollout_dir, guidance_flag, job.config, flow_model, job.sweep)
        
if __name__ == "__main__":
    main()