import argparse
import logging
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from itertools import product

import xarray as xr

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


def get_container_args(rollout_type):
    match rollout_type:
        case "ung":
            return [("ung", False)]
        case "gui":
            return [
                ("grads", True),
                ("vfs", True),
                ("gui_vfs", True),
                ("clean_preds", True),
                ("gui", False),
                ("ung_gui", False)
            ]
        case _:
            return []


def find_resume_index(rollout_dir, rollout_type, sweep_params):
    """Return the index into the product-ordered sweep list at which to resume.

    Walks the sweeps in the same order as build_jobs. A sweep is "filled" only
    when every container has no NaN in that sweep's (m, n, ...) region. Returns
    the index of the first not-fully-filled sweep (half-filled or empty), so the
    loop skips completed sweeps and overwrites the first incomplete one.
    """
    container_args = get_container_args(rollout_type)

    datasets = []
    for (container_type, _) in container_args:
        path = rollout_dir / f"{container_type}.zarr"
        if not path.exists():
            return 0  # missing container -> nothing reliably filled, run all
        datasets.append(xr.open_zarr(path))

    sweeps = list(iter_sweeps(sweep_params))
    for i, sweep in enumerate(sweeps):
        for ds in datasets:
            probe = ds[list(ds.data_vars)[0]].sel(sweep)
            if not bool(probe.notnull().all().compute()):
                return i  # first not-fully-filled sweep
    return len(sweeps)  # all sweeps already filled


def create_zarr_containers(rollout_type, rollout_id, M, N, sweep_params):
    container_args = get_container_args(rollout_type)

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

    for key, values in sweep_params.items():
        assert len(values) == len(set(values)), (
            f"sweep_params['{key}'] has duplicate values {values}; "
            f"each sweep axis must be uniquely valued or the zarr region write fails. "
            f"Fix {get_rollout_dir(args.rollout_id) / 'sweep_params.json'}."
        )

    # a store from a prior run is resumed, not recreated: create_zarr_containers
    # uses mode="w" and would wipe partially-filled sweeps.
    container_args = get_container_args(args.rollout_type)
    store_exists = all(
        (rollout_dir / f"{c}.zarr").exists() for (c, _) in container_args
    )

    if store_exists:
        resume_idx = find_resume_index(rollout_dir, args.rollout_type, sweep_params)
        logger.info("Existing store found, resuming sweep at index %d", resume_idx)
    else:
        create_zarr_containers(args.rollout_type, args.rollout_id, config.M, config.N, sweep_params)
        resume_idx = 0

    for i, job in enumerate(build_jobs(config, sweep_params)):
        if i < resume_idx:
            logger.info("Skipping filled %s", job.label)
            continue

        logger.info("Running %s", job.label)

        rollout(rollout_dir, guidance_flag, job.config, flow_model, job.sweep)
        
if __name__ == "__main__":
    main()