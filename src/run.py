import argparse
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from itertools import product

import xarray as xr

from src.rollout import rollout
from src.rollout_config import RolloutConfig, GUIDANCE_METHOD_HYPERS
from src.utils import get_model, get_config, get_sweep_dict, get_rollout_dir, ensure_rollout_dir
from src.utils import get_device
from src.utils import create_full_zarr_container, sweep_coord_label


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


# TODO: correct and check this
def iter_sweeps(sweep_params: dict[str, list]) -> Iterator[dict]:
    # Mode-restricted product: for each GUIDANCE_MODE value, only that mode's specific
    # hypers vary; axes that belong to *some* mode but not this one are pinned to their
    # index-0 value (the GUIDANCE_MODE coord disambiguates the index-0 writes in the
    # union zarr). Mask hypers (sigma_div) apply to every mask mode, so they vary
    # unrestricted. Yields full job dicts (every union key present).
    all_keys = list(sweep_params)

    # unguided (no GUIDANCE_MODE axis, e.g. ung sweep_params={}): plain product
    # (the empty dict yields a single empty job).
    if "GUIDANCE_MODE" not in sweep_params:
        for values in product(*(sweep_params[k] for k in all_keys)):
            yield dict(zip(all_keys, values))
        return

    specific_axes = set().union(*[set(v) for v in GUIDANCE_METHOD_HYPERS.values()])

    for mode in sweep_params["GUIDANCE_MODE"]:
        active = GUIDANCE_METHOD_HYPERS[mode]
        axis_values = []
        for k in all_keys:
            if k == "GUIDANCE_MODE":
                axis_values.append([mode])
            elif k in specific_axes and k not in active:
                axis_values.append([sweep_params[k][0]])  # pin to index 0
            else:
                axis_values.append(sweep_params[k])
        for values in product(*axis_values):
            yield dict(zip(all_keys, values))


def build_jobs(
    base_config: RolloutConfig,
    sweep_params: dict[str, list],
) -> Iterator[RolloutJob]:
    for sweep in iter_sweeps(sweep_params):
        config = deepcopy(base_config)

        # set value in rollout config
        for key, value in sweep.items():
            setattr(config, key, value)

        # TODO: only keep the parameters that are being swepts as sweep
        label = ", ".join(
            [f"{k}={v}" for k, v in sweep.items()]
        )

        coord_sweep = {
            k: sweep_coord_label(k, v, sweep_params) for k, v in sweep.items()
        }

        yield RolloutJob(config=config, sweep=coord_sweep, label=label)


def get_container_args(rollout_type):
    # tuples: (file type, t_dim_flag)
    match rollout_type:
        case "ung":
            # ung holds the FULL flow-step trajectory (final state = t=-1 slice)
            return [("ung", True)]
        case "gui":
            # raw primitives only: grads = dL/dz, vfs = u*s_t, res = noisy state z_t.
            # gui_vec / gui_vf / gui_res / clean_preds are reconstructed in the UI.
            return [
                ("grads", True),
                ("vfs", True),
                ("res", True),
                ("gui", False),
                ("gui_det", False),
                ("gui_ung", True)
            ]
        case _:
            return []

# TODO: check, this
def written_containers(rollout_type, guidance_mode):
    """Container types a single sweep point actually fills."""
    if rollout_type == "ung":
        return {"ung"}
    return {"gui", "gui_det", "gui_ung", "grads", "vfs", "res"}


def find_resume_index(rollout_dir, rollout_type, sweep_params):
    """Return the index into the product-ordered sweep list at which to resume.

    Walks the sweeps in the same order as build_jobs. A sweep is "filled" only
    when every container has no NaN in that sweep's (m, n, ...) region. Returns
    the index of the first not-fully-filled sweep (half-filled or empty), so the
    loop skips completed sweeps and overwrites the first incomplete one.
    """
    container_args = get_container_args(rollout_type)

    datasets = {}
    for (container_type, _) in container_args:
        path = rollout_dir / f"{container_type}.zarr"
        if not path.exists():
            return 0  # missing container -> nothing reliably filled, run all
        datasets[container_type] = xr.open_zarr(path)

    sweeps = list(iter_sweeps(sweep_params))
    for i, sweep in enumerate(sweeps):
        coord_sweep = {
            k: sweep_coord_label(k, v, sweep_params) for k, v in sweep.items()
        }
        # only require the containers this sweep point's guidance_mode actually writes
        expected = written_containers(rollout_type, sweep.get("GUIDANCE_MODE"))
        for container_type in expected:
            ds = datasets[container_type]
            probe = ds[list(ds.data_vars)[0]].sel(coord_sweep)
            if not bool(probe.notnull().all().compute()):
                return i  # first not-fully-filled sweep
    return len(sweeps)  # all sweeps already filled


def create_zarr_containers(rollout_type, rollout_id, M, N, T, sweep_params):
    container_args = get_container_args(rollout_type)

    ensure_rollout_dir(rollout_id)

    for (container_type, t_dim_flag) in container_args:
        save_path = get_rollout_dir(rollout_id) / f"{container_type}.zarr"
        # existing containers are kept (a store from a prior run is resumed, not
        # wiped); only missing ones are created -- e.g. gui_det on older stores
        if save_path.exists():
            continue
        container_ds = create_full_zarr_container(M, N, t_dim_flag, T, sweep_params)
        # compute=False: write only metadata + coords (dask-backed NaN chunks are
        # never materialized); append_to_zarr fills real (m, n, sweep) chunks later.
        container_ds.to_zarr(save_path, mode="w", compute=False)

def main() -> None:
    args = parse_args()

    print("Loading model")
    flow_model = get_model(get_device())

    config = get_config(args.rollout_id)
    rollout_dir = get_rollout_dir(args.rollout_id)

    # number of flow/sampling steps comes from the config (set in unguided mode);
    # fall back to 25 for older configs that predate the T field.
    T = config.T if config.T is not None else 25

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
        unique = [v for i, v in enumerate(values) if v not in values[:i]]
        assert len(values) == len(unique), (
            f"sweep_params['{key}'] has duplicate values {values}; "
            f"each sweep axis must be uniquely valued or the zarr region write fails. "
            f"Fix {get_rollout_dir(args.rollout_id) / 'sweep_params.json'}."
        )

    # create any missing containers; existing ones are kept, so a store from a
    # prior run is resumed rather than wiped (e.g. gui_det added on older stores
    # starts empty -> resume probes it and reruns what is needed)
    container_args = get_container_args(args.rollout_type)
    any_existing = any(
        (rollout_dir / f"{c}.zarr").exists() for (c, _) in container_args
    )
    create_zarr_containers(args.rollout_type, args.rollout_id, config.M, config.N, T, sweep_params)
    if any_existing:
        resume_idx = find_resume_index(rollout_dir, args.rollout_type, sweep_params)
        print(f"Existing store found, resuming sweep at index {resume_idx}")
    else:
        resume_idx = 0

    for i, job in enumerate(build_jobs(config, sweep_params)):
        if i < resume_idx:
            print("Skipping filled %s", job.label)
            continue

        print("Running %s", job.label)

        rollout(rollout_dir, guidance_flag, job.config, flow_model, T, job.sweep)
        
if __name__ == "__main__":
    main()