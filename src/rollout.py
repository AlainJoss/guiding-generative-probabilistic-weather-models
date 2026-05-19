import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)

import xarray as xr
import torch

from src.utils.read_write import (
    save_to_json,
    get_td_ds,
    update_experiment_params,
)
from src.utils.dataset_utils import get_x_cond_from_ts
from src.utils.converters import (
    rollout_to_xarray,
    batchify_and_move,
    list_tensors_to_floats,
    xr_rollout_slice_to_tdict,
)
from src.utils.setup import ensure_rollout_dir, get_device
from src.funcs import make_hash
from src.config import GUIDANCE_PARAMS
from src.target import get_reference_trajectory, get_target_trajectory
from src.mask import get_mask_tdict

from geoarches.lightning_modules.guided_diffusion import GuidedFlow
from tensordict.tensordict import TensorDict


def _config_to_serializable(config) -> dict:
    d = asdict(config)
    lam = d.get("lambda_")
    if isinstance(lam, list) and lam and hasattr(lam[0], "item"):
        d["lambda_"] = list_tensors_to_floats(lam)
    return d


def rollout(
    config,
    flow_model: GuidedFlow | None = None,
    test: bool = False,
):
    rollout_dir = ensure_rollout_dir(config.rollout_id)
    if config.guidance_flag:
        update_experiment_params(rollout_dir, config, GUIDANCE_PARAMS)

    ds = get_td_ds(multistep=config.N)
    x_cond, _ = get_x_cond_from_ts(ds, config.timestamp)

    device = flow_model.device if flow_model is not None else get_device()
    x_cond = batchify_and_move(x_cond, device)
    lead_time_hours = int(x_cond["lead_time_hours"].cpu().flatten()[0].item())

    build_ground_truth = (not config.guidance_flag) or test
    if build_ground_truth:
        ground_truth = torch.cat(
            [x_cond["state"].unsqueeze(1), x_cond["future_states"]],
            dim=1,
        )
        ground_truth = rollout_to_xarray(
            ds=ds,
            sample_multistep=ground_truth,
            init_timestamp=x_cond["timestamp"],
            member=-1,
            lead_time_hours=lead_time_hours,
            include_init=True,
        )

    member_datasets = []
    for m in range(config.M):
        logger.info(f"sampling member={m}")

        if config.guidance_flag and not test and config.reference != "sampled_trajectory":
            reference_trajectory = get_reference_trajectory(config, m)
            target_trajectory = get_target_trajectory(config, reference_trajectory)
            target_tdicts = [
                xr_rollout_slice_to_tdict(target_trajectory.isel(time=n))[None].to(device)
                for n in range(config.N)
            ]
            masks = [
                get_mask_tdict(config, target_trajectory.isel(time=n), x_cond["state"])
                for n in range(config.N)
            ]
        else:
            target_tdicts = None
            masks = None

        if not test:
            sample_multistep = flow_model.sample_rollout(
                config,
                m=m,
                x_cond=x_cond,
                y=target_tdicts,
                masks=masks,
            )
        else:
            sample_multistep = x_cond["future_states"]

        xr_member = rollout_to_xarray(
            ds=ds,
            sample_multistep=sample_multistep,
            init_timestamp=x_cond["timestamp"],
            member=m,
            lead_time_hours=lead_time_hours,
        )
        member_datasets.append(xr_member)

    xr_pred = xr.concat(member_datasets, dim="member")

    config_dict = _config_to_serializable(config)

    if config.guidance_flag:
        params = {
            "guidance_mode": config.guidance_mode,
            "alpha": config.alpha,
            "w": config.w,
        }

        guided_id = make_hash(params)
        guided_path = rollout_dir / "guided" / guided_id
        guided_path.mkdir(parents=True, exist_ok=True)

        xr_pred.to_netcdf(guided_path / "guided.nc")
        save_to_json(config_dict, guided_path, "config")
    else:
        save_to_json(config_dict, rollout_dir, "config")
        xr_pred.to_netcdf(rollout_dir / "unguided.nc")
        ground_truth.to_netcdf(rollout_dir / "ground_truth.nc")
