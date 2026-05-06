from pathlib import Path

import xarray as xr
import torch
from tensordict.tensordict import TensorDict

from src.utils import (
    save_to_json,
    rollout_to_xarray,
    batchify_and_move,
    get_dataset,
    get_x_cond,
    tensor_timestamp_to_string
)
from src.funcs import get_mask_tensordict
from src.interaction import get_mask_from_corners

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.lightning_modules.guided_diffusion import GuidedFlow


def rollout(
        guidance_flag: bool,  # either guiding or not the sampling
        rollout_dir: Path, 
        flow_model: GuidedFlow, 
        timestamp: str,
        mask_corners: list[float], 
        init_mask_term: float = None,  # same as ground truth, for visualization purposes
        y: list[torch.Tensor] = None, 
        lambda_: list[torch.Tensor] = None,
        N: int = 1,
        partition: str = None, 
        level_idx: int = None, 
        var_idx: int = None, 
        M: int = 1,
        test: bool = False,
    ):

    ds = get_dataset(multistep=N)
    x_cond, timestamp_idx = get_x_cond(ds, timestamp)

    device = flow_model.device
    x_cond = batchify_and_move(x_cond, device)
    lead_time_hours = int(x_cond["lead_time_hours"].cpu().flatten()[0].item())

    if guidance_flag:
        y = [torch.tensor(y[n]).to(device) for n in range(0, N + 1)]  # 0 because I need all ys
        lambda_ = [torch.tensor(lambda_[t]).to(device) for t in range(25)]
        mask = get_mask_from_corners(*mask_corners)
        mask = mask.to(device)
        mask = get_mask_tensordict(x_cond["state"][0], partition, var_idx, level_idx, mask)
    if not guidance_flag or test:
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
        mask = None

    M_mask_terms = {}
    member_datasets = []
    for m in range(1, M+1):
        print(f"member={m}")
        if not test:
            sample_multistep, mask_terms = flow_model.sample_rollout(
                N=N,
                member=m, 
                x_cond=x_cond,
                y=y,
                mask=mask,
                lambda_=lambda_
            )
            M_mask_terms[f"{m}"] = [init_mask_term] + mask_terms
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

    if guidance_flag:
        xr_pred.to_netcdf(rollout_dir / "guided.nc")
        save_to_json(M_mask_terms, rollout_dir, "mask_terms")
    else:
        xr_pred.to_netcdf(rollout_dir / "unguided.nc")

    ground_truth.to_netcdf(rollout_dir / "ground_truth.nc")