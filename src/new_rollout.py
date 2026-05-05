from pathlib import Path

import xarray as xr
import torch
from tensordict.tensordict import TensorDict

from src.utils import (
    save_state, 
    save_to_json,
    rollout_to_xarray
)
from src.funcs import get_mask_tensordict
from src.interaction import get_mask_from_corners
from src.paths import ROLLOUTS

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.lightning_modules.guided_diffusion import GuidedFlow


def rollout(
        guidance_flag: bool,  # either guiding or not the sampling
        rollout_id: Path, 
        ds: Era5Forecast, 
        x_cond: dict[TensorDict], 
        flow_model: GuidedFlow, 
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

    device = flow_model.device
    y = [y[n].to(device) for n in range(N)]
    lambda_ = [lambda_[t].to(device) for t in range(25)]

    if guidance_flag:
        mask = get_mask_from_corners(*mask_corners)
        mask = mask.to(device)
        mask = get_mask_tensordict(x_cond["state"][0], partition, var_idx, level_idx, mask)

    M_mask_terms = {}
    member_datasets = []
    for m in range(1, M+1):
        print(f"member={m}")
        if guidance_flag:
            sample_multistep, mask_terms = flow_model.sample_rollout(
                x_cond,
                member=m,
                iterations=N,
                # ... # y, lambda ... init_mask_term (check in particular)
            )
            M_mask_terms[f"{m}"] = mask_terms
        else: 
            sample_multistep = flow_model.sample_rollout(
                        x_cond,
                        batch_nb=0, # TODO: fix
                        member=m,  
                        iterations=N,
                    )
        xr_member = rollout_to_xarray(
            ds=ds,
            sample_multistep=sample_multistep,
            init_timestamp=x_cond["timestamp"],
            member=m,
        )
        member_datasets.append(xr_member)

    xr_pred = xr.concat(member_datasets, dim="member")
    out_file = ROLLOUTS / f"{rollout_id}" / f"guided.nc"
    xr_pred.to_netcdf(out_file)

    if guidance_flag:
        ... # save the mask terms




