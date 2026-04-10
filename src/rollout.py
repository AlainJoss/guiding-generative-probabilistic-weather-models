from pathlib import Path

from tensordict.tensordict import TensorDict

from src.utils import (
    save_state, 
    save_to_json,
)
from src.funcs import get_mask_tensordict
from src.interaction import get_mask_from_corners

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.lightning_modules.guided_diffusion import GuidedFlow
from geoarches.utils.tensordict_utils import tensordict_apply, tensordict_cat

# TODO: create run func and call from marimo after setup!

##### load #####

def rollout(
        sampling_flag: bool,
        guidance_flag: bool,  # either guiding or not the sampling
        ensemble_flag: bool,  # if M forecasts are being produced it's relevant for the saving logic
        rollout_dir: Path, 
        ds: Era5Forecast, x_start: dict[TensorDict], 
        gen_model: GuidedFlow, 
        mask_corners, y, lambda_, N,
        partition, level_idx, var_idx, timestamp_idx,
        ensemble_step: int | None = None,
    ):
    """
    Switch "guidance" ON/OFF using the mask: None=OFF, torch.Tensor=ON.
    """
    ### init

    device = gen_model.device

    if guidance_flag:
        y = y.to(device)
        mask = get_mask_from_corners(*mask_corners)
        mask = mask.to(device)
        mask = get_mask_tensordict(x_start["state"][0], partition, var_idx, level_idx, mask)
    else:
        mask=None

    x_cond = x_start
    lead_time_seconds = x_start["lead_time_hours"] * 3600

    ### iter

    mask_terms = []  # realized guidance
    for n in range(1, N+1):
        if sampling_flag:
            state, mask_term = gen_model.rollout_step(
                x_cond=x_cond, 
                mask=mask,
                y_n=y[n] if guidance_flag else None,
                lambda_=lambda_
            )
        else:
            timestamp_idx+=1
            state = ds[timestamp_idx]["next_state"].unsqueeze(0)
            mask_term = 0

        ### save states

        # for testing 
        # mask_term = 0.0
        # state = x_start["state"]

        mask_term = float(mask_term)  # cast from torch
        mask_terms.append(mask_term)

        state_denorm = ds.denormalize(state).cpu()
        current_timestamp = x_cond["timestamp"].cpu() + lead_time_seconds.cpu()
        state_xr = ds.convert_to_xarray(state_denorm, current_timestamp)

        if ensemble_flag:
            # step=m (model id)
            save_state(rollout_dir, state_xr, n_step=n, m_step=ensemble_step)
        else:
            save_state(rollout_dir, state_xr, n_step=n, m_step=1)
        # build next conditioning batch 
        if n < N and sampling_flag:
            next_timestamp = x_cond["timestamp"] + lead_time_seconds
            x_cond = {
                "prev_state": x_cond["state"],
                "state": state,
                "timestamp": next_timestamp,
                "lead_time_hours": x_start["lead_time_hours"],
            }
    
    dict_ = {"mask_terms": mask_terms}
    save_to_json(dict_, rollout_dir, "mask_terms")