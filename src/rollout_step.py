from tensordict.tensordict import TensorDict

from src.utils import (
    save_state, 
    ensure_result_dir,
    save_config
)
from src.funcs import get_mask_tensordict
from src.interaction import get_mask_from_corners

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.lightning_modules.guided_diffusion import GuidedFlow
from geoarches.utils.tensordict_utils import tensordict_apply, tensordict_cat

# TODO: create run func and call from marimo after setup!

##### load #####

def rollout_step(
        result_dir, 
        ds: Era5Forecast, x_start: dict[TensorDict], 
        gen_model: GuidedFlow, 
        mask_corners, y, lambda_, N,
        partition, level_idx, var_idx
    ):
    ### init

    device = gen_model.device
    mask = get_mask_from_corners(*mask_corners)
    mask = mask.to(device)
    y = y.to(device)

    mask = get_mask_tensordict(x_start["state"][0], partition, var_idx, level_idx, mask)

    x_cond_guided = x_start
    x_cond_unguided = x_start

    lead_time_seconds = x_start["lead_time_hours"] * 3600

    ### iter

    for n in range(1, N+1):
        guided_state = gen_model.rollout_step(
            x_cond=x_cond_guided, 
            mask=mask,
            y_n=y[n],
            lambda_=lambda_
        )
        unguided_state = gen_model.rollout_step(
            x_cond=x_cond_unguided, 
            mask=None,
            y_n=None,
            lambda_=None
        )

        ### save states
            
        # guided_state = x_start["state"]
        # unguided_state = x_start["state"]

        # --- save denormalized xarray outputs ---
        guided_state_denorm = ds.denormalize(guided_state).cpu()
        unguided_state_denorm = ds.denormalize(unguided_state).cpu()

        current_timestamp = x_cond_guided["timestamp"].cpu() + lead_time_seconds.cpu()

        guided_state_xr = ds.convert_to_xarray(guided_state_denorm, current_timestamp)
        unguided_state_xr = ds.convert_to_xarray(unguided_state_denorm, current_timestamp)

        save_state(result_dir, guided_state_xr, "guided", step=n)
        save_state(result_dir, unguided_state_xr, "unguided", step=n)

        # --- build next conditioning batch ---
        if n < N:

            next_timestamp_guided = x_cond_guided["timestamp"] + lead_time_seconds
            next_timestamp_unguided = x_cond_unguided["timestamp"] + lead_time_seconds

            x_cond_guided = {
                "prev_state": x_cond_guided["state"],
                "state": guided_state,
                "timestamp": next_timestamp_guided,
                "lead_time_hours": x_start["lead_time_hours"],
            }

            x_cond_unguided = {
                "prev_state": x_cond_unguided["state"],
                "state": unguided_state,
                "timestamp": next_timestamp_unguided,
                "lead_time_hours": x_start["lead_time_hours"],
            }