from tensordict.tensordict import TensorDict

from src.utils import (
    save_state, 
    save_to_json,
    save_config,
)
from src.funcs import get_mask_tensordict
from src.interaction import get_mask_from_corners

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.lightning_modules.guided_diffusion import GuidedFlow
from geoarches.utils.tensordict_utils import tensordict_apply, tensordict_cat

# TODO: create run func and call from marimo after setup!

##### load #####

def rollout(
        experiment_type,
        result_dir, 
        ds: Era5Forecast, x_start: dict[TensorDict], 
        gen_model: GuidedFlow, 
        mask_corners, y, lambda_, N,
        partition, level_idx, var_idx
    ):
    """
    Switch "guidance" ON/OFF using the mask: None=OFF, torch.Tensor=ON.
    """
    ### init

    device = gen_model.device

    if experiment_type=="guided":
        y = y.to(device)
        mask = get_mask_from_corners(*mask_corners)
        mask = mask.to(device)
        mask = get_mask_tensordict(x_start["state"][0], partition, var_idx, level_idx, mask)
    elif experiment_type=="unguided" or isinstance(experiment_type, int):
        mask=None
    else:
        raise ValueError(f"no such experiment type: {experiment_type}")

    x_cond = x_start

    lead_time_seconds = x_start["lead_time_hours"] * 3600

    ### iter

    mask_terms = []
    for n in range(1, N+1):
        state, mask_term = gen_model.rollout_step(
            x_cond=x_cond, 
            mask=mask,
            y_n=y[n] if experiment_type=="guided" else None,
            lambda_=lambda_
        )

        ### save states

        # for testing 
        # mask_term = 0.0
        # state = x_start["state"]

        # --- save denormalized xarray outputs ---

        mask_term = float(mask_term)
        mask_terms.append(mask_term)

        state_denorm = ds.denormalize(state).cpu()
        current_timestamp = x_cond["timestamp"].cpu() + lead_time_seconds.cpu()
        state_xr = ds.convert_to_xarray(state_denorm, current_timestamp)

        if isinstance(experiment_type, int):
            # step=m (model id)
            save_state(result_dir, state_xr, f"{n}", step=experiment_type)
        else:
            save_state(result_dir, state_xr, experiment_type, n)


        # build next conditioning batch 
        if n < N:
            next_timestamp = x_cond["timestamp"] + lead_time_seconds
            x_cond = {
                "prev_state": x_cond["state"],
                "state": state,
                "timestamp": next_timestamp,
                "lead_time_hours": x_start["lead_time_hours"],
            }
    
    dict_ = {"mask_terms": mask_terms}
    save_to_json(dict_, result_dir, "mask_terms")