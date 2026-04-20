from pathlib import Path

from tensordict.tensordict import TensorDict

from src.utils import (
    save_state, 
    save_to_json
)
from src.funcs import get_mask_tensordict, get_guidance
from src.interaction import get_mask_from_corners

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.lightning_modules.guided_diffusion import GuidedFlow
from geoarches.utils.tensordict_utils import tensordict_apply, tensordict_cat

# TODO: create run func and call from marimo after setup!

##### load #####

def rollout(
        guidance_flag: bool,  # either guiding or not the sampling
        rollout_dir: Path, 
        ds: Era5Forecast, x_start: dict[TensorDict], 
        gen_model: GuidedFlow, 
        mask_corners, init_mask_term,
        y, lambda_, N,
        partition, level_idx, var_idx, m: int = 1,
        seeded_run: bool | None = None
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
        mask_term = float(init_mask_term)
        final_mask_terms = [mask_term]
        all_mask_terms = []
    else:
        mask = None

    x_cond = x_start
    lead_time_seconds = 6 * 3600

    ### iter

    seed=None
    for n in range(1, N+1):
        if seeded_run:
            seed = 1000 * n
        if guidance_flag:
            # NOTE: y[0] == 0 and will be ignored since we start at n=1, nice
            y_n = get_guidance(y[n], mask_term)
        else:
            y_n = None

        TEST = False
        if not TEST:
            state, mask_terms_n = gen_model.rollout_step(
                x_cond=x_cond,
                mask=mask,
                y_n=y_n,
                lambda_=lambda_,
                seed=seed
            )
        else:
            mask_terms_n = [0.0]*25
            state = x_start["state"]

        if guidance_flag:
            mask_term = float(mask_terms_n[-1])
            final_mask_terms.append(float(mask_term))
            all_mask_terms.append([float(mt) for mt in mask_terms_n])

        ### save states
        state_denorm = ds.denormalize(state).cpu()
        current_timestamp = x_cond["timestamp"].cpu() + lead_time_seconds
        state_xr = ds.convert_to_xarray(state_denorm, current_timestamp)

        save_state(rollout_dir, state_xr, n=n, m=m)

        # build next conditioning batch 
        if n < N:
            next_timestamp = x_cond["timestamp"] + lead_time_seconds
            x_cond = {
                "prev_state": x_cond["state"],
                "state": state,
                "timestamp": next_timestamp,
                "lead_time_hours": x_start["lead_time_hours"],
            }
    
    if guidance_flag:
        dict_ = {
            "final_mask_terms": final_mask_terms,
            "all_mask_terms": all_mask_terms
        }
        save_to_json(dict_, rollout_dir, "mask_terms")