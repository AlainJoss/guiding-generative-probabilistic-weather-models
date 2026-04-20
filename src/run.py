from src.utils import (
    save_state,
    state_to_device,
    save_to_json,
    ensure_rollout_dir,
)

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast

##### load #####

# data
ds = Era5Forecast(
    path="data/era5_240/full",  # default path
    load_prev=True,  # whether to load previous state
    norm_scheme="pangu",  # default normalization scheme
    domain="test",  # domain to consider. domain = 'test' loads the 2020 period
)

# model
model = 0
module_target = "geoarches.lightning_modules.{}"
MODELS = ["diffusion.DiffusionModule", "guided_diffusion.GuidedFlow"]
device = "mps"
gen_model, gen_config = load_module(
    "archesweathergen",
    module_target=module_target.format(MODELS[model]),
)
gen_model = gen_model.to(device)

##### set up #####

# select X_start
X_start = ds[0]
X_start = state_to_device(X_start, device)

##### run #####

N = 2
M = 1  # ensemble size
lead_s = int(X_start["lead_time_hours"].item()) * 3600
start_timestamp = X_start["timestamp"].cpu()
rollout_dir = ensure_rollout_dir("old_model", N=N)

for m in range(1, M + 1):
    samples = gen_model.sample_rollout(
        batch=X_start,
        iterations=N,
        member=0,
    ).cpu()  # [B=1, iterations, ...]

    for n in range(1, N + 1):
        sample = samples[:, n - 1]  # keep batch dim → [B=1, ...]
        st = ds.denormalize(sample)
        ts = start_timestamp + n * lead_s
        st_xr = ds.convert_to_xarray(st, ts)
        save_state(rollout_dir, st_xr, n=n, m=m)

config = {
    "rollout_dir": str(rollout_dir),
    "M": M,
    "N": N,
    "timestamp": str(start_timestamp),
}

save_to_json(config, rollout_dir, "config")
