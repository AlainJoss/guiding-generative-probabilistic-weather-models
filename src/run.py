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
    lead_time_hours=6,
)

# model
device = "mps"
module_target = "geoarches.lightning_modules.{}"
MODELS = ["diffusion.DiffusionModule", "guided_diffusion.GuidedFlow"]

model = 0
old_model, old_config = load_module(
    "archesweathergen",
    module_target=module_target.format(MODELS[model]),
)
old_model = old_model.to(device)

# model = 1
# new_model, new_config = load_module(
#     "archesweathergen",
#     module_target=module_target.format(MODELS[model]),
# )
# new_model = new_model.to(device)

##### set up #####

# select X_start
x_start = ds[0]
x_start = state_to_device(x_start, device)

##### run #####
TIMESTAMPS = [str(ts[2]).split('.')[0] for ts in ds.timestamps][1:-1]
timestamp = TIMESTAMPS[0]

N = 12
M = 1  # ensemble size
lead_s = int(x_start["lead_time_hours"].item()) * 3600
start_timestamp = x_start["timestamp"].cpu()
print(start_timestamp, timestamp)
# import sys
# sys.exit(0)
old_rollout_dir = ensure_rollout_dir("old_model", N=N)
# new_rollout_dir = ensure_rollout_dir("new_model", N=N)

# old

for m in range(1, M + 1):
    samples = old_model.sample_rollout(
        batch=x_start,
        iterations=N,
        member=0,
    ).cpu()  # [B=1, iterations, ...]
    
    # N=1
    # state = old_model.sample(X_start, seed=0)
    # samples = state.unsqueeze(1)  # [B, iterations=1, ...] so samples[:, n-1] works

    for n in range(1, N + 1):
        sample = samples[:, n - 1]  # keep batch dim → [B=1, ...]
        st = ds.denormalize(sample)
        ts = start_timestamp + n * lead_s
        st_xr = ds.convert_to_xarray(st, ts)
        save_state(old_rollout_dir, st_xr, n=n, m=m)

config = {
    "rollout_dir": str(old_rollout_dir),
    "M": M,
    "N": N,
    "timestamp": str(start_timestamp),
    
}

save_to_json(config, old_rollout_dir, "config")

# new

# for m in range(1, M + 1):
    
#     samples = new_model.sample_rollout(N, x_start, seed=None)

#     # state, _ = new_model.rollout_step(X_start, seed=0)
#     # samples = state.unsqueeze(1)  

#     for n in range(1, N + 1):
#         sample = samples[n - 1]  # tdict at step n
#         st = ds.denormalize(sample)
#         ts = start_timestamp + n * lead_s
#         st_xr = ds.convert_to_xarray(st, ts)
#         save_state(new_rollout_dir, st_xr, n=n, m=m)

# config = {
#     "rollout_dir": str(new_rollout_dir),
#     "M": M,
#     "N": N,
#     "timestamp": timestamp,
#     "level": 0,
#     "partition": "surface",
#     "var": "2m_temperature"
# }

# save_to_json(config, new_rollout_dir, "config")