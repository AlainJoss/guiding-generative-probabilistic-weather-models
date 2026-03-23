import torch

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from geoarches.lightning_modules import load_module
from geoarches.dataloaders.era5 import Era5Forecast


# TODO: normalize the colorbar across

def plot_sample(sample, title=""):
    sample = sample.cpu()

    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(8, 4))
    fig.suptitle(title)

    vmax = sample.abs().max().item()
    if vmax == 0:
        vmax = 1e-8
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(sample, cmap="RdBu_r", norm=norm)
    ax.set_title("tensor")
    ax.set_xticks([])
    ax.set_yticks([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.show()


def plot_state(sample, partition: str, var: str, var_idx: int, level: int):
    sample = sample[var_idx, level]
    title = f"{partition} - {var} - {level}"
    plot_sample(sample, title)

# print(pred_state.shape)

def state_to_device(state):
    return {k: v[None].to(device) for k, v in state.items()}


device = "mps"
# model, config = load_module("archesweather-m-seed0")

ds = Era5Forecast(
    path="data/era5_240/full",  # default path
    domain="test", # domain to consider. domain = 'test' loads the 2020 period
    load_prev=True,  # whether to load previous state
    norm_scheme="pangu",  # default normalization scheme
    lead_time_hours=6
)

sample_state = ds[0]
gt_state = sample_state["next_state"]
X_start = state_to_device(sample_state)

gen_model, gen_config = load_module("archesweathergen")
gen_model = gen_model.to(device)

sampled_state = gen_model.rollout_step(
    X_start, 
    y_t=torch.tensor(1, device=X_start["state"].device, dtype=torch.float32), 
    mask=torch.zeros((121, 240), device=X_start["state"].device, dtype=torch.float32)
).cpu()

# variable_type = "surface"
# level = 1
# variable = "2tm_temperature"
# plot_state(sampled_state[variable_type][0], variable_type, variable, var_idx, level_idx)