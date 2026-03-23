import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_state(state, partition, var_idx, level_idx):
    """
    ensure following about state:
    - batch dim is gone 
    - state in on cpu and detached 

    """
    state = state[partition]
    state = state[var_idx, level_idx]
    plt.figure(dpi=1000)
    norm = mcolors.TwoSlopeNorm(vmin=state.min(), vcenter=0, vmax = state.max())
    plt.imshow(state, cmap="RdBu_r", norm=norm)
    plt.colorbar()
    plt.show()