import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## setup
    """)
    return


@app.cell
def _():
    from pathlib import Path 
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from wigglystuff import ChartPuck
    from datetime import datetime, timedelta
    import geopandas as gpd
    import geodatasets
    import torch
    import matplotlib.colors as mcolors

    from scipy.interpolate import (
        CubicSpline,
        PchipInterpolator,
        Akima1DInterpolator,
        interp1d,
        BarycentricInterpolator,
    )

    import matplotlib.patches as mpatches
    import cartopy.feature as cfeature

    from cartopy.crs import PlateCarree
    from matplotlib import colors

    return (
        ChartPuck,
        CubicSpline,
        Path,
        datetime,
        geodatasets,
        gpd,
        mcolors,
        mo,
        mpatches,
        np,
        plt,
        timedelta,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Guidance experiments
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## funcs
    """)
    return


@app.cell
def _(ChartPuck, CubicSpline, n_pucks_slider, np):
    # Internal state for interpolation method (not a marimo dependency)
    method_state = {"value": "CubicSpline"}

    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 100

    MAX_PERC_DELTA = 200/100

    def draw_spline_chart(ax, x_pucks, y_pucks, method):
        # Add fixed left anchor at (0, 0)
        x_all = np.concatenate([[0], x_pucks])
        y_all = np.concatenate([[0], y_pucks])

        # Sort points by x for proper spline fitting
        sorted_indices = np.argsort(x_all)
        x_sorted = x_all[sorted_indices]
        y_sorted = y_all[sorted_indices]

        # Ensure strictly increasing x by adding small perturbations to duplicates
        for i in range(1, len(x_sorted)):
            if x_sorted[i] <= x_sorted[i - 1]:
                x_sorted[i] = x_sorted[i - 1] + 1e-6

        # Dense x values over the full axis 0 ... N-1
        x_dense = np.linspace(0, n_pucks_slider.value, 200)

        spline = CubicSpline(x_sorted, y_sorted)
        y_dense = spline(x_dense)

        ax.plot(x_dense, y_dense, "b-", linewidth=2)
        ax.set_xlim(0, n_pucks_slider.value)
        ax.set_ylim(-MAX_PERC_DELTA, MAX_PERC_DELTA)
        ax.set_xticks(np.arange(0, n_pucks_slider.value, 1))
        ax.set_xlabel("N")
        ax.set_ylabel("Percentage change")
        ax.set_title("Trajectory")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5)

        # Fixed anchor
        ax.plot([0], [0], "ko", markersize=8, zorder=5)

    def draw_spline(ax, widget):
        # N total points INCLUDING the fixed anchor => N draggable pucks
        fixed_x = np.linspace(1, n_pucks_slider.value, len(widget.y)).tolist()

        try:
            widget.x = fixed_x
        except Exception:
            pass

        draw_spline_chart(ax, fixed_x, list(widget.y), method_state["value"])

    # Slider value N = total number of points INCLUDING the anchor
    _n = n_pucks_slider.value

    # Draggable pucks are at x = 1, 2, ..., N
    _init_x = np.linspace(1, _n, _n).tolist()
    _init_y = (0.15 * np.sin(np.pi * np.array(_init_x) / (_n))).tolist()

    spline_puck = ChartPuck.from_callback(
        draw_fn=draw_spline,
        x_bounds=(1, _n),
        y_bounds=(-MAX_PERC_DELTA, MAX_PERC_DELTA),
        figsize=(6, 4),
        x=_init_x,
        y=_init_y,
        puck_color="#9c27b0",
        drag_y_bounds=(-MAX_PERC_DELTA, MAX_PERC_DELTA),
    )
    return method_state, spline_puck


@app.cell
def _(method_state, spline_puck):
    def on_method_change(new_val):
        method_state["value"] = new_val
        spline_puck.redraw()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## State
    """)
    return


@app.cell
def _(spline_widget):
    drift = spline_widget.y
    return (drift,)


@app.cell
def _(mo):
    from geoarches.lightning_modules import load_module
    from geoarches.dataloaders.era5 import Era5Forecast

    device = "mps"
    model, config = load_module("archesweather-m-seed0")

    ds = Era5Forecast(
        path="data/era5_240/full",  # default path
        domain="test", # domain to consider. domain = 'test' loads the 2020 period
        load_prev=True,  # whether to load previous state
        norm_scheme="pangu",  # default normalization scheme
        lead_time_hours=6
    )

    gen_model, gen_config = load_module("archesweathergen")

    gen_model = gen_model.to(device)

    MODELS = ["ArchesWeather"]
    model = mo.ui.dropdown(MODELS, value=MODELS[0], label="model: ")
    model
    return device, ds, gen_model


@app.cell
def _(mo):
    VARIABLE_TYPES = ["surface", "level"]
    variable_type = mo.ui.dropdown(VARIABLE_TYPES, value=VARIABLE_TYPES[0], label="variable type: ")
    variable_type
    return (variable_type,)


@app.cell
def _(mo, variable_type):
    LEVELS_DICT = {
        "surface": [i+1 for i in range(1)],
        "level": [i+1 for i in range(13)]
    }
    LEVELS = LEVELS_DICT[variable_type.value]
    level = mo.ui.dropdown(LEVELS, value=LEVELS[0], label="level: ")
    level
    return LEVELS, level


@app.cell
def _(mo, variable_type):
    VARIABLES_DICT = {
        "surface": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure"
        ],
        "level": [
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
            "temperature",
            "specific_humidity",
            "vertical_velocity"
        ]
    }

    VARIABLES = VARIABLES_DICT[variable_type.value]
    variable = mo.ui.dropdown(VARIABLES, value=VARIABLES[0], label="variable : ")
    variable
    return VARIABLES, variable


@app.cell
def _(datetime, mo, timedelta):
    def dates_2020_6h():
        start = datetime(2020, 1, 1, 0)
        end = datetime(2021, 1, 1, 0)

        dates = []
        current = start
        while current < end:
            dates.append(f"{current:%Y-%m-%d} - {current.hour}h")
            current += timedelta(hours=6)

        # removes first and last
        return dates[1:-1]

    TIMESTAMPS = dates_2020_6h()
    start_ts = mo.ui.dropdown(TIMESTAMPS, value=TIMESTAMPS[0], label="start state : ")
    start_ts
    return TIMESTAMPS, start_ts


@app.cell
def _(LEVELS, TIMESTAMPS, VARIABLES, level, start_ts, variable):
    start_ts_idx = TIMESTAMPS.index(start_ts.value)
    var_idx = VARIABLES.index(variable.value)
    level_idx = LEVELS.index(level.value) - 1
    return level_idx, start_ts_idx, var_idx


@app.cell
def _(ds, start_ts_idx):
    start_state = ds[start_ts_idx]
    return (start_state,)


@app.cell
def _(ChartPuck, geodatasets, gpd, mo, mpatches, np):
    def lonlat_to_ij(lon, lat, lon_c, lat_c):
        j = int(np.argmin(np.abs(lon_c - lon)))
        i = int(np.argmin(np.abs(lat_c - lat)))
        return i, j

    def visualize_map_with_rectangle(array_2d, cmap="coolwarm", norm=None):
        ny, nx = array_2d.shape
        assert (ny, nx) == (121, 240), f"Expected (121, 240), got {(ny, nx)}"

        # undo the 180° roll applied in convert_to_tensordict
        array_2d_unrolled = np.roll(array_2d, -nx // 2, axis=1)

        # original ERA5 grid in [0, 360]
        lon_e = np.linspace(0.0, 360.0, nx + 1, endpoint=True)
        lat_e = np.linspace(90.0, -90.0, ny + 1, endpoint=True)
        lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
        lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

        # convert longitudes to [-180, 180]
        lon_c_plot = ((lon_c + 180) % 360) - 180
        sort_idx = np.argsort(lon_c_plot)
        lon_c_plot = lon_c_plot[sort_idx]
        array_2d_plot = array_2d_unrolled[:, sort_idx]

        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

        def draw_map(ax, widget):
            ax.clear()

            ax.pcolormesh(
                lon_c_plot,
                lat_c,
                array_2d_plot,
                cmap=cmap,
                norm=norm,
                shading="nearest",
            )

            world.boundary.plot(ax=ax, color="black", linewidth=0.4, zorder=5)

            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Drag pucks to define rectangle")

            x0, x1 = widget.x
            y0, y1 = widget.y

            lon_left = min(x0, x1)
            lon_right = max(x0, x1)
            lat_bottom = min(y0, y1)
            lat_top = max(y0, y1)

            try:
                widget.x = [lon_left, lon_right]
                widget.y = [lat_top, lat_bottom]
            except Exception:
                pass

            ul_i, ul_j = lonlat_to_ij(lon_left, lat_top, lon_c_plot, lat_c)
            lr_i, lr_j = lonlat_to_ij(lon_right, lat_bottom, lon_c_plot, lat_c)

            rect = mpatches.Rectangle(
                (lon_left, lat_bottom),
                lon_right - lon_left,
                lat_top - lat_bottom,
                fill=False,
                edgecolor="red",
                linewidth=1.5,
                zorder=10,
            )
            ax.add_patch(rect)

            ax.plot(
                [lon_left, lon_right],
                [lat_top, lat_bottom],
                "o",
                color="red",
                markersize=5,
                zorder=11,
            )
        rect_puck = ChartPuck.from_callback(
            draw_fn=draw_map,
            x=[20.0, 60.0],
            y=[50.0, 20.0],
            puck_color=["green", "red"],
            figsize=(10, 5),
            x_bounds=(-180.0, 180.0),
            y_bounds=(-90.0, 90.0),
        )

        return mo.ui.anywidget(rect_puck)

    return (visualize_map_with_rectangle,)


@app.cell
def _(mo):
    mo.md("""
    Spatial mask:
    """)
    return


@app.cell
def _(
    ds,
    level_idx,
    start_state,
    var_idx,
    variable_type,
    visualize_map_with_rectangle,
):
    array_2d_denormalized = ds.denormalize(start_state["state"])[variable_type.value][var_idx, level_idx]
    array_2d = start_state["state"][variable_type.value][var_idx, level_idx]

    map_widget = visualize_map_with_rectangle(array_2d)
    map_widget
    return array_2d, array_2d_denormalized, map_widget


@app.cell
def _():
    # TODO: denormalize and renormalize!
    return


@app.cell
def _(map_widget, np):
    x0, x1 = map_widget.value["x"]
    y0, y1 = map_widget.value["y"]

    lon_left, lon_right = sorted([x0, x1])
    lat_bottom, lat_top = sorted([y0, y1])

    lon_e = np.linspace(0.0, 360.0, 240 + 1, endpoint=True)
    lat_e = np.linspace(90.0, -90.0,  121 + 1, endpoint=True)
    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])

    lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)

    spatial_mask = (
        (lon_grid >= lon_left)
        & (lon_grid <= lon_right)
        & (lat_grid >= lat_bottom)
        & (lat_grid <= lat_top)
    ).astype(np.uint8)

    # spatial_mask
    return (spatial_mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Trajectory
    """)
    return


@app.cell
def _(mo):
    n_pucks_slider = mo.ui.slider(3, 20, value=6, label="N: ")
    n_pucks_slider
    return (n_pucks_slider,)


@app.cell
def _(mo, spline_puck):
    spline_widget = mo.ui.anywidget(spline_puck)
    spline_widget
    return (spline_widget,)


@app.cell
def _(array_2d, array_2d_denormalized, drift, np, spatial_mask):
    # compute initial avg: init_avg = avg(mask*state)
    # compute guidance: guidance = drift_perc*init_avg

    def init_avg(mask, state):
        return np.mean(mask * state)

    def guidance_terms(drift, init_avg):
        return [init_avg + d * np.abs(init_avg) for d in drift]

    init_avg_denormalized = init_avg(spatial_mask, np.asarray(array_2d_denormalized))
    guidance_terms_denormalized = guidance_terms(drift, init_avg_denormalized)

    init_avg = init_avg(spatial_mask, np.asarray(array_2d))
    guidance_terms = guidance_terms(drift, init_avg)

    # '{:f}'.format(init_avg_denormalized)
    # '{:f}'.format(init_avg)
    return guidance_terms_denormalized, init_avg_denormalized


@app.cell
def _(guidance_terms_denormalized, init_avg_denormalized, np, plt, variable):
    trajectory = [init_avg_denormalized] + guidance_terms_denormalized
    def plot_guidance_trajectory(trajectory):
        x = np.arange(len(trajectory))

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot(x, trajectory, "b-", linewidth=2)
        ax.plot(x, trajectory, "o", color="#9c27b0", markersize=5)

        ax.set_xlim(0, len(trajectory) - 1)
        ax.set_xticks(np.arange(len(trajectory)))
        ax.set_xlabel("N")
        ax.set_ylabel(f"{variable.value}")
        ax.set_title("Trajectory")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5)

        return fig

    fig = plot_guidance_trajectory(trajectory)
    # mo.vstack([
    #     mo.md("Planned trajectory: "),
    #     fig
    # ])
    fig
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Guidance
    """)
    return


@app.cell
def _(mo):
    GUIDANCE_METHODS = ["CFG"]
    guidance_method = mo.ui.dropdown(GUIDANCE_METHODS, value=GUIDANCE_METHODS[0], label="Guidance method: ")
    guidance_method
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate
    """)
    return


@app.cell
def _():
    # can retrieve ground truth state on ds using timestamp
    return


@app.cell
def _(device, gen_model, start_state):
    ##### sample

    SEED = 0
    T = 25

    def sample_next_state(current_state):
        batch = {k: v[None].to(device) for k, v in current_state.items()}
        sampled_state = gen_model.guided_sampling(
            batch, seed=SEED, num_steps=T, scale_input_noise=1.05
        ).cpu()
    
        return sampled_state

    sample_next_state(start_state)
    return (T,)


@app.cell
def _(
    Path,
    T,
    level_idx,
    mcolors,
    plt,
    start_state,
    torch,
    var_idx,
    variable,
    variable_type,
):
    def get_last_experiment_id():
        paths = Path("experiments").glob("2026*")
        paths = sorted(paths)
        return paths[-1]

    def read_experiment_tensor(id, t, partition):
        path = Path(id, str(t), f"{partition}.pt")
        tensor = torch.load(path)
        return tensor.cpu()

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def plot_sample(ground_truth, pred_state, tensor, title=""):
        ground_truth = ground_truth.cpu()
        pred_state = pred_state.cpu()
        tensor = tensor.cpu()

        combined = pred_state + tensor
        error = ground_truth - combined

        det_error = pred_state - ground_truth
        abs_error = torch.abs(det_error)
        sum_error = torch.sum(abs_error)
        print(f"total det error: {sum_error}")


        abs_error = torch.abs(error)
        sum_error = torch.sum(abs_error)
        print(f"total error: {sum_error}")

        fig, axes = plt.subplots(1, 4, dpi=1000, figsize=(16, 4))
        fig.suptitle(title)

        fields = [
            ("tensor", tensor),
            ("pred_state + tensor", combined),
            ("ground_truth", ground_truth),
            ("error", error),
        ]

        for ax, (name, field) in zip(axes, fields):
            vmax = field.abs().max().item()
            if vmax == 0:
                vmax = 1e-8
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            im = ax.imshow(field, cmap="RdBu_r", norm=norm)
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

        plt.tight_layout()
        plt.show()

    def plot_state(ground_truth, pred_state, id, t, partition: str, var: str, var_idx: int, level: int):
        tensor = read_experiment_tensor(id, t, partition)
        tensor = tensor[var_idx, level]
        pred_state = pred_state[var_idx, level]

        title = f"{t} - {partition} - {var} - {level}"
    
        plot_sample(ground_truth, pred_state, tensor, title)

    id = get_last_experiment_id()

    gt_state = start_state["next_state"]
    ground_truth = gt_state[variable_type.value][var_idx, level_idx]
    pred_state = read_experiment_tensor(id, "pred_state", variable_type.value)
    # print(pred_state.shape)

    for t in range(T):
        plot_state(ground_truth, pred_state, id, t, variable_type.value, variable.value, var_idx, level_idx)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
