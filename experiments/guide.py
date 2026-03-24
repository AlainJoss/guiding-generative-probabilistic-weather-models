import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", css_file="")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## todos
    - refactor plots to use a dix min and max
    - solve how to steer in denormalized and latent space
    - refactor guided_flow model to take in partition, level_index, variable_index
    - refactor this whole notebook to outsource functionality

    ## ideas
    - steer model towards grount truth in mask and measure divergence across states
    - generate multiple (vars, levels, partitions) no-guidance N rollouts and save in experiment folder -> need a way to encode the experiment (log of experiments or yaml with keys).
    """)
    return


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
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from cartopy.crs import PlateCarree
    from matplotlib import colors

    return (
        ChartPuck,
        CubicSpline,
        Path,
        datetime,
        geodatasets,
        gpd,
        make_axes_locatable,
        mcolors,
        mo,
        mpatches,
        np,
        plt,
        timedelta,
        torch,
    )


@app.cell
def _():
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 100
    return


@app.cell
def _():
    # TODO: the start date is 18:00 the 31.12.2019!
    return


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
    ## Model
    """)
    return


@app.cell
def _(mo):
    from geoarches.lightning_modules import load_module
    from geoarches.dataloaders.era5 import Era5Forecast

    device = "mps"

    ds = Era5Forecast(
        path="data/era5_240/full",  # default path
        domain="test", # domain to consider. domain = 'test' loads the 2020 period
        load_prev=True,  # whether to load previous state
        norm_scheme="pangu",  # default normalization scheme
        lead_time_hours=6
    )

    module_target = "geoarches.lightning_modules.{}"
    MODELS = ["diffusion.DiffusionModule", "guided_diffusion.GuidedFlow"]
    model = mo.ui.dropdown(MODELS, value=MODELS[1], label="model: ")
    model
    return device, ds, load_module, model, module_target


@app.cell
def _(device, load_module, model, module_target):
    gen_model, gen_config = load_module(
        "archesweathergen",
        module_target=module_target.format(model.value),
    )
    gen_model = gen_model.to(device)
    return (gen_model,)


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
    VARIABLE_TYPES = ["surface", "level"]
    variable_type = mo.ui.dropdown(VARIABLE_TYPES, value=VARIABLE_TYPES[0], label="variable type: ")
    variable_type, mo.md("temperature is in kelvin, my god")
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
    variable = mo.ui.dropdown(VARIABLES, value=VARIABLES[2], label="variable : ")
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
def _(level_idx, var_idx):
    var_idx, level_idx
    return


@app.cell
def _(ds, start_ts_idx):
    start_state = ds[start_ts_idx]
    return (start_state,)


@app.cell
def _(
    ChartPuck,
    geodatasets,
    gpd,
    level,
    make_axes_locatable,
    mcolors,
    mo,
    mpatches,
    np,
    variable,
):

    def lonlat_to_ij(lon, lat, lon_c, lat_c):
        j = int(np.argmin(np.abs(lon_c - lon)))
        i = int(np.argmin(np.abs(lat_c - lat)))
        return i, j


    def visualize_map(
        array_2d,
        cmap="coolwarm",
        vmin=None,
        vmax=None,
        center=None,
        figsize=(10, 5),
        with_rectangle=False,
        rectangle_x=(20.0, 60.0),
        rectangle_y=(50.0, 20.0),
    ):
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

        if center is not None:
            if vmin is None or vmax is None:
                absmax = np.nanmax(np.abs(array_2d_plot))
                if not np.isfinite(absmax) or absmax == 0:
                    absmax = 1e-8
                if vmin is None:
                    vmin = -absmax
                if vmax is None:
                    vmax = absmax
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        def draw_base(ax):
            im = ax.pcolormesh(
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
            ax.set_title(f"{variable.value} @ level {level.value}")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            ax.figure.colorbar(im, cax=cax)

            return im

        if not with_rectangle:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, dpi=300, figsize=figsize)
            draw_base(ax)
            plt.tight_layout()
            plt.show()
            return

        def draw_map(ax, widget):
            ax.clear()

            # remove previous colorbar axes if redraw happens
            fig = ax.figure
            while len(fig.axes) > 1:
                fig.delaxes(fig.axes[-1])

            draw_base(ax)

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
            x=list(rectangle_x),
            y=list(rectangle_y),
            puck_color=["green", "red"],
            figsize=figsize,
            x_bounds=(-180.0, 180.0),
            y_bounds=(-90.0, 90.0),
        )

        return mo.ui.anywidget(rect_puck)

    return (visualize_map,)


@app.cell
def _(mo):
    mo.md("""
    Spatial mask:
    """)
    return


@app.cell
def _(ds, level_idx, start_state, var_idx, variable_type, visualize_map):
    array_2d_denormalized = ds.denormalize(start_state["state"])[variable_type.value][var_idx, level_idx]
    array_2d = start_state["state"][variable_type.value][var_idx, level_idx]

    map_widget = visualize_map(array_2d, vmin=-3, vmax=3, with_rectangle=True)

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

    def avg_over_mask(mask, state):
        avg = np.sum(mask * state) / np.sum(mask)
        return avg

    def guidance_terms(drift, init_avg):
        return [init_avg + d * np.abs(init_avg) for d in drift]

    avg_over_mask_denormalized  = avg_over_mask(spatial_mask, np.asarray(array_2d_denormalized))
    guidance_terms_denormalized  = guidance_terms(drift, avg_over_mask_denormalized)

    avg_over_mask_normalized = avg_over_mask(spatial_mask, np.asarray(array_2d))
    guidance_terms_normalized = guidance_terms(drift, avg_over_mask_normalized)

    avg_over_mask_denormalized, guidance_terms_denormalized, avg_over_mask_normalized, guidance_terms_normalized
    return avg_over_mask_denormalized, guidance_terms_denormalized


@app.cell
def _():
    # TODO: need to compute guidance in residual space!
    return


@app.cell
def _(
    avg_over_mask_denormalized,
    guidance_terms_denormalized,
    np,
    plt,
    variable,
):
    trajectory = [avg_over_mask_denormalized] + guidance_terms_denormalized
    def plot_trajectory(trajectory):
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

    fig = plot_trajectory(trajectory)
    # mo.vstack([
    #     mo.md("Planned trajectory: "),
    #     fig
    # ])
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
def _(device):
    def state_to_device(state):
        return {k: v[None].to(device) for k, v in state.items()}

    return (state_to_device,)


@app.cell
def _(mo):
    ##### sample

    run_button = mo.ui.run_button(label="Run experiment")
    run_button
    return (run_button,)


@app.cell
def _(mo):
    get_sampled_state, set_sampled_state = mo.state(None)
    return get_sampled_state, set_sampled_state


@app.cell
def _(device, gen_model, mo, state_to_device, torch):
    @mo.cache
    def run_rollout(start_state, guidance_terms_denormalized, spatial_mask):
        batch = state_to_device(start_state)
        guidance = torch.tensor(guidance_terms_denormalized)
        mask = torch.tensor(spatial_mask, device=device, dtype=torch.float32)
        return gen_model.rollout_step(batch, guidance[0], mask).cpu()

    return (run_rollout,)


@app.cell
def _(
    Path,
    datetime,
    guidance_terms_denormalized,
    level_idx,
    model,
    run_button,
    run_rollout,
    sampled_state,
    set_sampled_state,
    spatial_mask,
    start_state,
    torch,
    var_idx,
    variable_type,
):
    def save_run(sampled_state):
        date, time = str(datetime.now().replace(microsecond=0)).split(" ")
        now = date + "_" + time
        experiment_path = Path("experiments", f"{now}")
        experiment_path.mkdir(parents=True, exist_ok=True)
        experiment_path = experiment_path /f"{model.value}.pt"
        torch.save(sampled_state[variable_type.value][0][var_idx, level_idx], experiment_path)
        print("run saved")

    if run_button.value:
        result = run_rollout(start_state, guidance_terms_denormalized, spatial_mask)
        set_sampled_state(result)
        save_run(sampled_state)
    return


@app.cell
def _():
    vmin = -3
    vmax = 3
    return vmax, vmin


@app.cell
def _(get_sampled_state, mo):

    # plot_state(start_state["next_state"][variable_type.value], variable_type.value, variable.value, var_idx, level_idx)

    sampled_state = get_sampled_state()

    mo.stop(sampled_state is None, mo.md("Press **Run rollout** first."))

    # DO THINGS WITH sampled_state

    # visualize_map(sampled_state[variable_type.value][0][var_idx, level_idx], vmin=vmin, vmax=vmax)
    # visualize_map(start_state["next_state"][variable_type.value][var_idx, level_idx], vmin=vmin, vmax=vmax)
    return (sampled_state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## compare
    """)
    return


@app.cell
def _(torch):
    path_ = "experiments/generated/{}.pt"
    diffusion_tensor = torch.load(path_.format("diffusion.DiffusionModule"), weights_only=False)
    flow_tensor = torch.load(path_.format("guided_diffusion.GuidedFlow"), weights_only=False)
    torch.allclose(diffusion_tensor, flow_tensor, rtol=1e-6, atol=1e-7)
    return diffusion_tensor, flow_tensor


@app.cell
def _(diffusion_tensor, flow_tensor, visualize_map, vmax, vmin):
    visualize_map(diffusion_tensor, vmin=vmin, vmax=vmax)
    visualize_map(flow_tensor, vmin=vmin, vmax=vmax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Guided roll-out
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
