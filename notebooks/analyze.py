import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import xarray as xr

    return mo, np, xr


@app.cell
def _():
    from src.paths import ROLLOUTS
    from src.utils import (
        read_nc,
        read_json,
        get_dataset,
        get_slice,
        get_rollout_dir,
        get_experiment_ids,
        make_hash
    )
    from src.config import PARTITIONS, LEVELS_DICT, VARIABLES_DICT
    from src.ui.interaction import (
        visualize_map,
        get_mask_from_corners,
        get_mask_center,
        plot_rmse_over_n,
        plot_trajectory,
        plot_trajectories_over_n,
        plot_states_over_n,
    )
    from src.ui.plot_variable_change_parallel import plot_variable_change_parallel

    return (
        LEVELS_DICT,
        PARTITIONS,
        VARIABLES_DICT,
        get_experiment_ids,
        get_mask_center,
        get_mask_from_corners,
        get_rollout_dir,
        make_hash,
        plot_variable_change_parallel,
        read_json,
        read_nc,
        visualize_map,
    )


@app.cell
def _():
    from src.funcs import safe_abs_limits

    return (safe_abs_limits,)


@app.cell
def _():
    from src.ui.analysis.analysis_plots import plot_guidance_tracking
    from src.ui.analysis.helpers import format_time_value, get_time_values, clean_coord_value, get_member_values, member_label, as_numpy_2d, select_xr_field, xr_field_to_array, build_mask_rollouts, mean_rollout_terms, member_rollout_terms

    return (
        build_mask_rollouts,
        get_member_values,
        member_rollout_terms,
        plot_guidance_tracking,
        xr_field_to_array,
    )


@app.cell
def _(xr):
    import torch
    from geoarches.dataloaders.era5 import (
        STATS_PATH,
        pressure_levels,
        surface_variables,
        level_variables,
    )

    _PANGU_STATS = torch.load(
        STATS_PATH / "pangu_norm_stats2_with_w.pt", weights_only=True
    )
    _S_MEAN = _PANGU_STATS["surface_mean"].squeeze().numpy()
    _S_STD = _PANGU_STATS["surface_std"].squeeze().numpy()
    _L_MEAN = _PANGU_STATS["level_mean"].squeeze().numpy()
    _L_STD = _PANGU_STATS["level_std"].squeeze().numpy()


    def normalize_xr(xr_state):
        out = {}
        for i, v in enumerate(surface_variables):
            if v in xr_state.data_vars:
                out[v] = (xr_state[v] - _S_MEAN[i]) / _S_STD[i]
        for i, v in enumerate(level_variables):
            if v in xr_state.data_vars:
                mean_da = xr.DataArray(
                    _L_MEAN[i], dims=["level"], coords={"level": pressure_levels}
                )
                std_da = xr.DataArray(
                    _L_STD[i], dims=["level"], coords={"level": pressure_levels}
                )
                out[v] = (xr_state[v] - mean_da) / std_da
        return xr.Dataset(out, coords=xr_state.coords, attrs=xr_state.attrs)


    def denormalize_xr(xr_state):
        out = {}
        for i, v in enumerate(surface_variables):
            if v in xr_state.data_vars:
                out[v] = xr_state[v] * _S_STD[i] + _S_MEAN[i]
        for i, v in enumerate(level_variables):
            if v in xr_state.data_vars:
                mean_da = xr.DataArray(
                    _L_MEAN[i], dims=["level"], coords={"level": pressure_levels}
                )
                std_da = xr.DataArray(
                    _L_STD[i], dims=["level"], coords={"level": pressure_levels}
                )
                out[v] = xr_state[v] * std_da + mean_da
        return xr.Dataset(out, coords=xr_state.coords, attrs=xr_state.attrs)

    return (normalize_xr,)


@app.cell
def _(mo):
    mo.md(f"""
    # PHASE 3 - results analysis
    """)
    return


@app.cell
def _():
    # Select anylsis elements of interest: 
    # - {realized_guidance_checkbox}
    # - {inspection_ui_checkbox}
    return


@app.cell
def _():
    # realized_guidance_checkbox = mo.ui.checkbox(label="Realized guidance")
    # inspection_ui_checkbox = mo.ui.checkbox(label="Inspection widget")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Experiment
    """)
    return


@app.cell
def _(mo):
    refresh_button = mo.ui.button(label="refresh")
    return (refresh_button,)


@app.cell
def _(get_experiment_ids, mo, refresh_button):
    if refresh_button.value:
        pass

    unguided_rollouts = get_experiment_ids("unguided")
    print(unguided_rollouts)
    pick_rollout_dropdown = mo.ui.dropdown(
        options=unguided_rollouts,
        value=unguided_rollouts[0],
        label="Rollout: ",
    )
    return pick_rollout_dropdown, unguided_rollouts


@app.cell
def _(mo, pick_rollout_dropdown, refresh_button, unguided_rollouts):
    experiment_selector = mo.hstack(
        [
            pick_rollout_dropdown,
            refresh_button,
        ],
        justify="start",
    ) if len(unguided_rollouts) > 0 else refresh_button

    # experiment_selector = mo.vstack(
    #     [
    #         experiment_selector,
    #         mo.accordion(
    #             {
    #                 "Experiment params": mo.md(
    #                     "<br>".join(f"{k}: {v}" for k, v in config.items())
    #                 ),
    #             }
    #         ),
    #     ]
    # )
    return (experiment_selector,)


@app.cell
def _(get_rollout_dir, pick_rollout_dropdown):
    rollout_dir = get_rollout_dir(pick_rollout_dropdown.value)
    return (rollout_dir,)


@app.cell
def _(read_nc, rollout_dir):
    unguided_xr = read_nc(rollout_dir, "unguided")
    ground_truth_xr = read_nc(rollout_dir, "ground_truth")
    return ground_truth_xr, unguided_xr


@app.cell
def _(experiment_selector):
    experiment_selector
    return


@app.cell
def _(mo, read_json, rollout_dir):
    experiment_params = read_json(rollout_dir, "experiment_params")

    guidance_mode_dropdown = mo.ui.dropdown(
        options=experiment_params["guidance_mode"],
        value=experiment_params["guidance_mode"][0],
        label="guidance mode",
    )

    alpha_slider = mo.ui.slider(
        steps=experiment_params["alpha"],
        value=experiment_params["alpha"][0],
        label="alpha",
        debounce=True,
        show_value=True
    )

    w_slider = mo.ui.slider(
        steps=experiment_params["w"],
        value=experiment_params["w"][0],
        label="w",
        debounce=True,
        show_value=True
    )

    mo.vstack([
        mo.md("Select experiment: "),
        guidance_mode_dropdown,
        alpha_slider,
        w_slider,
    ])
    return alpha_slider, guidance_mode_dropdown, w_slider


@app.cell
def _(
    alpha_slider,
    guidance_mode_dropdown,
    make_hash,
    rollout_dir,
    w_slider,
    xr,
):
    hash_params = {
        "guidance_mode": guidance_mode_dropdown.value,
        "alpha": alpha_slider.value,
        "w": w_slider.value,
    }

    guided_id = make_hash(hash_params)
    guided_xr = xr.open_dataset(rollout_dir / "guided" / guided_id / "guided.nc")
    return guided_id, guided_xr


@app.cell
def _(guided_id, read_json, rollout_dir):
    config = read_json(rollout_dir / "guided" / guided_id, "config")
    return (config,)


@app.cell
def _(PARTITIONS, mo):
    partition = mo.ui.dropdown(
        PARTITIONS,
        value=PARTITIONS[0],
        label="partition: ",
    )
    return (partition,)


@app.cell
def _(LEVELS_DICT, mo, partition):
    LEVELS = LEVELS_DICT[partition.value]

    level = mo.ui.slider(
        steps=LEVELS,
        value=LEVELS[-1], 
        label="level: ",
        show_value=True,
        debounce=True,
    )
    return (level,)


@app.cell
def _(VARIABLES_DICT, mo, partition):
    VARIABLES = VARIABLES_DICT[partition.value]
    if partition.value == "surface":
        variable_default = VARIABLES[2]
    else:
        variable_default = VARIABLES[3]
    var = mo.ui.dropdown(VARIABLES, value=variable_default, label="variable: ")
    return (var,)


@app.cell
def _(config, mo):
    N = config["N"]
    n_slider = mo.ui.slider(
        steps=range(1, N + 1),
        value=1,
        label="n: ",
        show_value=True,
        debounce=True,
    )
    return N, n_slider


@app.cell
def _(n_slider):
    n = n_slider.value
    return (n,)


@app.cell
def _(config, get_member_values, guided_xr, mo, unguided_xr):
    M = config["M"]
    member_values = get_member_values(unguided_xr)
    guided_member_values = get_member_values(guided_xr)

    m_slider = mo.ui.slider(
        steps=member_values,
        value=member_values[0],
        label="m: ",
        show_value=True,
        debounce=True,
    )
    return M, guided_member_values, m_slider, member_values


@app.cell
def _(m_slider):
    m = m_slider.value
    return (m,)


@app.cell
def _(guided_member_values, m, member_values):
    member_index = member_values.index(m)

    if len(guided_member_values) == len(member_values):
        guided_member = guided_member_values[member_index]
    elif len(guided_member_values) == 1:
        guided_member = guided_member_values[0]
    else:
        guided_member = guided_member_values[min(member_index, len(guided_member_values) - 1)]

    unguided_member = m
    return guided_member, member_index, unguided_member


@app.cell
def _(mo):
    zoom_slider = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=1,
        label="zoom: ",
    )
    return (zoom_slider,)


@app.cell
def _(zoom_slider):
    zoom = zoom_slider.value
    return


@app.cell
def _(mo):
    show_mask_switch = mo.ui.checkbox(label="show mask", value=True)
    return (show_mask_switch,)


@app.cell
def _(show_mask_switch):
    show_mask = show_mask_switch.value
    return (show_mask,)


@app.cell
def _(mo):
    analysis_types = ["absolute", "difference"]

    analysis_type_dropdown = mo.ui.dropdown(
        analysis_types,
        value=analysis_types[0],
        label="analysis type: ",
    )
    return (analysis_type_dropdown,)


@app.cell
def _(analysis_type_dropdown):
    analysis_type = analysis_type_dropdown.value
    return (analysis_type,)


@app.cell
def _(mo):
    show_values_checkbox = mo.ui.checkbox(label="show values")
    return (show_values_checkbox,)


@app.cell
def _(show_values_checkbox):
    show_values = show_values_checkbox.value
    return (show_values,)


@app.cell
def _(config, get_mask_from_corners):
    mask_corners = tuple(config["mask_corners"])
    mask = get_mask_from_corners(*mask_corners)
    return (mask,)


@app.cell
def _(get_mask_center, mask):
    zoom_centers = get_mask_center(mask)
    return (zoom_centers,)


@app.cell
def _(
    build_mask_rollouts,
    ground_truth_xr,
    guided_xr,
    level,
    mask,
    member_index,
    member_rollout_terms,
    unguided_xr,
    var,
):
    (
        timestamps,
        ground_truth_terms,
        unguided_rollout_terms,
        guided_rollout_terms,
    ) = build_mask_rollouts(
        ground_truth_xr=ground_truth_xr,
        unguided_xr=unguided_xr,
        guided_xr=guided_xr,
        var=var.value,
        level=level.value,
        mask=mask,
    )

    # if I ever needed it
    # mean_unguided_rollout_terms = mean_rollout_terms(unguided_rollout_terms)
    # mean_guided_rollout_terms = mean_rollout_terms(guided_rollout_terms)

    selected_guided_terms = member_rollout_terms(
        guided_rollout_terms,
        member_index,
    )

    selected_unguided_terms = member_rollout_terms(
        unguided_rollout_terms,
        member_index,
    )
    return (
        ground_truth_terms,
        guided_rollout_terms,
        selected_guided_terms,
        selected_unguided_terms,
        timestamps,
        unguided_rollout_terms,
    )


@app.cell
def _(config, var):
    if var.value == config["var"]:
        planned_guidance = config.get("y", config.get("planned_guidance", None))
    else:
        planned_guidance=None
    return (planned_guidance,)


@app.cell
def _(ground_truth_xr, guided_xr, n, unguided_xr):
    ground_truth_times = list(ground_truth_xr.time.values)
    guided_times = list(guided_xr.time.values)
    unguided_times = list(unguided_xr.time.values)

    current_time = ground_truth_times[n - 1]
    ground_truth_time = ground_truth_times[n]
    guided_time = guided_times[n - 1]
    unguided_time = unguided_times[n - 1]
    guided_time_prev = guided_times[n - 2]
    unguided_time_prev = unguided_times[n - 2]
    return (
        current_time,
        ground_truth_time,
        guided_time,
        guided_time_prev,
        unguided_time,
        unguided_time_prev,
    )


@app.cell
def _(
    current_time,
    ground_truth_time,
    ground_truth_xr,
    guided_member,
    guided_time,
    guided_time_prev,
    guided_xr,
    level,
    unguided_member,
    unguided_time,
    unguided_time_prev,
    unguided_xr,
    var,
    xr_field_to_array,
):
    current_slice = xr_field_to_array(
        ground_truth_xr,
        var=var.value,
        timestamp=current_time,
        level=level.value,
    )

    next_slice = xr_field_to_array(
        ground_truth_xr,
        var=var.value,
        timestamp=ground_truth_time,
        level=level.value,
    )

    unguided_slice = xr_field_to_array(
        unguided_xr,
        var=var.value,
        timestamp=unguided_time,
        level=level.value,
        member=unguided_member,
    )

    unguided_slice_prev = xr_field_to_array(
        unguided_xr,
        var=var.value,
        timestamp=unguided_time_prev,
        level=level.value,
        member=unguided_member,
    )

    guided_slice = xr_field_to_array(
        guided_xr,
        var=var.value,
        timestamp=guided_time,
        level=level.value,
        member=guided_member,
    )

    guided_slice_prev = xr_field_to_array(
        guided_xr,
        var=var.value,
        timestamp=guided_time_prev,
        level=level.value,
        member=guided_member,
    )

    guided_unguided = guided_slice - unguided_slice
    guided_gt = guided_slice - next_slice
    unguided_gt = unguided_slice - next_slice
    next_current = next_slice - current_slice
    guided_current = guided_slice - current_slice
    unguided_current = unguided_slice - current_slice
    unguided_unguided = unguided_slice - unguided_slice_prev
    guided_guided = guided_slice - guided_slice_prev
    return (
        current_slice,
        guided_current,
        guided_gt,
        guided_guided,
        guided_slice,
        guided_unguided,
        next_current,
        next_slice,
        unguided_current,
        unguided_gt,
        unguided_slice,
        unguided_unguided,
    )


@app.cell
def _(guided_unguided, mo, np):
    # text thresh
    center_value_for_threshold = 0.0

    max_abs_diff_for_threshold = float(
        max(
            np.nanmax(guided_unguided),
            abs(np.nanmin(guided_unguided)),
        )
    )

    if max_abs_diff_for_threshold <= 0:
        max_abs_diff_for_threshold = 1e-8

    default_value_threshold = max_abs_diff_for_threshold * 0.9

    text_thresh = mo.ui.slider(
        start=center_value_for_threshold,
        stop=max_abs_diff_for_threshold,
        step=max_abs_diff_for_threshold / 20,
        value=default_value_threshold,
        label="text thresh: ",
    )
    return (text_thresh,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Realized guidance
    """)
    return


@app.cell
def _(
    ground_truth_terms,
    guided_rollout_terms,
    m,
    n,
    planned_guidance,
    plot_guidance_tracking,
    selected_guided_terms,
    selected_unguided_terms,
    timestamps,
    unguided_rollout_terms,
):
    realized_guidance_plot = plot_guidance_tracking(
        timestamps=timestamps,
        m=m,
        n=n,
        guided_member=selected_guided_terms,
        unguided_member=selected_unguided_terms,
        target_schedule=planned_guidance,
        reference=ground_truth_terms,
        unguided_ensemble=unguided_rollout_terms,
        guided_ensemble=guided_rollout_terms,
        show_unguided_mean=False,
        show_guided_mean=False,
        title="Realized guidance analysis",
        subtitle="Mask terms along the rollout",
        figsize=(22, 6)
    )
    return (realized_guidance_plot,)


@app.cell
def _(realized_guidance_plot):
    realized_guidance_plot
    return


@app.cell(hide_code=True)
def _(mo):
    norm_modes = ["own_scale", "same_scale"]
    norm_mode = mo.ui.dropdown(
        norm_modes,
        value=norm_modes[0],
        label="norm mode: ",
    )
    return (norm_mode,)


@app.cell
def _(
    analysis_type,
    current_slice,
    dpi,
    guided_current,
    guided_gt,
    guided_guided,
    guided_slice,
    guided_unguided,
    mask,
    next_current,
    next_slice,
    norm_mode,
    np,
    safe_abs_limits,
    show_mask,
    show_values,
    text_thresh,
    unguided_current,
    unguided_gt,
    unguided_slice,
    unguided_unguided,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    mask_np = np.asarray(mask)

    if analysis_type == "absolute":
        absolute_panels = [
            ("$x_t$", current_slice),
            ("$x_{t+1}$", next_slice),
            ("$x_{t+1}^{unguided}$", unguided_slice),
            ("$x_{t+1}^{guided}$", guided_slice),
        ]

        abs_vmin, abs_vmax, abs_center = safe_abs_limits(
            [arr for _, arr in absolute_panels]
        )

        absolute_maps = {}

        for label, arr in absolute_panels:
            absolute_maps[label] = visualize_map(
                arr,
                mask_2d=mask_np,
                title=label,
                vmin=abs_vmin,
                vmax=abs_vmax,
                center=abs_center,
                show_mask=show_mask,
                zoom=zoom_slider.value,
                zoom_center_lon=zoom_centers[0],
                zoom_center_lat=zoom_centers[1],
                dpi=dpi.value,
            )

        state_map = absolute_maps["$x_t$"]
        next_map = absolute_maps["$x_{t+1}$"]
        unguided_map = absolute_maps["$x_{t+1}^{unguided}$"]
        guided_map = absolute_maps["$x_{t+1}^{guided}$"]

        next_current_map = None
        unguided_gt_map = None
        guided_gt_map = None
        guided_unguided_map = None
        unguided_unguided_map = None
        guided_guided_map = None

    else:
        difference_panels = [
            ("$x_{t+1} - x_t$", next_current),
            ("$x_{t+1}^{unguided} - x_{t+1}$", unguided_gt),
            ("$x_{t+1}^{guided} - x_{t+1}$", guided_gt),
            ("$x_{t+1}^{guided} - x_{t+1}^{unguided}$", guided_unguided),
            ("$x_{t+1}^{guided} - x_{t}^{guided}$", guided_guided),
            ("$x_{t+1}^{unguided} - x_{t}^{unguided}$", unguided_unguided),
            ("$x_{t+1}^{guided} - x_{t}$", guided_current),
            ("$x_{t+1}^{unguided} - x_{t}$", unguided_current),
        ]

        diff_vmin = min(float(np.nanmin(arr)) for _, arr in difference_panels)
        diff_vmax = max(float(np.nanmax(arr)) for _, arr in difference_panels)

        difference_maps = {}

        for label, arr in difference_panels:
            is_guided_unguided = label == "$x_{t+1}^{guided} - x_{t+1}^{unguided}$"

            if norm_mode.value == "own_scale":
                v_min = min(float(np.nanmin(arr)), -1e-12)
                v_max = max(float(np.nanmax(arr)), 1e-12)
            else:
                v_min, v_max = diff_vmin, diff_vmax

            difference_maps[label] = visualize_map(
                arr,
                mask_2d=mask_np,
                title=label,
                vmin=v_min,
                vmax=v_max,
                center=0.0,
                show_mask=show_mask,
                zoom=zoom_slider.value,
                zoom_center_lon=zoom_centers[0],
                zoom_center_lat=zoom_centers[1],
                show_values=show_values if is_guided_unguided else False,
                value_threshold=text_thresh.value if is_guided_unguided else None,
                value_fontsize=5,
                dpi=dpi.value,
            )

        next_current_map = difference_maps["$x_{t+1} - x_t$"]
        unguided_gt_map = difference_maps["$x_{t+1}^{unguided} - x_{t+1}$"]
        guided_gt_map = difference_maps["$x_{t+1}^{guided} - x_{t+1}$"]
        guided_unguided_map = difference_maps[
            "$x_{t+1}^{guided} - x_{t+1}^{unguided}$"
        ]
        unguided_unguided_map = difference_maps[
            "$x_{t+1}^{unguided} - x_{t}^{unguided}$"
        ]
        guided_guided_map = difference_maps["$x_{t+1}^{guided} - x_{t}^{guided}$"]
        unguided_current_map = difference_maps["$x_{t+1}^{unguided} - x_{t}$"]
        guided_current_map = difference_maps["$x_{t+1}^{guided} - x_{t}$"]

        state_map = None
        next_map = None
        unguided_map = None
        guided_map = None
    return (
        guided_current_map,
        guided_gt_map,
        guided_guided_map,
        guided_map,
        guided_unguided_map,
        next_current_map,
        next_map,
        state_map,
        unguided_current_map,
        unguided_gt_map,
        unguided_map,
        unguided_unguided_map,
    )


@app.cell
def _(mo):
    dpi = mo.ui.slider(start=100, stop=500, step=25, debounce=False, show_value=True, label="dpi: ")
    return (dpi,)


@app.cell
def _(
    analysis_type,
    analysis_type_dropdown,
    dpi,
    guided_current_map,
    guided_gt_map,
    guided_guided_map,
    guided_map,
    guided_unguided_map,
    level,
    m_slider,
    mo,
    n_slider,
    next_current_map,
    next_map,
    norm_mode,
    partition,
    show_mask_switch,
    show_values_checkbox,
    state_map,
    text_thresh,
    unguided_current_map,
    unguided_gt_map,
    unguided_map,
    unguided_unguided_map,
    var,
    zoom_slider,
):
    common_controls = [
        mo.hstack([analysis_type_dropdown, dpi], justify="start"),
        mo.hstack([n_slider, m_slider], justify="start"),
        mo.hstack(
            [partition, var, level],
            justify="start",
        ),
    ]

    if analysis_type == "absolute":
        inspect_states_ui = mo.vstack(
            [
                *common_controls,
                mo.hstack([show_mask_switch, zoom_slider], justify="start"),
                mo.md("Absolute states:"),
                mo.hstack(
                    [state_map, next_map, guided_map, unguided_map],
                    justify="start",
                ),
            ],
            justify="start",
        )

    else:
        inspect_states_ui = mo.vstack(
            [
                *common_controls,
                mo.hstack([show_mask_switch, zoom_slider, norm_mode, show_values_checkbox, text_thresh], justify="start"),
                mo.md("Difference over states:"),
                mo.hstack(
                    [
                        next_current_map,
                        guided_unguided_map,
                        unguided_unguided_map,
                        guided_guided_map,
                    ],
                    justify="start",
                ),
                mo.hstack(
                    [
                        guided_gt_map,
                        unguided_gt_map,
                        guided_current_map,
                        unguided_current_map,
                    ],
                    justify="start",
                ),
            ],
            justify="start",
        )

    inspect_states_ui
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## State change analysis

    \(
    \text{norm}(\sum_{i,j} |x^{guided}_{n,m,i,j} - x^{unguided}_{n,m,i,j}|)
    \)

    Mean |guided − unguided| per cell in z-score state space, across the rollout:

    - **per variable-level** — 82 channels (4 surface + 6 × 13 pressure levels), reveals which exact channel guidance hits hardest
    - **level-aggregated** — 10 variables (level-means), reveals which physical variable absorbs most of the guidance
    """)
    return


@app.cell
def _(mo):
    top_k_slider = mo.ui.slider(
        start=1,
        stop=4+13*6,
        value=12,
        step=1,
        label="top-k channels",
        debounce=True
    )

    rank_by_radio = mo.ui.radio(
        options=["max", "mean"],
        value="max",
        label="rank by",
        inline=True
    )

    variable_change_controls = mo.hstack(
        [top_k_slider, rank_by_radio],
        justify="start",
    )
    return rank_by_radio, top_k_slider, variable_change_controls


@app.cell
def _(variable_change_controls):
    variable_change_controls
    return


@app.cell
def _(np):
    def per_var_level_mean_abs_from_xr_diff(diff_xr):
        out = {}

        for var_name, da in diff_xr.data_vars.items():
            da_abs = abs(da)

            if "level" in da_abs.dims:
                for level_value in da_abs.level.values:
                    out[f"{var_name}-{int(level_value)}"] = float(
                        da_abs.sel(level=level_value).mean(skipna=True).item()
                    )
            else:
                out[f"{var_name}-surface"] = float(da_abs.mean(skipna=True).item())

        return out


    def aggregate_variable_change_dicts_over_members(dict_list, error="std"):
        keys = list(dict_list[0].keys())
        arr = np.array([[d[k] for k in keys] for d in dict_list], dtype=float)

        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        sem = std / np.sqrt(arr.shape[0])

        return {
            "keys": keys,
            "mean": mean,
            "err": std if error == "std" else sem,
            "values": arr,
        }


    def build_raw_zdiff_scores_from_xr(
        *,
        guided_xr,
        unguided_xr,
        N,
        M,
    ):
        analysis_list = []

        for step_idx in range(int(N)):
            per_member_scores = []

            for m in range(int(M)):
                guided_step = guided_xr.isel(time=step_idx, member=m)
                unguided_step = unguided_xr.isel(time=step_idx, member=m)

                diff_step = guided_step - unguided_step

                per_member_scores.append(
                    per_var_level_mean_abs_from_xr_diff(diff_step)
                )

            analysis_list.append(per_member_scores)

        return analysis_list

    return (
        aggregate_variable_change_dicts_over_members,
        build_raw_zdiff_scores_from_xr,
        per_var_level_mean_abs_from_xr_diff,
    )


@app.cell
def _(
    M,
    N,
    aggregate_variable_change_dicts_over_members,
    build_raw_zdiff_scores_from_xr,
    guided_xr,
    normalize_xr,
    plot_variable_change_parallel,
    rank_by_radio,
    top_k_slider,
    unguided_xr,
):
    raw_zdiff_analysis_list = build_raw_zdiff_scores_from_xr(
        guided_xr=normalize_xr(guided_xr),
        unguided_xr=normalize_xr(unguided_xr),
        N=N,
        M=M,
    )

    aggregated_raw_zdiff_per_n = [
        aggregate_variable_change_dicts_over_members(dict_list, error="std")
        for dict_list in raw_zdiff_analysis_list
    ]

    rollout_var_change_fig, _ = plot_variable_change_parallel(
        aggregated_raw_zdiff_per_n,
        top_k=top_k_slider.value,
        rank_by=rank_by_radio.value,
        title="Variable change across rollout steps",
        subtitle=(
            "mean |guided - unguided| per cell in z-score state space; "
            "all 82 channels shown faded, top-k highlighted; error bars = ensemble std"
        ),
        ylim=None,
        ylabel="mean |z-diff| per cell",
        show_unselected=True,
    )
    return (rollout_var_change_fig,)


@app.cell
def _(
    M,
    N,
    aggregate_variable_change_dicts_over_members,
    guided_xr,
    normalize_xr,
    per_var_level_mean_abs_from_xr_diff,
    unguided_xr,
):
    def collapse_level_keys_to_vars_mean(d):
        sums = {}
        counts = {}

        for key, val in d.items():
            var_name, _ = key.rsplit("-", 1)
            sums[var_name] = sums.get(var_name, 0.0) + val
            counts[var_name] = counts.get(var_name, 0) + 1

        return {k: sums[k] / counts[k] for k in sums}


    def build_raw_zdiff_scores_level_agg_from_xr(
        *,
        guided_xr,
        unguided_xr,
        N,
        M,
    ):
        analysis_list = []

        for step_idx in range(int(N)):
            per_member_scores = []

            for m in range(int(M)):
                diff_step = guided_xr.isel(
                    time=step_idx, member=m
                ) - unguided_xr.isel(time=step_idx, member=m)

                raw_dict = per_var_level_mean_abs_from_xr_diff(diff_step)
                per_member_scores.append(
                    collapse_level_keys_to_vars_mean(raw_dict)
                )

            analysis_list.append(per_member_scores)

        return analysis_list


    raw_zdiff_level_agg_list = build_raw_zdiff_scores_level_agg_from_xr(
        guided_xr=normalize_xr(guided_xr),
        unguided_xr=normalize_xr(unguided_xr),
        N=N,
        M=M,
    )

    aggregated_raw_zdiff_level_agg_per_n = [
        aggregate_variable_change_dicts_over_members(dict_list, error="std")
        for dict_list in raw_zdiff_level_agg_list
    ]
    return (aggregated_raw_zdiff_level_agg_per_n,)


@app.cell
def _(
    aggregated_raw_zdiff_level_agg_per_n,
    plot_variable_change_parallel,
    rank_by_radio,
    top_k_slider,
):
    var_change_fig, _ = plot_variable_change_parallel(
        aggregated_raw_zdiff_level_agg_per_n,
        top_k=top_k_slider.value,
        rank_by=rank_by_radio.value,
        title="Variable change across rollout steps (level-aggregated)",
        subtitle=(
            "mean |guided - unguided| per cell in z-score state space, "
            "averaged over levels; error bars = ensemble std"
        ),
        ylim=None,
        ylabel="mean |z-diff| per cell",
        show_unselected=True,
    )
    return (var_change_fig,)


@app.cell
def _(mo, rollout_var_change_fig, var_change_fig):
    mo.hstack([rollout_var_change_fig, var_change_fig], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gradient and vector field analysis
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
