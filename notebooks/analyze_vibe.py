import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import xarray as xr

    return mo, np, xr


@app.cell
def _():
    from src.ui.interaction import (
        visualize_map,
        get_mask_from_corners,
        get_mask_center,
        plot_rmse_over_n,
        plot_variable_change_parallel,
        plot_trajectory,
        plot_trajectories_over_n,
        plot_states_over_n,
    )

    from src.constants import PARTITIONS, LEVELS_DICT, VARIABLES_DICT

    from src.funcs import get_guidance

    from src.utils import (
        read_nc,
        read_json,
        get_dataset,
        get_slice,
        get_rollout_dir,
        get_experiment_ids,
        make_hash
    )

    from src.paths import ROLLOUTS

    return (
        LEVELS_DICT,
        PARTITIONS,
        VARIABLES_DICT,
        get_dataset,
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
    from src.ui.analysis.analysis_plots import plot_guidance_tracking
    from src.ui.analysis.helpers import format_time_value, get_time_values, clean_coord_value, get_member_values, member_label, as_numpy_2d, select_xr_field, xr_field_to_array, build_mask_rollouts, mean_rollout_terms, member_rollout_terms

    return (
        build_mask_rollouts,
        format_time_value,
        get_member_values,
        mean_rollout_terms,
        member_rollout_terms,
        plot_guidance_tracking,
        xr_field_to_array,
    )


@app.cell
def _(get_dataset):
    ds = get_dataset()
    return


@app.cell
def _(inspection_ui_checkbox, mo, realized_guidance_checkbox):
    mo.md(f"""
    # PHASE 3 - results analysis

    Select anylsis elements of interest: 
    - {realized_guidance_checkbox}
    - {inspection_ui_checkbox}
    """)
    return


@app.cell
def _(mo):
    realized_guidance_checkbox = mo.ui.checkbox(label="Realized guidance")
    inspection_ui_checkbox = mo.ui.checkbox(label="Inspection widget")
    return inspection_ui_checkbox, realized_guidance_checkbox


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
    pick_unguided_rollout_dropdown = mo.ui.dropdown(
        label="Experiment: ", value=unguided_rollouts[0], options=unguided_rollouts
    )

    pick_rollout_dropdown = mo.ui.dropdown(
        options=unguided_rollouts,
        value=unguided_rollouts[0],
        label="Pick rollout: ",
    )
    return (
        pick_rollout_dropdown,
        pick_unguided_rollout_dropdown,
        unguided_rollouts,
    )


@app.cell
def _(config, mo, pick_rollout_dropdown, refresh_button, unguided_rollouts):
    selector = mo.hstack(
        [
            pick_rollout_dropdown,
            refresh_button,
        ],
        justify="start",
    ) if len(unguided_rollouts) > 0 else refresh_button

    experiment_picker = mo.vstack(
        [
            selector,
            mo.accordion(
                {
                    "Experiment params": mo.md(
                        "<br>".join(f"{k}: {v}" for k, v in config.items())
                    ),
                }
            ),
        ]
    )
    return (experiment_picker,)


@app.cell
def _(get_rollout_dir, pick_unguided_rollout_dropdown):
    rollout_dir = get_rollout_dir(pick_unguided_rollout_dropdown.value)
    return (rollout_dir,)


@app.cell
def _(read_nc, rollout_dir):
    unguided_xr = read_nc(rollout_dir, "unguided")
    ground_truth_xr = read_nc(rollout_dir, "ground_truth")
    return ground_truth_xr, unguided_xr


@app.cell
def _(experiment_picker):
    experiment_picker
    return


@app.cell
def _(experiment_params):
    experiment_params
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
        debounce=True
    )

    w_slider = mo.ui.slider(
        steps=experiment_params["w"],
        value=experiment_params["w"][0],
        label="w",
        debounce=True
    )

    mo.vstack([
        mo.md("Slide experiment by params:"),
        guidance_mode_dropdown,
        alpha_slider,
        w_slider,
    ])
    return alpha_slider, experiment_params, guidance_mode_dropdown, w_slider


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
    level = mo.ui.slider(steps=LEVELS, value=LEVELS[0], label="level  ", show_value=True, debounce=True)
    return (level,)


@app.cell
def _(VARIABLES_DICT, mo, partition):
    VARIABLES = VARIABLES_DICT[partition.value]
    if partition.value == "surface":
        variable_default = VARIABLES[2]
    else:
        variable_default = VARIABLES[3]
    var = mo.ui.dropdown(VARIABLES, value=variable_default, label="variable : ")
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
        label="Zoom",
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
    mean_rollout_terms,
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

    mean_unguided_rollout_terms = mean_rollout_terms(unguided_rollout_terms)
    mean_guided_rollout_terms = mean_rollout_terms(guided_rollout_terms)

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
    return current_time, ground_truth_time, guided_time, unguided_time


@app.cell
def _(
    current_time,
    ground_truth_time,
    ground_truth_xr,
    guided_member,
    guided_time,
    guided_xr,
    level,
    unguided_member,
    unguided_time,
    unguided_xr,
    var,
    xr_field_to_array,
):
    current_slice = xr_field_to_array(
        ground_truth_xr,
        var=var.value,
        timestamp=current_time,
        level=level,
    )

    next_slice = xr_field_to_array(
        ground_truth_xr,
        var=var.value,
        timestamp=ground_truth_time,
        level=level,
    )

    unguided_slice = xr_field_to_array(
        unguided_xr,
        var=var.value,
        timestamp=unguided_time,
        level=level,
        member=unguided_member,
    )

    guided_slice = xr_field_to_array(
        guided_xr,
        var=var.value,
        timestamp=guided_time,
        level=level,
        member=guided_member,
    )

    guided_unguided = guided_slice - unguided_slice
    guided_gt = guided_slice - next_slice
    unguided_gt = unguided_slice - next_slice
    next_current = next_slice - current_slice
    guided_current = guided_slice - current_slice
    unguided_current = unguided_slice - current_slice
    return (
        current_slice,
        guided_current,
        guided_gt,
        guided_slice,
        guided_unguided,
        next_current,
        next_slice,
        unguided_current,
        unguided_gt,
        unguided_slice,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make a vline at the current timestep!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - make checkbox for each component of the plot!
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
    )
    return (realized_guidance_plot,)


@app.cell
def _(m_slider, mo, realized_guidance_plot):
    mo.vstack([m_slider, realized_guidance_plot])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspect states

    Analyze all relevant weather states interactively.
    """)
    return


@app.cell
def _(
    analysis_type,
    current_slice,
    guided_current,
    guided_gt,
    guided_slice,
    guided_unguided,
    mask,
    next_current,
    next_slice,
    np,
    show_mask,
    show_values,
    text_thresh,
    unguided_current,
    unguided_gt,
    unguided_slice,
    visualize_map,
    zoom_centers,
    zoom_slider,
):
    def safe_abs_limits(arrays):
        vmin = min(float(np.nanmin(np.asarray(arr))) for arr in arrays)
        vmax = max(float(np.nanmax(np.asarray(arr))) for arr in arrays)

        if vmax <= vmin:
            vmax = vmin + 1e-9

        center = 0.5 * (vmin + vmax)
        center = min(max(center, vmin + 1e-9), vmax - 1e-9)

        return vmin, vmax, center


    def safe_diff_absmax(arrays):
        absmax = max(
            float(np.nanmax(np.abs(np.asarray(arr))))
            for arr in arrays
        )

        if absmax <= 0:
            absmax = 1e-8

        return absmax


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
            )

        state_map = absolute_maps["$x_t$"]
        next_map = absolute_maps["$x_{t+1}$"]
        unguided_map = absolute_maps["$x_{t+1}^{unguided}$"]
        guided_map = absolute_maps["$x_{t+1}^{guided}$"]

        next_current_map = None
        unguided_current_map = None
        guided_current_map = None
        unguided_gt_map = None
        guided_gt_map = None
        guided_unguided_map = None

    else:
        difference_panels = [
            ("$x_{t+1} - x_t$", next_current),
            ("$x_{t+1}^{unguided} - x_t$", unguided_current),
            ("$x_{t+1}^{guided} - x_t$", guided_current),
            ("$x_{t+1}^{unguided} - x_{t+1}$", unguided_gt),
            ("$x_{t+1}^{guided} - x_{t+1}$", guided_gt),
            ("$x_{t+1}^{guided} - x_{t+1}^{unguided}$", guided_unguided),
        ]

        diff_absmax = safe_diff_absmax(
            [arr for _, arr in difference_panels]
        )

        difference_maps = {}

        for label, arr in difference_panels:
            difference_maps[label] = visualize_map(
                arr,
                mask_2d=mask_np,
                title=label,
                vmin=-diff_absmax,
                vmax=diff_absmax,
                center=0.0,
                show_mask=show_mask,
                zoom=zoom_slider.value,
                zoom_center_lon=zoom_centers[0],
                zoom_center_lat=zoom_centers[1],
                show_values=show_values,
                value_threshold=text_thresh.value,
                value_fontsize=5,
            )

        next_current_map = difference_maps["$x_{t+1} - x_t$"]
        unguided_current_map = difference_maps["$x_{t+1}^{unguided} - x_t$"]
        guided_current_map = difference_maps["$x_{t+1}^{guided} - x_t$"]
        unguided_gt_map = difference_maps["$x_{t+1}^{unguided} - x_{t+1}$"]
        guided_gt_map = difference_maps["$x_{t+1}^{guided} - x_{t+1}$"]
        guided_unguided_map = difference_maps[
            "$x_{t+1}^{guided} - x_{t+1}^{unguided}$"
        ]

        state_map = None
        next_map = None
        unguided_map = None
        guided_map = None
    return (
        guided_gt_map,
        guided_map,
        guided_unguided_map,
        next_current_map,
        next_map,
        state_map,
        unguided_gt_map,
        unguided_map,
    )


@app.cell
def _(
    analysis_type,
    analysis_type_dropdown,
    current_time,
    format_time_value,
    ground_truth_time,
    guided_gt_map,
    guided_map,
    guided_time,
    guided_unguided_map,
    level,
    m_slider,
    mo,
    n_slider,
    next_current_map,
    next_map,
    partition,
    show_mask_switch,
    show_values_checkbox,
    state_map,
    text_thresh,
    unguided_gt_map,
    unguided_map,
    unguided_time,
    var,
    zoom_slider,
):
    time_info = mo.accordion(
        {
            "Selected times": mo.md(
                "<br>".join(
                    [
                        f"current / init: `{format_time_value(current_time)}`",
                        f"ground truth target: `{format_time_value(ground_truth_time)}`",
                        f"unguided forecast: `{format_time_value(unguided_time)}`",
                        f"guided forecast: `{format_time_value(guided_time)}`",
                    ]
                )
            )
        }
    )

    common_controls = [
        analysis_type_dropdown,
        mo.hstack([n_slider, m_slider], justify="start"),
        mo.hstack(
            [
                partition,
                var,
                level
            ],
            justify="start",
        ),
        time_info,
    ]

    if analysis_type == "absolute":
        inspect_states_ui = mo.vstack(
            [
                *common_controls,
                mo.hstack([show_mask_switch], justify="start"),
                zoom_slider,
                mo.md("Absolute states:"),
                mo.hstack([state_map, next_map], justify="start"),
                mo.hstack([unguided_map, guided_map], justify="start"),
            ], justify="start",
        )

    else:
        inspect_states_ui = mo.vstack(
            [
                *common_controls,
                mo.hstack(
                    [
                        show_mask_switch,
                        show_values_checkbox,
                        text_thresh,
                    ],
                    justify="start",
                ),
                zoom_slider,
                mo.md("Difference over states:"),
                mo.hstack([next_current_map, guided_unguided_map], justify="start"),
                # mo.hstack([unguided_current_map, guided_current_map], justify="start"),
                mo.hstack([unguided_gt_map, guided_gt_map], justify="start"),
            ], justify="start",
        )

    inspect_states_ui
    return


@app.cell
def _(current_slice, guided_slice, next_slice, np, unguided_slice):
    slice_shapes = {
        "current_slice": np.asarray(current_slice).shape,
        "next_slice": np.asarray(next_slice).shape,
        "unguided_slice": np.asarray(unguided_slice).shape,
        "guided_slice": np.asarray(guided_slice).shape,
    }
    return (slice_shapes,)


@app.cell
def _(mo, slice_shapes):
    mo.accordion(
        {
            "Slice shapes": mo.md(
                "<br>".join(f"{k}: `{v}`" for k, v in slice_shapes.items())
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## State change analysis

    For each rollout step \(n = 1, \dots, N\) we measure:
    \(
    \sum_{i,j} |x^{guided}_{n,m,i,j} - x^{unguided}_{n,m,i,j}|
    \).

    Scores are min-max normalized within each member and rollout step, so
    the strongest changed channels at each step sit near 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Per variable-level change
    """)
    return


@app.cell
def _(np):
    def per_var_level_abs_sum_from_xr_diff(diff_xr):
        out = {}

        for var_name, da in diff_xr.data_vars.items():
            da_abs = abs(da)

            if "level" in da_abs.dims:
                for level_value in da_abs.level.values:
                    out[f"{var_name}-{int(level_value)}"] = float(
                        da_abs.sel(level=level_value).sum(skipna=True).item()
                    )
            else:
                out[f"{var_name}-surface"] = float(
                    da_abs.sum(skipna=True).item()
                )

        return out


    def minmax_normalize_dict(d):
        vals = list(d.values())
        vmin = min(vals)
        vmax = max(vals)

        if vmax == vmin:
            return {k: 0.0 for k in d}

        return {
            k: (v - vmin) / (vmax - vmin)
            for k, v in d.items()
        }


    def aggregate_dicts(dict_list, error="std"):
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


    def build_variable_change_scores_from_xr(
        *,
        guided_xr,
        unguided_xr,
        N,
        M,
    ):
        minmax_analysis_list = []

        for step_idx in range(int(N)):
            per_member_scores = []

            for m in range(int(M)):
                guided_step = guided_xr.isel(time=step_idx, member=m)
                unguided_step = unguided_xr.isel(time=step_idx, member=m)

                diff_step = guided_step - unguided_step

                abs_sum_dict = per_var_level_abs_sum_from_xr_diff(diff_step)
                abs_sum_dict_minmax = minmax_normalize_dict(abs_sum_dict)

                per_member_scores.append(abs_sum_dict_minmax)

            minmax_analysis_list.append(per_member_scores)

        return minmax_analysis_list

    return aggregate_dicts, build_variable_change_scores_from_xr


@app.cell
def _(mo):
    top_k_slider = mo.ui.slider(
        start=1,
        stop=50,
        value=12,
        step=1,
        label="top-k channels",
    )

    rank_by_radio = mo.ui.radio(
        options=["max", "mean"],
        value="max",
        label="rank by",
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
def _(
    M,
    N,
    build_variable_change_scores_from_xr,
    guided_xr,
    normalize_xr,
    unguided_xr,
):
    minmax_analysis_list = build_variable_change_scores_from_xr(
        guided_xr=normalize_xr(guided_xr),
        unguided_xr=normalize_xr(unguided_xr),
        N=N,
        M=M,
    )
    return (minmax_analysis_list,)


@app.cell
def _(
    aggregate_dicts,
    minmax_analysis_list,
    plot_variable_change_parallel,
    rank_by_radio,
    top_k_slider,
):
    aggregated_per_n = [
        aggregate_dicts(dict_list, error="std")
        for dict_list in minmax_analysis_list
    ]

    var_change_fig, _ = plot_variable_change_parallel(
        aggregated_per_n,
        top_k=top_k_slider.value,
        rank_by=rank_by_radio.value,
        title="Variable change across rollout steps",
        subtitle="min-max normalized |guided - unguided| per variable-level; error bars = ensemble std",
    )
    return (var_change_fig,)


@app.cell
def _(var_change_fig):
    var_change_fig
    return


@app.cell
def _(np):
    def per_var_level_abs_sum_for_rollout_norm(diff_xr):
        out = {}

        for var_name, da in diff_xr.data_vars.items():
            da_abs = abs(da)

            if "level" in da_abs.dims:
                for level_value in da_abs.level.values:
                    out[f"{var_name}-{int(level_value)}"] = float(
                        da_abs.sel(level=level_value).sum(skipna=True).item()
                    )
            else:
                out[f"{var_name}-surface"] = float(
                    da_abs.sum(skipna=True).item()
                )

        return out


    def normalize_scores_per_variable_over_rollout(raw_analysis_list):
        keys = list(raw_analysis_list[0][0].keys())

        per_key_values = {
            key: []
            for key in keys
        }

        for per_step_scores in raw_analysis_list:
            for member_scores in per_step_scores:
                for key in keys:
                    per_key_values[key].append(member_scores[key])

        per_key_minmax = {}

        for key, values in per_key_values.items():
            values = np.asarray(values, dtype=float)

            per_key_minmax[key] = {
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }

        normalized_analysis_list = []

        for per_step_scores in raw_analysis_list:
            normalized_per_step_scores = []

            for member_scores in per_step_scores:
                normalized_member_scores = {}

                for key in keys:
                    value = member_scores[key]
                    vmin = per_key_minmax[key]["min"]
                    vmax = per_key_minmax[key]["max"]

                    if vmax == vmin:
                        normalized_member_scores[key] = 0.0
                    else:
                        normalized_member_scores[key] = (value - vmin) / (vmax - vmin)

                normalized_per_step_scores.append(normalized_member_scores)

            normalized_analysis_list.append(normalized_per_step_scores)

        return normalized_analysis_list


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


    def build_rollout_normalized_variable_change_scores_from_xr(
        *,
        guided_xr,
        unguided_xr,
        N,
        M,
    ):
        raw_analysis_list = []

        for step_idx in range(int(N)):
            per_member_scores = []

            for m in range(int(M)):
                guided_step = guided_xr.isel(time=step_idx, member=m)
                unguided_step = unguided_xr.isel(time=step_idx, member=m)

                diff_step = guided_step - unguided_step

                abs_sum_dict = per_var_level_abs_sum_for_rollout_norm(diff_step)
                per_member_scores.append(abs_sum_dict)

            raw_analysis_list.append(per_member_scores)

        rollout_normalized_analysis_list = normalize_scores_per_variable_over_rollout(
            raw_analysis_list
        )

        return rollout_normalized_analysis_list

    return (
        aggregate_variable_change_dicts_over_members,
        build_rollout_normalized_variable_change_scores_from_xr,
    )


@app.cell
def _(
    M,
    N,
    aggregate_variable_change_dicts_over_members,
    build_rollout_normalized_variable_change_scores_from_xr,
    guided_xr,
    normalize_xr,
    plot_variable_change_parallel,
    rank_by_radio,
    top_k_slider,
    unguided_xr,
):
    rollout_normalized_analysis_list = build_rollout_normalized_variable_change_scores_from_xr(
        guided_xr=normalize_xr(guided_xr),
        unguided_xr=normalize_xr(unguided_xr),
        N=N,
        M=M,
    )

    aggregated_rollout_normalized_per_n = [
        aggregate_variable_change_dicts_over_members(dict_list, error="std")
        for dict_list in rollout_normalized_analysis_list
    ]

    rollout_var_change_fig, _ = plot_variable_change_parallel(
        aggregated_rollout_normalized_per_n,
        top_k=top_k_slider.value,
        rank_by=rank_by_radio.value,
        title="Variable change across rollout steps",
        subtitle=(
            "per-variable rollout min-max normalized |guided - unguided| "
            "per variable-level; error bars = ensemble std"
        ),
    )
    return (rollout_var_change_fig,)


@app.cell
def _(rollout_var_change_fig):
    rollout_var_change_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Need to change something here, the spike is just because we go from zero to that, but is not correct.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Per variable change (level-aggregated)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Do not drop surface variables only!
    """)
    return


@app.cell
def _(aggregate_dicts, minmax_analysis_list, plot_variable_change_parallel):
    def collapse_level_keys_to_vars(d):
        out = {}

        for key, val in d.items():
            var_name, suffix = key.rsplit("-", 1)

            if suffix == "surface":
                continue

            out[var_name] = out.get(var_name, 0.0) + val

        return out


    var_analysis_list = [
        [collapse_level_keys_to_vars(d) for d in dict_list]
        for dict_list in minmax_analysis_list
    ]

    aggregated_vars_per_n = [
        aggregate_dicts(dict_list, error="std")
        for dict_list in var_analysis_list
    ]

    var_agg_fig, _ = plot_variable_change_parallel(
        aggregated_vars_per_n,
        top_k=None,
        rank_by="max",
        title="Variable change across rollout steps, level-aggregated",
        subtitle="pressure-level keys summed per variable; error bars = ensemble std",
        ylim=None,
        ylabel="score, summed over levels",
    )
    return (var_agg_fig,)


@app.cell
def _(var_agg_fig):
    var_agg_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Whole state change
    """)
    return


@app.cell
def _():
    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metric change analysis
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Paste after here:
    """)
    return


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


    def normalize_xr(ds):
        out = {}
        for i, v in enumerate(surface_variables):
            if v in ds.data_vars:
                out[v] = (ds[v] - _S_MEAN[i]) / _S_STD[i]
        for i, v in enumerate(level_variables):
            if v in ds.data_vars:
                mean_da = xr.DataArray(
                    _L_MEAN[i], dims=["level"], coords={"level": pressure_levels}
                )
                std_da = xr.DataArray(
                    _L_STD[i], dims=["level"], coords={"level": pressure_levels}
                )
                out[v] = (ds[v] - mean_da) / std_da
        return xr.Dataset(out, coords=ds.coords, attrs=ds.attrs)


    def denormalize_xr(ds):
        out = {}
        for i, v in enumerate(surface_variables):
            if v in ds.data_vars:
                out[v] = ds[v] * _S_STD[i] + _S_MEAN[i]
        for i, v in enumerate(level_variables):
            if v in ds.data_vars:
                mean_da = xr.DataArray(
                    _L_MEAN[i], dims=["level"], coords={"level": pressure_levels}
                )
                std_da = xr.DataArray(
                    _L_STD[i], dims=["level"], coords={"level": pressure_levels}
                )
                out[v] = ds[v] * std_da + mean_da
        return xr.Dataset(out, coords=ds.coords, attrs=ds.attrs)

    return (normalize_xr,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
