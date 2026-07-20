import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from src.ui.comparison import (
        channel,
        default_selection,
        gt_states,
        load_rollout,
        masked_mean,
        open_store,
        select_point,
        sweep_points,
    )
    from src.ui.plot_trajectory import plot_trajectory
    from src.utils import get_rollout_ids

    return (
        channel,
        default_selection,
        get_rollout_ids,
        gt_states,
        load_rollout,
        masked_mean,
        mo,
        np,
        open_store,
        plot_trajectory,
        plt,
        select_point,
        sweep_points,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Compare: trajectories

    Mask-averaged state trajectories over forecast steps $n$, one line per sweep point
    (method x hyperparameters), with the independent unguided rollout and ground truth
    as references. All at the rollout's guided channel.
    """)
    return


@app.cell
def _(get_rollout_ids, mo):
    rollout_ids = get_rollout_ids("gui")
    rollout_dropdown = mo.ui.dropdown(rollout_ids, value=rollout_ids[0], label="rollout: ")
    rollout_dropdown
    return (rollout_dropdown,)


@app.cell
def _(load_rollout, rollout_dropdown, sweep_points):
    rollout_dir, config, sweep_values, records, mask = load_rollout(rollout_dropdown.value)
    points = sweep_points(sweep_values, records)
    VAR = config["VAR"]
    return VAR, config, mask, points, rollout_dir, sweep_values


@app.cell
def _(config, default_selection, mo, points):
    points_multiselect = mo.ui.multiselect(list(points), value=default_selection(points), label="runs: ")
    m_slider = mo.ui.slider(1, config["M"], step=1, value=1, label="m: ", show_value=True, debounce=True)
    targets_checkbox = mo.ui.checkbox(label="show (1+δ) targets", value=True)
    return m_slider, points_multiselect, targets_checkbox


@app.cell
def _(
    VAR,
    channel,
    config,
    gt_states,
    m_slider,
    mask,
    masked_mean,
    mo,
    np,
    open_store,
    plot_trajectory,
    plt,
    points,
    points_multiselect,
    rollout_dir,
    select_point,
    sweep_values,
    targets_checkbox,
):
    _m = m_slider.value - 1
    _tab10 = plt.get_cmap("tab10").colors
    _lines, _colors, _styles = {}, {}, {}
    for _i, _label in enumerate(points_multiselect.value):
        _sel = points[_label]
        _color = _tab10[_i % len(_tab10)]
        _gui = channel(select_point(open_store(rollout_dir, "gui", VAR), _sel), config)
        _lines[_label] = masked_mean(np.asarray(_gui.isel(m=_m), dtype=float), mask)
        _colors[_label] = _color
        if targets_checkbox.value:
            # per-run target trajectory: (1 + delta_n) * this run's gui_ung twin
            _twin = channel(select_point(open_store(rollout_dir, "gui_ung", VAR), _sel), config)
            _twin = _twin.isel(t=-1) if "t" in _twin.dims else _twin
            _twin_mm = masked_mean(np.asarray(_twin.isel(m=_m), dtype=float), mask)
            _delta = np.asarray(sweep_values["GUIDANCE_DELTA"][_sel["GUIDANCE_DELTA"]], dtype=float)[: len(_twin_mm)]
            _key = f"{_label} target"
            _lines[_key] = (1.0 + _delta) * _twin_mm
            _colors[_key] = _color
            _styles[_key] = ":"
    try:
        _ung = channel(open_store(rollout_dir, "ung", VAR), config)
        _lines["ung"] = masked_mean(np.asarray(_ung.isel(m=min(_m, _ung.sizes["m"] - 1)), dtype=float), mask)
        _colors["ung"] = "#888888"
    except FileNotFoundError:
        pass
    _gt = channel(gt_states(config)[VAR], config)
    _lines["gt"] = masked_mean(np.asarray(_gt.isel(time=slice(1, None)), dtype=float), mask)
    _colors["gt"] = "#111111"

    _fig = plot_trajectory(
        _lines,
        title="Mask-averaged trajectories over $n$",
        subtitle=f"{VAR} | m={m_slider.value} | dotted = (1+δ)·twin targets",
        xlabel="$n$",
        color_map=_colors,
        linestyle_map=_styles or None,
        figsize=(14, 6),
        prepend_zero=False,
        start_index=1,
    ) if points_multiselect.value else mo.md("_select at least one run_")
    mo.vstack(
        [mo.hstack([points_multiselect, m_slider, targets_checkbox], justify="start", align="start"), _fig],
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
