import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np

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
        select_point,
        sweep_points,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Compare: cross checks

    Per-step metrics over $n$ at the guided channel, one line per sweep point:
    the mask-averaged gap to ground truth, the gap to each run's `ung_gui` twin
    (= realized guidance effect), and the mask-weighted gradient magnitude
    (summed over the flow).
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
    metric_dropdown = mo.ui.dropdown(
        ["gui − gt", "gui − ung_gui", "grad norm (Σ over t)"],
        value="gui − ung_gui",
        label="metric: ",
    )
    m_slider = mo.ui.slider(1, config["M"], step=1, value=1, label="m: ", show_value=True, debounce=True)
    return m_slider, metric_dropdown, points_multiselect


@app.cell
def _(
    VAR,
    channel,
    config,
    gt_states,
    m_slider,
    mask,
    masked_mean,
    metric_dropdown,
    mo,
    np,
    open_store,
    plot_trajectory,
    points,
    points_multiselect,
    rollout_dir,
    select_point,
):
    _m = m_slider.value - 1
    _gt_mm = masked_mean(
        np.asarray(channel(gt_states(config)[VAR], config).isel(time=slice(1, None)), dtype=float), mask
    )

    def _metric(label):
        """(N,) metric line for one sweep point."""
        _sel = points[label]
        if metric_dropdown.value == "grad norm (Σ over t)":
            _g = np.asarray(channel(select_point(open_store(rollout_dir, "grads", VAR), _sel), config).isel(m=_m), dtype=float)
            _w = np.asarray(mask, dtype=float)
            return np.sqrt(((_g ** 2) * _w).sum(axis=(-2, -1)) / _w.sum()).sum(axis=-1)
        _gui_mm = masked_mean(
            np.asarray(channel(select_point(open_store(rollout_dir, "gui", VAR), _sel), config).isel(m=_m), dtype=float), mask
        )
        if metric_dropdown.value == "gui − gt":
            return _gui_mm - _gt_mm
        _twin = channel(select_point(open_store(rollout_dir, "ung_gui", VAR), _sel), config)
        _twin = _twin.isel(t=-1) if "t" in _twin.dims else _twin
        return _gui_mm - masked_mean(np.asarray(_twin.isel(m=_m), dtype=float), mask)

    _lines = {_label: _metric(_label) for _label in points_multiselect.value}
    _fig = plot_trajectory(
        _lines,
        title=f"{metric_dropdown.value} over $n$",
        subtitle=f"{VAR} | mask-averaged | m={m_slider.value}",
        xlabel="$n$",
        figsize=(14, 6),
        prepend_zero=False,
        start_index=1,
    ) if _lines else mo.md("_select at least one run_")
    mo.vstack(
        [mo.hstack([points_multiselect, metric_dropdown, m_slider], justify="start", align="start"), _fig],
        align="start",
    )
    return


@app.cell
def _(
    VAR,
    channel,
    config,
    mask,
    masked_mean,
    mo,
    np,
    open_store,
    points,
    rollout_dir,
    select_point,
    sweep_values,
):
    # ===== leaderboard: landing quality of every completed run (m=1) =====
    _rows = []
    for _label, _sel in points.items():
        _gui_mm = masked_mean(
            np.asarray(channel(select_point(open_store(rollout_dir, "gui", VAR), _sel), config).isel(m=0), dtype=float), mask
        )
        _twin = channel(select_point(open_store(rollout_dir, "ung_gui", VAR), _sel), config)
        _twin = _twin.isel(t=-1) if "t" in _twin.dims else _twin
        _twin_mm = masked_mean(np.asarray(_twin.isel(m=0), dtype=float), mask)
        _delta = np.asarray(sweep_values["GUIDANCE_DELTA"][_sel["GUIDANCE_DELTA"]], dtype=float)[: len(_gui_mm)]
        _gap = _gui_mm - (1.0 + _delta) * _twin_mm
        _active = _delta != 0
        if not _active.any():
            continue
        _rows.append({
            "run": _label,
            "mean |gap| (K)": round(float(np.nanmean(np.abs(_gap[_active]))), 4),
            "final gap (K)": round(float(_gap[_active][-1]), 4),
            "mean push (K)": round(float(np.nanmean((_gui_mm - _twin_mm)[_active])), 4),
            "asked push (K)": round(float(np.nanmean((_delta * _twin_mm)[_active])), 4),
        })
    _rows.sort(key=lambda r: abs(r["mean |gap| (K)"]))
    mo.vstack(
        [
            mo.md(
                "### Landing leaderboard — gap to the $(1+\\delta)\\cdot$twin target, guided steps only (m=1). "
                "Every run of the rollout, ranked by mean |gap|."
            ),
            mo.ui.table(_rows, selection=None, pagination=False),
        ],
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
