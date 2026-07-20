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
        open_store,
        select_point,
        sweep_points,
    )
    from src.ui.map import visualize_map
    from src.utils import get_rollout_ids

    return (
        channel,
        default_selection,
        get_rollout_ids,
        gt_states,
        load_rollout,
        mo,
        np,
        open_store,
        plt,
        select_point,
        sweep_points,
        visualize_map,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Compare: inspect states

    State maps across sweep points (rows) and the temporal context $n-1,\ n,\ n+1$
    (columns), at the rollout's guided channel. The object dropdown switches between the
    guided state and its differences to ground truth / the `gui_ung` twin. Color scale is
    shared within each row.
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
    return VAR, config, mask, points, rollout_dir


@app.cell
def _(config, default_selection, mo, points):
    points_multiselect = mo.ui.multiselect(list(points), value=default_selection(points), label="runs: ")
    object_dropdown = mo.ui.dropdown(
        ["gui", "gui − gt", "gui − gui_ung"], value="gui − gui_ung", label="object: "
    )
    m_slider = mo.ui.slider(1, config["M"], step=1, value=1, label="m: ", show_value=True, debounce=True)
    n_slider = mo.ui.slider(1, config["N"], step=1, value=min(2, config["N"]), label="n: ", show_value=True, debounce=True)
    show_mask_checkbox = mo.ui.checkbox(label="show mask", value=True)
    shared_scale_checkbox = mo.ui.checkbox(label="share scale across runs", value=True)
    white_zero_checkbox = mo.ui.checkbox(label="white zeros")
    white_thr_slider = mo.ui.slider(start=0.0, stop=20.0, step=0.5, value=5.0, label="white below (% of max |v|): ", show_value=True, debounce=True)
    return (
        m_slider,
        n_slider,
        object_dropdown,
        points_multiselect,
        shared_scale_checkbox,
        show_mask_checkbox,
        white_thr_slider,
        white_zero_checkbox,
    )


@app.cell
def _(
    VAR,
    channel,
    config,
    gt_states,
    m_slider,
    mask,
    mo,
    n_slider,
    np,
    object_dropdown,
    open_store,
    plt,
    points,
    points_multiselect,
    rollout_dir,
    select_point,
    shared_scale_checkbox,
    show_mask_checkbox,
    visualize_map,
    white_thr_slider,
    white_zero_checkbox,
):
    _m, _n = m_slider.value - 1, n_slider.value - 1
    _steps = [i for i in (_n - 1, _n, _n + 1) if 0 <= i < config["N"]]
    _is_diff = "−" in object_dropdown.value
    _gt = np.asarray(channel(gt_states(config)[VAR], config).isel(time=slice(1, None)), dtype=float)

    def _fields(label):
        """(N, lat, lon) object for one sweep point."""
        _sel = points[label]
        _gui = np.asarray(channel(select_point(open_store(rollout_dir, "gui", VAR), _sel), config).isel(m=_m), dtype=float)
        if object_dropdown.value == "gui − gt":
            return _gui - _gt
        if object_dropdown.value == "gui − gui_ung":
            _twin = channel(select_point(open_store(rollout_dir, "gui_ung", VAR), _sel), config)
            _twin = _twin.isel(t=-1) if "t" in _twin.dims else _twin
            return _gui - np.asarray(_twin.isel(m=_m), dtype=float)
        return _gui

    _data = {_label: _fields(_label) for _label in points_multiselect.value}

    def _limits(vals):
        if _is_diff:
            _vmax = float(np.nanmax(np.abs(vals))) or 1.0
            return -_vmax, _vmax, 0
        return float(np.nanmin(vals)), float(np.nanmax(vals)), float(np.nanmean(vals))

    _shared = _limits(np.stack([_f[_steps] for _f in _data.values()])) if (_data and shared_scale_checkbox.value) else None
    _white_cmap = plt.get_cmap("coolwarm").copy()
    _white_cmap.set_bad("white")

    def _row(label):
        _f = _data[label]
        _vmin, _vmax, _center = _shared if _shared else _limits(_f[_steps])
        _maps = []
        for _i in _steps:
            _fld, _cmap = _f[_i], "coolwarm"
            if white_zero_checkbox.value and _is_diff:
                _thr = white_thr_slider.value / 100.0 * float(np.nanmax(np.abs(_f[_steps])))
                _fld = np.where(np.abs(_fld) <= _thr, np.nan, _fld)
                _cmap = _white_cmap
            _maps.append(visualize_map(
                _fld, cmap=_cmap, mask_2d=mask, show_mask=show_mask_checkbox.value,
                title=f"{label} | {object_dropdown.value} | n={_i + 1}",
                vmin=_vmin, vmax=_vmax, center=_center, figsize=(9, 5.5), dpi=100,
            )[0])
        return mo.hstack(_maps, justify="start")

    mo.vstack(
        [
            mo.hstack([points_multiselect, object_dropdown], justify="start", align="start"),
            mo.hstack([m_slider, n_slider, show_mask_checkbox, shared_scale_checkbox, white_zero_checkbox, white_thr_slider], justify="start"),
            *[_row(_label) for _label in _data],
        ],
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
