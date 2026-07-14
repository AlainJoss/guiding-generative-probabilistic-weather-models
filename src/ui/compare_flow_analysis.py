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
        clean_pred_trajectory,
        default_selection,
        guidance_vector,
        load_rollout,
        open_store,
        residual_scaler,
        select_point,
        sweep_points,
    )
    from src.ui.map import visualize_map
    from src.utils import get_rollout_ids

    return (
        channel,
        clean_pred_trajectory,
        default_selection,
        get_rollout_ids,
        guidance_vector,
        load_rollout,
        mo,
        np,
        open_store,
        plt,
        residual_scaler,
        select_point,
        sweep_points,
        visualize_map,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Compare: flow analysis

    Per-flow-step maps across sweep points (rows) and the temporal context
    $t-1,\ t,\ t+1$ (columns), at forecast step $n$ and the guided channel.
    Objects: raw gradients, the applied guidance vector $\lambda_t \nabla$, the
    velocity field, the noisy state $z_t$, and the reconstructed clean prediction.
    Color scale is shared within each row.
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
    VAR, LEVEL = config["VAR"], config["LEVEL"]
    return LEVEL, VAR, config, mask, points, records, rollout_dir


@app.cell
def _(config, default_selection, mo, points):
    points_multiselect = mo.ui.multiselect(list(points), value=default_selection(points), label="runs: ")
    object_dropdown = mo.ui.dropdown(
        ["grads", "gui_vec", "vfs", "res", "clean_pred"], value="gui_vec", label="object: "
    )
    m_slider = mo.ui.slider(1, config["M"], step=1, value=1, label="m: ", show_value=True, debounce=True)
    n_slider = mo.ui.slider(1, config["N"], step=1, value=min(2, config["N"]), label="n: ", show_value=True, debounce=True)
    t_slider = mo.ui.slider(1, config["T"], step=1, value=max(1, config["T"] // 2), label="t: ", show_value=True, debounce=True)
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
        t_slider,
        white_thr_slider,
        white_zero_checkbox,
    )


@app.cell
def _(
    LEVEL,
    VAR,
    channel,
    clean_pred_trajectory,
    config,
    guidance_vector,
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
    records,
    residual_scaler,
    rollout_dir,
    select_point,
    shared_scale_checkbox,
    show_mask_checkbox,
    t_slider,
    visualize_map,
    white_thr_slider,
    white_zero_checkbox,
):
    _m, _n, _t = m_slider.value - 1, n_slider.value - 1, t_slider.value - 1
    _steps = [i for i in (_t - 1, _t, _t + 1) if 0 <= i < config["T"]]
    _signed = object_dropdown.value != "clean_pred"

    def _trace(label):
        """(T, lat, lon) object for one sweep point, or None without records."""
        _sel = points[label]
        if object_dropdown.value == "gui_vec":
            return guidance_vector(rollout_dir, records, _sel, _m, _n, VAR, level=LEVEL)
        if object_dropdown.value == "clean_pred":
            _c = residual_scaler(config["PARTITION"], VAR, LEVEL)
            return clean_pred_trajectory(rollout_dir, records, _sel, _m, _n, VAR, _c, level=LEVEL)
        _da = channel(select_point(open_store(rollout_dir, object_dropdown.value, VAR), _sel), config)
        return np.asarray(_da.isel(m=_m, n=_n), dtype=float)

    _data = {}
    for _label in points_multiselect.value:
        _f = _trace(_label)
        if _f is not None and np.isfinite(np.asarray(_f)[_steps]).any():
            _data[_label] = np.asarray(_f)

    def _limits(vals):
        if _signed:
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
            if white_zero_checkbox.value and _signed:
                _thr = white_thr_slider.value / 100.0 * float(np.nanmax(np.abs(_f[_steps])))
                _fld = np.where(np.abs(_fld) <= _thr, np.nan, _fld)
                _cmap = _white_cmap
            _maps.append(visualize_map(
                _fld, cmap=_cmap, mask_2d=mask, show_mask=show_mask_checkbox.value,
                title=f"{label} | {object_dropdown.value} | t={_i + 1}",
                vmin=_vmin, vmax=_vmax, center=_center, figsize=(9, 5.5), dpi=100,
            )[0])
        return mo.hstack(_maps, justify="start")

    _missing = [_l for _l in points_multiselect.value if _l not in _data]
    mo.vstack(
        [
            mo.hstack([points_multiselect, object_dropdown], justify="start", align="start"),
            mo.hstack([m_slider, n_slider, t_slider, show_mask_checkbox, shared_scale_checkbox, white_zero_checkbox, white_thr_slider], justify="start"),
            *[_row(_label) for _label in _data],
            *([mo.md(f"_no trace at this selection: {', '.join(_missing)}_")] if _missing else []),
        ],
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
