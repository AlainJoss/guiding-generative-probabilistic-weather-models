import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from src.spectrum import log_spectral_distance, power_spectrum, spectral_bias
    from src.ui.comparison import (
        channel,
        clean_pred_trajectory,
        default_selection,
        gt_states,
        load_rollout,
        open_store,
        residual_scaler,
        select_point,
        sweep_points,
    )
    from src.ui.plot_trajectory import plot_trajectory
    from src.utils import get_rollout_ids

    return (
        channel,
        clean_pred_trajectory,
        default_selection,
        get_rollout_ids,
        gt_states,
        load_rollout,
        log_spectral_distance,
        mo,
        np,
        open_store,
        plot_trajectory,
        plt,
        power_spectrum,
        residual_scaler,
        select_point,
        spectral_bias,
        sweep_points,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Physical realism

    Spherical-harmonic spectral realism of one sweep point, at the rollout's guided channel.
    Per rollout step $n$ the power spectra of each source (ung / ung_gui / gui) are compared
    to ground truth; over flow $t$ the guided clean-pred trajectory is compared to gt at step
    $n$. The bottom row overlays the spatial value distribution of every source at $(m, n)$.
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
    VAR, LEVEL, PARTITION = config["VAR"], config["LEVEL"], config["PARTITION"]
    N, T = config["N"], config["T"]
    return LEVEL, N, PARTITION, T, VAR, config, points, records, rollout_dir


@app.cell
def _(N, T, config, default_selection, mo, points):
    run_dropdown = mo.ui.dropdown(list(points), value=default_selection(points)[0], label="run: ")
    m_slider = mo.ui.slider(1, config["M"], step=1, value=1, label="m: ", show_value=True, debounce=True)
    n_slider = mo.ui.slider(1, N, step=1, value=min(2, N), label="n: ", show_value=True, debounce=True)
    t_slider = mo.ui.slider(1, T, step=1, value=max(1, T // 2), label="t: ", show_value=True, debounce=True)
    pr_row_checks = mo.ui.dictionary({k: mo.ui.checkbox(label=k, value=True) for k in (
        "spectral distance", "spectral bias", "total power", "distributional skill",
    )})
    return m_slider, n_slider, pr_row_checks, run_dropdown, t_slider


@app.cell
def _(
    LEVEL,
    N,
    PARTITION,
    T,
    VAR,
    channel,
    clean_pred_trajectory,
    config,
    gt_states,
    log_spectral_distance,
    m_slider,
    n_slider,
    np,
    open_store,
    points,
    power_spectrum,
    records,
    residual_scaler,
    rollout_dir,
    run_dropdown,
    select_point,
    spectral_bias,
):
    # ===== per-source spectral metric traces (LSD / bias / power) =====
    _sel = points[run_dropdown.value]
    _m, _n = m_slider.value - 1, n_slider.value - 1

    def _src(store):
        return np.asarray(channel(select_point(open_store(rollout_dir, store, VAR), _sel), config).isel(m=_m), dtype=float)

    _gt_da = channel(gt_states(config)[VAR], config)
    _lat = _gt_da.latitude.values
    _gt = np.asarray(_gt_da.isel(time=slice(1, None)), dtype=float)  # (N, lat, lon); [ni] valid day ni+1
    _ung = _src("ung")
    _gui = _src("gui")
    _twin = channel(select_point(open_store(rollout_dir, "ung_gui", VAR), _sel), config)
    _twin = _twin.isel(t=-1) if "t" in _twin.dims else _twin
    _ung_gui = np.asarray(_twin.isel(m=_m), dtype=float)

    def _spec(_f):
        return power_spectrum(np.asarray(_f), _lat)[1]

    _gt_spec = {_ni: _spec(_gt[_ni]) for _ni in range(N)}
    _src_slices = {"ung": _ung, "ung_gui": _ung_gui, "gui": _gui}
    pr_n = {"lsd": {}, "bias": {}, "power": {}}
    for _k, _arr in _src_slices.items():
        _specs = [_spec(_arr[_ni]) for _ni in range(N)]
        pr_n["lsd"][_k] = [log_spectral_distance(_specs[_ni], _gt_spec[_ni]) for _ni in range(N)]
        pr_n["bias"][_k] = [spectral_bias(_specs[_ni], _gt_spec[_ni]) for _ni in range(N)]
        pr_n["power"][_k] = [float(np.sum(_specs[_ni][1:])) for _ni in range(N)]
    pr_n["power"]["gt"] = [float(np.sum(_gt_spec[_ni][1:])) for _ni in range(N)]

    # over flow t: guided clean-pred trajectory (reconstructed) vs gt at step n
    _clean = clean_pred_trajectory(
        rollout_dir, records, _sel, _m, _n, VAR, residual_scaler(PARTITION, VAR, LEVEL), level=LEVEL
    )
    if _clean is not None:
        _clean_specs = [_spec(_clean[_ti]) for _ti in range(T)]
        pr_t = {
            "lsd": {"gui": [log_spectral_distance(_clean_specs[_ti], _gt_spec[_n]) for _ti in range(T)]},
            "bias": {"gui": [spectral_bias(_clean_specs[_ti], _gt_spec[_n]) for _ti in range(T)]},
            "power": {"gui": [float(np.sum(_clean_specs[_ti][1:])) for _ti in range(T)],
                      "gt": [float(np.sum(_gt_spec[_n][1:]))] * T},
        }
    else:
        pr_t = {"lsd": {}, "bias": {}, "power": {}}  # no trace -> plots show a placeholder

    # distributional skill: spatial value distribution at (m, n), all sources
    pr_dist = {
        "gt": _gt[_n].ravel(),
        "ung": _ung[_n].ravel(),
        "ung_gui": _ung_gui[_n].ravel(),
        "gui": _gui[_n].ravel(),
    }
    return pr_dist, pr_n, pr_t


@app.cell
def _(
    LEVEL,
    N,
    PARTITION,
    T,
    VAR,
    m_slider,
    mo,
    n_slider,
    np,
    plot_trajectory,
    plt,
    pr_dist,
    pr_n,
    pr_row_checks,
    pr_t,
    run_dropdown,
    t_slider,
):
    # ===== spectral-realism plots, cross-var-check style =====
    _m, _n, _t = m_slider.value - 1, n_slider.value - 1, t_slider.value - 1
    _src_colors = {"gt": "#222222", "ung": "#1f77b4", "ung_gui": "#2ca02c", "gui": "#d62728"}
    _wn = min(22.0, max(8.0, 3.4 + 0.78 * N))
    _wt = min(22.0, max(8.0, 3.4 + 0.78 * T))

    def _pr_plot(_tr, _title, _sub, _axis):
        if not any(np.isfinite(np.asarray(_v, dtype=float)).any() for _v in _tr.values()):
            return mo.md(f"_{_title}: no finite data for this selection_")
        return plot_trajectory(
            _tr, title=_title, subtitle=_sub, xlabel=f"${_axis}$",
            step=(_n + 1 if _axis == "n" else _t + 1),
            color_map={_k: _src_colors[_k] for _k in _tr},
            figsize=((_wn if _axis == "n" else _wt), 6),
            prepend_zero=(_axis == "n"), start_index=(1 if _axis == "t" else 0),
            mirror_right_axis=True,
        )

    def _dist_plot(dist):
        _order = [s for s in ("gt", "ung", "ung_gui", "gui") if s in dist]
        _fin = {s: dist[s][np.isfinite(dist[s])] for s in _order}
        _present = [s for s in _order if _fin[s].size]
        _lvl = f" @ {LEVEL} hPa" if PARTITION == "level" else ""
        with plt.rc_context({"font.size": 10, "axes.titlesize": 14, "legend.fontsize": 9}):
            _fig, _ax = plt.subplots(figsize=(_wn, 6), dpi=130)
            if _present:
                _allv = np.concatenate([_fin[s] for s in _present])
                _bins = np.linspace(float(_allv.min()), float(_allv.max()), 41)
                for s in _present:
                    _ax.hist(_fin[s], bins=_bins, density=True, histtype="step",
                             linewidth=1.9, color=_src_colors[s], label=s, zorder=3)
            _ax.set_xlabel(f"{VAR}{_lvl} value")
            _ax.set_ylabel("density")
            _ax.set_title(f"Value distribution — {VAR}{_lvl}  (m={m_slider.value}, n={n_slider.value})",
                          loc="left", fontweight="bold")
            for _sp in ("top", "right"):
                _ax.spines[_sp].set_visible(False)
            _ax.legend(frameon=False, loc="upper right")
            _fig.tight_layout()
        return _fig

    _pr_rows = {
        "spectral distance": [
            _pr_plot(pr_n["lsd"], r"Log-spectral distance (LSD) over $n$",
                     r"$\mathrm{LSD}_n=\sqrt{\frac{1}{L}\sum_{\ell\geq1}\left(\ln\mathrm{PS}_n(\ell)-\ln\mathrm{PS}^{\mathrm{gt}}_n(\ell)\right)^2}$", "n"),
            _pr_plot(pr_t["lsd"], r"Log-spectral distance (LSD) over $t$",
                     r"$\mathrm{LSD}_t=\sqrt{\frac{1}{L}\sum_{\ell\geq1}\left(\ln\mathrm{PS}_t(\ell)-\ln\mathrm{PS}^{\mathrm{gt}}_n(\ell)\right)^2}$", "t"),
        ],
        "spectral bias": [
            _pr_plot(pr_n["bias"], r"Spectral bias over $n$",
                     r"$\mathrm{bias}_n=\frac{1}{L}\sum_{\ell\geq1}\ln\frac{\mathrm{PS}_n(\ell)}{\mathrm{PS}^{\mathrm{gt}}_n(\ell)}$", "n"),
            _pr_plot(pr_t["bias"], r"Spectral bias over $t$",
                     r"$\mathrm{bias}_t=\frac{1}{L}\sum_{\ell\geq1}\ln\frac{\mathrm{PS}_t(\ell)}{\mathrm{PS}^{\mathrm{gt}}_n(\ell)}$", "t"),
        ],
        "total power": [
            _pr_plot(pr_n["power"], r"Total power over $n$",
                     r"$P_n=\sum_{\ell\geq1}\mathrm{PS}_n(\ell)$", "n"),
            _pr_plot(pr_t["power"], r"Total power over $t$",
                     r"$P_t=\sum_{\ell\geq1}\mathrm{PS}_t(\ell)$", "t"),
        ],
        "distributional skill": [
            _dist_plot(pr_dist),
        ],
    }

    mo.vstack([
        mo.hstack([run_dropdown, m_slider, n_slider, t_slider], justify="start", align="start"),
        mo.hstack([
            mo.vstack(list(pr_row_checks.values()), justify="start", align="start").style(width="fit-content"),
            mo.vstack(
                [mo.hstack(_plots, justify="start") for _key, _plots in _pr_rows.items() if pr_row_checks[_key].value],
                align="start",
            ).style(width="fit-content"),
        ], align="start", justify="start"),
    ], align="start")
    return


if __name__ == "__main__":
    app.run()
