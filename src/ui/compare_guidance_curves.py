import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    from src.ui.comparison import (
        applied_lambda,
        clean_pred_trajectory,
        default_selection,
        flow_grids,
        load_rollout,
        masked_mean,
        open_store,
        residual_scaler,
        select_point,
        sweep_points,
    )
    from src.ui.plot_trajectory import plot_trajectory
    from src.utils import get_rollout_ids

    return (
        applied_lambda,
        clean_pred_trajectory,
        default_selection,
        flow_grids,
        get_rollout_ids,
        load_rollout,
        masked_mean,
        mo,
        np,
        open_store,
        plot_trajectory,
        plt,
        residual_scaler,
        select_point,
        sweep_points,
        xr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Compare: realized guidance curves

    How does each run **close the gap over the flow steps $t$** — and what does that cost?

    The curve of one guided step $n$ is the remaining-gap fraction
    $\rho_t = r_t / r_0$, with $r_t = \mathrm{mean}_{\mathrm{mask}}(\hat{x}_t) - (1+\delta_n)\cdot\mathrm{mean}_{\mathrm{mask}}(x^{gui\_ung}_n)$
    measured on the clean prediction *before* each step ($\rho_0 = 1$, $0$ = target hit).
    Dashed: the run's own prescribed schedule (recorded $a_t$), where one exists.

    1. **Across runs** — all selected runs at one $n$: who closes faster, who overshoots.
    2. **Across $n$** — one panel per run, all guided steps: is the shape stable over the forecast?
    3. **Metrics table** — landing, effort (*less push is better*), pushback, shape statistics.
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
    VAR, LEVEL, N_STEPS = config["VAR"], config["LEVEL"], config["N"]
    return (
        LEVEL,
        N_STEPS,
        VAR,
        config,
        mask,
        points,
        records,
        rollout_dir,
        sweep_values,
    )


@app.cell
def _(config, default_selection, mo, points):
    points_multiselect = mo.ui.multiselect(list(points), value=default_selection(points), label="runs: ")
    m_slider = mo.ui.slider(1, config["M"], step=1, value=1, label="m: ", show_value=True, debounce=True)
    n_slider = mo.ui.slider(1, config["N"], step=1, value=min(2, config["N"]), label="n: ", show_value=True, debounce=True)

    TRAJ_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]

    def run_colors(labels):
        order = list(points)
        return {lab: TRAJ_COLORS[order.index(lab) % len(TRAJ_COLORS)] for lab in labels}

    return m_slider, n_slider, points_multiselect, run_colors


@app.cell
def _(
    LEVEL,
    N_STEPS,
    VAR,
    applied_lambda,
    clean_pred_trajectory,
    config,
    flow_grids,
    m_slider,
    mask,
    masked_mean,
    np,
    open_store,
    points,
    records,
    residual_scaler,
    rollout_dir,
    select_point,
    sweep_values,
    xr,
):
    # ===== curves: one pass per (run, guided n) at the selected member =====
    # pre_t  : measured gap before step t (clean pred, K)
    # post_t : pre_t + first-order through-Jacobian claim of the kick at t
    # effort : ||applied kick||_z per step = h_t * lambda_t * ||g_t||  (z-space, full domain)
    # a_t    : the run's recorded schedule (target remaining fraction), if any
    _m = m_slider.value - 1
    RESID_SCALER = residual_scaler(config["PARTITION"], VAR, LEVEL)
    _mask_sum = float(np.asarray(mask).sum())

    def _recorded_a_t(sel, n):
        for r in records:
            if r["m"] == _m and r["n"] == n and all(
                r["sweep"].get(k) == v for k, v in sel.items() if k in r["sweep"]
            ):
                return np.asarray(r["a_t"], dtype=float)
        return None

    def _grads_norm2_full(sel, n):
        gds = xr.open_zarr(rollout_dir / "grads.zarr")
        g2 = None
        for v in gds.data_vars:
            da = select_point(gds[v], sel)
            term = (da ** 2).sum(dim=[d for d in da.dims if d not in ("m", "n", "t")])
            g2 = term if g2 is None else g2 + term
        return np.asarray(g2.isel(m=_m, n=n), dtype=float)

    def _twin_final_mm(sel):
        tw = select_point(open_store(rollout_dir, "gui_ung", VAR), sel)
        tw = tw.sel(level=LEVEL) if "level" in tw.dims else tw
        tw = tw.isel(t=-1) if "t" in tw.dims else tw
        return masked_mean(np.asarray(tw.isel(m=_m), dtype=float), mask)

    def _gui_final_mm(sel):
        da = select_point(open_store(rollout_dir, "gui", VAR), sel)
        da = da.sel(level=LEVEL) if "level" in da.dims else da
        return masked_mean(np.asarray(da.isel(m=_m), dtype=float), mask)

    run_curves = {}
    for _label, _sel in points.items():
        _delta = np.asarray(sweep_values["GUIDANCE_DELTA"][_sel["GUIDANCE_DELTA"]], dtype=float)[:N_STEPS]
        _twin_mm, _gui_mm = _twin_final_mm(_sel), _gui_final_mm(_sel)
        _per_n = {}
        for _ni in range(N_STEPS):
            if _delta[_ni] == 0.0:
                continue  # unguided step: no trace, trivially on target
            _lam = applied_lambda(records, _sel, _m, _ni)
            _traj = clean_pred_trajectory(rollout_dir, records, _sel, _m, _ni, VAR, RESID_SCALER, level=LEVEL)
            if _lam is None or _traj is None or not np.isfinite(_traj).any():
                continue
            _target = (1.0 + _delta[_ni]) * _twin_mm[_ni]
            _pre = masked_mean(_traj, mask) - _target
            _g2 = _grads_norm2_full(_sel, _ni)
            _s, _h = flow_grids(len(_pre))
            _r_sum = _pre * _mask_sum
            _dS = np.where(np.abs(_r_sum) > 1e-12, -_h * _lam * _g2 / (2.0 * _r_sum), 0.0)
            _per_n[_ni] = {
                "pre": _pre,
                "post": _pre + _dS,
                "effort": _h * _lam * np.sqrt(_g2),
                "a_t": _recorded_a_t(_sel, _ni),
                "final_gap": float(_gui_mm[_ni] - _target),
            }
        if _per_n:
            run_curves[_label] = _per_n
    return (run_curves,)


@app.cell
def _(
    m_slider,
    mo,
    n_slider,
    np,
    plot_trajectory,
    points_multiselect,
    run_colors,
    run_curves,
):
    # ===== 1. shapes across runs, at one n =====
    _ni = n_slider.value - 1
    _lines, _styles = {}, {}
    _colors = run_colors(points_multiselect.value)
    for _label in points_multiselect.value:
        _c = run_curves.get(_label, {}).get(_ni)
        if _c is None:
            continue
        _r0 = _c["pre"][0]
        if abs(_r0) < 1e-12:
            continue
        _lines[_label] = _c["pre"] / _r0
        if _c["a_t"] is not None:
            # prescribed remaining fraction BEFORE step t: 1 at t=1, then a_{t-1}
            _sched = np.concatenate([[1.0], _c["a_t"][:-1]])
            _lines[f"{_label} (schedule)"] = _sched
            _styles[f"{_label} (schedule)"] = ":"
            _colors[f"{_label} (schedule)"] = _colors[_label]
    mo.vstack(
        [
            mo.md(
                "## 1 · Shapes across runs\n\nRemaining-gap fraction $\\rho_t = r_t/r_0$ at the selected $n$ "
                "(solid: measured; dotted: the run's prescribed schedule). Below 0 = overshoot."
            ),
            mo.hstack([points_multiselect, m_slider, n_slider], justify="start", align="start"),
            plot_trajectory(
                _lines,
                title="Remaining gap fraction over $t$",
                subtitle=r"$\rho_t = r_t / r_0$ (clean-pred gap before step $t$)" + f"  |  n={n_slider.value}, m={m_slider.value}",
                xlabel="$t$", color_map=_colors, linestyle_map=_styles or None,
                figsize=(13, 5.5), prepend_zero=False, start_index=1,
            ) if _lines else mo.md("_no guided traces at this selection_"),
        ],
        align="start",
    )
    return


@app.cell
def _(m_slider, mo, np, plt, points_multiselect, run_curves):
    # ===== 2. shapes across n, per run =====
    _figs = []
    for _label in points_multiselect.value:
        _per_n = run_curves.get(_label, {})
        _valid = {ni: c for ni, c in _per_n.items() if abs(c["pre"][0]) > 1e-12}
        if not _valid:
            _figs.append(mo.md(f"_{_label}: no guided traces_"))
            continue
        _cmap = plt.get_cmap("viridis")
        with plt.rc_context({"font.size": 9, "axes.titlesize": 11}):
            _fig, _ax = plt.subplots(figsize=(6.0, 4.2), dpi=110)
            for _j, (_ni, _c) in enumerate(sorted(_valid.items())):
                _rho = _c["pre"] / _c["pre"][0]
                _ax.plot(np.arange(1, len(_rho) + 1), _rho,
                         color=_cmap(_j / max(len(_valid) - 1, 1)), linewidth=1.6, alpha=0.9,
                         label=f"n={_ni + 1}")
            _ax.axhline(0.0, color="#888888", linewidth=0.9, alpha=0.8)
            _ax.set_title(_label, loc="left", fontweight="bold")
            _ax.set_xlabel("$t$"); _ax.set_ylabel(r"$\rho_t$")
            for _sp in ("top", "right"):
                _ax.spines[_sp].set_visible(False)
            _ax.yaxis.grid(True, color="#D7D7D7", linewidth=0.6, alpha=0.5)
            _ax.legend(fontsize=8, frameon=False)
            _fig.tight_layout()
        _figs.append(_fig)
    mo.vstack(
        [
            mo.md(
                "## 2 · Shapes across $n$\n\nOne panel per run, one curve per guided forecast step "
                "(dark → light = early → late $n$). Overlapping curves = the closure shape is stable "
                "along the forecast; spreading curves = the method behaves differently as errors accumulate."
                f"  (m={m_slider.value})"
            ),
            mo.hstack(_figs, justify="start", align="start", wrap=True),
        ],
        align="start",
    )
    return


@app.cell
def _(m_slider, mo, np, points, run_curves):
    # ===== 3. metrics =====
    # per (run, n), then averaged over the guided n's:
    #   final |gap|  : realized landing distance (gui store, K)          -> smaller better
    #   total push   : sum_t ||applied kick||_z = sum h_t lam_t ||g_t||  -> smaller better
    #   pushback     : sum_t max(0, |pre_{t+1}| - |post_t|)  (K) — model undoing the claim
    #   pushback %   : pushback / sum_t |post_t - pre_t|  — undone fraction of claimed work
    #   closure AUC  : mean_t |rho_t| — area under the remaining-gap curve -> faster close = smaller
    #   monotone %   : fraction of steps with |pre| strictly decreasing
    #   sched RMSE   : rho vs the recorded a_t path (eta methods only)
    def _metrics_one(c):
        pre, post = c["pre"], c["post"]
        r0 = pre[0]
        if abs(r0) < 1e-12:
            return None
        rho = pre / r0
        claimed = np.abs(post - pre).sum()
        pushback = np.maximum(0.0, np.abs(pre[1:]) - np.abs(post[:-1])).sum()
        out = {
            "final |gap| (K)": abs(c["final_gap"]),
            "total push (z)": float(c["effort"].sum()),
            "pushback (K)": float(pushback),
            "pushback %": float(100.0 * pushback / claimed) if claimed > 1e-12 else None,
            "closure AUC": float(np.mean(np.abs(rho))),
            "monotone %": float(100.0 * np.mean(np.abs(pre[1:]) < np.abs(pre[:-1]))),
        }
        if c["a_t"] is not None:
            sched = np.concatenate([[1.0], c["a_t"][:-1]])
            out["sched RMSE"] = float(np.sqrt(np.mean((rho - sched) ** 2)))
        else:
            out["sched RMSE"] = None
        return out

    _cols = [
        ("final |gap| (K)", "min", 4), ("total push (z)", "min", 1), ("pushback (K)", "min", 4),
        ("pushback %", "min", 1), ("closure AUC", "min", 3), ("monotone %", "max", 1),
        ("sched RMSE", "min", 3),
    ]
    _rows = []
    for _label in points:
        _per_n = run_curves.get(_label, {})
        _ms = [m for m in (_metrics_one(c) for c in _per_n.values()) if m is not None]
        if not _ms:
            continue
        _row = {"run": _label, "n's": len(_ms)}
        for _c, _mode, _nd in _cols:
            _vals = [m[_c] for m in _ms if m[_c] is not None]
            _row[_c] = round(float(np.mean(_vals)), _nd) if _vals else None
        _rows.append(_row)
    _rows.sort(key=lambda r: (r["final |gap| (K)"] is None, r["final |gap| (K)"]))
    _best = {}
    for _c, _mode, _nd in _cols:
        _vals = [_r[_c] for _r in _rows if _r[_c] is not None]
        if _vals:
            _best[_c] = min(_vals) if _mode == "min" else max(_vals)
    _hdr = ["run", "n's"] + [_c + (" ↓" if _mode == "min" else " ↑") for _c, _mode, _nd in _cols]
    _md = ["| " + " | ".join(_hdr) + " |", "|" + "---|" * len(_hdr)]
    for _r in _rows:
        _cells = [_r["run"], str(_r["n's"])]
        for _c, _mode, _nd in _cols:
            _s = "—" if _r[_c] is None else f"{_r[_c]}"
            if _r[_c] is not None and _r[_c] == _best.get(_c):
                _s = f"**{_s}**"
            _cells.append(_s)
        _md.append("| " + " | ".join(_cells) + " |")
    mo.vstack(
        [
            mo.md(
                "## 3 · Metrics\n\nMean over the guided $n$'s (m={}), ranked by final |gap|; best per column **bold**.\n\n"
                "`final |gap|`: realized landing distance (K). "
                "`total push`: $\\Sigma_t\\, h_t \\lambda_t \\lVert g_t \\rVert$ — total applied kick in latent space; "
                "*for the same landing, less push is better*. "
                "`pushback`: how much of the claimed per-step progress the model undid before the next step "
                "($\\Sigma_t \\max(0, |r_{{t+1}}| - |r^{{post}}_t|)$, K), also as % of the total claimed progress. "
                "`closure AUC`: mean $|\\rho_t|$ — small = the gap closes early. "
                "`monotone %`: steps where the measured gap actually shrank. "
                "`sched RMSE`: distance of the realized $\\rho_t$ from the run's own prescribed schedule."
                .format(m_slider.value)
            ),
            mo.md("\n".join(_md)) if _rows else mo.md("_no guided traces in this rollout_"),
        ],
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
