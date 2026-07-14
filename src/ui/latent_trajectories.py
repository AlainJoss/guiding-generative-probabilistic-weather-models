import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    from datetime import datetime, timedelta
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    from src.ui.comparison import (
        applied_lambda,
        channel,
        clean_pred_trajectory,
        default_selection,
        flow_grids,
        gt_states,
        load_rollout,
        masked_mean,
        open_store,
        residual_scaler,
        select_point,
        sweep_points,
    )
    from src.ui.plot_trajectory import plot_trajectory
    from src.rollout_config import GUIDANCE_METHOD_HYPERS
    from src.utils import get_gt_rollout, get_rollout_ids

    return (
        GUIDANCE_METHOD_HYPERS,
        Path,
        applied_lambda,
        channel,
        clean_pred_trajectory,
        datetime,
        flow_grids,
        get_gt_rollout,
        get_rollout_ids,
        gt_states,
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
        timedelta,
        xr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Latent trajectories (PCA) — comparison lab

    3D PCA latent-trajectory analysis (arXiv:2605.14317-style), now across **runs**
    (methods × hyperparameters), **forecast steps $n$** and **members $m$**, on any rollout.

    **Solid curves**: guided clean-pred trajectories $\hat{x}_t$ (mask-bbox pixels, flattened).
    **Dashed curves**: each run's own `ung_gui` twin (same seed & conditioning — coincides at
    $t=0$). **Gray cloud**: daily ERA5 states in the bbox, the fixed PCA frame. **○** = the
    independent unguided rollout's final state at $n$; **★** = GT at the valid day.
    The table at the bottom scores every run at every $n$ in latent space.
    """)
    return


@app.cell
def _(get_rollout_ids, mo):
    rollout_ids = get_rollout_ids("gui")
    rollout_dropdown = mo.ui.dropdown(rollout_ids, value=rollout_ids[0], label="rollout: ")
    rollout_dropdown
    return (rollout_dropdown,)


@app.cell
def _(
    GUIDANCE_METHOD_HYPERS,
    applied_lambda,
    channel,
    clean_pred_trajectory,
    datetime,
    flow_grids,
    get_gt_rollout,
    gt_states,
    load_rollout,
    masked_mean,
    np,
    open_store,
    residual_scaler,
    rollout_dropdown,
    select_point,
    sweep_points,
    timedelta,
    xr,
):
    # ===== rollout, latent helpers, reference cloud, PCA frame =====
    rollout_dir, config, sweep_values, records, mask = load_rollout(rollout_dropdown.value)
    points = sweep_points(sweep_values, records)
    VAR, LEVEL, N_STEPS = config["VAR"], config["LEVEL"], config["N"]
    ROLLOUT_START = datetime.fromisoformat(config["START_TS"])
    RESID_SCALER = residual_scaler(config["PARTITION"], VAR, LEVEL)
    TRAJ_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]

    # ---------- latent construction ----------
    def mask_bbox(mask_2d, rel_threshold=0.5):
        """Row/col slices of the mask footprint at rel_threshold * max."""
        rows, cols = np.where(np.asarray(mask_2d) >= rel_threshold * float(np.asarray(mask_2d).max()))
        return slice(int(rows.min()), int(rows.max()) + 1), slice(int(cols.min()), int(cols.max()) + 1)

    BBOX = mask_bbox(mask)

    def bbox_latent(field):
        """Flatten the bbox region of (..., lat, lon) fields into feature vectors."""
        field = np.asarray(field, dtype=float)
        return field[..., BBOX[0], BBOX[1]].reshape(*field.shape[:-2], -1)

    def gt_reference_latents(days_back):
        """Daily GT states ending just before the rollout start (shrinks if the store is short)."""
        err = None
        for days in (days_back, days_back // 2, days_back // 4, 7):
            try:
                da = channel(get_gt_rollout(days, ROLLOUT_START - timedelta(days=days + 2))[VAR], config)
                return bbox_latent(da.values)
            except Exception as e:
                err = e
        raise RuntimeError(f"no GT cloud loadable: {err}")

    def pca_frame(reference, n_components=3):
        """PCA basis of `reference` rows; returns (project, explained_variance_ratio)."""
        mu = reference.mean(axis=0)
        _, sv, vt = np.linalg.svd(reference - mu, full_matrices=False)
        basis = vt[:n_components].T
        return (lambda x: (x - mu) @ basis), (sv ** 2 / (sv ** 2).sum())[:n_components]

    pca_cloud = gt_reference_latents(60)
    pca_project, pca_evr = pca_frame(pca_cloud)
    pca_cloud_proj = pca_project(pca_cloud)
    pca_targets = bbox_latent(np.asarray(channel(gt_states(config)[VAR], config).isel(time=slice(1, None)), dtype=float))
    try:
        pca_ung_finals = bbox_latent(np.asarray(channel(open_store(rollout_dir, "ung", VAR), config).isel(m=0), dtype=float))
    except FileNotFoundError:
        pca_ung_finals = None

    # ---------- per-run data access ----------
    def guided_latents(sel, m, n):
        """Guided clean-pred trajectory as bbox latents (T, F), or None without records."""
        traj = clean_pred_trajectory(rollout_dir, records, sel, m, n, VAR, RESID_SCALER, level=LEVEL)
        return None if traj is None else bbox_latent(traj)

    def twin_latents(sel, m, n):
        """The run's ung_gui twin trajectory as bbox latents (T, F)."""
        tw = channel(select_point(open_store(rollout_dir, "ung_gui", VAR), sel), config)
        return bbox_latent(np.asarray(tw.isel(m=m, n=n), dtype=float))

    def twin_final_mm(sel, m):
        """Mask-averaged final ung_gui twin states, per n."""
        tw = channel(select_point(open_store(rollout_dir, "ung_gui", VAR), sel), config)
        tw = tw.isel(t=-1) if "t" in tw.dims else tw
        return masked_mean(np.asarray(tw.isel(m=m), dtype=float), mask)

    def gui_final_mm(sel, m):
        """Mask-averaged final guided states, per n."""
        da = channel(select_point(open_store(rollout_dir, "gui", VAR), sel), config)
        return masked_mean(np.asarray(da.isel(m=m), dtype=float), mask)

    def delta_of(sel):
        """The run's authored delta trajectory (N,)."""
        return np.asarray(sweep_values["GUIDANCE_DELTA"][sel["GUIDANCE_DELTA"]], dtype=float)[:N_STEPS]

    def recorded_a_t(sel, m, n):
        """The a_t schedule recorded for one run at (m, n), or None."""
        for r in records:
            if r["m"] == m and r["n"] == n and all(
                r["sweep"].get(k) == v for k, v in sel.items() if k in r["sweep"]
            ):
                return np.asarray(r["a_t"], dtype=float)
        return None

    def grads_norm2_full(sel, m, n):
        """Full-domain ||dL/dz||^2 per t at (m, n) -- all channels, lazily summed."""
        gds = xr.open_zarr(rollout_dir / "grads.zarr")
        g2 = None
        for v in gds.data_vars:
            da = select_point(gds[v], sel)
            term = (da ** 2).sum(dim=[d for d in da.dims if d not in ("m", "n", "t")])
            g2 = term if g2 is None else g2 + term
        return np.asarray(g2.isel(m=m, n=n), dtype=float)

    # ---------- run selection ----------
    def runs_matching(axis_choices):
        """Runs whose sweep matches the pinned axes; "compare" axes fan out.

        A method hyper (e.g. fgwnolr_eta) is only enforced for runs of its own mode,
        so pinning it never excludes the other method's runs."""
        def _ok(sel):
            mode = sel["GUIDANCE_MODE"]
            for k, choice in axis_choices.items():
                if choice == "compare":
                    continue
                if k == "GUIDANCE_MODE":
                    if mode != choice:
                        return False
                elif k == "GUIDANCE_DELTA":
                    if f"δ#{sel[k]}" != choice:
                        return False
                elif k in GUIDANCE_METHOD_HYPERS.get(mode, ()):
                    if str(sel[k]) != choice:
                        return False
            return True
        return {label: sel for label, sel in points.items() if _ok(sel)}

    def run_colors(labels):
        """Stable color per run label (iteration order of the matched runs)."""
        return {label: TRAJ_COLORS[i % len(TRAJ_COLORS)] for i, label in enumerate(labels)}

    # ---------- derived quantities per section ----------
    def projected_runs(axis_choices, m, n):
        """label -> (guided_proj, twin_proj) for matched runs with traces at (m, n)."""
        out = {}
        for label, sel in runs_matching(axis_choices).items():
            g = guided_latents(sel, m, n)
            if g is not None and np.isfinite(g).any():
                out[label] = (pca_project(g), pca_project(twin_latents(sel, m, n)))
        return out

    def alignment_lines(axis_choices, m, n):
        """label -> deviation-alignment series at (m, n)."""
        out = {}
        for label, sel in runs_matching(axis_choices).items():
            g = guided_latents(sel, m, n)
            if g is not None and np.isfinite(g).any():
                out[label] = alignment_series(g, twin_latents(sel, m, n), pca_targets[n])
        return out

    def alignment_series(gui, twin, target):
        """cos(guidance displacement, twin->GT direction) per flow step."""
        def _cos(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        return [_cos(g - u, target - u) for g, u in zip(gui, twin)]

    def convergence_gap_n(sel, m):
        """Realized final gap to the (1+delta)·twin target, per n."""
        return gui_final_mm(sel, m) - (1.0 + delta_of(sel)) * twin_final_mm(sel, m)

    def waterfall_gaps(sel, m, n):
        """(pre, post) gap series of the within-step split, or None without traces.
        post = pre - h*lambda*||g||^2 / (2 r): the through-Jacobian first-order claim."""
        lam = applied_lambda(records, sel, m, n)
        traj = clean_pred_trajectory(rollout_dir, records, sel, m, n, VAR, RESID_SCALER, level=LEVEL)
        if lam is None or traj is None or not np.isfinite(traj).any():
            return None
        target = (1.0 + delta_of(sel)[n]) * twin_final_mm(sel, m)[n]
        pre = masked_mean(traj, mask) - target
        s, h = flow_grids(len(pre))
        r_sum = pre * float(np.asarray(mask).sum())
        dS = np.where(np.abs(r_sum) > 1e-12, -h * lam * grads_norm2_full(sel, m, n) / (2.0 * r_sum), 0.0)
        return pre, pre + dS

    def informativeness(sel, m):
        """(jsd_align, nss), each (N, T); NaN at flow steps with no applied gradient."""
        mask_np = np.asarray(mask, dtype=float)
        pm = mask_np / mask_np.sum()
        region = mask_np >= 0.5 * mask_np.max()
        g = np.asarray(channel(select_point(open_store(rollout_dir, "grads", VAR), sel), config).isel(m=m), dtype=float)
        G = g ** 2
        tot = G.sum(axis=(-2, -1), keepdims=True)
        informative = tot.squeeze((-2, -1)) > 0
        pg = G / np.where(tot > 0, tot, 1.0)

        def plogq(p, q):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(p > 0, p * np.log(p / np.where(q > 0, q, 1.0)), 0.0)

        mavg = 0.5 * (pg + pm)
        jsd = 0.5 * plogq(pg, mavg).sum(axis=(-2, -1)) + 0.5 * plogq(np.broadcast_to(pm, pg.shape), mavg).sum(axis=(-2, -1))
        jsd_align = np.where(informative, 1.0 - jsd / np.log(2.0), np.nan)
        sd = G.std(axis=(-2, -1), keepdims=True)
        z = (G - G.mean(axis=(-2, -1), keepdims=True)) / np.where(sd > 0, sd, np.nan)
        nss = np.where(informative, z[..., region].mean(axis=-1), np.nan)
        return jsd_align, nss

    print(f"cloud: {pca_cloud.shape} | EVR(3): {float(pca_evr.sum()):.3f} | runs: {len(points)}")
    return (
        N_STEPS,
        TRAJ_COLORS,
        alignment_lines,
        alignment_series,
        config,
        convergence_gap_n,
        delta_of,
        guided_latents,
        informativeness,
        pca_cloud_proj,
        pca_evr,
        pca_project,
        pca_targets,
        pca_ung_finals,
        points,
        projected_runs,
        recorded_a_t,
        run_colors,
        runs_matching,
        sweep_values,
        twin_latents,
        waterfall_gaps,
    )


@app.cell
def _(N_STEPS, config, mo, sweep_values):
    # per-axis comparison control: "compare" fans an axis out; a value pins it
    _varying = {k: v for k, v in sweep_values.items() if isinstance(v, list) and len(v) > 1}
    axis_options = {
        k: ([f"δ#{i}" for i in range(len(v))] if k == "GUIDANCE_DELTA" else [str(x) for x in v])
        for k, v in _varying.items()
    }
    axis_selectors = mo.ui.dictionary({
        k: mo.ui.dropdown(
            ["compare"] + opts,
            value=("compare" if k == "GUIDANCE_MODE" else opts[0]),
            label=f"{k}: ",
        )
        for k, opts in axis_options.items()
    })
    n_slider = mo.ui.slider(1, N_STEPS, step=1, value=min(2, N_STEPS), label="n: ", show_value=True, debounce=True)
    m_slider = mo.ui.slider(1, config["M"], step=1, value=1, label="m: ", show_value=True, debounce=True)

    def selection_row(*extra):
        """The shared run-selection controls (same elements everywhere -> always in sync)."""
        return mo.hstack(
            [axis_selectors[k] for k in axis_options] + [n_slider, m_slider, *extra],
            justify="start", align="start",
        )

    # delta is an EXPERIMENT hyper (not a method hyper): it always splits the
    # ranking by default -- pooling across different asked pushes would mix regimes
    jsd_split_selectors = mo.ui.dictionary({
        k: mo.ui.dropdown(
            ["aggregate", "compare"],
            value=("compare" if k in ("GUIDANCE_MODE", "GUIDANCE_DELTA") else "aggregate"),
            label=f"{k}: ",
        )
        for k in axis_options
    })
    return (
        axis_options,
        axis_selectors,
        jsd_split_selectors,
        m_slider,
        n_slider,
        selection_row,
    )


@app.cell
def _(mo):
    elev_slider = mo.ui.slider(0, 90, step=5, value=25, label="elev: ", show_value=True, debounce=True)
    azim_slider = mo.ui.slider(-180, 180, step=5, value=-60, label="azim: ", show_value=True, debounce=True)
    zoom_traj_checkbox = mo.ui.checkbox(label="zoom to trajectories")
    return azim_slider, elev_slider, zoom_traj_checkbox


@app.cell
def _(
    axis_selectors,
    azim_slider,
    elev_slider,
    m_slider,
    mo,
    n_slider,
    np,
    pca_cloud_proj,
    pca_evr,
    pca_project,
    pca_targets,
    pca_ung_finals,
    plt,
    projected_runs,
    run_colors,
    selection_row,
    zoom_traj_checkbox,
):
    # ===== section: 3D PCA trajectories =====
    _m, _n = m_slider.value - 1, n_slider.value - 1
    _runs = projected_runs(axis_selectors.value, _m, _n)
    _colors = run_colors(_runs)

    def _draw_trajectory(ax, pts, color, label):
        ax.plot(*pts.T, "-", color=color, linewidth=1.7, alpha=0.9, label=label)
        ax.scatter(*pts.T, s=np.linspace(8, 46, len(pts)), color=color, alpha=0.9, depthshade=False)
        ax.scatter(*pts[-1], marker="D", s=70, color=color, edgecolors="white", depthshade=False)
        ax.text(*pts[-1], f"  {label}", color=color, fontsize=8)

    def _draw_twin(ax, pts, color, label):
        ax.plot(*pts.T, "--", color=color, linewidth=1.2, alpha=0.65, label=label)
        ax.scatter(*pts[-1], marker="X", s=80, color=color, alpha=0.8, edgecolors="white", depthshade=False)

    def _draw_marker(ax, pt, text, text_color, **scatter_kw):
        ax.scatter(*pt, depthshade=False, **scatter_kw)
        ax.text(*pt, f"  {text}", color=text_color, fontsize=9, fontweight="bold")

    def _zoom_to(ax, point_groups, pad_frac=0.12):
        pts = np.vstack(point_groups)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = pad_frac * float((hi - lo).max())
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_zlim(lo[2] - pad, hi[2] + pad)

    pca_fig = plt.figure(figsize=(24, 16), dpi=500)
    _ax3 = pca_fig.add_subplot(projection="3d")
    _ax3.scatter(*pca_cloud_proj.T, color="#BBBBBB", s=14, alpha=0.5, depthshade=False,
                 label=f"ERA5 cloud ({pca_cloud_proj.shape[0]} days)")
    for _label, (_gp, _up) in _runs.items():
        _draw_trajectory(_ax3, _gp, _colors[_label], _label)
        _draw_twin(_ax3, _up, _colors[_label], f"ung_gui {_label}")
    _tp = pca_project(pca_targets[_n])
    _draw_marker(_ax3, _tp, "GT", "black", marker="*", s=320, color="black", label="GT (valid day)")
    _groups = [p for pair in _runs.values() for p in pair] + [_tp[None, :]]
    if pca_ung_finals is not None:
        _op = pca_project(pca_ung_finals[_n])
        _draw_marker(_ax3, _op, "ung", "#111111", marker="o", s=130, facecolors="none",
                     edgecolors="#111111", linewidths=2.0, label="ung rollout (final)")
        _groups.append(_op[None, :])
    if zoom_traj_checkbox.value and _runs:
        _zoom_to(_ax3, _groups)
    _ax3.set_xlabel("PC1"); _ax3.set_ylabel("PC2"); _ax3.set_zlabel("PC3")
    _ax3.set_title(
        f"Clean-pred trajectories in the ERA5 PCA frame  (n={_n + 1}, m={m_slider.value}, EVR={pca_evr.sum():.0%}, {len(_runs)} runs)",
        loc="left",
    )
    _ax3.view_init(elev=elev_slider.value, azim=azim_slider.value)
    _ax3.legend(loc="upper left", fontsize=8)
    mo.vstack(
        [
            mo.md("## 3D PCA trajectories"),
            selection_row(elev_slider, azim_slider, zoom_traj_checkbox),
            pca_fig,
        ],
        align="start",
    )
    return (pca_fig,)


@app.cell
def _(Path, mo, n_slider, pca_fig, rollout_dropdown, save_fig_button):
    # ===== thesis-figure export (high-dpi PNG + vector PDF) =====
    _msg = "export the 3D figure at the current view"
    if save_fig_button.value:
        _out_dir = Path("figures")
        _out_dir.mkdir(exist_ok=True)
        _stem = _out_dir / f"pca_trajectories_{rollout_dropdown.value.replace(':', '-')}_n{n_slider.value}"
        pca_fig.savefig(f"{_stem}.png", dpi=300, bbox_inches="tight")
        pca_fig.savefig(f"{_stem}.pdf", bbox_inches="tight")
        _msg = f"saved `{_stem}.png` and `{_stem}.pdf`"
    mo.hstack([save_fig_button, mo.md(f"_{_msg}_")], justify="start", align="center")
    return


@app.cell
def _(mo):
    save_fig_button = mo.ui.run_button(label="save figure")
    return (save_fig_button,)


@app.cell
def _(
    alignment_lines,
    axis_selectors,
    m_slider,
    mo,
    n_slider,
    plot_trajectory,
    run_colors,
    selection_row,
):
    # ===== section: deviation alignment =====
    _m, _n = m_slider.value - 1, n_slider.value - 1
    _lines = alignment_lines(axis_selectors.value, _m, _n)
    mo.vstack(
        [
            mo.md(r"""
    ## Deviation alignment

    $\cos(\hat{x}^{gui}_t - \hat{x}^{ung\_gui}_t,\; x^{gt} - \hat{x}^{ung\_gui}_t)$ per flow step —
    $+1$: the push has the spatial pattern of a real correction toward the atmosphere;
    $0$: orthogonal (mask-shaped fabrication); negative: against it.
    """),
            selection_row(),
            plot_trajectory(
                _lines,
                title="Deviation alignment over $t$",
                subtitle=f"n={_n + 1}, m={m_slider.value}",
                xlabel="$t$", color_map=run_colors(_lines),
                figsize=(12, 5), prepend_zero=False, start_index=1,
            ) if _lines else mo.md("_no runs with traces at this selection_"),
        ],
        align="start",
    )
    return


@app.cell
def _(
    N_STEPS,
    alignment_series,
    guided_latents,
    m_slider,
    mo,
    np,
    pca_cloud_proj,
    pca_project,
    pca_targets,
    points,
    twin_latents,
):
    # ===== section: latent scoreboard (every run × every n) =====
    _m = m_slider.value - 1
    _pc_std = pca_cloud_proj.std(axis=0)
    _sqf = float(np.sqrt(pca_targets.shape[1]))
    _rows = []
    for _label, _sel in points.items():
        for _ni in range(N_STEPS):
            _g = guided_latents(_sel, _m, _ni)
            if _g is None or not np.isfinite(_g).any():
                continue
            _tw = twin_latents(_sel, _m, _ni)
            _align = alignment_series(_g, _tw, pca_targets[_ni])
            _gp = pca_project(_g)
            _rows.append({
                "run": _label,
                "n": _ni + 1,
                "dist to GT (K)": round(float(np.linalg.norm(_g[-1] - pca_targets[_ni]) / _sqf), 3),
                "push (K)": round(float(np.linalg.norm(_g[-1] - _tw[-1]) / _sqf), 3),
                "mean align": round(float(np.mean(_align)), 3),
                "final align": round(float(_align[-1]), 3),
                "path len (PCA)": round(float(np.linalg.norm(np.diff(_gp, axis=0), axis=1).sum()), 1),
                "off-manifold z": round(float(np.sqrt(((_gp[-1] / _pc_std) ** 2).sum())), 2),
            })
    mo.vstack(
        [
            mo.md(
                "## Latent scoreboard\n\nEvery run × every $n$ (member m={}). ".format(m_slider.value)
                + "`dist to GT` / `push`: per-pixel RMS of the endpoint vs GT / vs the twin; "
                + "`align`: deviation alignment; `path len`: trajectory length in the 3-PC frame; "
                + "`off-manifold z`: endpoint's Mahalanobis distance from the ERA5 cloud (per-PC std units)."
            ),
            mo.hstack([m_slider], justify="start"),
            mo.ui.table(_rows, selection=None, pagination=False),
        ],
        align="start",
    )
    return


@app.cell
def _(
    axis_selectors,
    convergence_gap_n,
    delta_of,
    m_slider,
    mo,
    n_slider,
    np,
    plot_trajectory,
    plt,
    recorded_a_t,
    run_colors,
    runs_matching,
    selection_row,
    waterfall_gaps,
):
    # ===== section: guidance convergence (guidance.py-style) =====
    _m, _n = m_slider.value - 1, n_slider.value - 1
    _runs = runs_matching(axis_selectors.value)
    _colors = run_colors(_runs)

    _over_n = {label: convergence_gap_n(sel, _m) for label, sel in _runs.items()}
    _delta_idxs = {sel["GUIDANCE_DELTA"] for sel in _runs.values()}
    _shared_delta = delta_of(next(iter(_runs.values()))) if len(_delta_idxs) == 1 else None
    _fig_conv_n = plot_trajectory(
        _over_n,
        title="Guidance convergence over $n$",
        subtitle=r"$\mathrm{mean}_{\mathrm{mask}}(x^{gui}_{n,t=T}) - \mathrm{target}_n$" + f"  |  m={m_slider.value}",
        xlabel="$n$", color_map=_colors,
        right_trajectory=_shared_delta, right_label=r"$\delta_n$", right_color="#8A2BE2", right_percentage=True,
        figsize=(12, 6), prepend_zero=False, start_index=1,
    ) if _over_n else mo.md("_no runs matched_")

    def _waterfall(label, sel):
        """The within-step split panel for one run (pre circles, claim diamonds, bars, drift)."""
        gaps = waterfall_gaps(sel, _m, _n)
        if gaps is None:
            return mo.md(f"_{label}: no trace at this selection_")
        pre, post = gaps
        T_len = len(pre)
        xt = np.arange(1, T_len + 1).astype(float)
        wt = min(22.0, max(8.0, 3.4 + 0.78 * T_len))
        with plt.rc_context({"font.size": 10, "axes.titlesize": 14, "legend.fontsize": 9}):
            fig, ax = plt.subplots(figsize=(wt, 6), dpi=110)
            bar_off = 0.16
            ax.bar(xt + bar_off, post - pre, bottom=pre, width=0.28,
                   color="#2E86C1", alpha=0.35, zorder=3, label="guidance claim (1st order, via Jacobian)")
            if T_len > 1:
                ax.bar(xt[1:] - bar_off, pre[1:] - post[:-1], bottom=post[:-1], width=0.28,
                       color="#C0392B", alpha=0.35, zorder=3, label="realization gap  (pre$_{t+1}$ − post$_t$)")
            for i in range(T_len - 1):
                ax.plot([xt[i], xt[i + 1]], [post[i], pre[i + 1]], "-", color="#B7950B",
                        alpha=0.5, linewidth=1.1, zorder=4, label="model drift" if i == 0 else "_nolegend_")
            a_t = recorded_a_t(sel, _m, _n) if sel["GUIDANCE_MODE"] == "FGWNOGAP" else None
            if a_t is not None:
                ax.plot(xt, pre[0] * a_t, "--", color="#888888", linewidth=1.2, alpha=0.9,
                        zorder=2, label=r"NOGAP schedule  $r_0\,a_t$")
            ax.plot(xt, post, "D", color="#2E86C1", markersize=5, zorder=6, label=r"claimed after step $t$")
            ax.plot(xt, pre, "o", markerfacecolor="none", markeredgecolor="#B7950B",
                    markeredgewidth=1.8, markersize=8, linestyle="none", zorder=7,
                    label=r"measured before step $t$  (clean pred)")
            ax.axhline(0.0, color="#888888", linewidth=1.0, alpha=0.8, zorder=1)
            ax.set_xlim(0.6, T_len + 0.4)
            ax.set_xticks(xt)
            ax.set_xlabel("$t$"); ax.set_ylabel("masked mean − target")
            ax.set_title(f"Within-step split  |  {label}  (n={_n + 1}, m={m_slider.value})",
                         loc="left", fontweight="bold")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            ax.yaxis.grid(True, color="#D7D7D7", linewidth=0.7, alpha=0.55)
            fig.tight_layout(rect=(0, 0, 0.82, 1))
        return fig

    mo.vstack(
        [
            mo.md("## Guidance convergence\n\nGap to the $(1+\\delta)\\cdot$twin target: realized per $n$ (left axis, with the $\\delta$ schedule on the right), and the within-step split over $t$ per run."),
            selection_row(),
            _fig_conv_n,
            *[_waterfall(_label, _sel) for _label, _sel in _runs.items()],
        ],
        align="start",
    )
    return


@app.cell
def _(
    TRAJ_COLORS,
    axis_selectors,
    informativeness,
    m_slider,
    mo,
    n_slider,
    np,
    plot_trajectory,
    runs_matching,
    selection_row,
):
    # ===== section: gradient informativeness =====
    _m, _n = m_slider.value - 1, n_slider.value - 1
    _jsd_n, _jsd_t, _nss_n, _nss_t, _colors_gi = {}, {}, {}, {}, {}
    for _i, (_label, _sel) in enumerate(runs_matching(axis_selectors.value).items()):
        _jsd, _nss = informativeness(_sel, _m)
        if not np.isfinite(_nss).any():
            continue
        _jsd_n[_label] = np.nansum(_jsd, axis=1)
        _nss_n[_label] = np.nansum(_nss, axis=1)
        _jsd_t[_label] = _jsd[_n]
        _nss_t[_label] = _nss[_n]
        _colors_gi[_label] = TRAJ_COLORS[_i % len(TRAJ_COLORS)]

    def _gi_fig(lines, title, subtitle, xlabel):
        return plot_trajectory(
            lines, title=title, subtitle=subtitle, xlabel=xlabel,
            color_map=_colors_gi, figsize=(11, 5), prepend_zero=False, start_index=1,
        ) if lines else mo.md("_no gradients at this selection_")

    mo.vstack(
        [
            mo.md(r"""
    ## Gradient informativeness — NSS and JSD alignment of the applied gradient pattern

    All metrics operate on the **normalized gradient energy pattern** of the guided channel at
    flow step $t$: with $G_t(x) = g_t(x)^2$,

    $$p_{g,t}(x) = \frac{G_t(x)}{\sum_{x'} G_t(x')},\qquad
    p_m(x) = \frac{m(x)}{\sum_{x'} m(x')},\qquad
    R = \{x : m(x) \ge \tfrac{1}{2}\max m\},$$

    so every scalar factor ($w_t a_t$, residual size) cancels — only the spatial pattern counts.
    Steps with no applied gradient ($\sum G_t = 0$) are excluded.

    **NSS** (normalized scanpath saliency) — how many standard deviations above the field's own
    spatial average the gradient energy sits inside the mask region:

    $$\mathrm{NSS}_t = \frac{1}{|R|} \sum_{x \in R} \frac{G_t(x) - \mu(G_t)}{\sigma(G_t)},$$

    with $\mu, \sigma$ the spatial mean/std over the full domain. $0$ ≈ chance, larger = more
    concentrated in the region; unbounded above.

    **JSD align** — full-distribution similarity between the gradient pattern and the mask density,
    via the Jensen–Shannon divergence with $M = \tfrac{1}{2}(p_{g,t} + p_m)$:

    $$\mathrm{JSD}(p_{g,t} \Vert p_m) = \tfrac{1}{2}\,\mathrm{KL}(p_{g,t} \Vert M) + \tfrac{1}{2}\,\mathrm{KL}(p_m \Vert M),\qquad
    \mathrm{JSDalign}_t = 1 - \frac{\mathrm{JSD}(p_{g,t} \Vert p_m)}{\ln 2} \in [0, 1].$$

    $1$: the gradient pattern equals the mask density (shape included); $0$: disjoint support.

    **Over $n$ (left column)** — the sum over the informative flow steps of step $n$:

    $$\mathrm{NSS}(n) = \sum_{t\,:\,\sum G_t > 0} \mathrm{NSS}_t,\qquad
    \mathrm{JSDalign}(n) = \sum_{t\,:\,\sum G_t > 0} \mathrm{JSDalign}_t.$$

    **Over $t$ (right column)** — the per-step series at the selected $n$.
    """),
            selection_row(),
            mo.hstack(
                [
                    _gi_fig(_nss_n, "NSS over $n$", r"$\Sigma_t\ \mathrm{NSS}_t$  (mean $z$-score of $g^2$ in the mask region)" + f" | m={m_slider.value}", "$n$"),
                    _gi_fig(_nss_t, "NSS over $t$", f"n={_n + 1}, m={m_slider.value}", "$t$"),
                ],
                justify="start",
            ),
            mo.hstack(
                [
                    _gi_fig(_jsd_n, "JSD align over $n$", r"$\Sigma_t\,(1 - \mathrm{JSD}(p_g\Vert p_m)/\ln 2)$" + f" | m={m_slider.value}", "$n$"),
                    _gi_fig(_jsd_t, "JSD align over $t$", f"n={_n + 1}, m={m_slider.value}", "$t$"),
                ],
                justify="start",
            ),
        ],
        align="start",
    )
    return


@app.cell
def _(config, informativeness, np, points):
    # ===== JSD pool: per-run alignment values over all members, steps and informative t =====
    # computed once per rollout; the summary table below only regroups these (cheap)
    jsd_pool = {}
    for _label, _sel in points.items():
        _vals = [
            _jsd[np.isfinite(_jsd)]
            for _mi in range(config["M"])
            for _jsd, _ in [informativeness(_sel, _mi)]
        ]
        jsd_pool[_label] = np.concatenate(_vals) if _vals else np.array([])
    return (jsd_pool,)


@app.cell
def _(
    GUIDANCE_METHOD_HYPERS,
    axis_options,
    jsd_pool,
    jsd_split_selectors,
    mo,
    np,
    points,
):
    # ===== section: JSD summary =====
    def _group_key(sel):
        """Group label from the axes set to "compare"; aggregated axes pool together."""
        parts = []
        mode = sel["GUIDANCE_MODE"]
        for k, choice in jsd_split_selectors.value.items():
            if choice != "compare":
                continue
            if k == "GUIDANCE_MODE":
                parts.append(mode)
            elif k == "GUIDANCE_DELTA":
                parts.append(f"δ#{sel[k]}")
            elif k in GUIDANCE_METHOD_HYPERS.get(mode, ()):
                parts.append(f"{k}={sel[k]}")
        return " ".join(parts) or "all runs"

    _groups = {}
    for _label, _sel in points.items():
        _groups.setdefault(_group_key(_sel), []).append(jsd_pool[_label])
    _stats = sorted(
        (
            (key, float(np.mean(v)), float(np.std(v)), int(v.size))
            for key, arrs in _groups.items()
            if (v := np.concatenate(arrs)).size
        ),
        key=lambda r: -r[1],
    )
    _md = ["| rank | group | mean JSD align | std | samples |", "|---|---|---|---|---|"]
    for _i, (_key, _mean, _std, _cnt) in enumerate(_stats, 1):
        _b = "**" if _i == 1 else ""
        _md.append(f"| {_i} | {_b}{_key}{_b} | {_b}{_mean:.4f}{_b} | {_std:.4f} | {_cnt} |")
    mo.vstack(
        [
            mo.md(
                "## JSD summary\n\nMean ± std of the JSD alignment of the applied gradient pattern, "
                "pooled over every axis set to *aggregate* (plus members $m$, steps $n$ and informative "
                "flow steps $t$). Axes set to *compare* split the ranking into separate rows; "
                "the winner is **bold**."
            ),
            mo.hstack([jsd_split_selectors[_k] for _k in axis_options], justify="start", align="start"),
            mo.md("\n".join(_md)),
        ],
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
