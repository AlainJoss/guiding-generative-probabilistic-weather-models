import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    from src.ui.comparison import (
        load_rollout, sweep_points,
        clean_pred_trajectory_primitive, guided_velocity_primitive, convergence_state_line,
        residual_scaler, channel, open_store, select_point,
    )
    from src.pca_basis import mask_bbox, bbox_latent, ensure_basis, project, pathlength, cloud_sample
    from src import flow_time_tests as ftt
    from src.mask import get_masked_mean
    from src.ui.plot_trajectory import plot_trajectory
    from src.utils import get_rollout_ids, get_gt_rollout, get_rollout_dir, find_era5_input
    from datetime import datetime

    # petroff6 house style; the 6 schedules map 1:1 to its 6 colours (shared by every chart)
    plt.style.use("petroff6")
    PETROFF6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
    # same warm half of RdBu_r as the mask map in guidance.py (white at 0 -> red at max)
    WARM_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "rdbu_warm", plt.get_cmap("RdBu_r")(np.linspace(0.5, 1.0, 256)))
    return (
        PETROFF6,
        WARM_CMAP,
        bbox_latent,
        channel,
        clean_pred_trajectory_primitive,
        cloud_sample,
        convergence_state_line,
        datetime,
        ensure_basis,
        find_era5_input,
        ftt,
        get_gt_rollout,
        get_masked_mean,
        get_rollout_dir,
        get_rollout_ids,
        guided_velocity_primitive,
        load_rollout,
        mask_bbox,
        mcolors,
        mo,
        np,
        open_store,
        pathlength,
        plt,
        project,
        residual_scaler,
        select_point,
        sweep_points,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Flow-time tests

    Flow step $t$ = the $T$ Euler steps inside one `sample()` call. Every metric scores the guided
    channel `VAR@LEVEL`, pooled over members $m$, guided steps $n$, and the selected experiments,
    per schedule. Grids are experiments (rows) $\times$ $m$ (cols) at step $n$; press **compute metrics**.

    - **Reliability** — is the target reached? ($\varepsilon_t \to 0$)
    - **Interference** — how much guidance it took, and how the convergence behaves.
    - **Closeness** — how far the guided flow strays from the unguided.
    - **Realism** — does the guidance footprint look like the mask?
    """)
    return


@app.cell
def _(get_rollout_ids, mo):
    # OUTER experiment selector; metrics are averaged over ALL its date-subexperiments
    # (which share the same sweep). Values are outer experiment ids ("<exp>").
    _exp_ids = sorted({rid.split("/", 1)[0] for rid in get_rollout_ids("gui")})
    experiment_multiselect = mo.ui.multiselect(
        _exp_ids, value=_exp_ids[:1] if _exp_ids else [], label="experiment: "
    )
    return (experiment_multiselect,)


@app.cell
def _(
    experiment_multiselect,
    get_rollout_ids,
    load_rollout,
    mask_bbox,
    np,
    sweep_points,
):
    # all date-subexperiments of the selected experiment(s) -> metrics are pooled and
    # averaged over them (they share VAR/LEVEL/mask/sweep, hence one bbox + one PCA basis)
    _sel_exps = set(experiment_multiselect.value)
    EXPS = [rid for rid in get_rollout_ids("gui") if rid.split("/", 1)[0] in _sel_exps]
    _rid0 = EXPS[0]
    _dir0, config0, sweep_values0, records0, mask0 = load_rollout(_rid0)
    mask0 = np.asarray(mask0, dtype=float)
    VAR = config0["VAR"]
    LEVEL = config0["LEVEL"]
    PARTITION = config0["PARTITION"]
    N = int(config0["N"])
    M = int(config0["M"])
    BBOX = mask_bbox(mask0)
    points = sweep_points(sweep_values0, records0)
    return BBOX, EXPS, LEVEL, M, N, PARTITION, VAR, points


@app.cell
def _(mo, points):
    point_multiselect = mo.ui.multiselect(
        list(points), value=list(points), label="sweep points: "
    )
    compute_button = mo.ui.run_button(label="compute metrics")
    return compute_button, point_multiselect


@app.cell(hide_code=True)
def _(N, mo):
    # global step selector (m is the grid columns; PCA view controls live in the Closeness section)
    n_slider = mo.ui.slider(0, max(N - 1, 1), value=0, step=1, label="n: ", show_value=True)
    return (n_slider,)


@app.cell(hide_code=True)
def _(mo):
    # collapse the m-columns of the first two sections into one chart per experiment,
    # with the spread across m shown as shading behind the mean line
    aggregate_m_checkbox = mo.ui.checkbox(value=False, label="aggregate m (spread as shading)")
    return (aggregate_m_checkbox,)


@app.cell
def _(
    EXPS,
    compute_button,
    experiment_multiselect,
    mo,
    n_slider,
    point_multiselect,
):
    mo.vstack([
        experiment_multiselect,
        mo.md(f"_averaging over **{len(EXPS)}** subexperiment(s): "
              f"{', '.join(e.split('/')[-1] for e in EXPS)}_"),
        mo.hstack([point_multiselect, compute_button, n_slider], justify="start", align="center"),
    ], align="start")
    return


@app.cell
def _(
    BBOX,
    EXPS,
    LEVEL,
    N,
    PARTITION,
    PETROFF6,
    VAR,
    basis,
    bbox_latent,
    channel,
    clean_pred_trajectory_primitive,
    compute_button,
    convergence_state_line,
    datetime,
    find_era5_input,
    ftt,
    get_gt_rollout,
    get_masked_mean,
    get_rollout_dir,
    guided_velocity_primitive,
    load_rollout,
    np,
    open_store,
    pathlength,
    point_multiselect,
    project,
    residual_scaler,
    select_point,
    sweep_points,
):
    # ---- compute: pooled metrics + per-(exp, m, n) grid trajectories ----
    metrics = None
    traj_grid = None
    gt_proj = None
    sched_colors = None
    if compute_button.value:
        _labels = list(point_multiselect.value)
        sched_colors = {_l: PETROFF6[_i % len(PETROFF6)] for _i, _l in enumerate(_labels)}
        _keys = ("eps_final", "eps_rel_final", "total_guidance", "pushback_count", "pushback_amount",
                 "overshoot_count", "overshoot_amount", "L", "L_twin", "guidance_steps", "pushback_steps", "end_dist", "realism_tv")
        _acc = {lb: {k: [] for k in _keys} for lb in _labels}
        _eps_curves = {lb: [] for lb in _labels}
        traj_grid = {}
        gt_proj = {}
        ung_proj = {}
        mask_bbox_ref = None
        for _ei, _rid in enumerate(EXPS):
            _dir, _cfg, _sv, _recs, _mask = load_rollout(_rid)
            _mask = np.asarray(_mask, dtype=float)
            mask_bbox_ref = _mask[BBOX[0], BBOX[1]]
            _c = residual_scaler(PARTITION, VAR, LEVEL)
            _pts = {lb: sel for lb, sel in sweep_points(_sv, _recs).items() if lb in _labels}
            try:
                _ung_final = np.asarray(channel(open_store(_dir, "ung", VAR), _cfg).isel(t=-1), float)
                _ung_base = get_masked_mean(_ung_final, _mask)
            except Exception:
                _ung_base = None
            try:
                _gtr = get_gt_rollout(N + 1, datetime.fromisoformat(_cfg["START_TS"]),
                                      input_path=find_era5_input(get_rollout_dir(_rid)))
                _gt_field = np.asarray(channel(_gtr[VAR], _cfg), float)
                _gt_base = get_masked_mean(_gt_field, _mask)
            except Exception:
                _gt_field = None; _gt_base = None
            for _lb, _sel in _pts.items():
                _ref = str(_sel.get("GUI_REF") or (_sv.get("GUI_REF", ["UNG"])[0]))
                _p = np.asarray(_sv["GUIDANCE_DELTA"][_sel["GUIDANCE_DELTA"]], float)[:N]
                _gui = channel(select_point(open_store(_dir, "gui", VAR), _sel), _cfg)
                _tw = channel(select_point(open_store(_dir, "gui_ung", VAR), _sel), _cfg)
                for _m in range(_gui.sizes["m"]):
                    for _n in range(N):
                        if float(_p[_n]) == 0.0:
                            continue
                        if _ref == "GT":
                            if _gt_base is None: continue
                            _base = float(_gt_base[_n + 1])
                        else:
                            if _ung_base is None: continue
                            _base = float(_ung_base[_m, _n])
                        _A = float(ftt.target_A(_base, _p[_n]))
                        _traj = clean_pred_trajectory_primitive(_dir, _sel, _m, _n, VAR, _c, level=LEVEL)
                        _eps, _eps_rel = ftt.eps_trajectory(get_masked_mean(_traj, _mask), _A)
                        _guif = np.asarray(_gui.isel(m=_m, n=_n), float)
                        _s_gui = float(get_masked_mean(_guif, _mask))
                        _states, _land = convergence_state_line(_dir, _sel, _m, _n, VAR, _c, _mask, _A, level=LEVEL)
                        _inter = ftt.interference_from_convergence(_states, _land)
                        _g = bbox_latent(_traj, BBOX)
                        _pg = project(basis, _g)
                        _ungf = np.asarray(_tw.isel(m=_m, n=_n).isel(t=-1), float)
                        _u = bbox_latent(np.asarray(_tw.isel(m=_m, n=_n), float), BBOX)
                        _pu = project(basis, _u)
                        _L = pathlength(_pg); _Lt = pathlength(_pu)
                        _vel = guided_velocity_primitive(_dir, _sel, _m, _n, VAR, _c, level=LEVEL)
                        _kick = bbox_latent(_traj - _c * _vel["s"][:, None, None] * _vel["gui_vec"], BBOX)
                        _gstep, _pstep = ftt.step_sums(_g, _kick)
                        _end = float(np.linalg.norm(_g[-1] - _u[-1]) / np.sqrt(_g.shape[1]))
                        _tv = ftt.realism_tv(_guif, _ungf, _mask)
                        _gfield = np.abs(_guif - _ungf)[BBOX[0], BBOX[1]]
                        _a = _acc[_lb]
                        _a["eps_final"].append(ftt.final_gap(_s_gui, _A)); _a["eps_rel_final"].append(float(_eps_rel[-1]))
                        for _k in ("total_guidance", "pushback_count", "pushback_amount", "overshoot_count", "overshoot_amount"):
                            _a[_k].append(_inter[_k])
                        _a["L"].append(_L); _a["L_twin"].append(_Lt); _a["guidance_steps"].append(_gstep); _a["pushback_steps"].append(_pstep); _a["end_dist"].append(_end); _a["realism_tv"].append(_tv)
                        _eps_curves[_lb].append(_eps_rel)
                        traj_grid.setdefault((_ei, _m, _n), {})[_lb] = {
                            "pg": _pg, "pu": _pu, "states": _states, "land_ung": _land, "gfield": _gfield, "eps_rel": _eps_rel,
                            "gvec_avg": np.abs(_vel["gui_vec"]).mean(axis=0)[BBOX[0], BBOX[1]]}
            if _ung_base is not None:
                for _m in range(_ung_final.shape[0]):
                    for _n in range(N):
                        ung_proj[(_ei, _m, _n)] = project(basis, bbox_latent(_ung_final[_m, _n], BBOX))
            if _gt_field is not None:
                for _n in range(N):
                    gt_proj[(_ei, _n)] = project(basis, bbox_latent(_gt_field[_n + 1], BBOX))
        metrics = {}
        for _lb in _labels:
            _a = _acc[_lb]; _rec = {k: ftt.aggregate_mean_std(v) for k, v in _a.items()}
            _rec["n_pool"] = len(_a["eps_final"])
            _rec["eps_rel_curve"] = (np.nanmean(np.stack(_eps_curves[_lb]), axis=0) if _eps_curves[_lb] else None)
            metrics[_lb] = _rec
    return gt_proj, mask_bbox_ref, metrics, sched_colors, traj_grid, ung_proj


@app.cell(hide_code=True)
def _(aggregate_m_checkbox, mo):
    mo.vstack([
        mo.md(r"""## Reliability
    - Gap $\varepsilon_t = S(\hat x_t) - A \to 0$ over flow steps ($\varepsilon_t/\varepsilon_0$, one line per schedule).
    - Table: total guidance."""),
        aggregate_m_checkbox,
    ])
    return


@app.cell
def _(
    EXPS,
    M,
    aggregate_m_checkbox,
    metrics,
    mo,
    n_slider,
    np,
    plt,
    sched_colors,
    traj_grid,
):
    # ---- Reliability: normalized-gap grid ----
    if metrics is None or traj_grid is None:
        t1_view = mo.md("_press **compute metrics**_")
    else:
        _n = int(n_slider.value); _agg = aggregate_m_checkbox.value; _labels = list(sched_colors)
        if _agg:
            _f, _axs = plt.subplots(len(EXPS), 1, figsize=(7.5, 3.1*len(EXPS)), squeeze=False)
            for _ei in range(len(EXPS)):
                _ax = _axs[_ei][0]
                for _lb in _labels:
                    _cs = [np.asarray(traj_grid[(_ei, _m, _n)][_lb]["eps_rel"]) for _m in range(M)
                           if (_ei, _m, _n) in traj_grid and _lb in traj_grid[(_ei, _m, _n)]]
                    if not _cs: continue
                    _arr = np.stack(_cs); _x = np.arange(1, _arr.shape[1]+1); _c = sched_colors[_lb]
                    _ax.fill_between(_x, np.nanmin(_arr, 0), np.nanmax(_arr, 0), color=_c, alpha=0.18, linewidth=0)
                    _ax.plot(_x, np.nanmean(_arr, 0), "-", color=_c, linewidth=1.9, label=_lb.split()[-1])
                _ax.axhline(0.0, color="#888888", linewidth=0.8); _ax.set_title(EXPS[_ei].split('/')[-1], fontsize=10)
                _ax.set_xlabel("$t$"); _ax.set_ylabel(r"$\varepsilon_t/\varepsilon_0$")
                if _ei == 0: _ax.legend(fontsize=7)
        else:
            _f, _axs = plt.subplots(len(EXPS), M, figsize=(4.6*M, 3.0*len(EXPS)), squeeze=False)
            for _ei in range(len(EXPS)):
                for _m in range(M):
                    _ax = _axs[_ei][_m]; _cell = traj_grid.get((_ei, _m, _n), {})
                    for _lb, _d in _cell.items():
                        _er = np.asarray(_d["eps_rel"])
                        _ax.plot(np.arange(1, len(_er)+1), _er, "-", color=sched_colors[_lb], linewidth=1.5, label=_lb.split()[-1])
                    _ax.axhline(0.0, color="#888888", linewidth=0.8); _ax.set_title(f"{EXPS[_ei].split('/')[-1]} - m={_m}", fontsize=9)
                    _ax.set_xlabel("$t$", fontsize=8)
                    if _m == 0: _ax.set_ylabel(r"$\varepsilon_t/\varepsilon_0$", fontsize=8)
                    if _ei == 0 and _m == M-1 and _cell: _ax.legend(fontsize=6)
        _f.suptitle(f"Reliability - normalized gap over flow steps, n={_n}" + ("  (m aggregated)" if _agg else ""), fontsize=12)
        _f.tight_layout()
        _brel = min(abs(_r["eps_final"][0]) for _r in metrics.values())
        _rows = ["| sweep point | n_pool | target diff (mean +/- std) |", "|---|---|---|"]
        for _lb, _r in metrics.items():
            _v = _r["eps_final"]; _c = f"{_v[0]:+.4f} +/- {_v[1]:.4f}"
            _rows.append(f"| {_lb} | {_r['n_pool']} | {('**'+_c+'**') if abs(_v[0])==_brel else _c} |")
        t1_view = mo.vstack([mo.as_html(_f), mo.md("\n".join(_rows))])
    t1_view
    return


@app.cell(hide_code=True)
def _(aggregate_m_checkbox, mo):
    mo.vstack([
        mo.md(r"""## Interference
    - Guided-state convergence $M(\hat x^{\mathrm{det}}+\sigma_r z_t)-A$ per schedule (average mask space, target variable).
    - Table: total guidance $=\sum_t|\text{guidance move}|$."""),
        aggregate_m_checkbox,
    ])
    return


@app.cell
def _(
    EXPS,
    M,
    aggregate_m_checkbox,
    metrics,
    mo,
    n_slider,
    np,
    plt,
    sched_colors,
    traj_grid,
):
    # ---- Interference: guided-state convergence grid (no ung lines) ----
    if metrics is None or traj_grid is None:
        t2_view = mo.md("_press **compute metrics**_")
    else:
        _n = int(n_slider.value); _agg = aggregate_m_checkbox.value; _labels = list(sched_colors)
        if _agg:
            _f, _axs = plt.subplots(len(EXPS), 1, figsize=(7.5, 3.1*len(EXPS)), squeeze=False)
            for _ei in range(len(EXPS)):
                _ax = _axs[_ei][0]
                for _lb in _labels:
                    _cs = [np.asarray(traj_grid[(_ei, _m, _n)][_lb]["states"]) for _m in range(M)
                           if (_ei, _m, _n) in traj_grid and _lb in traj_grid[(_ei, _m, _n)]]
                    if not _cs: continue
                    _arr = np.stack(_cs); _x = np.arange(_arr.shape[1]); _c = sched_colors[_lb]
                    _ax.fill_between(_x, np.nanmin(_arr, 0), np.nanmax(_arr, 0), color=_c, alpha=0.18, linewidth=0)
                    _ax.plot(_x, np.nanmean(_arr, 0), "-", color=_c, linewidth=1.9, label=_lb.split()[-1])
                _ax.axhline(0.0, color="#888888", linewidth=0.8); _ax.set_title(EXPS[_ei].split('/')[-1], fontsize=10)
                _ax.set_xlabel("$t$"); _ax.set_ylabel("masked mean - target")
                if _ei == 0: _ax.legend(fontsize=7)
        else:
            _f, _axs = plt.subplots(len(EXPS), M, figsize=(5.0*M, 3.2*len(EXPS)), squeeze=False)
            for _ei in range(len(EXPS)):
                for _m in range(M):
                    _ax = _axs[_ei][_m]; _cell = traj_grid.get((_ei, _m, _n), {})
                    for _lb, _d in _cell.items():
                        _st = np.asarray(_d["states"])
                        _ax.plot(np.arange(len(_st)), _st, "-", color=sched_colors[_lb], linewidth=1.5, label=_lb.split()[-1])
                    _ax.axhline(0.0, color="#888888", linewidth=0.8); _ax.set_title(f"{EXPS[_ei].split('/')[-1]} - m={_m}", fontsize=9)
                    _ax.set_xlabel("$t$", fontsize=8)
                    if _m == 0: _ax.set_ylabel("masked mean - target", fontsize=8)
                    if _ei == 0 and _m == M-1 and _cell: _ax.legend(fontsize=6)
        _f.suptitle(f"Interference - convergence of the guided state, n={_n}" + ("  (m aggregated)" if _agg else ""), fontsize=12)
        _f.tight_layout()
        _best = min(_r["total_guidance"][0] for _r in metrics.values())
        _rows = ["| sweep point | n_pool | total guidance |", "|---|---|---|"]
        for _lb, _r in metrics.items():
            _v = _r["total_guidance"]; _c = f"{_v[0]:.3f}+/-{_v[1]:.3f}"
            _rows.append(f"| {_lb} | {_r['n_pool']} | {('**'+_c+'**') if _v[0]==_best else _c} |")
        t2_view = mo.vstack([mo.as_html(_f), mo.md("\n".join(_rows))])
    t2_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Closeness
    - **Deviance** $=\sum_t\lVert c\,s_t\,\mathrm{gui\_vec}_t\rVert/\sqrt F$ — per-step guidance vs the unguided step from the same state (any schedule).
    - **End distance from gui\_ung** $=\lVert g_T-u_T\rVert/\sqrt F$ (**higher = better**).
    - PCA grid: ERA5 cloud, guided (solid), gui\_ung twin (dashed), GT $\star$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # PCA view controls (3D angle + zoom-to-trajectories) -- live in the Closeness subsection
    elev_slider = mo.ui.slider(0, 90, value=22, step=2, label="elev: ", show_value=True)
    azim_slider = mo.ui.slider(-180, 180, value=-60, step=5, label="azim: ", show_value=True)
    zoom_traj_checkbox = mo.ui.checkbox(label="zoom to trajectories")
    zoom_pad_slider = mo.ui.slider(-0.4, 2.0, step=0.05, value=0.12, label="zoom pad: ", show_value=True)
    mo.hstack([elev_slider, azim_slider, zoom_traj_checkbox, zoom_pad_slider], justify="start", align="center")
    return azim_slider, elev_slider, zoom_pad_slider, zoom_traj_checkbox


@app.cell
def _(BBOX, LEVEL, VAR, ensure_basis, mo):
    # persisted PCA basis: fit once on the 2020 climatology (~1464 steps) for this region,
    # save the 3 PCs to file, reuse (project). (cloud background lives in the Closeness section.)
    basis = ensure_basis(VAR, LEVEL, BBOX, source="era5")
    basis_info = mo.md(
        f"**PCA basis** - region F={basis.F} px - fit on **{basis.meta['n_points']}** climatology "
        f"states (2020) - EVR(3) = "
        f"{', '.join(f'{e:.0%}' for e in basis.evr)} (sum={float(basis.evr.sum()):.0%})"
    )
    basis_info
    return (basis,)


@app.cell(hide_code=True)
def _(BBOX, LEVEL, VAR, basis, cloud_sample, project):
    # PCA-plot background: a projected subsample of the climatology cloud the basis was fit on
    cloud_proj = project(basis, cloud_sample(VAR, LEVEL, BBOX, "era5", max_points=400))
    return (cloud_proj,)


@app.cell
def _(
    EXPS,
    M,
    azim_slider,
    basis,
    cloud_proj,
    elev_slider,
    gt_proj,
    metrics,
    mo,
    n_slider,
    np,
    plt,
    sched_colors,
    traj_grid,
    ung_proj,
    zoom_pad_slider,
    zoom_traj_checkbox,
):
    # ---- Closeness: full PCA path grid (ported), rows=exp, cols=m, with zoom-to-trajectories ----
    if metrics is None or traj_grid is None:
        t3_view = mo.md("_press **compute metrics**_")
    else:
        _n = int(n_slider.value); _nr, _nc = len(EXPS), M
        _f, _axs = plt.subplots(_nr, _nc, figsize=(5.2*_nc, 4.6*_nr), subplot_kw={"projection": "3d"}, squeeze=False, dpi=100)
        for _ei in range(_nr):
            for _m in range(_nc):
                _ax = _axs[_ei][_m]; _grp = []
                _ax.scatter(cloud_proj[:, 0], cloud_proj[:, 1], cloud_proj[:, 2], color="#BBBBBB", s=8, alpha=0.45,
                            depthshade=False, label=f"ERA5 cloud ({cloud_proj.shape[0]})")
                _cell = traj_grid.get((_ei, _m, _n), {})
                for _lb, _d in _cell.items():
                    _pg = np.asarray(_d["pg"]); _c = sched_colors[_lb]
                    _ax.plot(_pg[:, 0], _pg[:, 1], _pg[:, 2], "-", color=_c, linewidth=1.4, alpha=0.9, label=_lb.split()[-1])
                    _ax.scatter(_pg[-1, 0], _pg[-1, 1], _pg[-1, 2], marker="o", s=45, color=_c, depthshade=False)
                    _pu = np.asarray(_d["pu"])
                    _ax.plot(_pu[:, 0], _pu[:, 1], _pu[:, 2], "--", color=_c, linewidth=1.0, alpha=0.6)
                    _ax.scatter(_pu[-1, 0], _pu[-1, 1], _pu[-1, 2], marker="o", s=34, color="black", depthshade=False)
                    _ax.text(_pu[-1, 0], _pu[-1, 1], _pu[-1, 2], "  gui_ung", fontsize=7, color="#111111")
                    _grp.extend([_pg, _pu])
                if (_ei, _n) in gt_proj:
                    _tp = np.asarray(gt_proj[(_ei, _n)])
                    _ax.scatter(_tp[0], _tp[1], _tp[2], marker="*", s=120, color="black", depthshade=False)
                    _ax.text(_tp[0], _tp[1], _tp[2], "  gt", fontsize=7, fontweight="bold"); _grp.append(_tp[None, :])
                if (_ei, _m, _n) in ung_proj:
                    _op = np.asarray(ung_proj[(_ei, _m, _n)])
                    _ax.scatter(_op[0], _op[1], _op[2], marker="o", s=34, color="black", depthshade=False)
                    _ax.text(_op[0], _op[1], _op[2], "  ung", fontsize=7, color="#111111"); _grp.append(_op[None, :])
                if _cell:
                    _sp = np.asarray(next(iter(_cell.values()))["pg"])[0]
                    _ax.scatter(_sp[0], _sp[1], _sp[2], marker="o", s=34, color="black", depthshade=False)
                    _ax.text(_sp[0], _sp[1], _sp[2], "  start", fontsize=7)
                if zoom_traj_checkbox.value and _grp:
                    _pts = np.vstack(_grp); _lo, _hi = _pts.min(0), _pts.max(0)
                    _pad = zoom_pad_slider.value * float((_hi - _lo).max())
                    _ax.set_xlim(_lo[0]-_pad, _hi[0]+_pad); _ax.set_ylim(_lo[1]-_pad, _hi[1]+_pad); _ax.set_zlim(_lo[2]-_pad, _hi[2]+_pad)
                _ax.set_xlabel("PC1", fontsize=7); _ax.set_ylabel("PC2", fontsize=7); _ax.set_zlabel("PC3", fontsize=7)
                _ax.view_init(elev=elev_slider.value, azim=azim_slider.value)
                _ax.set_title(f"{EXPS[_ei].split('/')[-1]} - m={_m}", fontsize=9)
                if _ei == 0 and _m == _nc-1 and _cell:
                    _ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        _f.suptitle(f"Closeness - latent paths (PCA), EVR={basis.evr.sum():.0%}, n={_n}  (solid=guided, dashed=gui_ung, *=GT, o=ung)", fontsize=12)
        _f.tight_layout()
        _bg = min(_r["guidance_steps"][0] for _r in metrics.values())
        _bp = min(_r["pushback_steps"][0] for _r in metrics.values())
        _be = max(_r["end_dist"][0] for _r in metrics.values())
        _rows = ["| sweep point | n_pool | guidance steps | pushback steps | end dist from gui_ung |", "|---|---|---|---|---|"]
        for _lb, _r in metrics.items():
            _gs = _r["guidance_steps"]; _gc = f"{_gs[0]:.3g}+/-{_gs[1]:.2g}"; _gc = f"**{_gc}**" if _gs[0]==_bg else _gc
            _ps = _r["pushback_steps"]; _pc = f"{_ps[0]:.3g}+/-{_ps[1]:.2g}"; _pc = f"**{_pc}**" if _ps[0]==_bp else _pc
            _ed = _r["end_dist"]; _ec = f"{_ed[0]:.3g}+/-{_ed[1]:.2g}"; _ec = f"**{_ec}**" if _ed[0]==_be else _ec
            _rows.append(f"| {_lb} | {_r['n_pool']} | {_gc} | {_pc} | {_ec} |")
        t3_view = mo.vstack([mo.as_html(_f), mo.md("\n".join(_rows))])
    t3_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Realism
    - One figure per experiment. Row 1: guidance effect $|x^{\mathrm{gui}}-x^{\mathrm{ung}}|$; Row 2: trajectory-avg guidance vector (mask-normalized), per guidance mode.
    - Mask beside spike@24; per-row $0\to$max colorbar (mask-map colouring). TV between guidance change and mask (**higher = better**).
    """)
    return


@app.cell
def _(
    EXPS,
    WARM_CMAP,
    mask_bbox_ref,
    mcolors,
    metrics,
    mo,
    n_slider,
    np,
    plt,
    sched_colors,
    traj_grid,
):
    # ---- Realism: one figure per experiment; row 1 = guidance effect, row 2 = traj-avg guidance vector ----
    if metrics is None or traj_grid is None:
        t4_view = mo.md("_press **compute metrics**_")
    else:
        _n = int(n_slider.value); _labels = list(sched_colors)
        _cols = ["mask"] + _labels
        _mk = np.asarray(mask_bbox_ref, float); _mkp = _mk/_mk.sum() if _mk.sum() > 0 else _mk
        _figs = []
        for _ei in range(len(EXPS)):
            _cell = traj_grid.get((_ei, 0, _n), {})
            _row0, _row1 = [], []
            for _col in _cols:
                if _col == "mask":
                    _row0.append(_mkp); _row1.append(None)
                elif _col in _cell:
                    _gf = np.asarray(_cell[_col]["gfield"], float); _row0.append(_gf/_gf.sum() if _gf.sum() > 0 else _gf)
                    _gv = np.asarray(_cell[_col]["gvec_avg"], float); _row1.append(_gv/_gv.sum() if _gv.sum() > 0 else _gv)
                else:
                    _row0.append(None); _row1.append(None)
            _m0 = max((float(np.nanmax(_p)) for _p in _row0 if _p is not None), default=1.0)
            _m1 = max((float(np.nanmax(_p)) for _p in _row1 if _p is not None), default=1.0)
            _f = plt.figure(figsize=(2.4*len(_cols) + 0.7, 5.0), dpi=110)
            _gs = _f.add_gridspec(2, len(_cols) + 1, width_ratios=[1]*len(_cols) + [0.07], wspace=0.08, hspace=0.12)
            for _ci in range(len(_cols)):
                _a0 = _f.add_subplot(_gs[0, _ci])
                if _row0[_ci] is not None:
                    _a0.imshow(_row0[_ci], cmap=WARM_CMAP, vmin=0.0, vmax=_m0)
                _a0.set_xticks([]); _a0.set_yticks([])
                _a0.set_title("mask" if _cols[_ci] == "mask" else _cols[_ci].split()[-1], fontsize=8)
                if _ci == 0:
                    _a0.set_ylabel("guidance effect", fontsize=7)
                _a1 = _f.add_subplot(_gs[1, _ci])
                if _row1[_ci] is not None:
                    _a1.imshow(_row1[_ci], cmap=WARM_CMAP, vmin=0.0, vmax=_m1)
                _a1.set_xticks([]); _a1.set_yticks([])
                if _ci == 0:
                    _a1.set_ylabel("guidance vector\\n(traj-avg)", fontsize=7)
            _c0 = _f.add_subplot(_gs[0, len(_cols)])
            _f.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0.0, _m0), cmap=WARM_CMAP), cax=_c0); _c0.tick_params(labelsize=6)
            _c1 = _f.add_subplot(_gs[1, len(_cols)])
            _f.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0.0, _m1), cmap=WARM_CMAP), cax=_c1); _c1.tick_params(labelsize=6)
            _f.suptitle(f"Realism - {EXPS[_ei].split('/')[-1]}  (m=0, n={_n})", fontsize=11)
            _figs.append(mo.as_html(_f))
        _besttv = max(_r["realism_tv"][0] for _r in metrics.values())
        _rows = ["| sweep point | n_pool | realism TV (mean+/-std) |", "|---|---|---|"]
        for _lb, _r in metrics.items():
            _v = _r["realism_tv"]; _c = f"{_v[0]:.3f}+/-{_v[1]:.3f}"
            _rows.append(f"| {_lb} | {_r['n_pool']} | {('**'+_c+'**') if _v[0]==_besttv else _c} |")
        t4_view = mo.vstack(_figs + [
            mo.md(r"Row 1: guidance effect $|x^{gui}-x^{ung}|$. Row 2: trajectory-avg $|$guidance vector$|$, mask-normalized. TV vs mask (**higher = better**)."),
            mo.md("\n".join(_rows))])
    t4_view
    return


@app.cell
def _(metrics, mo):
    if metrics is None:
        leaderboard = mo.md("")
    else:
        _best = {_k: (max if _k == "realism_tv" else min)(_r[_k][0] for _r in metrics.values()) for _k in ("total_guidance", "guidance_steps", "realism_tv")}
        _rows = ["| sweep point | total guidance | guidance steps | realism TV |", "|---|---|---|---|"]
        for _lb, _r in metrics.items():
            _tg = f"{_r['total_guidance'][0]:.3f}"; _tg = f"**{_tg}**" if _r['total_guidance'][0] == _best['total_guidance'] else _tg
            _l = f"{_r['guidance_steps'][0]:.3g}"; _l = f"**{_l}**" if _r['guidance_steps'][0] == _best['guidance_steps'] else _l
            _tv = f"{_r['realism_tv'][0]:.3f}"; _tv = f"**{_tv}**" if _r['realism_tv'][0] == _best['realism_tv'] else _tv
            _rows.append(f"| {_lb} | {_tg} | {_l} | {_tv} |")
        leaderboard = mo.vstack([mo.md("### Leaderboard  (best per column in bold)"), mo.md("\n".join(_rows))])
    leaderboard
    return


if __name__ == "__main__":
    app.run()
