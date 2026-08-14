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
    from src.pca_basis import mask_bbox, bbox_latent, ensure_basis, project, pathlength, cloud_sample, region_bool, region_latent, ensure_region_basis, region_cloud_sample
    from src import flow_time_tests as ftt
    from src.mask import get_masked_mean
    from src.ui.plot_trajectory import plot_trajectory
    from src.utils import get_rollout_ids, get_gt_rollout, get_rollout_dir, find_era5_input
    from datetime import datetime
    import xarray as xr
    from src.dimensions import LEVELS_DICT

    # line colours: sampled per run from a colormap sized to the number of schedules
    # (see sched_colors), so any count of schedules gets all-distinct colours
    # same warm half of RdBu_r as the mask map in guidance.py (white at 0 -> red at max)
    WARM_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "rdbu_warm", plt.get_cmap("RdBu_r")(np.linspace(0.5, 1.0, 256)))


    def glabel(lb):
        """Display label: the sweep key `a_t_mode` IS the flow-time profile gamma, so
        render it as 'γ' in every table/legend/title (report notation)."""
        return str(lb).replace("a_t_mode", "γ")


    return (
        LEVELS_DICT,
        WARM_CMAP,
        channel,
        clean_pred_trajectory_primitive,
        convergence_state_line,
        datetime,
        ensure_region_basis,
        find_era5_input,
        ftt,
        get_gt_rollout,
        get_masked_mean,
        get_rollout_dir,
        get_rollout_ids,
        glabel,
        guided_velocity_primitive,
        load_rollout,
        mask_bbox,
        mo,
        np,
        open_store,
        pathlength,
        plt,
        project,
        region_bool,
        region_cloud_sample,
        region_latent,
        residual_scaler,
        select_point,
        sweep_points,
        xr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Flow-time tests

    Flow step $t$ = the $T$ internal Euler steps of one `sample()` call; $n$ indexes the weather step and
    $\hat{x}_{n,t}$ the clean-state estimate. Metrics score the guided channel `VAR@LEVEL` through the
    masked average $\mathcal{M}_m$ (mask $m$), pooled over ensemble members, guided weather steps $n$,
    and the selected experiment's subexperiments, per guidance schedule (flow-time profile $\gamma_t$,
    here `spike@k`). Metrics recompute automatically on selection.

    - **T1 Reliability**: did we reach the target? Final gap $\xi_n=\mathcal{M}_m(x_n^{\mathrm{gui}})-y_n^\star$.
    - **T2 Guidance effect**: how much change was inflicted to other variables.
    - **T3 Closeness**: how far the guided flow strays from the unguided one (PCA latent space).
    - **T4 Realism**: does the guidance footprint match the mask?
    """)
    return


@app.cell
def _(get_rollout_ids, mo):
    # OUTER experiment selector (single dropdown; defaults to the LATEST experiment). Metrics are
    # averaged over ALL its date-subexperiments (which share the same sweep).
    _exp_ids = sorted({rid.split("/", 1)[0] for rid in get_rollout_ids("gui")})
    experiment_dropdown = mo.ui.dropdown(
        _exp_ids, value=_exp_ids[-1] if _exp_ids else None, label="experiment: "
    )
    return (experiment_dropdown,)


@app.cell
def _(
    experiment_dropdown,
    get_rollout_ids,
    load_rollout,
    mask_bbox,
    np,
    sweep_points,
):
    # all date-subexperiments of the selected experiment(s) -> metrics are pooled and
    # averaged over them (they share VAR/LEVEL/mask/sweep, hence one bbox + one PCA basis)
    _sel_exps = {experiment_dropdown.value}
    EXPS = [rid for rid in get_rollout_ids("gui") if rid.split("/", 1)[0] in _sel_exps]
    _rid0 = EXPS[0]
    _dir0, config0, sweep_values0, records0, mask0 = load_rollout(_rid0)
    mask0 = np.asarray(mask0, dtype=float)
    MASK0 = mask0
    VAR = config0["VAR"]
    LEVEL = config0["LEVEL"]
    PARTITION = config0["PARTITION"]
    N = int(config0["N"])
    M = int(config0["M"])
    BBOX = mask_bbox(mask0)
    points = sweep_points(sweep_values0, records0)
    # order the sweep points by the a_t_mode order authored in sweep_params.json
    # (sweep_values0), not the zarr coord/append order
    _atm_ord = {str(_v): _i for _i, _v in enumerate(sweep_values0.get('a_t_mode', []))}
    points = dict(sorted(points.items(),
                         key=lambda _kv: _atm_ord.get(str(_kv[1].get('a_t_mode', '')), len(_atm_ord))))
    return BBOX, EXPS, LEVEL, M, MASK0, N, PARTITION, VAR, points


@app.cell
def _(mo, points):
    point_multiselect = mo.ui.multiselect(
        list(points), value=list(points), label="sweep points: "
    )
    return (point_multiselect,)


@app.cell(hide_code=True)
def _(N, mo):
    # global step selector (m is the grid columns; PCA view controls live in the Closeness section)
    n_slider = mo.ui.slider(1, max(N, 2), value=1, step=1, label="n: ", show_value=True)
    return (n_slider,)


@app.cell(hide_code=True)
def _(mo):
    # collapse the m-columns of the first two sections into one chart per experiment,
    # with the spread across m shown as shading behind the mean line
    aggregate_m_checkbox = mo.ui.checkbox(value=False, label="aggregate m (spread as shading)")
    return (aggregate_m_checkbox,)


@app.cell
def _(EXPS, experiment_dropdown, mo, n_slider, point_multiselect):
    mo.vstack([
        experiment_dropdown,
        mo.md(f"_averaging over **{len(EXPS)}** subexperiment(s): "
              f"{', '.join(e.split('/')[-1] for e in EXPS)}_"),
        mo.hstack([point_multiselect, n_slider], justify="start", align="center"),
    ], align="start")
    return


@app.cell
def _(
    BBOX,
    EXPS,
    LEVEL,
    N,
    PARTITION,
    PCA_REGION,
    VAR,
    basis,
    channel,
    clean_pred_trajectory_primitive,
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
    plt,
    point_multiselect,
    project,
    region_latent,
    residual_scaler,
    select_point,
    sweep_points,
):
    # ---- compute: pooled metrics + per-(exp, m, n) grid trajectories ----
    metrics = None
    traj_grid = None
    gt_proj = None
    sched_colors = None
    ref_metrics = None
    if True:  # auto-compute
        _labels = list(point_multiselect.value)
        sched_colors = {_l: plt.get_cmap("turbo")(_i / max(len(_labels) - 1, 1)) for _i, _l in enumerate(_labels)}
        _keys = ("eps_final", "total_guidance", "pushback_count", "pushback_amount",
                 "overshoot_count", "overshoot_amount", "L", "L_twin", "guidance_steps", "pushback_steps", "end_dist", "realism_tv", "dpc1", "dpc2", "dpc3", "gvec_res_sim")
        _acc = {lb: {k: [] for k in _keys} for lb in _labels}
        _reached_acc = {lb: [] for lb in _labels}
        traj_grid = {}
        gt_proj = {}
        ung_proj = {}
        ung_traj = {}
        guiung_traj = {}
        mask_bbox_ref = None
        _ref_acc = {"ung": {"L": [], "dpc1": [], "dpc2": [], "dpc3": []},
                    "gui_ung": {"L": [], "dpc1": [], "dpc2": [], "dpc3": []}}
        for _ei, _rid in enumerate(EXPS):
            _dir, _cfg, _sv, _recs, _mask = load_rollout(_rid)
            _mask = np.asarray(_mask, dtype=float)
            mask_bbox_ref = _mask[BBOX[0], BBOX[1]]
            _c = residual_scaler(PARTITION, VAR, LEVEL)
            _sp = sweep_points(_sv, _recs)
            _pts = {lb: _sp[lb] for lb in _labels if lb in _sp}
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
                        _guif = np.asarray(_gui.isel(m=_m, n=_n), float)
                        _s_gui = float(get_masked_mean(_guif, _mask))
                        _states, _land = convergence_state_line(_dir, _sel, _m, _n, VAR, _c, _mask, _A, level=LEVEL)
                        _inter = ftt.interference_from_convergence(_states, _land)
                        _g = region_latent(_traj, PCA_REGION)
                        _pg = project(basis, _g)
                        _ungf = np.asarray(_tw.isel(m=_m, n=_n).isel(t=-1), float)
                        _u = region_latent(np.asarray(_tw.isel(m=_m, n=_n), float), PCA_REGION)
                        _pu = project(basis, _u)
                        _L = pathlength(_pg); _Lt = pathlength(_pu)
                        _vel = guided_velocity_primitive(_dir, _sel, _m, _n, VAR, _c, level=LEVEL)
                        # pingpong in FULL bbox-latent space (per-pixel RMS): base_t -> kick_t = GUIDANCE
                        # step, kick_t -> base_{t+1} = PUSHBACK. The 3-PC plot understates the pushback
                        # (mostly out-of-plane -> inverts the ratio), so we report the faithful full-field
                        # deviation; kick = base - c*s*gui_vec (clean-pred branch, step-t kick removed).
                        _kick = region_latent(_traj - _c * _vel["s"][:, None, None] * _vel["gui_vec"], PCA_REGION)
                        _pk = project(basis, _kick)
                        _vfs_lat = region_latent(_c * _vel["vfs"], PCA_REGION)   # vfs field -> region latents
                        _pgx = project(basis, _g - _vfs_lat)                       # actual state x_t = clean-pred - c*vfs
                        _pkx = project(basis, _kick - _vfs_lat)                    # x_t kick branch
                        _gstep, _pstep = ftt.step_sums(_g, _kick)
                        _end = float(np.linalg.norm(_g[-1] - _u[-1]) / np.sqrt(_g.shape[1]))
                        _tv = ftt.realism_tv(_guif, _ungf, _mask)
                        _gfield = np.abs(_guif - _ungf)[BBOX[0], BBOX[1]]
                        _gvec_avg = np.abs(_vel["gui_vec"]).mean(axis=0)[BBOX[0], BBOX[1]]
                        # similarity (cosine) of the normalized avg-guidance vector to the
                        # guidance residual |x_gui - x_ung| over the mask bbox
                        _pe = _gfield.ravel().astype(float); _pv = _gvec_avg.ravel().astype(float)
                        _sim = float(_pe @ _pv / (np.linalg.norm(_pe) * np.linalg.norm(_pv) + 1e-12))
                        _a = _acc[_lb]
                        for _k in ("total_guidance", "pushback_count", "pushback_amount", "overshoot_count", "overshoot_amount"):
                            _a[_k].append(_inter[_k])
                        _a["L"].append(_L); _a["L_twin"].append(_Lt); _a["guidance_steps"].append(_gstep); _a["pushback_steps"].append(_pstep); _a["end_dist"].append(_end); _a["realism_tv"].append(_tv)
                        _a["eps_final"].append(float(_s_gui - _A))
                        _reached_acc[_lb].append(int(abs((_s_gui - _A) / max(abs(_A - _base), 1e-12)) <= 0.01))
                        _dpc = _pg[-1] - _pu[-1]
                        _a["dpc1"].append(float(_dpc[0])); _a["dpc2"].append(float(_dpc[1])); _a["dpc3"].append(float(_dpc[2])); _a["gvec_res_sim"].append(_sim)
                        traj_grid.setdefault((_ei, _m, _n), {})[_lb] = {
                            "pg": _pg, "pk": _pk, "pgx": _pgx, "pkx": _pkx, "pu": _pu, "states": _states, "land_ung": _land, "gfield": _gfield,
                            "gvec_avg": _gvec_avg}
            # reference rows (ung + gui_ung): PCA path length + endpoint PC-deviation from
            # gui_ung; guidance/pushback are 0 (unguided). twin is a_t_mode-independent -> any sel.
            if _pts:
                _sel0 = next(iter(_pts.values()))
                _p0 = np.asarray(_sv["GUIDANCE_DELTA"][_sel0["GUIDANCE_DELTA"]], float)[:N]
                _ung_da = channel(open_store(_dir, "ung", VAR), _cfg)
                _giu_da = channel(select_point(open_store(_dir, "gui_ung", VAR), _sel0), _cfg)
                for _m in range(_ung_da.sizes["m"]):
                    for _n in range(N):
                        if float(_p0[_n]) == 0.0:
                            continue
                        _pu_ref = project(basis, region_latent(np.asarray(_giu_da.isel(m=_m, n=_n), float), PCA_REGION))
                        _pun = project(basis, region_latent(np.asarray(_ung_da.isel(m=_m, n=_n), float), PCA_REGION))
                        ung_traj[(_ei, _m, _n)] = _pun
                        guiung_traj[(_ei, _m, _n)] = _pu_ref
                        _ref_acc["gui_ung"]["L"].append(pathlength(_pu_ref))
                        _ref_acc["ung"]["L"].append(pathlength(_pun))
                        _dref = _pun[-1] - _pu_ref[-1]
                        for _j, _kk in enumerate(("dpc1", "dpc2", "dpc3")):
                            _ref_acc["gui_ung"][_kk].append(0.0)
                            _ref_acc["ung"][_kk].append(float(_dref[_j]))
            if _ung_base is not None:
                for _m in range(_ung_final.shape[0]):
                    for _n in range(N):
                        ung_proj[(_ei, _m, _n)] = project(basis, region_latent(_ung_final[_m, _n], PCA_REGION))
            if _gt_field is not None:
                for _n in range(N):
                    gt_proj[(_ei, _n)] = project(basis, region_latent(_gt_field[_n + 1], PCA_REGION))
        metrics = {}
        for _lb in _labels:
            _a = _acc[_lb]; _rec = {k: ftt.aggregate_mean_std(v) for k, v in _a.items()}
            _rec["n_pool"] = len(_a["guidance_steps"])
            _rec["reached_count"] = int(sum(_reached_acc[_lb]))
            metrics[_lb] = _rec
        ref_metrics = {}
        for _rlb in ("gui_ung", "ung"):
            _ra = _ref_acc[_rlb]
            if _ra["L"]:
                _rrec = {_k: ftt.aggregate_mean_std(_v) for _k, _v in _ra.items()}
                _rrec["guidance_steps"] = (0.0, 0.0); _rrec["pushback_steps"] = (0.0, 0.0)
                _rrec["n_pool"] = len(_ra["L"])
                ref_metrics[_rlb] = _rrec
    return (
        gt_proj,
        guiung_traj,
        mask_bbox_ref,
        metrics,
        ref_metrics,
        sched_colors,
        traj_grid,
        ung_traj,
    )


@app.cell
def _(glabel, interf_data, metrics, mo, np):
    if metrics is None:
        leaderboard = mo.md("## Leaderboard\n\n_press **compute metrics**_")
    else:
        _has_i = interf_data is not None
        def _push(_lb):
            return interf_data["avg_score_ms"].get(_lb) if _has_i else None
        def _reached(_r):
            _n = _r["n_pool"]; _p = (_r["reached_count"] / _n) if _n else float("nan")
            _sd = float(np.sqrt(max(_p * (1.0 - _p), 0.0) / _n)) if _n else float("nan")
            return (_p, _sd)
        def _dist_ms(_r):
            # PC-space dist to gui_ung + first-order propagated std from the dPC stds
            _mu = (_r["dpc1"][0], _r["dpc2"][0], _r["dpc3"][0])
            _sd = (_r["dpc1"][1], _r["dpc2"][1], _r["dpc3"][1])
            _d = float(np.sqrt(sum(_x * _x for _x in _mu)))
            _ds = float(np.sqrt(sum((_x * _y) ** 2 for _x, _y in zip(_mu, _sd))) / _d) if _d > 0 else 0.0
            return (_d, _ds)
        # per-column winners: max everywhere, except dPC = max |deviation|
        _bre = max((_reached(_r)[0] for _r in metrics.values() if np.isfinite(_reached(_r)[0])), default=None)
        _bpu = max((v[0] for v in (_push(_lb) for _lb in metrics) if v is not None and np.isfinite(v[0])), default=None)
        _bd = {_k: max(abs(_r[_k][0]) for _r in metrics.values()) for _k in ("dpc1", "dpc2", "dpc3")}
        _bdi = max(_dist_ms(_r)[0] for _r in metrics.values())
        _bt = max(_r["realism_tv"][0] for _r in metrics.values())
        def _f(_ms, _is_best, signed=False):
            if _ms is None or not np.isfinite(_ms[0]):
                return "—"
            _s = f"{_ms[0]:+.3f} ± {_ms[1]:.3f}" if signed else f"{_ms[0]:.3f} ± {_ms[1]:.3f}"
            return f"**{_s}**" if _is_best else _s
        _rows = ["| sweep point | reached ratio | push score | ΔPC1 | ΔPC2 | ΔPC3 | dist to gui_ung | realism TV |",
                 "|---|---|---|---|---|---|---|---|"]
        for _lb, _r in metrics.items():
            _rc = _reached(_r); _dm = _dist_ms(_r); _pu = _push(_lb)
            _rows.append(
                f"| {glabel(_lb)} "
                f"| {_f(_rc, _bre is not None and _rc[0] == _bre)} "
                f"| {_f(_pu, _pu is not None and _bpu is not None and _pu[0] == _bpu)} "
                f"| {_f(_r['dpc1'], abs(_r['dpc1'][0]) == _bd['dpc1'], signed=True)} "
                f"| {_f(_r['dpc2'], abs(_r['dpc2'][0]) == _bd['dpc2'], signed=True)} "
                f"| {_f(_r['dpc3'], abs(_r['dpc3'][0]) == _bd['dpc3'], signed=True)} "
                f"| {_f(_dm, _dm[0] == _bdi)} "
                f"| {_f(_r['realism_tv'], _r['realism_tv'][0] == _bt)} |")
        leaderboard = mo.vstack([
            mo.md("## Leaderboard"),
            mo.md("mean ± std over the pool (3 dp); **bold** = winner per column. **reached ratio** = fraction of "
                  "pooled runs within 1% of the target (± binomial std). Winners are **max** everywhere except "
                  "ΔPCk = **max |deviation|** (most divergent on that PC) and dist to gui_ung = **max** (most divergent). "
                  "**push score** = variable-normalized avg-space push (needs the **interference profiles** computed, else —). "
                  "ΔPCk = signed endpoint deviation from gui_ung along PCk; dist to gui_ung = its PC-space norm."),
            mo.md("\n".join(_rows))])
    leaderboard
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## T1: Reliability

    Did the guidance hit the target $y_n^\star=(1+\rho_n)\,\mathcal{M}_m(x_n^{\mathrm{ung}})$?

    - **final gap** $\xi_n=\mathcal{M}_m(x_n^{\mathrm{gui}})-y_n^\star$ (masked average), mean $\pm$ std over the pool.
    - **reached** = pooled runs within a 1% relative miss $|\xi_n|/|y_n^\star-\mathcal{M}_m(x_n^{\mathrm{ung}})|\le0.01$.
    - **target reached** = 🎯 if ALL pooled runs are within 1%, else ❌.
    - **convergence** (one figure per experiment, below): the gap $\xi_{n,t}=\mathcal{M}_m(\hat{x}_{n,t})-y_n^\star$ over flow time converging to $0$ (toggle `aggregate m`).
    """)
    return


@app.cell
def _(
    EXPS,
    M,
    aggregate_m_checkbox,
    glabel,
    metrics,
    mo,
    n_slider,
    np,
    plt,
    sched_colors,
    traj_grid,
):
    # ---- Reliability: guided-state convergence over flow time, one figure per experiment ----
    # The gap xi_{n,t} = M_m(x_hat_{n,t}) - y_n* per schedule, converging to 0 (moved here from
    # the Interference section; the bar chart + total-guidance table were removed).
    if metrics is None or traj_grid is None:
        reliability_conv_view = mo.md("_press **compute metrics**_")
    else:
        _n = int(n_slider.value) - 1; _agg = aggregate_m_checkbox.value; _labels = list(sched_colors)
        _items = []
        for _ei in range(len(EXPS)):
            _items.append(mo.md(f"**{EXPS[_ei].split(chr(47))[-1]}**  (n={_n + 1})"))
            if _agg:
                _f, _ax = plt.subplots(1, 1, figsize=(6.0, 2.8), dpi=120)
                for _lb in _labels:
                    _cs = [np.asarray(traj_grid[(_ei, _m, _n)][_lb]["states"]) for _m in range(M)
                           if (_ei, _m, _n) in traj_grid and _lb in traj_grid[(_ei, _m, _n)]]
                    if not _cs: continue
                    _arr = np.stack(_cs); _x = np.arange(_arr.shape[1]); _c = sched_colors[_lb]
                    _ax.fill_between(_x, np.nanmin(_arr, 0), np.nanmax(_arr, 0), color=_c, alpha=0.18, linewidth=0)
                    _ax.plot(_x, np.nanmean(_arr, 0), "-", color=_c, linewidth=1.8, label=glabel(_lb).split()[-1])
                _ax.axhline(0.0, color="#888888", linewidth=0.8)
                _ax.set_xlabel("$t$"); _ax.set_ylabel(r"gap $\xi_{n,t}$")
                _ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
            else:
                _f, _axs = plt.subplots(1, M, figsize=(3.6*M, 2.8), squeeze=False, dpi=120)
                for _m in range(M):
                    _ax = _axs[0][_m]; _cell = traj_grid.get((_ei, _m, _n), {})
                    for _lb in _labels:
                        if _lb not in _cell: continue
                        _st = np.asarray(_cell[_lb]["states"])
                        _ax.plot(np.arange(len(_st)), _st, "-", color=sched_colors[_lb], linewidth=1.5, label=glabel(_lb).split()[-1])
                    _ax.axhline(0.0, color="#888888", linewidth=0.8); _ax.set_title(f"m={_m}", fontsize=9)
                    _ax.set_xlabel("$t$", fontsize=8)
                    if _m == 0: _ax.set_ylabel(r"gap $\xi_{n,t}$", fontsize=8)
                    if _m == M - 1 and _cell: _ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
            _f.tight_layout(pad=0.3)
            _items.append(mo.as_html(_f))
        reliability_conv_view = mo.vstack([aggregate_m_checkbox, *_items], align="start")
    reliability_conv_view
    return


@app.cell
def _(glabel, metrics, mo):
    # ---- Reliability: target-diff + within-1% target-reached (no plots) ----
    if metrics is None:
        t1_view = mo.md("_press **compute metrics**_")
    else:
        _rows = ["| sweep point | n_pool | final gap $\\xi_n$ (mean +/- std) | reached | target reached |",
                 "|---|---|---|---|---|"]
        for _lb, _r in metrics.items():
            _v = _r["eps_final"]; _td = f"{_v[0]:+.4f} +/- {_v[1]:.4f}"
            _rc = _r["reached_count"]; _np = _r["n_pool"]
            _rows.append(f"| {glabel(_lb)} | {_np} | {_td} | {_rc}/{_np} | {chr(0x1F3AF) if _rc == _np else chr(0x274C)} |")
        t1_view = mo.md("\n".join(_rows))
    t1_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## T2: Guidance effect

    How much the guidance perturbs the flow, profiled across **all** variables/levels.
    **T2a — Push across all variables** gives three masked-mean profiles per level variable (aggregated over
    experiments): the **absolute** guided intensity $\mathcal{M}_m(x^{\mathrm{gui}})$, its deviation from
    **ground truth** $\mathcal{M}_m(x^{\mathrm{gui}})-\mathcal{M}_m(x^{\mathrm{gt}})$, and the **push**
    relative to the unguided twin $\mathcal{M}_m(x^{\mathrm{gui}})-\mathcal{M}_m(x^{\mathrm{gui\_ung}})$.
    **T2b** — push as the guidance kick $\|\nabla\mathcal{L}\|$ (lower = better).
    """)
    return


@app.cell
def _(mo, sched_colors):
    # T2 interference subtests (T2a/T2b): the per-channel push across ALL variables is heavy
    # (~1-2 min over the pool), so it is gated behind a button; the top-k slider then re-renders
    # the tables/grids instantly from the cached result.
    interf_button = mo.ui.run_button(label="compute interference profiles (~1-2 min)")
    interf_k_slider = mo.ui.slider(1, max(len(sched_colors), 2), value=min(3, max(len(sched_colors), 1)),
                                   step=1, label="top-k schedules: ", show_value=True)
    mo.hstack([interf_button, interf_k_slider], justify="start", align="center")
    return interf_button, interf_k_slider


@app.cell
def _(LEVELS_DICT):
    # constants for the interference subtests: the 6 level variables and their channel order
    # ('sfc' ALWAYS first, then levels bottom->top), like intensity_comparison's intensity_channels.
    INTERF_LEVEL_VARS = ["geopotential", "u_component_of_wind", "v_component_of_wind",
                         "temperature", "specific_humidity", "vertical_velocity"]
    INTERF_SURFACE_PAIR = {"temperature": "2m_temperature",
                           "u_component_of_wind": "10m_u_component_of_wind",
                           "v_component_of_wind": "10m_v_component_of_wind"}
    INTERF_VSHORT = {"geopotential": "z", "u_component_of_wind": "u", "v_component_of_wind": "v",
                     "temperature": "T", "specific_humidity": "q", "vertical_velocity": "w"}


    def interf_chan_order(_var):
        """Channel labels for one variable, surface->top. 'sfc' is ALWAYS the first channel (even
        for variables with no surface pair, e.g. geopotential) so every chart shares one aligned
        x-axis; variables without a surface value simply have a gap (NaN) at sfc."""
        return ["sfc"] + [f"L{_L}" for _L in reversed(LEVELS_DICT["level"])]

    return (
        INTERF_LEVEL_VARS,
        INTERF_SURFACE_PAIR,
        INTERF_VSHORT,
        interf_chan_order,
    )


@app.cell
def _(
    EXPS,
    INTERF_LEVEL_VARS,
    INTERF_SURFACE_PAIR,
    M,
    N,
    datetime,
    find_era5_input,
    get_gt_rollout,
    get_rollout_dir,
    interf_button,
    load_rollout,
    np,
    sched_colors,
    select_point,
    sweep_points,
    xr,
):
    # ---- heavy per-channel interference compute (button-gated) ----
    # For every selected schedule, pooled over subexperiments x members x guided n, compute three
    # per-channel pushes across ALL 6 level variables (+ their surface pair):
    #   avg  = M_mask(x_gui) - M_mask(x_gui_ung)                 (masked-mean of the guidance effect)
    #   kick = sqrt( sum_t sum_mask (dL/dz)^2 )                  (RAW grad norm; unscaled, all steps)
    #   appk = sqrt( sum_t lam_raw_t^2 sum_mask (dL/dz)^2 )      (APPLIED kick; lam_raw_t=w a c/||g||,
    #          zero on unguided steps -> schedule-aware, guided-steps-only)
    # Also the ABSOLUTE masked means M(x_gui), M(x_gui_ung) and the ground-truth M(x_gt) at valid time
    # n+1 (for the T2a absolute-intensity and vs-ground-truth profile grids).
    # Also per-variable totals and a variable-normalized rank score (equal weight per variable).
    interf_data = None
    if interf_button.value:
        _labels = list(sched_colors)
        _avg = {lb: {v: {} for v in INTERF_LEVEL_VARS} for lb in _labels}
        _kick = {lb: {v: {} for v in INTERF_LEVEL_VARS} for lb in _labels}
        _appk = {lb: {v: {} for v in INTERF_LEVEL_VARS} for lb in _labels}
        _absg = {lb: {v: {} for v in INTERF_LEVEL_VARS} for lb in _labels}   # M(x_gui)
        _absu = {lb: {v: {} for v in INTERF_LEVEL_VARS} for lb in _labels}   # M(x_gui_ung)
        _absgt = {lb: {v: {} for v in INTERF_LEVEL_VARS} for lb in _labels}  # M(x_gt) @ valid time n+1
        _any_gt = False

        def _lam_raw_for(_recs, _sel, _m, _n):
            # applied-kick multiplier per flow step: lam_raw_t = w_t a_t c_t / ||g_t|| (from the
            # guidance_schedule sidecar). Zero where a_t=0, so appk is schedule-aware + guided-only.
            for _r in _recs:
                if _r.get("m") == _m and _r.get("n") == _n and all(
                    _r["sweep"].get(_k) == _v for _k, _v in _sel.items() if _k in _r["sweep"]
                ):
                    _a = np.asarray(_r["a_t"], float); _c = np.asarray(_r["c_t"], float)
                    _w = np.asarray(_r["w_t"], float); _gn = np.asarray(_r["g_norm_t"], float)
                    return (_w * _a * _c) / np.clip(_gn, 1e-30, None)
            return None

        for _ei, _rid in enumerate(EXPS):
            _dir, _cfg, _sv, _recs, _mask = load_rollout(_rid)
            _mask = np.asarray(_mask, float); _msum = float(_mask.sum())
            _sp = sweep_points(_sv, _recs); _pts = {lb: _sp[lb] for lb in _labels if lb in _sp}
            _gui = xr.open_zarr(_dir / "gui.zarr"); _giu = xr.open_zarr(_dir / "gui_ung.zarr"); _gr = xr.open_zarr(_dir / "grads.zarr")
            try:
                # GT valid state (member-independent). N+1 times: index 0 = init, n+1 = forecast step n.
                _gtr = get_gt_rollout(N + 1, datetime.fromisoformat(_cfg["START_TS"]),
                                      input_path=find_era5_input(get_rollout_dir(_rid)))
                _have_gt = True; _any_gt = True
            except Exception:
                _gtr = None; _have_gt = False
            for _lb, _sel in _pts.items():
                _p = np.asarray(_sv["GUIDANCE_DELTA"][_sel["GUIDANCE_DELTA"]], float)[:N]
                _gns = [nn for nn in range(N) if _p[nn] != 0.0] or [0]
                for _m in range(M):
                    for _n in _gns:
                        _lam = _lam_raw_for(_recs, _sel, _m, _n)
                        _lam2 = None if _lam is None else (_lam ** 2)
                        for _var in INTERF_LEVEL_VARS:
                            _levs = list(_gui[_var]["level"].values)
                            _g = np.asarray(select_point(_gui[_var], _sel).isel(m=_m, n=_n), float)      # (level,lat,lon)
                            _ud = select_point(_giu[_var], _sel).isel(m=_m, n=_n)
                            _u = np.asarray(_ud.isel(t=-1) if "t" in _ud.dims else _ud, float)            # (level,lat,lon)
                            _gg = np.asarray(select_point(_gr[_var], _sel).isel(m=_m, n=_n), float)       # (t,level,lat,lon)
                            _ga = (_g * _mask).sum((-2, -1)) / _msum                                       # (level,) M(x_gui)
                            _ua = (_u * _mask).sum((-2, -1)) / _msum                                       # (level,) M(x_gui_ung)
                            _pa = _ga - _ua                                                                # (level,) push
                            _gta = None
                            if _have_gt:
                                _gt_lv = np.asarray(_gtr[_var].isel(time=_n + 1).sel(level=_levs), float)  # (level,lat,lon)
                                _gta = (_gt_lv * _mask).sum((-2, -1)) / _msum                              # (level,) M(x_gt)
                            _pt = ((_gg ** 2) * _mask).sum((-2, -1))                                       # (t,level) grad energy
                            _pk = np.sqrt(_pt.sum(axis=0))                                                 # (level,) raw
                            _pc = np.sqrt((_pt * _lam2[:, None]).sum(axis=0)) if _lam2 is not None else None  # (level,) applied
                            for _li, _L in enumerate(_levs):
                                _cl = f"L{int(_L)}"
                                _avg[_lb][_var].setdefault(_cl, []).append(float(_pa[_li]))
                                _absg[_lb][_var].setdefault(_cl, []).append(float(_ga[_li]))
                                _absu[_lb][_var].setdefault(_cl, []).append(float(_ua[_li]))
                                _kick[_lb][_var].setdefault(_cl, []).append(float(_pk[_li]))
                                if _gta is not None:
                                    _absgt[_lb][_var].setdefault(_cl, []).append(float(_gta[_li]))
                                if _pc is not None:
                                    _appk[_lb][_var].setdefault(_cl, []).append(float(_pc[_li]))
                            if _var in INTERF_SURFACE_PAIR:
                                _svar = INTERF_SURFACE_PAIR[_var]
                                _gs = np.asarray(select_point(_gui[_svar], _sel).isel(m=_m, n=_n), float)  # (lat,lon)
                                _usd = select_point(_giu[_svar], _sel).isel(m=_m, n=_n)
                                _us = np.asarray(_usd.isel(t=-1) if "t" in _usd.dims else _usd, float)
                                _grs = np.asarray(select_point(_gr[_svar], _sel).isel(m=_m, n=_n), float)  # (t,lat,lon)
                                _pts = ((_grs ** 2) * _mask).sum((-2, -1))                                  # (t,)
                                _ga_s = float((_gs * _mask).sum() / _msum)
                                _ua_s = float((_us * _mask).sum() / _msum)
                                _avg[_lb][_var].setdefault("sfc", []).append(_ga_s - _ua_s)
                                _absg[_lb][_var].setdefault("sfc", []).append(_ga_s)
                                _absu[_lb][_var].setdefault("sfc", []).append(_ua_s)
                                _kick[_lb][_var].setdefault("sfc", []).append(float(np.sqrt(_pts.sum())))
                                if _have_gt:
                                    _gts = np.asarray(_gtr[_svar].isel(time=_n + 1), float)                # (lat,lon)
                                    _absgt[_lb][_var].setdefault("sfc", []).append(float((_gts * _mask).sum() / _msum))
                                if _lam2 is not None:
                                    _appk[_lb][_var].setdefault("sfc", []).append(float(np.sqrt((_pts * _lam2).sum())))

        def _mean_map(_acc):
            return {lb: {v: {cl: float(np.mean(vs)) for cl, vs in _acc[lb][v].items() if vs} for v in INTERF_LEVEL_VARS} for lb in _labels}

        def _pvt(_D, _absv):
            return {lb: {v: float(sum((abs(x) if _absv else x) for x in _D[lb][v].values())) for v in INTERF_LEVEL_VARS} for lb in _labels}

        def _scores(_pv):
            _vmax = {v: (max(_pv[lb][v] for lb in _labels) or 1.0) for v in INTERF_LEVEL_VARS}
            return {lb: float(np.mean([_pv[lb][v] / _vmax[v] for v in INTERF_LEVEL_VARS])) for lb in _labels}

        _AVG = _mean_map(_avg); _KICK = _mean_map(_kick); _APPK = _mean_map(_appk)
        _ABSG = _mean_map(_absg); _ABSU = _mean_map(_absu); _ABSGT = _mean_map(_absgt)
        _apv = _pvt(_AVG, True); _kpv = _pvt(_KICK, False); _cpv = _pvt(_APPK, False)
        _npool = max((len(vs) for v in INTERF_LEVEL_VARS for vs in _avg[_labels[0]][v].values()), default=0) if _labels else 0
        # per-sample T2a score + T2d collateral -> (mean, std) over the (exp x m x n) pool. Fixed
        # normalizers (overall vmax / mean 2mT) so the mean matches the T2a/T2d table values.
        _avg_score = _scores(_apv)
        def _nsamp(_l):
            return min((len(vs) for v in INTERF_LEVEL_VARS for vs in _avg[_l][v].values()), default=0)
        def _pv_sample(_acc, _l, _i, _absv):
            return {v: float(sum((abs(_acc[_l][v][cl][_i]) if _absv else _acc[_l][v][cl][_i]) for cl in _acc[_l][v])) for v in INTERF_LEVEL_VARS}
        _avmax = {v: (max(_apv[lb][v] for lb in _labels) or 1.0) for v in INTERF_LEVEL_VARS}
        _avg_score_ms, _collateral_ms = {}, {}
        for _l in _labels:
            _ns = _nsamp(_l)
            _si = [float(np.mean([_pv_sample(_avg, _l, _i, True)[v] / _avmax[v] for v in INTERF_LEVEL_VARS])) for _i in range(_ns)]
            _avg_score_ms[_l] = (_avg_score[_l], float(np.std(_si)) if _si else float("nan"))
            _sfc = _APPK[_l]["temperature"].get("sfc", float("nan"))
            _ci = [sum(_pv_sample(_appk, _l, _i, False).values()) / _sfc - 1.0 for _i in range(_ns)] if (np.isfinite(_sfc) and _sfc != 0.0) else []
            _collateral_ms[_l] = (float(np.mean(_ci)), float(np.std(_ci))) if _ci else (float("nan"), float("nan"))
        interf_data = {"avg": _AVG, "kick": _KICK, "appk": _APPK,
                       "abs_gui": _ABSG, "abs_ung": _ABSU, "abs_gt": _ABSGT, "have_gt": _any_gt,
                       "avg_pvt": _apv, "kick_pvt": _kpv, "appk_pvt": _cpv,
                       "avg_score": _avg_score, "kick_score": _scores(_kpv), "appk_score": _scores(_cpv),
                       "avg_score_ms": _avg_score_ms, "collateral_ms": _collateral_ms,
                       "labels": _labels, "n_pool": _npool}
    return (interf_data,)


@app.cell
def _(
    INTERF_LEVEL_VARS,
    INTERF_VSHORT,
    glabel,
    interf_chan_order,
    interf_data,
    interf_k_slider,
    mo,
    np,
    plt,
    sched_colors,
):
    # shared renderer for a subtest: a per-variable table + a 6-chart perturbation-profile grid
    # (one chart per level variable, x = channels surface->top, lines = the top-k schedules).
    # _lower_better -> best is the MINIMUM. _normalize_by_2mT -> divide each line by its own 2mT
    # (surface-temperature) kick (all schedules = 1 at temp/sfc) and rank/score by COLLATERAL =
    # total non-target kick per unit 2mT (min = most targeted). _show_table=False -> grid only.
    def interf_render(_key, _title_md, _desc_md, _ylabel, _split_T_sfc=False, _lower_better=False,
                      _normalize_by_2mT=False, _show_table=True):
        if interf_data is None:
            return mo.vstack([mo.md(_title_md), mo.md("_press **compute interference profiles** above_")])
        _D = interf_data[_key]
        _pvt = interf_data[_key + "_pvt"]
        _score = interf_data[_key + "_score"]
        _labels = interf_data["labels"]
        _k = min(int(interf_k_slider.value), len(_labels))
        # ---- ranking (winner first) ----
        if _normalize_by_2mT:
            # collateral per unit 2mT = total_kick/2mT - 1 ; MIN = most targeted / least side-effect
            def _coll(_l):
                _nrm = _D[_l]["temperature"].get("sfc", np.nan)
                if not (np.isfinite(_nrm) and _nrm != 0.0):
                    return float("-inf")   # missing target -> never wins under max
                return sum(_pvt[_l][_v] for _v in INTERF_LEVEL_VARS) / _nrm - 1.0
            _rankval = {_l: _coll(_l) for _l in _labels}
            _ranked = sorted(_labels, key=lambda _l: _rankval[_l], reverse=True)  # descending: max collateral first (higher=better)
        else:
            _rankval = _score
            _ranked = sorted(_labels, key=lambda _l: _score[_l], reverse=not _lower_better)
        _winner = _ranked[0]
        _top = _ranked[:_k]
        # ---- 6-chart grid ----
        _fig, _axs = plt.subplots(2, 3, figsize=(15, 7), dpi=120)
        for _ax, _var in zip(_axs.ravel(), INTERF_LEVEL_VARS):
            _cos = interf_chan_order(_var); _xs = list(range(len(_cos)))
            for _lb in _top:
                _ys = [_D[_lb][_var].get(_cl, np.nan) for _cl in _cos]
                if _normalize_by_2mT:
                    _nrm = _D[_lb]["temperature"].get("sfc", np.nan)
                    _ys = [(_y / _nrm) if (np.isfinite(_nrm) and _nrm != 0.0) else np.nan for _y in _ys]
                _ax.plot(_xs, _ys, "-o", ms=3, lw=1.7, color=sched_colors[_lb], label=glabel(_lb))
            _ax.axhline(1.0 if _normalize_by_2mT else 0.0, color="#999999", lw=0.6,
                        ls="--" if _normalize_by_2mT else "-")
            _ax.set_xticks(_xs); _ax.set_xticklabels(_cos, fontsize=6, rotation=90)
            _ax.set_xlim(-0.5, len(_cos) - 0.5)   # half-slot pad: sfc is the first tick, off the y-axis
            _ax.set_title(_var, fontsize=9); _ax.set_ylabel(_ylabel, fontsize=7)
            for _s in ("top", "right"):
                _ax.spines[_s].set_visible(False)
        _axs.ravel()[0].legend(fontsize=6, loc="best")
        _fig.tight_layout()
        _grid = mo.as_html(_fig); plt.close(_fig)
        if not _show_table:
            return mo.vstack([mo.md(_title_md), mo.md(_desc_md), _grid], align="start")
        # ---- T2d collateral table (2mT-normalized) ----
        if _normalize_by_2mT:
            # per-variable kick per unit 2mT (temperature column = its LEVELS only, i.e. the collateral
            # WITHIN temperature; the 2mT reference is 1 by construction and omitted). MIN per column;
            # "collateral" = total non-target kick per unit 2mT; winner = min collateral.
            _cols = []
            for _v in INTERF_LEVEL_VARS:
                _cd = {}
                for _l in _labels:
                    _nrm = _D[_l]["temperature"].get("sfc", np.nan)
                    _raw = (_pvt[_l]["temperature"] - _D[_l]["temperature"].get("sfc", 0.0)) if _v == "temperature" else _pvt[_l][_v]
                    _cd[_l] = (_raw / _nrm) if (np.isfinite(_nrm) and _nrm != 0.0) else float("nan")
                _cols.append((INTERF_VSHORT[_v], _cd, "max"))
            _cols.append(("collateral", {_l: _rankval[_l] for _l in _labels}, "max"))
            _best = {_h: max((_v for _v in _vv.values() if np.isfinite(_v)), default=float("nan")) for _h, _vv, _r in _cols}
            _rows = ["| schedule | " + " | ".join(_h for _h, _u1, _u2 in _cols) + " |",
                     "|" + "---|" * (len(_cols) + 1)]
            for _lb in _ranked:
                _cells = [(f"**{_v2[_lb]:.3g}**" if _v2[_lb] == _best[_h] else f"{_v2[_lb]:.3g}") for _h, _v2, _r in _cols]
                _star = "★ " if _lb in _top else ""
                _rows.append(f"| {_star}{glabel(_lb)} | " + " | ".join(_cells) + " |")
            return mo.vstack([
                mo.md(_title_md), mo.md(_desc_md), _grid,
                mo.md(f"_best-{_k} schedules (★) drawn. Each cell = applied kick **per unit of the 2mT surface "
                      f"kick** (temperature column = its **levels only**; 2mT itself is 1 by construction, omitted). "
                      f"**collateral** = total non-target kick per unit 2mT; **max = winner** (fullest physically-coupled "
                      f"response per unit of target kick). **bold** = per-column max. Pooled over "
                      f"{interf_data['n_pool']} (exp×m×n) samples._"),
                mo.md("\n".join(_rows)),
            ], align="start")
        # ---- default per-variable push table ----
        _defrule = "min" if _lower_better else "max"
        _cols = []  # (header, {lb: value}, "max"|"min")
        for _v in INTERF_LEVEL_VARS:
            if _v == "temperature" and _split_T_sfc:
                _sfc = {_l: abs(_D[_l]["temperature"].get("sfc", float("nan"))) for _l in _labels}
                _tlev = {_l: _pvt[_l]["temperature"] - _sfc[_l] for _l in _labels}
                _cols.append(("T", _tlev, _defrule))
                _lead2mT = ("2mT", _sfc, None)   # 2mT -> first column, no bold
            else:
                _cols.append((INTERF_VSHORT[_v], {_l: _pvt[_l][_v] for _l in _labels}, _defrule))
        if _split_T_sfc:
            _cols = [_lead2mT] + _cols
        _best = {_h: (min if _r == "min" else max)(_vv.values()) for _h, _vv, _r in _cols if _r is not None}
        _rows = ["| schedule | " + " | ".join(_h for _h, _u1, _u2 in _cols) + " | score |",
                 "|" + "---|" * (len(_cols) + 2)]
        for _lb in _ranked:
            _cells = []
            for _h, _vv, _r in _cols:
                _val = _vv[_lb]; _s = f"{_val:.3g}"
                _cells.append(f"**{_s}**" if (_r is not None and _val == _best[_h]) else _s)
            _sc = f"{_score[_lb]:.3f}"; _sc = f"**{_sc}**" if _lb == _winner else _sc
            _star = "★ " if _lb in _top else ""
            _rows.append(f"| {_star}{glabel(_lb)} | " + " | ".join(_cells) + f" | {_sc} |")
        return mo.vstack([
            mo.md(_title_md),
            mo.md(_desc_md),
            _grid,
            mo.md(f"_best-{_k} schedules (★) drawn, ranked by the variable-normalized push score"
                  + (" (**lower = better**)" if _lower_better else "")
                  + f"; **bold** = per-column {'min' if _lower_better else 'max'}"
                  + (" (**2mT** = surface temperature, first column, unbolded; **T** = temperature levels only)" if _split_T_sfc else "")
                  + "; **score** bold = the single winner (rank #1). "
                  f"Per-variable value = sum over that variable's channels; "
                  f"pooled over {interf_data['n_pool']} (exp×m×n) samples._"),
            mo.md("\n".join(_rows)),
        ], align="start")

    return (interf_render,)


@app.cell(hide_code=True)
def _(
    INTERF_LEVEL_VARS,
    glabel,
    interf_chan_order,
    interf_data,
    interf_k_slider,
    mo,
    np,
    plt,
    sched_colors,
):
    # grid-only profile renderers ported from intensity_comparison's "Guidance intensity" section:
    # per level variable a 6-chart grid (x = channels surface->top), one line per top-k schedule of the
    # masked-mean of the guided state. _mode="abs" -> absolute M(x_gui); _mode="vsgt" -> M(x_gui)-M(gt).
    # References: unguided twin (gui_ung, dashed grey) and ground truth (gt: dashed green / zero line).
    # Top-k + ordering follow the T2a push score, so all three T2a views show the SAME schedules. No table.
    def interf_profile_render(_mode, _title_md, _desc_md, _ylabel):
        if interf_data is None:
            return mo.vstack([mo.md(_title_md), mo.md("_press **compute interference profiles** above_")])
        _absg = interf_data["abs_gui"]; _absu = interf_data["abs_ung"]; _absgt = interf_data["abs_gt"]
        _labels = interf_data["labels"]; _score = interf_data["avg_score"]
        _have_gt = bool(interf_data.get("have_gt"))
        _k = min(int(interf_k_slider.value), len(_labels))
        _ranked = sorted(_labels, key=lambda _l: _score[_l], reverse=True)
        _top = _ranked[:_k]

        def _refline(_D, _var, _cos):
            # schedule-independent reference (gui_ung / gt): mean across drawn labels (collapses to value)
            _out = []
            for _cl in _cos:
                _vv = [_D[_l][_var][_cl] for _l in _top if _cl in _D.get(_l, {}).get(_var, {})]
                _out.append(float(np.mean(_vv)) if _vv else np.nan)
            return _out

        _fig, _axs = plt.subplots(2, 3, figsize=(15, 7), dpi=120)
        for _ax, _var in zip(_axs.ravel(), INTERF_LEVEL_VARS):
            _cos = interf_chan_order(_var); _xs = list(range(len(_cos)))
            _gtref = _refline(_absgt, _var, _cos) if _have_gt else [np.nan] * len(_cos)
            for _lb in _top:
                _ys = [_absg[_lb][_var].get(_cl, np.nan) for _cl in _cos]
                if _mode == "vsgt":
                    _ys = [(_y - _gt) if (np.isfinite(_y) and np.isfinite(_gt)) else np.nan
                           for _y, _gt in zip(_ys, _gtref)]
                _ax.plot(_xs, _ys, "-o", ms=3, lw=1.7, color=sched_colors[_lb], label=glabel(_lb))
            # unguided-twin reference line
            _uref = _refline(_absu, _var, _cos)
            if _mode == "vsgt":
                _uref = [(_uu - _gt) if (np.isfinite(_uu) and np.isfinite(_gt)) else np.nan
                         for _uu, _gt in zip(_uref, _gtref)]
            _ax.plot(_xs, _uref, "--", color="#555555", lw=1.5, label="unguided (gui_ung)")
            # ground-truth reference
            if _mode == "abs":
                if _have_gt:
                    _ax.plot(_xs, _gtref, "--", color="#009E73", lw=1.5, label="ground truth (gt)")
            else:
                _ax.axhline(0.0, color="#009E73", lw=1.5, ls="--", label="ground truth (gt)")
            _ax.set_xticks(_xs); _ax.set_xticklabels(_cos, fontsize=6, rotation=90)
            _ax.set_xlim(-0.5, len(_cos) - 0.5)
            _ax.set_title(_var, fontsize=9); _ax.set_ylabel(_ylabel, fontsize=7)
            for _s in ("top", "right"):
                _ax.spines[_s].set_visible(False)
        _axs.ravel()[0].legend(fontsize=6, loc="best")
        _fig.tight_layout()
        _grid = mo.as_html(_fig); plt.close(_fig)
        return mo.vstack([mo.md(_title_md), mo.md(_desc_md), _grid], align="start")

    return (interf_profile_render,)


@app.cell(hide_code=True)
def _(interf_profile_render):
    interf_abs_view = interf_profile_render(
        "abs",
        r"### Absolute intensity — $\mathcal{M}_m(x^{\mathrm{gui}})$ by level",
        r"Masked-mean of the **guided state** $\mathcal{M}_m(x^{\mathrm{gui}})$ per variable/level, one line "
        r"per top-$k$ schedule; dashed grey = unguided twin (gui\_ung), dashed green = ground truth (gt). "
        r"One chart per level variable — $x$ = channels (surface→top). Pooled across experiments (exp×m×n).",
        r"mask-mean  $\mathcal{M}_m(x^{gui})$",
    )
    interf_abs_view
    return


@app.cell(hide_code=True)
def _(interf_profile_render):
    interf_vsgt_view = interf_profile_render(
        "vsgt",
        r"### Relative to ground truth — $\mathcal{M}_m(x^{\mathrm{gui}})-\mathcal{M}_m(x^{\mathrm{gt}})$",
        r"The same masked-mean, minus the ground-truth valid state at lead $n{+}1$: how far each "
        r"variable/level sits from truth. GT is the zero line (green); dashed grey = unguided twin "
        r"(gui\_ung). One chart per level variable — $x$ = channels (surface→top). Pooled across "
        r"experiments (exp×m×n).",
        r"mask-mean  $\mathcal{M}_m(x^{gui})-\mathcal{M}_m(x^{gt})$",
    )
    interf_vsgt_view
    return


@app.cell
def _(interf_render):
    interf_A_view = interf_render(
        "avg",
        r"### Push relative to unguided — $\mathcal{M}_m(x^{\mathrm{gui}})-\mathcal{M}_m(x^{\mathrm{gui\_ung}})$",
        r"Per variable/level, the masked-mean of the **guidance effect** "
        r"$\mathcal{M}_m(x^{\mathrm{gui}})-\mathcal{M}_m(x^{\mathrm{gui\_ung}})$: how far each variable's "
        r"average state is pushed (relative to its unguided twin) to reach the target. One chart per "
        r"level variable — $x$ = channels (surface→top), one line per top-$k$ schedule.",
        r"push  $\mathcal{M}_m(x^{gui})-\mathcal{M}_m(x^{gui\_ung})$",
        _split_T_sfc=True,
    )
    interf_A_view
    return


@app.cell
def _(interf_render):
    interf_B_view = interf_render(
        "kick",
        r"### T2b — Raw guidance-gradient norm $\lVert\nabla\mathcal{L}\rVert$",
        r"Per variable/level, the mask-weighted norm of the **raw** loss gradient over **all** flow steps "
        r"$\lVert\nabla\mathcal{L}\rVert=\sqrt{\sum_t\sum_{\mathrm{mask}}(\nabla_z\mathcal{L})^2}$ (the `grads` store, "
        r"**unscaled** by the schedule). It shows the *shape/direction* of the guidance signal per level, but is "
        r"schedule-blind (independent of $w,a_t,c_t$) and sums over steps where **no** kick was applied — so it is "
        r"NOT the applied kick (see T2c). One chart per level variable — $x$ = channels (surface→top), top-$k$ lines.",
        r"raw  $\|\nabla\mathcal{L}\|$",
        _lower_better=True,
    )
    interf_B_view
    return


@app.cell
def _(interf_render):
    interf_C_view = interf_render(
        "appk",
        r"### T2c — Applied guidance kick $\lVert\lambda_t\nabla\mathcal{L}\rVert$",
        r"Per variable/level, the **actually applied** kick: each step's raw gradient weighted by the applied "
        r"multiplier $\lambda^{\mathrm{raw}}_t=w_t a_t c_t/\lVert g_t\rVert$ (from the guidance schedule sidecar), then "
        r"$\sqrt{\sum_t (\lambda^{\mathrm{raw}}_t)^2\sum_{\mathrm{mask}}(\nabla_z\mathcal{L})^2}$. Since $\lambda^{\mathrm{raw}}_t=0$ on "
        r"unguided steps, this is schedule-aware and isolates **where** the guidance acts (e.g. a single step for "
        r"`spike@k`) — the faithful guidance kick. One chart per level variable — $x$ = channels (surface→top), top-$k$ lines.",
        r"applied kick  $\|\lambda_t\nabla\mathcal{L}\|$",
        _lower_better=True,
    )
    interf_C_view
    return


@app.cell
def _(interf_render):
    interf_D_view = interf_render(
        "appk",
        r"### T2d — Applied kick, normalized to surface temperature (2mT)",
        r"The applied-kick profiles from T2c, but every line is divided by **its own** surface-temperature "
        r"kick (2mT), so all schedules pass through **1 at temperature/sfc** (dashed line) and every channel "
        r"reads as the applied kick **per unit of the target's surface kick**. The table ranks by "
        r"**collateral** = total non-target kick per unit 2mT (**higher = better** — the fullest physically-coupled "
        r"response per unit of target kick). One chart per level variable — $x$ = channels (surface→top), best-$k$ lines.",
        r"applied kick / 2mT",
        _lower_better=True,
        _normalize_by_2mT=True,
        _show_table=True,
    )
    interf_D_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## T3: Closeness
    How far the guided flow strays from the unguided, in the mask-region PCA latent space (fit on the 12:00 climatology).
    - **path length**: length of the clean-pred trajectory $\hat{x}_{n,t}$ in the 3-PC space.
    - **guidance / pushback steps**: per-flow-step norm of the guided step vs. the unguided pushback (full mask-bbox space).
    - **$\Delta$PC$_k$ / dist to gui\_ung**: signed endpoint deviation from gui\_ung per PC, and its norm $\lVert\mathrm{PC}(x^{\mathrm{gui}}_{n,T})-\mathrm{PC}(x^{\mathrm{ung}}_{n,T})\rVert$.
    - PCA grid (one per experiment): ERA5 12:00 cloud, guided (solid), and gui\_ung / ung / GT reference points.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # PCA view controls (3D angle + zoom-to-trajectories) -- live in the Closeness subsection
    elev_slider = mo.ui.slider(0, 90, value=0, step=5, label="elev: ", show_value=True)
    azim_slider = mo.ui.slider(-180, 180, value=0, step=5, label="azim: ", show_value=True)
    zoom_traj_checkbox = mo.ui.checkbox(label="zoom to trajectories", value=True)
    zoom_pad_slider = mo.ui.slider(-0.4, 2.0, step=0.05, value=0, label="zoom pad: ", show_value=True)
    xt_checkbox = mo.ui.checkbox(value=False, label="actual state x_t (drop vfs)")
    mo.hstack([elev_slider, azim_slider, zoom_traj_checkbox, zoom_pad_slider, xt_checkbox], justify="start", align="center")
    return (
        azim_slider,
        elev_slider,
        xt_checkbox,
        zoom_pad_slider,
        zoom_traj_checkbox,
    )


@app.cell
def _(mo):
    # PCA region: fit/project the latent space on the whole globe, the mask footprint, or its
    # complement (see what guidance does inside vs. outside the target)
    pca_region_dropdown = mo.ui.dropdown(["mask", "globe", "!mask"], value="mask", label="PCA region: ")
    pca_region_dropdown
    return (pca_region_dropdown,)


@app.cell
def _(MASK0, pca_region_dropdown, region_bool):
    PCA_REGION = region_bool(MASK0, pca_region_dropdown.value)
    PCA_REGION_MODE = pca_region_dropdown.value
    return PCA_REGION, PCA_REGION_MODE


@app.cell
def _(LEVEL, PCA_REGION, PCA_REGION_MODE, VAR, ensure_region_basis, mo):
    # PCA basis on the selected region (globe / mask / !mask), fit once on the 12:00
    # climatology states of 2020; reuse by projection.
    basis = ensure_region_basis(VAR, LEVEL, PCA_REGION, "era5", mode=PCA_REGION_MODE, time_hour=12)
    _evr = ", ".join(f"{e:.0%}" for e in basis.evr)
    _lev = f"@{LEVEL}" if LEVEL is not None else ""
    _npt = basis.meta["n_points"]
    basis_info = mo.md(
        f"**PCA basis** on the **{PCA_REGION_MODE}** region: **F = {basis.F} pixels** of {VAR}{_lev}, "
        f"fit once on the **{_npt} 12:00 ERA5 states** of 2020 (SVD). The 3 PCs explain **{_evr}** "
        f"(sum **{float(basis.evr.sum()):.0%}**) of the selected region. "
        f"mask / !mask / globe expose the guidance at different levels."
    )
    basis_info
    return (basis,)


@app.cell(hide_code=True)
def _(LEVEL, PCA_REGION, VAR, basis, project, region_cloud_sample):
    # PCA-plot background: a projected subsample of the climatology cloud the basis was fit on
    cloud_proj = project(basis, region_cloud_sample(VAR, LEVEL, PCA_REGION, "era5", time_hour=12, max_points=400))
    return (cloud_proj,)


@app.cell
def _(
    EXPS,
    M,
    azim_slider,
    basis,
    cloud_proj,
    elev_slider,
    glabel,
    gt_proj,
    guiung_traj,
    metrics,
    mo,
    n_slider,
    np,
    plt,
    ref_metrics,
    sched_colors,
    traj_grid,
    ung_traj,
    xt_checkbox,
    zoom_pad_slider,
    zoom_traj_checkbox,
):
    # ---- Closeness: full PCA path grid (ported), rows=exp, cols=m, with zoom-to-trajectories ----
    if metrics is None or traj_grid is None:
        t3_view = mo.md("_press **compute metrics**_")
    else:
        _n = int(n_slider.value) - 1; _labels = list(sched_colors)
        _items = []
        for _ei in range(len(EXPS)):
            _items.append(mo.md(f"**{EXPS[_ei].split(chr(47))[-1]}**  (n={_n + 1}, EVR={basis.evr.sum():.0%})"))
            _f, _axs = plt.subplots(1, M, figsize=(4.6*M, 4.4), subplot_kw={"projection": "3d"}, squeeze=False, dpi=110)
            for _m in range(M):
                _ax = _axs[0][_m]; _grp = []
                _ax.scatter(cloud_proj[:, 0], cloud_proj[:, 1], cloud_proj[:, 2], color="#BBBBBB", s=8, alpha=0.45, depthshade=False, label=f"ERA5 cloud ({cloud_proj.shape[0]})")
                _cell = traj_grid.get((_ei, _m, _n), {})
                for _lb in _labels:
                    if _lb not in _cell: continue
                    _pg = np.asarray(_cell[_lb]["pgx" if xt_checkbox.value else "pg"]); _c = sched_colors[_lb]
                    _ax.plot(_pg[:, 0], _pg[:, 1], _pg[:, 2], "-", color=_c, linewidth=1.4, alpha=0.9, label=glabel(_lb).split()[-1])
                    _ax.scatter(_pg[-1, 0], _pg[-1, 1], _pg[-1, 2], marker="o", s=45, color=_c, depthshade=False)
                    _grp.append(_pg)
                if (_ei, _n) in gt_proj:
                    _tp = np.asarray(gt_proj[(_ei, _n)])
                    _ax.scatter(_tp[0], _tp[1], _tp[2], marker="*", s=70, color="#FFD700", edgecolors="black", linewidths=1.0, depthshade=False, label="gt")
                    _grp.append(_tp[None, :])
                if (_ei, _m, _n) in ung_traj:
                    _pn = np.asarray(ung_traj[(_ei, _m, _n)])
                    _ax.plot(_pn[:, 0], _pn[:, 1], _pn[:, 2], "-", color="#111111", linewidth=1.2, alpha=0.85, label="ung")
                    _ax.scatter(_pn[-1, 0], _pn[-1, 1], _pn[-1, 2], marker="o", s=34, color="#111111", depthshade=False)
                    _grp.append(_pn)
                if _n > 0 and (_ei, _m, _n) in guiung_traj:
                    _pu = np.asarray(guiung_traj[(_ei, _m, _n)])
                    _ax.plot(_pu[:, 0], _pu[:, 1], _pu[:, 2], "--", color="#888888", linewidth=1.1, alpha=0.8, label="gui_ung")
                    _ax.scatter(_pu[-1, 0], _pu[-1, 1], _pu[-1, 2], marker="o", s=34, color="#888888", depthshade=False)
                    _grp.append(_pu)
                if _cell:
                    _sp = np.asarray(next(iter(_cell.values()))["pgx" if xt_checkbox.value else "pg"])[0]
                    _ax.scatter(_sp[0], _sp[1], _sp[2], marker="o", s=45, facecolors="#FFD700", edgecolors="black", linewidths=1.2, depthshade=False, label="start")
                    pass
                if zoom_traj_checkbox.value and _grp:
                    _pts = np.vstack(_grp); _lo, _hi = _pts.min(0), _pts.max(0)
                    _pad = zoom_pad_slider.value * float((_hi - _lo).max())
                    _ax.set_xlim(_lo[0]-_pad, _hi[0]+_pad); _ax.set_ylim(_lo[1]-_pad, _hi[1]+_pad); _ax.set_zlim(_lo[2]-_pad, _hi[2]+_pad)
                _ax.set_xlabel("PC1", fontsize=7); _ax.set_ylabel("PC2", fontsize=7); _ax.set_zlabel("PC3", fontsize=7)
                _ax.view_init(elev=elev_slider.value, azim=azim_slider.value)
                _ax.set_title(f"m={_m}", fontsize=9)
                if _m == M - 1 and _cell:
                    _ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
            _f.tight_layout(pad=0.3)
            _items.append(mo.as_html(_f))
        _ms = lambda v: f"{v[0]:.3g}+/-{v[1]:.2g}"
        _mss = lambda v: f"{v[0]:+.3g}+/-{v[1]:.2g}"  # signed (endpoint PC deviations)
        _dist = lambda r: (r["dpc1"][0]**2 + r["dpc2"][0]**2 + r["dpc3"][0]**2) ** 0.5  # PC-space endpoint distance to gui_ung
        _sweep = list(metrics)
        _bg = min(metrics[_l]["guidance_steps"][0] for _l in _sweep)
        _bp = min(metrics[_l]["pushback_steps"][0] for _l in _sweep)
        _bd = max(_dist(metrics[_l]) for _l in _sweep)  # highest divergence from the PCs (bold)
        _bd1 = max(abs(metrics[_l]["dpc1"][0]) for _l in _sweep)
        _bd2 = max(abs(metrics[_l]["dpc2"][0]) for _l in _sweep)
        _bd3 = max(abs(metrics[_l]["dpc3"][0]) for _l in _sweep)
        _rows = ["| row | n_pool | path length | guidance steps | pushback steps | \u0394PC1 | \u0394PC2 | \u0394PC3 | dist to gui_ung |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for _lb in _sweep:
            _r = metrics[_lb]; _npv = _r["n_pool"]; _Lc = _ms(_r["L"])
            _gs = _r["guidance_steps"]; _gc = _ms(_gs); _gc = f"**{_gc}**" if _gs[0] == _bg else _gc
            _ps = _r["pushback_steps"]; _pc = _ms(_ps); _pc = f"**{_pc}**" if _ps[0] == _bp else _pc
            _dv = _dist(_r); _dc = f"{_dv:.3g}"; _dc = f"**{_dc}**" if _dv == _bd else _dc
            _d1 = _mss(_r["dpc1"]); _d1 = f"**{_d1}**" if abs(_r["dpc1"][0]) == _bd1 else _d1
            _d2 = _mss(_r["dpc2"]); _d2 = f"**{_d2}**" if abs(_r["dpc2"][0]) == _bd2 else _d2
            _d3 = _mss(_r["dpc3"]); _d3 = f"**{_d3}**" if abs(_r["dpc3"][0]) == _bd3 else _d3
            _rows.append(f"| {glabel(_lb)} | {_npv} | {_Lc} | {_gc} | {_pc} | {_d1} | {_d2} | {_d3} | {_dc} |")
        for _rlb in ("gui_ung", "ung"):
            if ref_metrics and _rlb in ref_metrics:
                _r = ref_metrics[_rlb]; _npv = _r["n_pool"]; _Lc = _ms(_r["L"])
                _d1 = _mss(_r["dpc1"]); _d2 = _mss(_r["dpc2"]); _d3 = _mss(_r["dpc3"]); _dc = f"{_dist(_r):.3g}"
                _rows.append(f"| _{_rlb}_ | {_npv} | {_Lc} | 0 | 0 | {_d1} | {_d2} | {_d3} | {_dc} |")
        t3_view = mo.vstack(_items + [mo.md("\n".join(_rows)),
            mo.md(r"Length = PCA path length of the clean-pred trajectory. $\Delta$PC$_k$ = signed "
                  r"endpoint deviation from gui\_ung along PC$_k$. Guidance/pushback in full bbox "
                  r"space (0 for the unguided ung / gui\_ung reference rows).")])
    t3_view
    return


@app.cell
def _(
    EXPS,
    M,
    azim_slider,
    basis,
    cloud_proj,
    elev_slider,
    glabel,
    gt_proj,
    guiung_traj,
    metrics,
    mo,
    n_slider,
    np,
    plt,
    sched_colors,
    traj_grid,
    ung_traj,
    xt_checkbox,
    zoom_pad_slider,
    zoom_traj_checkbox,
):
    # ---- Closeness (ping-pong): per-step guidance kick vs. pushback anatomy, rows=exp, cols=m ----
    # Same PCA grid as T3 above, but every schedule is drawn as its ping-pong decomposition
    # (ported from the gui_ung compare plot in latent_trajectories.py): solid = guidance step
    # (base_t -> kick_t), dotted = pushback step (kick_t -> base_{t+1}); markers = kick endpoints.
    if metrics is None or traj_grid is None:
        t3pp_view = mo.md("_press **compute metrics**_")
    else:
        _n = int(n_slider.value) - 1; _labels = list(sched_colors)
        _items = [mo.md(r"### Ping-pong anatomy — solid = guidance step "
                        r"($\mathrm{base}_t\!	o\!\mathrm{kick}_t$), dotted = pushback "
                        r"($\mathrm{kick}_t\!	o\!\mathrm{base}_{t+1}$); per schedule, coloured as in T3.")]
        for _ei in range(len(EXPS)):
            _items.append(mo.md(f"**{EXPS[_ei].split(chr(47))[-1]}**  (n={_n + 1}, EVR={basis.evr.sum():.0%})"))
            _f, _axs = plt.subplots(1, M, figsize=(4.6*M, 4.4), subplot_kw={"projection": "3d"}, squeeze=False, dpi=110)
            for _m in range(M):
                _ax = _axs[0][_m]; _grp = []
                _ax.scatter(cloud_proj[:, 0], cloud_proj[:, 1], cloud_proj[:, 2], color="#BBBBBB", s=8, alpha=0.45, depthshade=False, label=f"ERA5 cloud ({cloud_proj.shape[0]})")
                _cell = traj_grid.get((_ei, _m, _n), {})
                for _lb in _labels:
                    if _lb not in _cell or "pk" not in _cell[_lb]: continue
                    _bp = np.asarray(_cell[_lb]["pgx" if xt_checkbox.value else "pg"]); _kp = np.asarray(_cell[_lb]["pkx" if xt_checkbox.value else "pk"]); _c = sched_colors[_lb]
                    for _t in range(len(_bp)):
                        _seg = np.stack([_bp[_t], _kp[_t]])
                        _ax.plot(_seg[:, 0], _seg[:, 1], _seg[:, 2], "-", color=_c, linewidth=1.4, alpha=0.9,
                                 label=(glabel(_lb).split()[-1] if _t == 0 else "_nolegend_"))
                        if _t + 1 < len(_bp):
                            _seg = np.stack([_kp[_t], _bp[_t + 1]])
                            _ax.plot(_seg[:, 0], _seg[:, 1], _seg[:, 2], ":", color=_c, linewidth=1.0, alpha=0.55, label="_nolegend_")
                    _ax.scatter(_kp[:, 0], _kp[:, 1], _kp[:, 2], s=np.linspace(6, 34, len(_kp)), color=_c, depthshade=False)
                    _ax.scatter(_kp[-1, 0], _kp[-1, 1], _kp[-1, 2], marker="o", s=42, color=_c, depthshade=False)
                    _grp.append(_bp); _grp.append(_kp)
                if (_ei, _n) in gt_proj:
                    _tp = np.asarray(gt_proj[(_ei, _n)])
                    _ax.scatter(_tp[0], _tp[1], _tp[2], marker="*", s=70, color="#FFD700", edgecolors="black", linewidths=1.0, depthshade=False, label="gt")
                    _grp.append(_tp[None, :])
                if (_ei, _m, _n) in ung_traj:
                    _pn = np.asarray(ung_traj[(_ei, _m, _n)])
                    _ax.scatter(_pn[-1, 0], _pn[-1, 1], _pn[-1, 2], marker="o", s=30, color="#111111", depthshade=False, label="ung")
                    _grp.append(_pn[-1][None, :])
                if _n > 0 and (_ei, _m, _n) in guiung_traj:
                    _pu = np.asarray(guiung_traj[(_ei, _m, _n)])
                    _ax.plot(_pu[:, 0], _pu[:, 1], _pu[:, 2], "--", color="#888888", linewidth=1.0, alpha=0.7, label="gui_ung")
                    _grp.append(_pu)
                if _cell:
                    _sp = np.asarray(next(iter(_cell.values()))["pgx" if xt_checkbox.value else "pg"])[0]
                    _ax.scatter(_sp[0], _sp[1], _sp[2], marker="o", s=45, facecolors="#FFD700", edgecolors="black", linewidths=1.2, depthshade=False, label="start")
                if zoom_traj_checkbox.value and _grp:
                    _pts = np.vstack(_grp); _lo, _hi = _pts.min(0), _pts.max(0)
                    _pad = zoom_pad_slider.value * float((_hi - _lo).max())
                    _ax.set_xlim(_lo[0]-_pad, _hi[0]+_pad); _ax.set_ylim(_lo[1]-_pad, _hi[1]+_pad); _ax.set_zlim(_lo[2]-_pad, _hi[2]+_pad)
                _ax.set_xlabel("PC1", fontsize=7); _ax.set_ylabel("PC2", fontsize=7); _ax.set_zlabel("PC3", fontsize=7)
                _ax.view_init(elev=elev_slider.value, azim=azim_slider.value)
                _ax.set_title(f"m={_m}", fontsize=9)
                if _m == M - 1 and _cell:
                    _ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
            _f.tight_layout(pad=0.3)
            _items.append(mo.as_html(_f))
        t3pp_view = mo.vstack(_items)
    t3pp_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## T4: Realism
    Does the guidance footprint match the mask $m$?
    - One figure per experiment. Row 1: guidance effect $|x_n^{\mathrm{gui}}-x_n^{\mathrm{ung}}|$; Row 2: trajectory-averaged $|$guidance vector$|$ (mask-normalized), per schedule.
    - One side colorbar (norm = common / experiment max). **realism TV** between footprint and mask (**higher = better**); **guidance vs residual** = cosine similarity of the guidance vector to $|x^{\mathrm{gui}}-x^{\mathrm{ung}}|$.
    """)
    return


@app.cell
def _(M, N, mo):
    realism_cmap_dropdown = mo.ui.dropdown(
        ["viridis", "magma", "inferno", "Reds", "hot", "warm (RdBu_r)"],
        value="viridis", label="realism cmap: ")
    realism_norm_dropdown = mo.ui.dropdown(
        ["common max", "experiment max"], value="common max", label="norm: ")
    realism_m_slider = mo.ui.slider(0, max(M - 1, 1), value=0, step=1, label="m: ", show_value=True)
    realism_n_slider = mo.ui.slider(1, max(N, 2), value=1, step=1, label="n: ", show_value=True)
    mo.hstack([realism_m_slider, realism_n_slider, realism_cmap_dropdown, realism_norm_dropdown], justify="start")
    return (
        realism_cmap_dropdown,
        realism_m_slider,
        realism_n_slider,
        realism_norm_dropdown,
    )


@app.cell
def _(
    EXPS,
    WARM_CMAP,
    glabel,
    mask_bbox_ref,
    metrics,
    mo,
    np,
    plt,
    realism_cmap_dropdown,
    realism_m_slider,
    realism_n_slider,
    realism_norm_dropdown,
    sched_colors,
    traj_grid,
):
    # ---- Realism: guidance footprint vs mask; correct aspect + one side colorbar per figure ----
    if metrics is None or traj_grid is None:
        t4_view = mo.md("_press **compute metrics**_")
    else:
        _m = int(realism_m_slider.value); _n = int(realism_n_slider.value) - 1
        _labels = list(sched_colors)
        _cmap = WARM_CMAP if realism_cmap_dropdown.value == "warm (RdBu_r)" else realism_cmap_dropdown.value
        _common = realism_norm_dropdown.value == "common max"
        _cols = ["mask"] + _labels
        _mk = np.asarray(mask_bbox_ref, float); _mkp = _mk/_mk.sum() if _mk.sum() > 0 else _mk
        _grids = []; _exmax = []
        for _ei in range(len(EXPS)):
            _cell = traj_grid.get((_ei, _m, _n), {})
            _r0, _r1 = [], []
            for _col in _cols:
                if _col == "mask":
                    _r0.append(_mkp); _r1.append(_mkp)
                elif _col in _cell:
                    _gf = np.asarray(_cell[_col]["gfield"], float); _r0.append(_gf/_gf.sum() if _gf.sum() > 0 else _gf)
                    _gv = np.asarray(_cell[_col]["gvec_avg"], float); _r1.append(_gv/_gv.sum() if _gv.sum() > 0 else _gv)
                else:
                    _r0.append(None); _r1.append(None)
            _grids.append((_r0, _r1))
            _exmax.append(max((float(np.nanmax(_p)) for _p in _r0 + _r1 if _p is not None), default=1.0))
        _gmax = max(_exmax) if _exmax else 1.0
        _H, _W = _mkp.shape; _ncol = len(_cols)
        _fw = 1.35 * _ncol + 1.0
        _fh = 2.0 * (0.85 * _fw / _ncol) * (_H / _W) / 0.84 + 0.35
        _figs = []
        for _ei, (_r0, _r1) in enumerate(_grids):
            _vmax = _gmax if _common else _exmax[_ei]
            _figs.append(mo.md(f"**{EXPS[_ei].split(chr(47))[-1]}**  (m={_m}, n={_n + 1})"))
            _f = plt.figure(figsize=(_fw, _fh), dpi=120)
            _gs = _f.add_gridspec(2, _ncol, left=0.045, right=0.9, top=0.88, bottom=0.04, wspace=0.06, hspace=0.08)
            _im = None
            for _ri, _rowd in enumerate((_r0, _r1)):
                for _ci in range(_ncol):
                    _ax = _f.add_subplot(_gs[_ri, _ci])
                    if _rowd[_ci] is not None:
                        _im = _ax.imshow(_rowd[_ci], cmap=_cmap, vmin=0.0, vmax=_vmax)
                    _ax.set_xticks([]); _ax.set_yticks([])
                    if _ri == 0:
                        _ax.set_title("mask" if _cols[_ci] == "mask" else glabel(_cols[_ci].split()[-1]), fontsize=8)
                    if _ci == 0:
                        _ax.set_ylabel("guidance effect" if _ri == 0 else "avg-guidance", fontsize=7)
            if _im is not None:
                _cax = _f.add_axes([0.915, 0.04, 0.011, 0.84])
                _cb = _f.colorbar(_im, cax=_cax)
                _cb.set_ticks(list(np.linspace(0.0, _vmax, 5))); _cax.tick_params(labelsize=7)
            _figs.append(mo.as_html(_f))
        _besttv = max(_r["realism_tv"][0] for _r in metrics.values())
        _rows = ["| sweep point | n_pool | realism TV |", "|---|---|---|"]
        for _lb, _r in metrics.items():
            _tv = _r["realism_tv"]; _tvc = f"{_tv[0]:.3f}+/-{_tv[1]:.3f}"; _tvc = f"**{_tvc}**" if _tv[0]==_besttv else _tvc
            _npv = _r["n_pool"]
            _rows.append(f"| {glabel(_lb)} | {_npv} | {_tvc} |")
        t4_view = mo.vstack([
            mo.md(f"**Realism** \u2014 top: guidance effect; bottom: avg-guidance (traj-avg |guidance vector|). "
                  f"Mask-normalized; colour scale = **{realism_norm_dropdown.value}**.  (m={_m}, n={_n + 1})"),
            *_figs,
            mo.md("TV vs mask (**higher = better**)."),
            mo.md("\n".join(_rows))])
    t4_view
    return


if __name__ == "__main__":
    app.run()
