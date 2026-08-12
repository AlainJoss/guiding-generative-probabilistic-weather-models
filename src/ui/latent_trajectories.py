import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from src.ui.comparison import (
        channel,
        clean_pred_branches_primitive,
        clean_pred_trajectory_primitive,
        load_rollout,
        open_store,
        residual_scaler,
        select_point,
        sweep_points,
    )
    from src.pca_basis import mask_bbox, ensure_basis, project, cloud_sample
    from src.utils import get_gt_rollout, get_rollout_ids, get_rollout_dir, find_era5_input

    return (
        Path,
        channel,
        clean_pred_branches_primitive,
        clean_pred_trajectory_primitive,
        cloud_sample,
        datetime,
        ensure_basis,
        find_era5_input,
        get_gt_rollout,
        get_rollout_dir,
        get_rollout_ids,
        load_rollout,
        mask_bbox,
        mo,
        np,
        open_store,
        plt,
        project,
        residual_scaler,
        select_point,
        sweep_points,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Latent trajectories (PCA)

    Guided clean-pred trajectories $\hat{x}_t$ (mask-bbox pixels, flattened) in a fixed PCA
    frame of daily ERA5 states (gray cloud). **Solid**: guided; **dashed**: the run's same-seed
    `gui_ung` twin; **★** GT at the valid day; **○** the independent unguided rollout's final.
    Pick what to **compare** (η, δ, … or $n$) — compared values fan out as fading greens;
    everything else is pinned. The table below ranks every run.
    """)
    return


@app.cell
def _(get_rollout_ids, mo):
    # experiment selector (outer folder)
    _ids = get_rollout_ids("gui")
    _exp_ids = sorted({_r.split("/", 1)[0] for _r in _ids})
    experiment_dropdown = mo.ui.dropdown(
        _exp_ids, value=_exp_ids[0] if _exp_ids else None, label="experiment: ")
    return (experiment_dropdown,)


@app.cell
def _(experiment_dropdown, get_rollout_ids, mo):
    # rollout selector: the start_ts sub-folders under the selected experiment
    _starts = [_r.split("/", 1)[1] for _r in get_rollout_ids("gui")
               if "/" in _r and _r.split("/", 1)[0] == experiment_dropdown.value]
    rollout_dropdown = mo.ui.dropdown(
        _starts, value=_starts[0] if _starts else None, label="rollout: ")
    mo.hstack([experiment_dropdown, rollout_dropdown], justify="start", align="start")
    return (rollout_dropdown,)


@app.cell
def _(experiment_dropdown, rollout_dropdown):
    # combine experiment + rollout selectors into "<exp>/<start_ts>"; a legacy single
    # rollout has no start subfolder, so the experiment id IS the rollout id
    rollout_id = (f"{experiment_dropdown.value}/{rollout_dropdown.value}"
                  if rollout_dropdown.value else experiment_dropdown.value)
    return (rollout_id,)


@app.cell
def _(
    channel,
    clean_pred_branches_primitive,
    clean_pred_trajectory_primitive,
    cloud_sample,
    datetime,
    ensure_basis,
    find_era5_input,
    get_gt_rollout,
    get_rollout_dir,
    load_rollout,
    mask_bbox,
    np,
    open_store,
    project,
    residual_scaler,
    rollout_id,
    select_point,
    sweep_points,
):
    # ===== rollout, latent helpers, persisted PCA basis =====
    rollout_dir, config, sweep_values, records, mask = load_rollout(rollout_id)
    points = sweep_points(sweep_values, records)
    VAR, LEVEL, N_STEPS = config["VAR"], config["LEVEL"], config["N"]
    ROLLOUT_START = datetime.fromisoformat(config["START_TS"])
    RESID_SCALER = residual_scaler(config["PARTITION"], VAR, LEVEL)

    BBOX = mask_bbox(mask)

    def bbox_latent(field):
        """Flatten the bbox region of (..., lat, lon) fields into feature vectors."""
        field = np.asarray(field, dtype=float)
        return field[..., BBOX[0], BBOX[1]].reshape(*field.shape[:-2], -1)

    # persisted PCA basis (2020 climatology), reused across experiments -- no rollout-relative refit
    _basis = ensure_basis(VAR, LEVEL, BBOX, "era5")
    pca_project = lambda x: project(_basis, x)
    pca_evr = _basis.evr
    pca_cloud = cloud_sample(VAR, LEVEL, BBOX, "era5", max_points=400)
    pca_cloud_proj = pca_project(pca_cloud)
    # GT targets from this experiment's era5_input.nc (valid days 1..N)
    _gt_input = find_era5_input(get_rollout_dir(rollout_id))
    pca_targets = bbox_latent(np.asarray(channel(
        get_gt_rollout(config["N"] + 1, ROLLOUT_START, input_path=_gt_input)[VAR], config
    ).isel(time=slice(1, None)), dtype=float))
    try:
        _ung = channel(open_store(rollout_dir, "ung", VAR), config).isel(m=0)
        if "t" in _ung.dims:
            _ung = _ung.isel(t=-1)
        pca_ung_finals = bbox_latent(np.asarray(_ung, dtype=float))
    except FileNotFoundError:
        pca_ung_finals = None

    def guided_latents(sel, m, n):
        """Guided clean-pred trajectory as bbox latents (T, F)."""
        traj = clean_pred_trajectory_primitive(rollout_dir, sel, m, n, VAR, RESID_SCALER, level=LEVEL)
        return None if traj is None else bbox_latent(traj)

    def branch_latents(sel, m, n):
        """(gui_ung_over_t, gui) bbox-latent pair (convention-robust primitives):
        base_t -> kick_t is the GUIDANCE step, kick_t -> base_{t+1} the PUSHBACK (pull-back)."""
        pair = clean_pred_branches_primitive(rollout_dir, sel, m, n, VAR, RESID_SCALER, level=LEVEL)
        return None if pair is None else (bbox_latent(pair[0]), bbox_latent(pair[1]))

    def twin_latents(sel, m, n):
        """The run's gui_ung twin trajectory as bbox latents (T, F)."""
        tw = channel(select_point(open_store(rollout_dir, "gui_ung", VAR), sel), config)
        return bbox_latent(np.asarray(tw.isel(m=m, n=n), dtype=float))

    def delta_of(sel):
        """The run's authored delta trajectory (N,)."""
        return np.asarray(sweep_values["GUIDANCE_DELTA"][sel["GUIDANCE_DELTA"]], dtype=float)[:N_STEPS]

    print(f"cloud: {pca_cloud.shape} | EVR(3): {float(pca_evr.sum()):.3f} | runs: {len(points)}")
    return (
        N_STEPS,
        branch_latents,
        config,
        delta_of,
        guided_latents,
        pca_cloud_proj,
        pca_evr,
        pca_project,
        pca_targets,
        pca_ung_finals,
        points,
        twin_latents,
    )


@app.cell
def _(N_STEPS, config, mo, points):
    # comparison controls: compare one axis ("n" = across forecast steps); pin the rest
    _axis_vals = {}
    for _sel in points.values():
        for _k, _v in _sel.items():
            _axis_vals.setdefault(_k, {})[str(_v)] = _v
    varying_axes = [_k for _k, _vs in _axis_vals.items() if len(_vs) > 1]
    compare_dropdown = mo.ui.dropdown(
        varying_axes + ["n", "gui_ung"],
        value=("eta" if "eta" in varying_axes else (varying_axes + ["n", "gui_ung"])[0]),
        label="compare: ",
    )
    pin_dropdowns = mo.ui.dictionary({
        _k: mo.ui.dropdown(list(_axis_vals[_k]), value=list(_axis_vals[_k])[0], label=f"{_k}: ")
        for _k in varying_axes
    })
    n_slider = mo.ui.slider(1, N_STEPS, step=1, value=min(2, N_STEPS), label="n: ", show_value=True, debounce=True)
    m_slider = mo.ui.slider(1, config["M"], step=1, value=1, label="m: ", show_value=True, debounce=True)
    return compare_dropdown, m_slider, n_slider, pin_dropdowns


@app.cell
def _(mo):
    elev_slider = mo.ui.slider(0, 90, step=5, value=25, label="elev: ", show_value=True, debounce=True)
    azim_slider = mo.ui.slider(-180, 180, step=5, value=-60, label="azim: ", show_value=True, debounce=True)
    zoom_traj_checkbox = mo.ui.checkbox(label="zoom to trajectories")
    dpi_slider = mo.ui.slider(100, 1000, step=100, value=300, label="dpi: ", show_value=True, debounce=True)
    zoom_pad_slider = mo.ui.slider(-0.4, 2.0, step=0.05, value=0.12, label="zoom pad: ", show_value=True, debounce=True)
    lw_plain_slider = mo.ui.slider(0.1, 5.0, step=0.1, value=1.5, label="solid lw: ", show_value=True, debounce=True)
    lw_dash_slider = mo.ui.slider(0.1, 4.0, step=0.1, value=1.0, label="dashed lw: ", show_value=True, debounce=True)
    return (
        azim_slider,
        dpi_slider,
        elev_slider,
        lw_dash_slider,
        lw_plain_slider,
        zoom_pad_slider,
        zoom_traj_checkbox,
    )


@app.cell
def _(
    N_STEPS,
    azim_slider,
    branch_latents,
    compare_dropdown,
    delta_of,
    dpi_slider,
    elev_slider,
    guided_latents,
    lw_dash_slider,
    lw_plain_slider,
    m_slider,
    mo,
    n_slider,
    np,
    pca_cloud_proj,
    pca_evr,
    pca_project,
    pca_targets,
    pca_ung_finals,
    pin_dropdowns,
    plt,
    points,
    twin_latents,
    zoom_pad_slider,
    zoom_traj_checkbox,
):
    # ===== 3D PCA trajectories =====
    _m = m_slider.value - 1
    _cmp = compare_dropdown.value
    _pins = pin_dropdowns.value

    def _fmt_val(_k, _v):
        return f"δ#{_v}" if _k == "GUIDANCE_DELTA" else f"{_k}={_v}"

    def _sort_key(_v):
        try:
            return (0, float(_v), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(_v))

    _gcmap = plt.get_cmap("Greens")
    def _fade(_i, _total):
        return _gcmap(0.7) if _total <= 1 else _gcmap(0.30 + 0.65 * _i / (_total - 1))

    def _match(_sel, _skip):
        return all(str(_sel[_k]) == _pins[_k] for _k in _pins if _k != _skip)

    # curves: (label, color, guided latents, twin latents, n index)
    _curves = []
    _branch_pair = None
    if _cmp == "n":
        _sel = next((_s for _l, _s in points.items() if _match(_s, None)), None)
        if _sel is not None:
            _ns = [_i for _i in range(N_STEPS) if delta_of(_sel)[_i] != 0]
            for _j, _ni in enumerate(_ns):
                _g = guided_latents(_sel, _m, _ni)
                if _g is not None and np.isfinite(_g).any():
                    _curves.append((f"{_ni + 1}", _fade(_j, len(_ns)), _g, twin_latents(_sel, _m, _ni), _ni))
    elif _cmp == "gui_ung":
        # one pinned run vs its unguided (gui_ung) counterpart, both first-class
        _n = n_slider.value - 1
        _sel = next((_s for _l, _s in points.items() if _match(_s, None)), None)
        if _sel is not None:
            _branch_pair = branch_latents(_sel, _m, _n)
            _tw = twin_latents(_sel, _m, _n)
            if _tw is not None and np.isfinite(_tw).any():
                _curves.append(("gui_ung", "#1F77B4", _tw, None, _n))
    else:
        _n = n_slider.value - 1
        _matched = sorted(
            ((_l, _s) for _l, _s in points.items() if _match(_s, _cmp)),
            key=lambda ls: _sort_key(ls[1].get(_cmp)),
        )
        for _j, (_l, _s) in enumerate(_matched):
            _g = guided_latents(_s, _m, _n)
            if _g is not None and np.isfinite(_g).any():
                _curves.append((_fmt_val(_cmp, _s.get(_cmp)), _fade(_j, len(_matched)), _g, twin_latents(_s, _m, _n), _n))

    plt.close("all")  # free stale canvases

    def _make_fig(_branch_style):
        """One complete 3D figure; _branch_style: None | "chain" | "pingpong"."""
        _fig = plt.figure(figsize=(24, 16), dpi=dpi_slider.value)  # export cell re-renders at dpi=300
        _ax3 = _fig.add_subplot(projection="3d")
        _ax3.scatter(*pca_cloud_proj.T, color="#BBBBBB", s=14, alpha=0.5, depthshade=False,
                     label=f"ERA5 cloud ({pca_cloud_proj.shape[0]} days)")
        _groups = []
        for _label, _color, _g, _tw, _ni in _curves:
            _gp = pca_project(_g)
            _ax3.plot(*_gp.T, "-", color=_color, linewidth=1.2 * lw_plain_slider.value, alpha=0.9, label=_label)
            _ax3.scatter(*_gp.T, s=np.linspace(8, 46, len(_gp)), color=_color, alpha=0.9, depthshade=False)
            _ax3.scatter(*_gp[-1], marker=("o" if _label == "gui_ung" else "D"), s=100, color="black", edgecolors="white", depthshade=False)
            _ax3.text(*_gp[-1], f"    {_label}", color="black", fontsize=9, fontweight="bold")
            if _tw is not None:
                _up = pca_project(_tw)
                _ax3.plot(*_up.T, "--", color=_color, linewidth=lw_dash_slider.value, alpha=0.6, label="_nolegend_")
                _ax3.scatter(*_up[-1], marker="o", s=100, color="black", alpha=0.85, edgecolors="white", depthshade=False)
                _ax3.text(*_up[-1], f"    {_label} (gui_ung)", color="black", fontsize=9, fontweight="bold")
                _groups.append(_up)
            _groups.append(_gp)
        if _cmp == "gui_ung" and _branch_pair is not None:
            _bp, _kp = pca_project(_branch_pair[0]), pca_project(_branch_pair[1])
            _gcol = "#B05C3A"  # terracotta for the gui branch elements
            if _branch_style == "pingpong":
                # TRUE event path: kick (base_t -> gui_t) solid green;
                # pull-back (gui_t -> base_{t+1}) dotted grey
                for _t in range(len(_bp)):
                    _seg = np.stack([_bp[_t], _kp[_t]])
                    _ax3.plot(*_seg.T, "-", color=_gcol, linewidth=lw_plain_slider.value, alpha=0.9,
                              label=("gui" if _t == 0 else "_nolegend_"))
                    if _t + 1 < len(_bp):
                        _seg = np.stack([_kp[_t], _bp[_t + 1]])
                        _ax3.plot(*_seg.T, ":", color="#800080", linewidth=0.7 * lw_dash_slider.value, alpha=0.8, label="_nolegend_")
                # the realized guided path (green) overlaid on the kick/pull-back anatomy
                _gpath = np.vstack([_bp[:1], _kp])
                _ax3.plot(*_gpath.T, "-", color="#2CA02C", linewidth=lw_plain_slider.value,
                          alpha=0.75, label="_nolegend_")
            else:
                # gui chain: start -> gui_0 -> gui_1 -> ...; dotted grey dead-end
                # branches gui_t -> base_{t+1} (counterfactual pull-backs)
                _gpath = np.vstack([_bp[:1], _kp])
                _ax3.plot(*_gpath.T, "-", color="#2CA02C", linewidth=lw_plain_slider.value, alpha=0.9, label="gui")
                for _t in range(len(_bp) - 1):
                    _seg = np.stack([_kp[_t], _bp[_t + 1]])
                    _ax3.plot(*_seg.T, ":", color="#800080", linewidth=0.7 * lw_dash_slider.value, alpha=0.8, label="_nolegend_")
            _ax3.scatter(*_kp.T, s=np.linspace(8, 46, len(_kp)), color=_gcol, alpha=0.9, depthshade=False)
            _ax3.scatter(*_bp.T, s=14, color="#800080", alpha=0.9, depthshade=False, label="gui_ung_over_t")
            _ax3.scatter(*_kp[-1], marker="D", s=100, color="black", edgecolors="white", depthshade=False)
            _ax3.text(*_kp[-1], "    gui", color="black", fontsize=9, fontweight="bold")
            _groups.extend([_bp, _kp])
        # shared starting latent (all curves at one n start from the same seed)
        if _curves:
            _sp = pca_project(_curves[0][2])[0]
            _ax3.scatter(*_sp, marker="o", s=40, color="black", depthshade=False)
            _ax3.text(*_sp, "    start", color="black", fontsize=9, fontweight="bold")
        for _ni in sorted({_c[4] for _c in _curves}):
            _tp = pca_project(pca_targets[_ni])
            _ax3.scatter(*_tp, marker="*", s=100, color="black", depthshade=False)
            _ax3.text(*_tp, "    gt", color="black", fontsize=9, fontweight="bold")
            _groups.append(_tp[None, :])
            if pca_ung_finals is not None:
                _op = pca_project(pca_ung_finals[_ni])
                _ax3.scatter(*_op, marker="o", s=100, facecolors="none", edgecolors="#111111",
                             linewidths=2.0, depthshade=False)
                _ax3.text(*_op, "    ung", color="#111111", fontsize=9, fontweight="bold")
                _groups.append(_op[None, :])
        if zoom_traj_checkbox.value and _groups:
            _pts = np.vstack(_groups)
            _lo, _hi = _pts.min(axis=0), _pts.max(axis=0)
            _pad = zoom_pad_slider.value * float((_hi - _lo).max())
            _ax3.set_xlim(_lo[0] - _pad, _hi[0] + _pad)
            _ax3.set_ylim(_lo[1] - _pad, _hi[1] + _pad)
            _ax3.set_zlim(_lo[2] - _pad, _hi[2] + _pad)
        _ax3.set_xlabel("PC1"); _ax3.set_ylabel("PC2"); _ax3.set_zlabel("PC3")
        _pin_desc = ", ".join(_fmt_val(_k, _pins[_k]) for _k in _pins if _k != _cmp)
        _style_tag = f"  |  {_branch_style}" if _branch_style else ""
        _ax3.set_title(
            f"compare {_cmp}{_style_tag}  |  {_pin_desc}  (m={m_slider.value}, EVR={pca_evr.sum():.0%})",
            loc="left",
        )
        _ax3.view_init(elev=elev_slider.value, azim=azim_slider.value)
        _ax3.legend(loc="upper left", fontsize=8)
        return _fig

    pca_fig = _make_fig("chain" if _cmp == "gui_ung" else None)
    pca_fig2 = _make_fig("pingpong") if (_cmp == "gui_ung" and _branch_pair is not None) else None

    mo.vstack(
        [
            mo.hstack(
                [compare_dropdown] + [pin_dropdowns[_k] for _k in pin_dropdowns.value if _k != _cmp]
                + ([n_slider] if _cmp != "n" else []) + [m_slider, elev_slider, azim_slider, dpi_slider, lw_plain_slider, lw_dash_slider, zoom_traj_checkbox, zoom_pad_slider],
                justify="start", align="start",
            ),
            mo.hstack([pca_fig] + ([pca_fig2] if pca_fig2 is not None else []),
                      justify="start", align="start"),
        ],
        align="start",
    )
    return pca_fig, pca_fig2


@app.cell
def _(Path, mo, n_slider, pca_fig, pca_fig2, rollout_id, save_fig_button):
    # ===== thesis-figure export (high-dpi PNG + vector PDF) =====
    _msg = "export the 3D figure at the current view"
    if save_fig_button.value:
        _out_dir = Path("figures")
        _out_dir.mkdir(exist_ok=True)
        _stem = _out_dir / f"pca_trajectories_{rollout_id.replace('/', '_').replace(':', '-')}_n{n_slider.value}"
        _saved = []
        for _f, _suffix in ((pca_fig, ""), (pca_fig2, "_pingpong")):
            if _f is None:
                continue
            _f.savefig(f"{_stem}{_suffix}.png", dpi=300, bbox_inches="tight")
            _f.savefig(f"{_stem}{_suffix}.pdf", bbox_inches="tight")
            _saved.append(f"`{_stem}{_suffix}`.png/.pdf")
        _msg = "saved " + " and ".join(_saved)
    mo.hstack([save_fig_button, mo.md(f"_{_msg}_")], justify="start", align="center")
    return


@app.cell
def _(mo):
    save_fig_button = mo.ui.run_button(label="save figure")
    return (save_fig_button,)


@app.cell(hide_code=True)
def _(
    N_STEPS,
    delta_of,
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
    # ===== leaderboard =====
    # every run, pooled over its guided n at the selected member; ranked by dist to GT
    _m = m_slider.value - 1
    _pc_std = pca_cloud_proj.std(axis=0)
    _sqf = float(np.sqrt(pca_targets.shape[1]))
    _rows = []
    for _label, _sel in points.items():
        _d, _p, _z = [], [], []
        for _ni in range(N_STEPS):
            if delta_of(_sel)[_ni] == 0.0:
                continue
            _g = guided_latents(_sel, _m, _ni)
            if _g is None or not np.isfinite(_g).any():
                continue
            _tw = twin_latents(_sel, _m, _ni)
            _gp = pca_project(_g)
            _d.append(float(np.linalg.norm(_g[-1] - pca_targets[_ni]) / _sqf))
            _p.append(float(np.linalg.norm(_g[-1] - _tw[-1]) / _sqf))
            _z.append(float(np.sqrt(((_gp[-1] / _pc_std) ** 2).sum())))
        if _d:
            _rows.append({"run": _label, "n": len(_d), "dist": float(np.mean(_d)),
                          "push": float(np.mean(_p)), "offz": float(np.mean(_z))})
    _rows.sort(key=lambda r: r["dist"])
    _best_d = min((_r["dist"] for _r in _rows), default=None)
    _best_z = min((_r["offz"] for _r in _rows), default=None)
    _md = ["| run | n's | dist to GT (K) ↓ | push (K) | off-manifold z ↓ |", "|---|---|---|---|---|"]
    for _r in _rows:
        _ds = f"**{_r['dist']:.3f}**" if _r["dist"] == _best_d else f"{_r['dist']:.3f}"
        _zs = f"**{_r['offz']:.2f}**" if _r["offz"] == _best_z else f"{_r['offz']:.2f}"
        _md.append(f"| {_r['run']} | {_r['n']} | {_ds} | {_r['push']:.3f} | {_zs} |")
    mo.vstack(
        [
            mo.md(
                "## Leaderboard\n\nEvery run, guided $n$'s pooled (m={}). `dist to GT` / `push`: per-pixel "
                "RMS of the endpoint vs GT / vs the twin (K); `off-manifold z`: endpoint Mahalanobis distance "
                "from the ERA5 cloud (per-PC std units).".format(m_slider.value)
            ),
            mo.md("\n".join(_md)) if _rows else mo.md("_no runs with traces_"),
        ],
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
