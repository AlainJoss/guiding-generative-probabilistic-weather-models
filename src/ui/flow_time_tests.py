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
    from src.ui.map import visualize_map, get_mask_center
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
        return str(lb).split(" δ#")[0].replace("a_t_mode", "γ")


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
        get_mask_center,
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
        region_bool,
        region_cloud_sample,
        region_latent,
        residual_scaler,
        select_point,
        sweep_points,
        visualize_map,
        xr,
    )


@app.cell(hide_code=True)
def _(open_store, residual_scaler, xr):
    # Reconstruct the unguided STATE store from the new latent format: x_t = x_det + sigma_res*z_t.
    # Drop-in for open_store(dir, prefix, var) where prefix in {"gui_ung","ung"} -- new rollouts persist the
    # latent (gui_ung_res / ung_res) + det core (gui_det / ung_det) instead of the physical state. The det
    # core is the guided pass's gui_det for the same-seed twin, but the independent run's own ung_det for
    # "ung". Falls back to the legacy {prefix} clean-pred store for pre-switch experiments.
    def open_unguided_state(rollout_dir, prefix, var):
        if not (rollout_dir / f"{prefix}_res.zarr").exists():
            return open_store(rollout_dir, prefix, var)               # legacy clean-pred store
        _det = "ung_det" if prefix == "ung" else "gui_det"
        if not (rollout_dir / f"{_det}.zarr").exists():
            _det = "gui_det"
        _z = open_store(rollout_dir, f"{prefix}_res", var)            # latent z_t: (..., t, [level], lat, lon)
        _dc = open_store(rollout_dir, _det, var)                      # x_det: (..., [level], lat, lon), no t
        if "level" in _z.dims:
            _sig = xr.DataArray([residual_scaler("level", var, int(_L)) for _L in _z["level"].values],
                                dims=("level",), coords={"level": _z["level"]})
        else:
            _sig = residual_scaler("surface", var)
        # match the res store's dim order (m,n,t,lat,lon,sweep) so lat/lon stay the trailing axes.
        # {prefix}_res holds z_0..z_T (length T+1), so x_t = x_det + sigma_res*z_t reaches the true
        # final x_T at t=-1 with no splice.
        return (_dc + _sig * _z).transpose(*_z.dims)

    return (open_unguided_state,)


@app.cell
def _(mo):
    md_title = r"""
    # Flow-time tests

    Flow step $t$ = the $T$ internal Euler steps of one `sample()` call; $n$ indexes the weather step and
    $x_{n,t}$ the actual flow state. Metrics score the guided channel `VAR@LEVEL` through the masked average
    $\mathcal{M}_\pi$ (mask $\pi$), pooled over ensemble members $m$, guided weather steps $n$, and the
    selected experiment's subexperiments — per **guidance schedule** (flow-time profile $\gamma_t$, e.g.
    `spike@k` / `spread@i-j`) and per **intensity level** $\rho_H$. The selectors up top pin the sweep
    context and pick the plotted intensity; metrics recompute automatically.

    - **T0 — Qualitative analysis**: descriptive **statistics** (T01), the **example maps** and
      $\Delta x^{\mathrm{GE}}$ across variables (T02–T03), the **absolute-intensity** (T04) / vs-ground-truth
      (T05) profiles, and the flow-time **convergence** grid (T06) with its **decomposition** (T07).
    - **T1 — Temporal localization of guidance**: multivariate propagation (T1.1, opening with the
      guidance-effect maps over flow time), spatial deviation from the mask (T1.2), and ensemble diversity
      of the footprint (T1.3).
    - **Support**: realized-target table, leaderboard, guidance-kick decomposition, latent-space closeness,
      and the prediction-noise check.
    """
    mo.md(md_title)
    return (md_title,)


@app.cell(hide_code=True)
def hdr_tables_readme(mo):
    md_reading = r"""
    ## Reading the tables — mean ± std

    Every table cell reports a score as **mean ± std** over a pool of realizations. For a pool $\mathcal{P}$ of
    scalar values $\{v_p\}_{p\in\mathcal{P}}$,

    $$\text{cell} \;=\; \mu_{\mathcal{P}} \pm \sigma_{\mathcal{P}}, \qquad
    \mu_{\mathcal{P}} = \frac{1}{|\mathcal{P}|}\sum_{p\in\mathcal{P}} v_p, \qquad
    \sigma_{\mathcal{P}} = \sqrt{\frac{1}{|\mathcal{P}|}\sum_{p\in\mathcal{P}} \big(v_p-\mu_{\mathcal{P}}\big)^2}\,,$$

    i.e. the **population** standard deviation (dividing by $|\mathcal{P}|$, not $|\mathcal{P}|-1$). The pool
    axes are ensemble members $m$, guided weather steps $n$, subexperiments/experiments $e$, and — for summary
    rows — intensity levels $\delta$. Each table's caption names the axis its $\pm$ runs over.

    - **Detailed tables** (rows = experiments $e$, columns = schedule $\gamma$; one table per intensity
      $\delta$): the per-instance score is first collapsed over the inner axis (the guided steps $n$) for each
      $(e,m)$, giving one value $S_{e,m}$ per member; the cell then reports **$\mu\pm\sigma$ over the remaining
      axis** named in the caption (members $m$, or steps $n$). So $\pm$ is that within-experiment spread.
    - **Summary tables** (one row per schedule $\gamma$): the same values pooled and reported as
      **$\mu\pm\sigma$ over all experiments × members × steps × intensity levels** — the overall ranking.
    - **Bold** marks the best schedule for the metric (largest or smallest, per its caption); summary rows are
      sorted by $\mu$.

    **T1.1 scores (per-member normalization).** For each experiment $e=(\text{startdate},\rho)$ and member $m$,
    each variable is reduced to $A_v=\sum_\ell \mathcal{M}_\pi(|\Delta x^{\mathrm{GE}}_{v\ell}|)$ (sum over
    levels of the masked-mean absolute guidance effect), normalized **per member** $N_v=A_v/\max_{\gamma'}A_v'$,
    and aggregated into a **single score** $R=\big(\sum_v N_v\big)/\max_{\gamma'}\sum_v N_v'$. Tables report $\mu\pm\sigma$ of $N_v$
    (per-variable views $V_2$) or $R$ (single-score views $V_1$); see T1.1 for the two paired summary views.

    **T1.2 / T1.3 scores.** Both reuse the T1.1 construction on their own object — the spatial-deviation $D$
    (T1.2) or the signed-effect ensemble spread $s=\mathcal{M}_\pi(\operatorname{std}_m\,\Delta x^{\mathrm{GE}})$
    (T1.3) — in place of $\mathcal{M}_\pi(|\Delta x^{\mathrm{GE}}|)$: per variable $A_v=\sum_\ell(\text{object})$,
    per-member normalized $N_v=A_v/\max_{\gamma'}A_v'$, single score $R=(\sum_v N_v)/\max_{\gamma'}\sum_v N_v'$,
    shown in the same paired per-variable ($V_2$, $\mu\pm\sigma$ of $N_v$) and single-score ($V_1$, of $R$)
    views. **T1.3 has no member axis** ($\operatorname{std}_m$ already consumes the members), so its cells carry
    a single value and any $\pm$ runs over startdates/$\rho$ only.

    **Descriptive statistics** (Support) pool differently: each cell is the ensemble avg $\pm$ std of a
    per-instance spatial statistic (min/max over the mask support; area-weighted masked mean $\mathcal{M}_\pi$ and weighted std) over the pool selected schedules
    $\times$ members $\times$ steps.
    """
    mo.md(md_reading)
    return (md_reading,)


@app.cell
def hdr_export(mo):
    md_export = r"""
    ## Exporting figures

    The **⬇ save all charts → figures/** button saves every currently-rendered chart to
    `docs/report/figures/` as a separate PNG (150 dpi), one file per panel, named by test id — e.g.
    `\includegraphics{figures/T02a.png}`:

    | test | ids |
    | --- | --- |
    | T02 | `T02a`, `T02b`, `T02c` |
    | T03 | `T03a`–`T03g` |
    | T04 | `T04` |
    | T05 | `T05` |
    | T06 | `T06a`–`T06d` |
    | T07 | `T07a`–`T07d` |
    | T1.1 / T1.2 / T1.3 charts | `T1.1`, `T1.2`, `T1.3` |
    | visual examples | `T1.1a`, `T1.2a`, `T1.3a` |

    Only currently-rendered charts are saved — first compute what you want (press **compute field profiles**
    for T04/T05 and the T1.1–T1.3 charts), then press export. Each press overwrites, so the files always match
    the current selection.
    """
    mo.md(md_export)
    return (md_export,)


@app.cell
def table_report_cell(
    copyable,
    md_closeness,
    md_descstats,
    md_export,
    md_kick,
    md_leaderboard,
    md_prednoise,
    md_reading,
    md_support,
    md_t0,
    md_t01,
    md_t02,
    md_t03,
    md_t04,
    md_t05,
    md_t0_gegrid,
    md_t0_maps,
    md_t1,
    md_t11,
    md_t11_def,
    md_t11_scores,
    md_t12,
    md_t12_def,
    md_t12_scores,
    md_t12_visual,
    md_t12_whymask,
    md_t13,
    md_t13_def,
    md_t13_scores,
    md_t13_visual,
    md_t14,
    md_target_real,
    md_title,
    mo,
    report_closeness,
    report_descstats,
    report_diversity,
    report_leaderboard,
    report_prednoise,
    report_propagation,
    report_realism,
    report_reliability,
    report_t14,
):
    # --- global "table report": the full notebook as one markdown chunk (section headers + explanations
    # interleaved with every table, in document order). Copyable via the button below. The inline chart
    # captions (rendered inside each plot's vstack) are view-dependent and NOT included; everything else is.
    _order = [
        ("md", md_title),
        ("md", md_reading),
        ("md", md_export),
        ("md", md_t0),
        ("md", md_descstats), ("tbl", report_descstats),
        ("md", md_t0_maps),
        ("md", md_t0_gegrid),
        ("md", md_t04),
        ("md", md_t05),
        ("md", md_t01),
        ("md", md_t02),
        ("md", md_t1),
        ("md", md_t11), ("md", md_t03), ("md", md_t11_def), ("md", md_t11_scores), ("tbl", report_propagation),
        ("md", md_t12), ("md", md_t12_visual), ("md", md_t12_def), ("md", md_t12_scores), ("tbl", report_realism), ("md", md_t12_whymask),
        ("md", md_t13), ("md", md_t13_visual), ("md", md_t13_def), ("md", md_t13_scores), ("tbl", report_diversity),
        ("md", md_t14), ("tbl", report_t14),
        ("md", md_support),
        ("md", md_target_real), ("tbl", report_reliability),
        ("md", md_leaderboard), ("tbl", report_leaderboard),
        ("md", md_kick),
        ("md", md_closeness), ("tbl", report_closeness),
        ("md", md_prednoise), ("tbl", report_prednoise),
    ]
    _chunks = []
    for _kind, _val in _order:
        if _kind == "md":
            _chunks.append(str(_val).strip())
        else:
            _chunks.extend(_val)
    table_report = mo.vstack([
        mo.md("**Table report** — copy the whole notebook (section headers, explanations, and every table) "
              "as one markdown chunk:"),
        copyable("\n\n".join(_chunks), "📋 copy full markdown report"),
    ], align="start")
    table_report
    return


@app.cell
def export_charts(mo):
    # ---- Export: one-click save of every plotted chart to the report figures folder (separate PNG per panel) ----
    from pathlib import Path
    FIG_DIR = Path("/Users/alain/Desktop/master-thesis/guiding-generative-probabilistic-weather-models/docs/report/figures")
    export_button = mo.ui.run_button(label="⬇ save all charts → figures/")
    def save_chart(_fig, _name):
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        _fig.savefig(FIG_DIR / f"{_name}.png", dpi=150, bbox_inches="tight")
        return _name
    mo.vstack([export_button, mo.md(f"_Press to save every plotted chart (T02–T07, T1.1–T1.3 + visual examples) to_ `docs/report/figures/` _as separate PNGs. Compute the relevant data first._")])
    return export_button, save_chart


@app.cell
def _(get_rollout_ids, mo, refresh_button):
    # OUTER experiment selector (single dropdown; defaults to the LATEST experiment). Metrics are
    # averaged over ALL its date-subexperiments (which share the same sweep).
    if refresh_button.value:      # press to re-read the rollout ids from disk (pick up new/removed runs)
        pass
    _exp_ids = sorted({rid.split("/", 1)[0] for rid in get_rollout_ids("gui")})
    experiment_dropdown = mo.ui.dropdown(
        _exp_ids, value=_exp_ids[-1] if _exp_ids else None, label="experiment: "
    )
    return (experiment_dropdown,)


@app.cell(hide_code=True)
def _(experiment_dropdown, get_rollout_ids, mo):
    # rollouts (start_ts) of the selected experiment; ALL selected by default. Untick the ones still
    # running to drop them from the pool and keep seeing intermediate results from the finished ones.
    _avail = sorted(rid for rid in get_rollout_ids("gui") if rid.split("/", 1)[0] == experiment_dropdown.value)
    _rollout_opts = {(rid.split("/", 1)[1] if "/" in rid else rid): rid for rid in _avail}   # start_ts -> full rid
    rollout_multiselect = mo.ui.multiselect(_rollout_opts, value=list(_rollout_opts.keys()), label="rollouts: ")
    return (rollout_multiselect,)


@app.cell
def _(get_rollout_dir, load_rollout, mask_bbox, mo, np, rollout_multiselect):
    # all date-subexperiments of the selected experiment(s) -> metrics are pooled and
    # averaged over them (they share VAR/LEVEL/mask/sweep, hence one bbox + one PCA basis)
    _selected = sorted(rollout_multiselect.value)   # rollouts ticked in the multiselect above
    def _rollout_ready(_rid):
        # analyzable once the guided pass wrote every store the T-cells read -> skip the ones
        # still running so "select all" shows whatever is finished instead of crashing
        _d = get_rollout_dir(_rid)
        if not (_d / "guidance_schedule.json").exists():
            return False
        if not all((_d / f"{_s}.zarr").exists() for _s in ("gui", "gui_det", "vfs", "grads")):
            return False
        return all((_d / f"{_a}.zarr").exists() or (_d / f"{_bb}.zarr").exists()
                   for _a, _bb in (("gui_res", "res"), ("gui_ung_res", "gui_ung"),
                                   ("ung_res", "ung"), ("ung_det", "gui_det")))
    EXPS = [rid for rid in _selected if _rollout_ready(rid)]
    ROLLOUTS_SKIPPED = [rid.split("/")[-1] for rid in _selected if rid not in EXPS]
    mo.stop(not EXPS, mo.md("**No ready rollouts.** "
                            + (f"Still running / incomplete: {', '.join(ROLLOUTS_SKIPPED)}. " if ROLLOUTS_SKIPPED else "")
                            + "Tick a finished rollout (or add guided data) and re-run."))
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
    return (
        BBOX,
        EXPS,
        LEVEL,
        M,
        MASK0,
        N,
        PARTITION,
        ROLLOUTS_SKIPPED,
        VAR,
        records0,
        sweep_values0,
    )


@app.cell(hide_code=True)
def _(mo, sweep_values0):
    # --- pin the non-schedule / non-intensity sweep axes (e.g. GUI_REF, MASK_MODE, mask_shift, sigma_div).
    # --- Schedule axes (GUIDANCE_MODE + its GUIDANCE_METHOD_HYPERS: a_t_mode / fgwnolr_w_init / eta / phi)
    # --- stay as the compared rows; GUIDANCE_DELTA is the intensity axis (handled separately). Single-valued
    # --- axes are pinned silently. Mirrors intensity_comparison.py's pin_dropdowns.
    from src.rollout_config import GUIDANCE_METHOD_HYPERS as _GMH
    _pin_skip = {"GUIDANCE_MODE", "GUIDANCE_DELTA"} | {_h for _hs in _GMH.values() for _h in _hs}
    _pin = {}
    for _k, _vals in sweep_values0.items():
        if _k in _pin_skip or not isinstance(_vals, list) or len(_vals) <= 1:
            continue
        _srt = sorted(_vals) if all(isinstance(_v, (int, float)) for _v in _vals) else list(_vals)
        _pin[_k] = mo.ui.dropdown([str(_v) for _v in _srt], value=str(_srt[0]), label=f"{_k}: ")
    pin_dropdowns = mo.ui.dictionary(_pin)
    def pinned_records(_recs, _base):
        # keep only records matching the pinned (coord-labeled) context; empty _base -> all records
        return [_r for _r in _recs if all(_r["sweep"].get(_k) == _v for _k, _v in _base.items())]

    return pin_dropdowns, pinned_records


@app.cell(hide_code=True)
def _(pin_dropdowns, sweep_values0):
    # --- coord-labeled selection for the pinned axes (recovered from the dropdowns); drives record
    # --- filtering. Empty when no non-schedule axis varies (current data). Mirrors base_sel.
    from src.utils import sweep_coord_label as _scl
    def _pin_recover(_k, _raw):
        for _v in sweep_values0[_k]:
            if str(_v) == _raw:
                return _v
        return sweep_values0[_k][0]
    base_pin = {_k: _scl(_k, _pin_recover(_k, _raw), sweep_values0) for _k, _raw in pin_dropdowns.value.items()}
    return (base_pin,)


@app.cell(hide_code=True)
def _(base_pin, pinned_records, records0, sweep_points, sweep_values0):
    # sweep points for the PINNED context. In THIS notebook a_t_mode is ALWAYS the analysis axis (the
    # schedule γ), never a collapsed sweepable — so it is injected into every label even when single-valued;
    # other varying mode-hypers and the delta index are kept as sweep_points authored them. Ordered by the
    # a_t_mode order in sweep_params.json. label_delta maps each label to its GUIDANCE_DELTA index.
    _raw = sweep_points(sweep_values0, pinned_records(records0, base_pin))
    points = {}
    for _lb, _sel in _raw.items():
        _am = _sel.get("a_t_mode")
        if _am is not None and "a_t_mode=" not in _lb:
            _bits = _lb.split(" ")
            _lb = " ".join([_bits[0], f"a_t_mode={_am}"] + _bits[1:])
        points[_lb] = _sel
    _atm_ord = {str(_v): _i for _i, _v in enumerate(sweep_values0.get('a_t_mode', []))}
    points = dict(sorted(points.items(),
                         key=lambda _kv: _atm_ord.get(str(_kv[1].get('a_t_mode', '')), len(_atm_ord))))
    label_delta = {_lb: _sel.get("GUIDANCE_DELTA", 0) for _lb, _sel in points.items()}
    return label_delta, points


@app.cell(hide_code=True)
def _(N, mo, sweep_values0):
    # intensity (GUIDANCE_DELTA) levels: enumerate + a PLOT-only selector to show one level at a time.
    # delta_labels[i] = peak%, delta_order = peak-sorted (lowest first). Mirrors intensity_comparison.py.
    _DELTAS_FT = sweep_values0["GUIDANCE_DELTA"]
    delta_order = sorted(range(len(_DELTAS_FT)), key=lambda _i: max(_DELTAS_FT[_i][:N]))
    delta_labels = {_i: f"peak@{100 * max(_DELTAS_FT[_i][:N]):+.3g}%" for _i in delta_order}
    intensity_dropdown = mo.ui.dropdown({delta_labels[_i]: _i for _i in delta_order},
                                        value=delta_labels[delta_order[0]], label="intensity (plots): ")
    return delta_labels, delta_order, intensity_dropdown


@app.cell(hide_code=True)
def _(intensity_dropdown, label_delta, selected_point_labels):
    # labels drawn in the PLOTS: the multiselected schedules at the chosen intensity level (== all
    # multiselected labels when there is a single delta). Tables/leaderboards keep every schedule x delta.
    plot_labels = [lb for lb in selected_point_labels if label_delta.get(lb) == intensity_dropdown.value]
    return (plot_labels,)


@app.cell
def _(gname, mo, points):
    # "sweep points" selector: only the a_t_modes (schedule γ), NOT the per-δ combos. Selecting a γ
    # auto-includes ALL its combos (every intensity δ × mask variant) downstream via selected_point_labels.
    # γ is shown in the chart form (back-shifted, LaTeX stripped) so it matches the plots/tables/leaderboards.
    _gammas = list(dict.fromkeys(
        gname(_lb).replace(r"\text{-}", "-").replace(r"$\mathrm{", "").replace("}$", "").replace("$", "")
        for _lb in points
    ))
    point_multiselect = mo.ui.multiselect(_gammas, value=_gammas, label="sweep points (γ): ")
    return (point_multiselect,)


@app.cell(hide_code=True)
def sweep_point_expand(gname, point_multiselect, points):
    # Expand the selected a_t_modes (γ) to every underlying combo label (all intensities δ × mask variants).
    # Downstream (sched_colors, plot_labels, tables) stays keyed by combos, so nothing else changes.
    def _gdisp(_lb):
        return gname(_lb).replace(r"\text{-}", "-").replace(r"$\mathrm{", "").replace("}$", "").replace("$", "")
    selected_point_labels = [_lb for _lb in points if _gdisp(_lb) in point_multiselect.value]
    return (selected_point_labels,)


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
    return


@app.cell
def _(
    EXPS,
    ROLLOUTS_SKIPPED,
    experiment_dropdown,
    intensity_dropdown,
    mo,
    n_slider,
    pin_dropdowns,
    point_multiselect,
    refresh_button,
    rollout_multiselect,
):
    mo.vstack([
        mo.hstack([experiment_dropdown, refresh_button], justify="start", align="center"),
        rollout_multiselect,
        mo.md(f"_averaging over **{len(EXPS)}** ready subexperiment(s): "
              f"{', '.join(e.split('/')[-1].split(chr(95))[0] for e in EXPS)}_"
              + (f"  ·  _skipped (not ready): {', '.join(ROLLOUTS_SKIPPED)}_" if ROLLOUTS_SKIPPED else "")),
        mo.hstack([point_multiselect, n_slider], justify="start", align="center"),
        mo.hstack([mo.md("_sweep:_"), *[pin_dropdowns[_k] for _k in pin_dropdowns.value], intensity_dropdown],
                  justify="start", align="center"),
    ], align="start")
    return


@app.cell
def _(mo, sched_colors):
    # T2 interference subtests (T2a/T2b): the per-channel push across ALL variables is heavy
    # (~1-2 min over the pool), so it is gated behind a button; the top-k slider then re-renders
    # the tables/grids instantly from the cached result.
    interf_button = mo.ui.run_button(label="compute field profiles: interference + realism (~1-2 min)")
    interf_k_slider = mo.ui.slider(1, max(len(sched_colors), 2), value=min(3, max(len(sched_colors), 1)),
                                   step=1, label="top-k schedules: ", show_value=True)
    mo.hstack([interf_button, interf_k_slider], justify="start", align="center")
    return interf_button, interf_k_slider


@app.cell
def _(mo):
    md_t0 = r"""
    ## T0: Qualitative analysis

    *Did the guidance realize the prescribed target, and how does its effect build up along flow time?*

    **Notation.** $\mathcal{M}_\pi$ is the area-weighted masked mean over the target region $\pi$. At weather
    step $n$ the target is $y_n^\star=(1+\rho_n)\,\mathcal{M}_\pi(x_n^{\mathrm{ung}})$ — the unguided regional
    mean scaled by the relative intensity $\rho_n$ (Ch. 3), with $x_n^{\mathrm{ung}}$ the matched unguided
    reference. The intermediate guided state is reconstructed as
    $\hat{x}_t=\hat{x}^{\mathrm{gui\_det}}+\sigma_r z_t$, and the **guidance effect** is the same-seed twin
    difference $\Delta x^{\mathrm{GE}}_t=x_t^{\mathrm{gui}}-x_t^{\mathrm{ung}\mid\mathrm{gui}}$ (guided minus
    unguided-under-guided-context).

    Seven views: descriptive **statistics** of the target field (T01); the **example maps**
    $x^{\mathrm{ung}\mid\mathrm{gui}}$, $x^{\mathrm{gui}}$, $\Delta x^{\mathrm{GE}}$ (T02) and $\Delta x^{\mathrm{GE}}$
    **across variables** (T03); the profiles of **absolute intensity** (T04) and **deviation from ground truth**
    (T05); and the flow-time **convergence** grid (T06) with its per-step **decomposition** (T07).
    """
    mo.md(md_t0)
    return (md_t0,)


@app.cell(hide_code=True)
def hdr_descstats(mo):
    md_descstats = r"""
    ### T01: Descriptive statistics

    For each intensity $\rho$ and experiment, the four per-instance spatial statistics
    $\{\min,\max,\mathrm{mean},\mathrm{std}\}$ of the **target field** (`VAR`) over the target region
    defined by the mask $\pi$ — **mean**/**std** use the area-weighted masked mean $\mathcal{M}_\pi(x)=\sum_{ij}\pi_{ij}x_{ij}/\sum_{ij}\pi_{ij}$ and the corresponding weighted std, while **min**/**max** are over the mask support ($\pi>0$), reported for the guided state $x^{\mathrm{gui}}$
    and its same-seed unguided twin $x^{\mathrm{ung}\mid\mathrm{gui}}$. Each cell is the **ensemble avg
    $\pm$ std** of that statistic over the pool (selected schedules $\times$ members $m$ $\times$ guided
    steps $n$); the four statistics are the columns, each split into an ung$\mid$gui and a gui subcolumn.
    """
    mo.md(md_descstats)
    return (md_descstats,)


@app.cell(hide_code=True)
def descstats_compute(
    EXPS,
    M,
    N,
    VAR,
    channel,
    gamma_key,
    label_delta,
    load_rollout,
    metrics,
    np,
    open_store,
    open_unguided_state,
    sched_colors,
    select_point,
    sweep_points,
):
    # ---- Descriptive statistics of the target field over the target region (Support) ----
    # For each intensity rho (delta) x experiment: the four per-instance spatial statistics
    # (min, max, mean, std) of the TARGET variable field over the target region defined by the mask:
    # mean/std are the area-weighted masked mean M_pi = sum(pi*x)/sum(pi) and its weighted std; min/max
    # are over the mask support (pi>0). For the guided state (gui) and same-seed unguided twin. Pool =
    # selected schedules x members m x guided steps n; the table reports the ensemble avg +/- std per stat.
    descstats = None
    if metrics is not None:
        _labs = sorted(sched_colors, key=gamma_key)   # all selected schedules across BOTH intensities
        _toff = 273.15 if VAR in ("2m_temperature", "temperature") else 0.0   # K -> C for temperature (std shift-invariant)
        def _four_stats(_f2d, _mk, _wsum, _supp):
            _f = np.asarray(_f2d, float) - _toff
            _mean = float((_f * _mk).sum() / _wsum)                          # area-weighted masked mean (M_pi)
            _std = float(np.sqrt((_mk * (_f - _mean) ** 2).sum() / _wsum))    # area-weighted masked std
            _px = _f[_supp]
            return {"min": float(_px.min()), "max": float(_px.max()), "mean": _mean, "std": _std}
        descstats = {}
        for _ei, _rid in enumerate(EXPS):
            _dir, _cfg, _sv, _recs, _mask = load_rollout(_rid)
            _mk = np.asarray(_mask, float); _wsum = float(_mk.sum()) or 1.0; _supp = _mk > 0   # per-experiment target region
            _sp = sweep_points(_sv, _recs)
            for _lb in _labs:
                if _lb not in _sp:
                    continue
                _dd = label_delta.get(_lb); _sel = _sp[_lb]
                _g = channel(select_point(open_store(_dir, "gui", VAR), _sel), _cfg)
                _t = channel(select_point(open_unguided_state(_dir, "gui_ung", VAR), _sel), _cfg)
                _p = np.asarray(_sv["GUIDANCE_DELTA"][_sel["GUIDANCE_DELTA"]], float)[:N]
                _gns = [nn for nn in range(N) if _p[nn] != 0.0] or [0]
                _acc = descstats.setdefault((_ei, _dd),
                    {"gui": {_s: [] for _s in ("min", "max", "mean", "std")},
                     "ung": {_s: [] for _s in ("min", "max", "mean", "std")}})
                for _m in range(M):
                    for _n in _gns:
                        _tf = _t.isel(m=_m, n=_n); _tf = _tf.isel(t=-1) if "t" in _tf.dims else _tf
                        _sg = _four_stats(_g.isel(m=_m, n=_n), _mk, _wsum, _supp); _st = _four_stats(_tf, _mk, _wsum, _supp)
                        for _s in ("min", "max", "mean", "std"):
                            _acc["gui"][_s].append(_sg[_s]); _acc["ung"][_s].append(_st[_s])
    return (descstats,)


@app.cell(hide_code=True)
def descstats_table(
    EXPS,
    VAR,
    copyable,
    delta_labels,
    delta_order,
    descstats,
    mo,
    np,
):
    report_descstats = []
    # two-level table: 4 stat columns (min/max/mean/std), each split into ung|gui and gui subcolumns;
    # rows = experiments; one table per intensity rho. Cell = ensemble avg +/- std of that statistic.
    if descstats is None:
        descstats_view = mo.md("_press **compute metrics**_")
    else:
        _stats = ["min", "max", "mean", "std"]
        _blocks = []
        for _dd in delta_order:
            _present = [(_ei, descstats[(_ei, _dd)]) for _ei in range(len(EXPS)) if (_ei, _dd) in descstats]
            if not _present:
                continue
            _rows_html = []
            _md = ["| experiment | " + " | ".join(f"{_s} ung/gui | {_s} gui" for _s in _stats) + " |",
                   "|" + "---|" * (2 * len(_stats) + 1)]
            for _ei, _acc in _present:
                _en = EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]
                _tds = []; _mds = []
                for _s in _stats:
                    for _side in ("ung", "gui"):
                        _v = _acc[_side][_s]
                        _txt = f"{np.mean(_v):.4g} ± {np.std(_v):.2g}" if _v else "—"
                        _tds.append(f"<td style='padding:2px 9px;text-align:right;white-space:nowrap'>{_txt}</td>")
                        _mds.append(_txt)
                _rows_html.append(f"<tr><td style='padding:2px 9px'>{_en}</td>" + "".join(_tds) + "</tr>")
                _md.append(f"| {_en} | " + " | ".join(_mds) + " |")
            _head = ("<thead><tr><th rowspan='2' style='padding:2px 9px;border-bottom:1px solid #999;text-align:left'>experiment</th>"
                     + "".join(f"<th colspan='2' style='padding:2px 9px;border-bottom:1px solid #999'>{_s}</th>" for _s in _stats)
                     + "</tr><tr>"
                     + "".join("<th style='padding:1px 9px;font-weight:normal;opacity:.75'>ung|gui</th><th style='padding:1px 9px'>gui</th>" for _ in _stats)
                     + "</tr></thead>")
            _tbl = f"<table style='border-collapse:collapse;font-size:12.5px'>{_head}<tbody>{''.join(_rows_html)}</tbody></table>"
            _rho = delta_labels.get(_dd, f"delta#{_dd}")
            _title = (f"**intensity {_rho}** — descriptive statistics of the target field "
                      f"`{VAR}` over the target region (mask π: weighted mean/std, min/max over support); each cell = ensemble avg ± std of the "
                      f"per-instance spatial statistic, pooled over selected schedules × members × steps.")
            report_descstats.append(_title + "\n\n" + "\n".join(_md))
            _blocks.append(mo.vstack([mo.md(_title), copyable(_title + "\n\n" + "\n".join(_md)), mo.Html(_tbl)], align="start"))
        descstats_view = mo.vstack(_blocks, align="start")
    descstats_view
    return (report_descstats,)


@app.cell(hide_code=True)
def md_t0_maps(mo):
    md_t0_maps = r"""
    ### T02: Example maps

    The three objects the evaluation is built on, for the selected experiment and schedule at member $m{=}0$
    (target field, mask bbox): the matched unguided twin $x^{\mathrm{ung}\mid\mathrm{gui}}$, the guided state
    $x^{\mathrm{gui}}$, and their difference — the **guidance effect**
    $\Delta x^{\mathrm{GE}} = x^{\mathrm{gui}} - x^{\mathrm{ung}\mid\mathrm{gui}}$ (diverging, white $=0$). The two
    state panels share a colour scale; each panel reports descriptive stats over the mask ($\mu\pm\sigma$,
    [min, max]) to orient oneself.
    """
    mo.md(md_t0_maps)
    return (md_t0_maps,)


@app.cell(hide_code=True)
def t0_selectors(
    EXPS,
    delta_order,
    gamma_key,
    gname,
    mo,
    mode_of,
    sched_colors,
):
    # T0 example selectors: browse the startdate (experiment) and the schedule γ (a_t_mode); intensity ρ and
    # all other sweep dims are fixed to the first (default). Only startdate and γ are user-selectable (m=0).
    _dd0 = delta_order[0] if delta_order else 0
    _t0_opts = {}
    for _ei in range(len(EXPS)):
        _date = EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]
        _key = _date if _date not in _t0_opts else f"{_date} #{_ei}"
        _t0_opts[_key] = (_ei, _dd0)
    t0_exp = mo.ui.dropdown(options=_t0_opts, value=(next(iter(_t0_opts)) if _t0_opts else None), label="example startdate: ")
    _t0_gs = {}
    for _l in sorted(sched_colors, key=gamma_key):
        _t0_gs.setdefault(mode_of(_l), gname(_l))
    t0_sched = mo.ui.dropdown(options=_t0_gs, value=(next(iter(_t0_gs)) if _t0_gs else None), label="example schedule γ: ")
    return t0_exp, t0_sched


@app.cell(hide_code=True)
def t0_maps(
    EXPS,
    N,
    VAR,
    WARM_CMAP,
    base_pin,
    channel,
    contour_checkbox_t02,
    contour_color_dropdown_t02,
    contour_levels_slider_t02,
    cool_half_cmap,
    delta_labels,
    export_button,
    gname,
    label_delta,
    load_rollout,
    metrics,
    mo,
    np,
    open_store,
    open_unguided_state,
    pinned_records,
    region_crop,
    sched_colors,
    select_point,
    sweep_points,
    t0_exp,
    t0_sched,
    viz_panel,
    white_zero_cmap,
    zoom_slider,
):
    # T0/T02 example maps — the exact guidance.py Inspect-states diff-mode panels: twin x^{ung|gui},
    # guided x^{gui}, and Δx^GE, via visualize_map with the zoom command (target field, m=0).
    if metrics is None or t0_exp.value is None or t0_sched.value is None:
        t0_maps_view = mo.md("_press **compute metrics** / pick an experiment_")
    else:
        _ei, _dd = t0_exp.value
        _dir, _cfg, _sv, _recs, _mask = load_rollout(EXPS[_ei])
        _sp = sweep_points(_sv, pinned_records(_recs, base_pin))
        _lb = next((l for l in sched_colors if gname(l) == t0_sched.value and label_delta.get(l) == _dd and l in _sp), None)
        if _lb is None:
            t0_maps_view = mo.md("_selected schedule not available for this experiment_")
        else:
            _sel = _sp[_lb]
            _p = np.asarray(_sv["GUIDANCE_DELTA"][_sel["GUIDANCE_DELTA"]], float)[:N]
            _n = next((nn for nn in range(N) if _p[nn] != 0.0), 0)
            _gt = channel(select_point(open_store(_dir, "gui", VAR), _sel), _cfg)
            _tt = channel(select_point(open_unguided_state(_dir, "gui_ung", VAR), _sel), _cfg)
            _toff = 273.15 if VAR in ("2m_temperature", "temperature") else 0.0   # K -> C for temperature
            _xgui = np.asarray(_gt.isel(m=0, n=_n), float) - _toff
            _xung = np.asarray(_tt.isel(m=0, n=_n).isel(t=-1), float) - _toff
            _en = f"{EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]} × {delta_labels.get(_dd, _dd)} · {t0_sched.value} · m=0"
            _zz = int(zoom_slider.value)
            _ws = region_crop(_xung, "globe", _zz); _wg = region_crop(_xgui, "globe", _zz)  # zoom-region windows
            _svmin = float(min(np.nanmin(_ws), np.nanmin(_wg))); _svmax = float(max(np.nanmax(_ws), np.nanmax(_wg)))
            if _svmin < 0.0 < _svmax:
                _scmap, _scen = white_zero_cmap, 0.0
            elif _svmin >= 0.0:
                _scmap, _scen = WARM_CMAP, None
            else:
                _scmap, _scen = cool_half_cmap, None
            _wge = region_crop(_xgui - _xung, "globe", _zz); _gm = (float(np.nanmax(np.abs(_wge))) if np.isfinite(_wge).any() else 0.0) or 1e-9
            _panels = mo.hstack([
                viz_panel(_xung, r"$x^{\mathrm{ung}\mid\mathrm{gui}}$", False, _ovmin=_svmin, _ovmax=_svmax, _ocmap=_scmap, _ocenter=_scen, _mask=_mask, _savename="T02a", _contour_on=contour_checkbox_t02.value, _contour_levels=contour_levels_slider_t02.value, _contour_color=contour_color_dropdown_t02.value, _do_save=export_button.value),
                viz_panel(_xgui, r"$x^{\mathrm{gui}}$", False, _ovmin=_svmin, _ovmax=_svmax, _ocmap=_scmap, _ocenter=_scen, _mask=_mask, _savename="T02b", _contour_on=contour_checkbox_t02.value, _contour_levels=contour_levels_slider_t02.value, _contour_color=contour_color_dropdown_t02.value, _do_save=export_button.value),
                viz_panel(_xgui - _xung, r"$\Delta x^{\mathrm{GE}}$", True, _ovmin=-_gm, _ovmax=_gm, _ocmap=white_zero_cmap, _ocenter=0.0, _mask=_mask, _savename="T02c", _contour_on=contour_checkbox_t02.value, _contour_levels=contour_levels_slider_t02.value, _contour_color=contour_color_dropdown_t02.value, _do_save=export_button.value),
            ], justify="start", align="start")
            t0_maps_view = mo.vstack([mo.md(f"_{_en}_"), _panels], align="start")
    mo.vstack([mo.hstack([t0_exp, t0_sched, zoom_slider, contour_checkbox_t02, contour_levels_slider_t02, contour_color_dropdown_t02], justify="start"), t0_maps_view], align="start")
    return


@app.cell(hide_code=True)
def md_t0_gegrid(mo):
    md_t0_gegrid = r"""
    ### T03: Δx^GE across selected variables

    The guidance effect $\Delta x^{\mathrm{GE}} = x^{\mathrm{gui}} - x^{\mathrm{ung}\mid\mathrm{gui}}$ at the
    final guided state, for the seven convergence variables (T06) (each at its surface value, else level $1000$) in the same
    $2\times4$ grid, for the selected experiment and schedule at $m{=}0$ (diverging, white $=0$; mask bbox).
    Shows how an intervention on $2\mathrm{m}$ temperature propagates across atmospheric variables and levels.
    """
    mo.md(md_t0_gegrid)
    return (md_t0_gegrid,)


@app.cell(hide_code=True)
def t0_gegrid(
    EXPS,
    INTERF_SHORT,
    INTERF_SURFACE_PAIR,
    N,
    VAR_ORDER,
    base_pin,
    contour_checkbox_t03,
    contour_color_dropdown_t03,
    contour_levels_slider_t03,
    delta_labels,
    export_button,
    gname,
    label_delta,
    load_rollout,
    metrics,
    mo,
    np,
    open_store,
    open_unguided_state,
    pinned_records,
    reliability_var_meta,
    sched_colors,
    select_point,
    sweep_points,
    t0_exp,
    t0_sched,
    viz_panel,
    zoom_slider,
):
    # T0/T03 — Δx^GE across the 7 convergence variables (final guided state minus twin) via the exact
    # guidance.py visualize_map renderer + zoom, m=0.
    if metrics is None or t0_exp.value is None or t0_sched.value is None:
        t0_gegrid_view = mo.md("_press **compute metrics** / pick an experiment_")
    else:
        _ei, _dd = t0_exp.value
        _dir, _cfg, _sv, _recs, _mask = load_rollout(EXPS[_ei])
        _sp = sweep_points(_sv, pinned_records(_recs, base_pin))
        _lb = next((l for l in sched_colors if gname(l) == t0_sched.value and label_delta.get(l) == _dd and l in _sp), None)
        if _lb is None:
            t0_gegrid_view = mo.md("_selected schedule not available for this experiment_")
        else:
            _sel = _sp[_lb]
            _p = np.asarray(_sv["GUIDANCE_DELTA"][_sel["GUIDANCE_DELTA"]], float)[:N]
            _n = next((nn for nn in range(N) if _p[nn] != 0.0), 0)
            def _resolve(_b):
                if _b in INTERF_SURFACE_PAIR:
                    return INTERF_SURFACE_PAIR[_b], "surface", 0
                if _b == "mean_sea_level_pressure":
                    return _b, "surface", 0
                return _b, "level", 1000
            _maps = []
            for _var in VAR_ORDER:
                _rv, _rp, _rl = _resolve(_var); _lev = _rl if _rp == "level" else None
                _gd = select_point(open_store(_dir, "gui", _rv), _sel)
                _ud = select_point(open_unguided_state(_dir, "gui_ung", _rv), _sel)
                _gd = _gd.sel(level=_lev) if (_lev is not None and "level" in _gd.dims) else _gd
                _ud = _ud.sel(level=_lev) if (_lev is not None and "level" in _ud.dims) else _ud
                _xg = np.asarray(_gd.isel(m=0, n=_n), float)
                _ux = _ud.isel(m=0, n=_n); _ux = np.asarray(_ux.isel(t=-1) if "t" in _ux.dims else _ux, float)
                _lbl = reliability_var_meta.get(_var, (INTERF_SHORT.get(_var, _var), False))[0]
                _maps.append(viz_panel(_xg - _ux, _lbl, True, _mask=_mask, _savename=f"T03{chr(97 + VAR_ORDER.index(_var))}", _contour_on=contour_checkbox_t03.value, _contour_levels=contour_levels_slider_t03.value, _contour_color=contour_color_dropdown_t03.value, _do_save=export_button.value))
            _rows = [mo.hstack(_maps[_i:_i + 4], justify="start", align="start") for _i in range(0, len(_maps), 4)]
            _en = f"{EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]} × {delta_labels.get(_dd, _dd)} · {t0_sched.value} · m=0 (final state)"
            t0_gegrid_view = mo.vstack([mo.md(rf"_$\Delta x^{{\mathrm{{GE}}}}$ — {_en}_")] + _rows, align="start")
    mo.vstack([mo.hstack([t0_exp, t0_sched, zoom_slider, contour_checkbox_t03, contour_levels_slider_t03, contour_color_dropdown_t03], justify="start"), t0_gegrid_view], align="start")
    return


@app.cell(hide_code=True)
def hdr_t04(mo):
    md_t04 = r"""
    ### T04: Absolute intensity

    The masked-mean of the guided state $\mathcal{M}_\pi(x^{\mathrm{gui}})$ per variable and vertical level, one
    line per top-$k$ schedule $\gamma$; dashed grey = unguided twin (ung$\mid$gui), dashed green = ground
    truth. One chart per level variable, $x$ = channels (surface→top); pooled across experiments
    (exp$\times m\times n$).
    """
    mo.md(md_t04)
    return (md_t04,)


@app.cell(hide_code=True)
def _(interf_profile_render):
    interf_abs_view = interf_profile_render(
        "abs",
        r"",
        r"Masked-mean of the **guided state** $\mathcal{M}_m(x^{\mathrm{gui}})$ per variable/level, one line "
        r"per top-$k$ schedule; dashed grey = unguided twin (ung|gui), dashed green = ground truth (gt). "
        r"One chart per level variable — $x$ = channels (surface→top). Pooled across experiments (exp×m×n).",
        r"$\mathcal{M}_m(x^{\mathrm{gui}})$",
    )
    interf_abs_view
    return


@app.cell(hide_code=True)
def hdr_t05(mo):
    md_t05 = r"""
    ### T05: Relative to ground truth

    The same masked-mean minus the ground-truth valid state at lead $n{+}1$:
    $\mathcal{M}_\pi(x^{\mathrm{gui}})-\mathcal{M}_\pi(x^{\mathrm{gt}})$ — how far each variable/level sits from
    truth (GT = zero line). Dashed grey = unguided twin (ung$\mid$gui). One chart per level variable, $x$ =
    channels (surface→top); pooled across experiments (exp$\times m\times n$).
    """
    mo.md(md_t05)
    return (md_t05,)


@app.cell(hide_code=True)
def _(interf_profile_render):
    interf_vsgt_view = interf_profile_render(
        "vsgt",
        r"",
        r"The same masked-mean, minus the ground-truth valid state at lead $n{+}1$: how far each "
        r"variable/level sits from truth. GT is the zero line (green); dashed grey = unguided twin "
        r"(ung|gui). One chart per level variable — $x$ = channels (surface→top). Pooled across "
        r"experiments (exp×m×n).",
        r"$\mathcal{M}_m(x^{\mathrm{gui}})-\mathcal{M}_m(x^{\mathrm{gt}})$",
    )
    interf_vsgt_view
    return


@app.cell(hide_code=True)
def hdr_r2(mo):
    md_t01 = r"""
    ### T06: Convergence over flow time

    The intermediate gap $\xi_{n,t}=\mathcal{M}_\pi(\hat{x}_{n,t})-y_n^\star$ along flow time $t$, converging
    to $0$ for the target field ($^\star$); the other six fields show the guidance effect
    $\mathcal{M}_\pi(\hat{x}_t)-\mathcal{M}_\pi(x^{\mathrm{ung}\mid\mathrm{gui}})$. A **7-field grid** (each
    field at its surface value, else level $1000$); solid = selected member $m$, dashed = ung$\mid$gui twin,
    band = min–max over members.
    """
    mo.md(md_t01)
    return (md_t01,)


@app.cell
def _(
    EXPS,
    M,
    VAR_ORDER,
    export_button,
    gamma_key,
    gname,
    metrics,
    mo,
    n_slider,
    np,
    plot_labels,
    plt,
    reliability_m_slider,
    reliability_shade_checkbox,
    reliability_states,
    reliability_twin,
    reliability_var_meta,
    save_chart,
    sched_colors,
):
    # ---- Reliability: guided-state convergence over flow time, 7-field grid per experiment ----
    # All 7 fields at surface (else level 1000). Target field (*) shows the gap xi_{n,t}->0; others show
    # the guidance effect M(x_t)-M(twin). Solid = selected member; dashed = ung|gui twin; band = min-max
    # over members. Legend in the upper-right slot.
    if metrics is None or reliability_states is None:
        reliability_conv_view = mo.md("_press **compute metrics**_")
    else:
        _rn = int(n_slider.value) - 1
        _m0 = int(reliability_m_slider.value)
        _shade = reliability_shade_checkbox.value
        _labels = sorted(plot_labels, key=gamma_key)
        _ncol, _nrow = 4, 2
        _slots = [(_r, _c) for _r in range(_nrow) for _c in range(_ncol) if (_r, _c) != (0, _ncol - 1)]
        _items = []
        for _ei in range(len(EXPS)):
            _items.append(mo.md(f"_{EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]}_  (n={_rn + 1}, member m={_m0})"))
            _f, _axs = plt.subplots(_nrow, _ncol, figsize=(3.1 * _ncol, 2.3 * _nrow), squeeze=False, dpi=120)
            for _vi, _var in enumerate(VAR_ORDER):
                _r, _c = _slots[_vi]; _ax = _axs[_r][_c]
                _lbl, _is_t = reliability_var_meta[_var]
                for _lb in _labels:
                    _cs = [reliability_states[(_ei, _m, _lb, _var)] for _m in range(M) if (_ei, _m, _lb, _var) in reliability_states]
                    if not _cs:
                        continue
                    _arr = np.stack(_cs); _x = np.arange(_arr.shape[1]); _col = sched_colors[_lb]
                    if _shade and _arr.shape[0] > 1:
                        _ax.fill_between(_x, np.nanmin(_arr, 0), np.nanmax(_arr, 0), color=_col, alpha=0.15, linewidth=0)
                    if (_ei, _m0, _lb, _var) in reliability_states:
                        _ax.plot(_x, reliability_states[(_ei, _m0, _lb, _var)], "-", color=_col, linewidth=1.4, label=gname(_lb))
                if (_ei, _m0, _var) in reliability_twin:
                    _tw = reliability_twin[(_ei, _m0, _var)]
                    _ax.plot(np.arange(len(_tw)), _tw, "--", color="#333333", linewidth=1.2, label=r"ung$\mid$gui")
                _ax.axhline(0.0, color="#888888", linewidth=0.8)
                _ax.set_title((_lbl[:-1] + r"^\star$") if _is_t else _lbl, fontsize=9)
                _ax.set_xlabel("$t$", fontsize=7); _ax.tick_params(labelsize=6)
            _lax = _axs[0][_ncol - 1]; _lax.axis("off")
            _h, _l = _axs[_slots[0][0]][_slots[0][1]].get_legend_handles_labels()
            if _h:
                _lax.legend(_h, _l, fontsize=7, loc="upper right", frameon=False)
            _f.tight_layout(pad=0.5)
            if export_button.value:
                save_chart(_f, f"T06{chr(97 + _ei)}")
            _items.append(mo.as_html(_f))
        reliability_conv_view = mo.vstack([
            mo.hstack([reliability_m_slider, reliability_shade_checkbox], justify="start"),
            mo.md(r"7 fields at **surface** (else **level 1000**). Target field ($^\star$): gap $\xi_{n,t}=\mathcal{M}_\pi(\hat{x}_t)-y^\star\!\to 0$; others: guidance effect $\mathcal{M}_\pi(\hat{x}_t)-\mathcal{M}_\pi(x^{\mathrm{ung}\mid\mathrm{gui}})$. Solid = selected member $m$; dashed = ung$\mid$gui twin; band = min-max over members."),
            *_items], align="start")
    reliability_conv_view
    return


@app.cell(hide_code=True)
def hdr_r3(mo):
    md_t02 = r"""
    ### T07: Convergence decomposition (landings)

    The same 7-field grid decomposed per flow step: the state $\mathcal{M}_\pi(\hat{x}_t)$ (gold) splits each
    step's move into the **flow move** ($\mathcal{M}_\pi(\hat{x}_t)\to$ ung$\mid$gui landing) and the
    **guidance move** (ung$\mid$gui landing $\to$ guided landing) — the guidance contribution injected at step
    $t$. One selected schedule $\gamma$ and member $m$; target field ($^\star$) descends to $0$.
    """
    mo.md(md_t02)
    return (md_t02,)


@app.cell(hide_code=True)
def wf_plot(
    EXPS,
    VAR_ORDER,
    export_button,
    metrics,
    mo,
    n_slider,
    np,
    plt,
    reliability_land,
    reliability_m_slider,
    reliability_states,
    reliability_var_meta,
    reliability_wf_sched,
    save_chart,
):
    # ---- Reliability 1.3: convergence decomposition (landings waterfall), 7-field grid ----
    # guidance.py "landings" grammar, thesis-ready: per flow step the state M_pi(x_t) (gold) splits into the
    # flow move (red: state_t -> ung|gui landing) and the guidance move (blue: ung|gui landing -> guided
    # landing). One selected schedule + member; target field (*) descends to 0.
    if metrics is None or reliability_states is None:
        reliability_wf_view = mo.md("_press **compute metrics**_")
    elif reliability_wf_sched.value is None:
        reliability_wf_view = mo.md("_no schedule selected_")
    else:
        _rn = int(n_slider.value) - 1
        _m0 = int(reliability_m_slider.value)
        _sched = reliability_wf_sched.value
        _ncol, _nrow = 4, 2
        _slots = [(_r, _c) for _r in range(_nrow) for _c in range(_ncol) if (_r, _c) != (0, _ncol - 1)]
        _off = 0.18
        _items = []
        for _ei in range(len(EXPS)):
            _items.append(mo.md(f"_{EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]}_  (n={_rn + 1}, member m={_m0})"))
            _f, _axs = plt.subplots(_nrow, _ncol, figsize=(3.7 * _ncol, 2.5 * _nrow), squeeze=False, dpi=120)
            for _vi, _var in enumerate(VAR_ORDER):
                _r, _c = _slots[_vi]; _ax = _axs[_r][_c]
                _lbl, _is_t = reliability_var_meta[_var]
                _lab = (lambda _s: _s) if _vi == 0 else (lambda _s: "_nolegend_")
                _k = (_ei, _m0, _sched, _var)
                if _k in reliability_states and _k in reliability_land and len(reliability_land[_k]):
                    _st = reliability_states[_k]; _ld = reliability_land[_k]
                    _T = len(_ld); _xt = np.arange(len(_st)).astype(float)
                    _fm = _ld - _st[:_T]; _gm = _st[1:] - _ld
                    _ax.bar(_xt[1:] - _off, _fm, bottom=_st[:_T], width=0.32, color="#C0392B", alpha=0.5, linewidth=0, zorder=3, label=_lab("flow move"))
                    _ax.bar(_xt[1:] + _off, _gm, bottom=_ld, width=0.32, color="#2E86C1", alpha=0.5, linewidth=0, zorder=3, label=_lab("guidance move"))
                    for _ti in range(_T):
                        _ax.plot([_xt[_ti], _xt[_ti + 1]], [_st[_ti], _ld[_ti]], "--", lw=0.7, color="#800080", alpha=0.45, zorder=4)
                    _ax.plot(_xt, _st, "-", color="#B7950B", alpha=0.6, linewidth=1.0, zorder=5)
                    _ax.plot(_xt, _st, "o", color="#B7950B", mec="white", mew=0.5, ms=3.5, ls="none", zorder=7, label=_lab("state"))
                    _ax.plot(_xt[1:], _ld, "s", color="#800080", mec="white", mew=0.5, ms=3.0, ls="none", zorder=6, label=_lab(r"ung$\mid$gui landing"))
                if _is_t:
                    _ax.axhline(0.0, color="#888888", linewidth=0.8, zorder=1)
                _ax.set_title((_lbl[:-1] + r"^\star$") if _is_t else _lbl, fontsize=9)
                _ax.set_xlabel("$t$", fontsize=7); _ax.tick_params(labelsize=6); _ax.margins(x=0.02)
            _lax = _axs[0][_ncol - 1]; _lax.axis("off")
            _h, _l = _axs[_slots[0][0]][_slots[0][1]].get_legend_handles_labels()
            if _h:
                _lax.legend(_h, _l, fontsize=8, loc="upper right", frameon=False)
            _f.tight_layout(pad=0.5)
            if export_button.value:
                save_chart(_f, f"T07{chr(97 + _ei)}")
            _items.append(mo.as_html(_f))
        reliability_wf_view = mo.vstack([
            mo.hstack([reliability_wf_sched, reliability_m_slider], justify="start"),
            mo.md(r"Per flow step the state $\mathcal{M}_\pi(\hat{x}_t)$ (gold) splits its move into the **flow move** (red: state$_t\to$ ung$\mid$gui landing) and the **guidance move** (blue: ung$\mid$gui landing $\to$ guided landing) — the guidance contribution injected at that step. Target field ($^\star$) descends to $0$."),
            *_items], align="start")
    reliability_wf_view
    return


@app.cell(hide_code=True)
def hdr_t1(mo):
    md_t1 = r"""
    ## T1: Temporal localization of guidance

    *Where does the guidance effect land — across variables, in space, and across the ensemble?* The guidance
    effect is the same-seed twin difference
    $\Delta x^{\mathrm{GE}}=x_n^{\mathrm{gui}}-x_n^{\mathrm{ung}\mid\mathrm{gui}}$ (guided minus
    unguided-under-guided-context). Three views: its **multivariate propagation** across variables/levels
    (T1.1), the **spatial deviation** of its footprint from the target mask (T1.2), and its **ensemble
    diversity** across members (T1.3).
    """
    mo.md(md_t1)
    return (md_t1,)


@app.cell(hide_code=True)
def hdr_t11(mo):
    md_t11 = r"""
    ### T1.1: Multivariate propagation

    *How does the guidance effect propagate across atmospheric variables and vertical levels, and which
    flow-time profile $\gamma$ drives the strongest response?* We first look at the effect directly (a visual
    example), then define the propagation score and its per-variable profiles, then rank the profiles with the
    score tables.
    """
    mo.md(md_t11)
    return (md_t11,)


@app.cell(hide_code=True)
def hdr_r4(mo):
    md_t03 = r"""
    ### Visual example

    A raw, **un-normalized** look at the object T1.1 scores below: the signed guidance effect
    $\Delta x^{\mathrm{GE}}_t=x_t^{\mathrm{gui}}-x_t^{\mathrm{ung}\mid\mathrm{gui}}$ (with
    $x_t=\hat{x}^{\mathrm{gui\_det}}+\sigma_r z_t$) as a spatial map over the mask bbox at each flow step $t$,
    per selected variable (one filmstrip each) and mode (rows), single experiment. Back-shifted by one flow
    step so spike@$t{=}k$ shows its effect from column $t{=}k$.
    """
    mo.md(md_t03)
    return (md_t03,)


@app.cell(hide_code=True)
def effect_plot(
    BBOX,
    EXPS,
    INTERF_SHORT,
    INTERF_SURFACE_PAIR,
    VAR_ORDER,
    base_pin,
    export_button,
    gamma_key,
    gname,
    load_rollout,
    metrics,
    mo,
    n_slider,
    np,
    open_store,
    open_unguided_state,
    pinned_records,
    plot_labels,
    plt,
    reliability_effect_modes,
    reliability_effect_vars,
    reliability_m_slider,
    reliability_var_meta,
    residual_scaler,
    save_chart,
    select_point,
    sweep_points,
):
    # ---- Reliability 1.4: guidance effect over flow time (maps), per selected variable + mode ----
    # Signed effect x_t^gui - x_t^{ung|gui} (x_t = gui_det + sigma_r z_t) as a map per flow step t
    # (mask bbox, diverging). Single experiment; rows = selected modes; one filmstrip per selected
    # variable. Back-shifted by one flow step so spike@t=k shows its effect from column t=k. No colorbar/title.
    if metrics is None:
        reliability_effect_view = mo.md("_press **compute metrics**_")
    elif not reliability_effect_modes.value or not reliability_effect_vars.value:
        reliability_effect_view = mo.md("_select at least one mode and one variable_")
    else:
        _rn = int(n_slider.value) - 1
        _m0 = int(reliability_m_slider.value)
        _modes = [_lb for _lb in sorted(plot_labels, key=gamma_key) if _lb in reliability_effect_modes.value]
        _vars = [_v for _v in VAR_ORDER if _v in reliability_effect_vars.value]
        _STEPS = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 24]
        _ei = 0
        _dir, _cfg, _sv, _recs, _rmask = load_rollout(EXPS[_ei])
        _sp = sweep_points(_sv, pinned_records(_recs, base_pin))

        def _resolve(_b):
            if _b in INTERF_SURFACE_PAIR:
                return INTERF_SURFACE_PAIR[_b], "surface", 0
            if _b == "mean_sea_level_pressure":
                return _b, "surface", 0
            return _b, "level", 1000

        _items = [mo.md(f"_{EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]}_  (n={_rn + 1}, member m={_m0})")]
        for _vb in _vars:
            _rv, _rp, _rl = _resolve(_vb); _lev = _rl if _rp == "level" else None
            _c = residual_scaler(_rp, _rv, _rl)
            _effs = {}
            for _lb in _modes:
                if _lb not in _sp:
                    continue
                _sel = _sp[_lb]
                _dd = select_point(open_store(_dir, "gui_det", _rv), _sel)
                _dd = _dd.sel(level=_lev) if (_lev is not None and "level" in _dd.dims) else _dd
                _det = np.asarray(_dd.isel(m=_m0, n=_rn), float)
                _rr = select_point(open_store(_dir, "gui_res", _rv), _sel)
                _rr = _rr.sel(level=_lev) if (_lev is not None and "level" in _rr.dims) else _rr
                _res = np.asarray(_rr.isel(m=_m0, n=_rn), float)
                _tt = select_point(open_unguided_state(_dir, "gui_ung", _rv), _sel)
                _tt = _tt.sel(level=_lev) if (_lev is not None and "level" in _tt.dims) else _tt
                _tw = np.asarray(_tt.isel(m=_m0, n=_rn), float)
                _effs[_lb] = ((_det[None] + _c * _res) - _tw)[:, BBOX[0], BBOX[1]]
            if not _effs:
                continue
            _T = next(iter(_effs.values())).shape[0]
            _vmax = max((float(np.nanmax(np.abs(_e))) for _e in _effs.values()), default=1.0) or 1.0
            _nrow, _ncol = len(_modes), len(_STEPS)
            _lbl = reliability_var_meta.get(_vb, (INTERF_SHORT.get(_vb, _vb), False))[0]
            _items.append(mo.md(f"**{_lbl}**"))
            _fig, _axs = plt.subplots(_nrow, _ncol, figsize=(_ncol * 0.85, _nrow * 0.85 * 0.55 + 0.45), squeeze=False, dpi=130)
            for _ri, _lb in enumerate(_modes):
                _e = _effs.get(_lb)
                for _ci, _k in enumerate(_STEPS):
                    _ax = _axs[_ri][_ci]
                    if _e is not None:
                        _idx = min(_k + 1, _T - 1)
                        _ax.imshow(_e[_idx], cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
                    _ax.set_xticks([]); _ax.set_yticks([])
                    if _ri == 0:
                        _ax.set_title(rf"$t={_k}$", fontsize=7)
                    if _ci == 0:
                        _ax.set_ylabel(gname(_lb), fontsize=10, rotation=0, ha="right", va="center", labelpad=20)
            _fig.subplots_adjust(left=0.08, right=0.995, top=0.85, bottom=0.03, wspace=0.06, hspace=0.12)
            if export_button.value:
                save_chart(_fig, f"T1.1{chr(97 + _vars.index(_vb))}")
            _items.append(mo.as_html(_fig)); plt.close(_fig)
        reliability_effect_view = mo.vstack([
            mo.hstack([reliability_effect_vars, reliability_effect_modes, reliability_m_slider], justify="start"),
            mo.md(r"Signed guidance effect $x_t^{\mathrm{gui}}-x_t^{\mathrm{ung}\mid\mathrm{gui}}$ at each flow step $t$ (mask bbox; diverging, white $=0$), rows = selected modes, one filmstrip per selected variable (single experiment). Back-shifted by one so spike@$t{=}k$ shows its effect from column $t=k$."),
            *_items], align="start")
    reliability_effect_view
    return


@app.cell(hide_code=True)
def md_t11_def(mo):
    md_t11_def = r"""
    ### Definition and per-variable profiles

    **Object.** The guidance effect is the same-seed twin difference
    $\Delta x^{\mathrm{GE}} = x^{\mathrm{gui}} - x^{\mathrm{ung}\mid\mathrm{gui}}$. For each experiment
    $e=(\text{startdate},\rho)$ ($E=4\times2=8$), member $m$ ($M=5$), profile $\gamma$, variable $v$ and level
    $\ell$ we reduce it to the **masked-mean absolute effect**
    $$g_{e,m,\gamma,v\ell} \;=\; \mathcal{M}_\pi\big(|\Delta x^{\mathrm{GE}}_{v\ell}|\big) \;=\; \frac{\sum_{ij}\pi_{ij}\,|\Delta x^{\mathrm{GE}}_{v\ell ij}|}{\sum_{ij}\pi_{ij}},$$
    the area-weighted mean of the pointwise magnitude over the target mask $\pi$ — the visual example above,
    reduced to one scalar per level.

    **Eval region.** A selector switches the spatial domain: **mask** = the masked mean $\mathcal{M}_\pi(|\Delta x^{\mathrm{GE}}|)$; **!mask** = the summed magnitude outside the mask $\sum_{\lnot\mathrm{mask}}|\Delta x^{\mathrm{GE}}|$; **full** = over the whole grid $\sum_{\mathrm{grid}}|\Delta x^{\mathrm{GE}}|$. The per-variable scores $N$/$R$ are scale-invariant, so they rank $\gamma$ within whichever region is selected.

    **Chart** (one per experiment, selectable): per variable, $x=$ channels (surface→top), one line per
    $\gamma$ of $g_{v\ell}$; the solid line is member $m{=}0$ and the band spans $[\min_m,\max_m]$ over the $M$
    members.
    """
    mo.md(md_t11_def)
    return (md_t11_def,)


@app.cell
def _(M, interf_data, mo, t11_exp, t11_region, t1x_chart):
    # T1.1 — per-experiment guidance-effect profile chart for the selected experiment + eval region.
    if interf_data is None or "chan_em" not in interf_data:
        t11_chart_view = mo.md("_press **compute field profiles** above_")
    elif t11_exp.value is None:
        t11_chart_view = mo.md("_no experiment selected_")
    else:
        _rk = t11_region.value
        _ryl = {"avgabs": r"$\mathcal{M}_\pi(|\Delta x^{\mathrm{GE}}|)$ (mask masked-mean)",
                "nabs": r"$\sum_{\lnot\mathrm{mask}}|\Delta x^{\mathrm{GE}}|$ (outside the mask)",
                "absum": r"$\sum_{\mathrm{grid}}|\Delta x^{\mathrm{GE}}|$ (whole grid)"}.get(_rk, "")
        t11_chart_view = t1x_chart(_rk, t11_exp.value[0], t11_exp.value[1],
            rf"{_ryl} per channel (surface→top); one line per $\gamma$, solid $=m{{=}}0$, band $=[\min_m,\max_m]$ over $M={M}$.", _savename="T1.1")
    mo.vstack([mo.hstack([t11_exp, t11_region], justify="start"), t11_chart_view], align="start")
    return


@app.cell(hide_code=True)
def md_t11_scores(mo):
    md_t11_scores = r"""
    ### Scores

    **Per-variable value:** sum over the variable's levels, $A_{e,m,\gamma,v} = \sum_{\ell} g_{e,m,\gamma,v\ell}$.

    **Per-variable score** (normalized **per member**, so each variable's best $\gamma$ is $1$ within that
    member): $\;N_{e,m,\gamma,v} = A_{e,m,\gamma,v} / \max_{\gamma'} A_{e,m,\gamma',v}.$

    **Single score** (aggregate the variable scores, then renormalize per member):
    $\;R_{e,m,\gamma} = \big(\sum_v N_{e,m,\gamma,v}\big) / \max_{\gamma'} \sum_v N_{e,m,\gamma',v}.$

    **Per-experiment table:** rows $=$ the seven variables plus a final **single-score** row; columns
    $=\gamma$; each cell is $\mathrm{mean}_m \pm \mathrm{std}_m$ over the $M$ members of $N_v$ (variable rows)
    or $R$ (single-score row).

    **Summary — two paired views.** For each intensity $\rho$ (and once pooled across both $\rho$) we show
    $V_1$ beside $V_2$:
    - **$V_1$ — single score, per experiment:** rows $=$ startdates, columns $=\gamma$; cell $=$ mean $\pm$ std
      of $R$ over the members $M$ (per $\rho$), or over $\rho\times M$ (across $\rho$).
    - **$V_2$ — per variable:** rows $=$ variables, columns $=\gamma$; cell $=$ mean $\pm$ std of $N_v$ over
      startdates $\times M$ (per $\rho$), or over startdates $\times\rho\times M=40$ (across $\rho$).

    All spreads are population std; **bold** marks the max $\gamma$ per row.
    """
    mo.md(md_t11_scores)
    return (md_t11_scores,)


@app.cell(hide_code=True)
def t11_tables(
    EXPS,
    M,
    delta_labels,
    interf_data,
    interf_score_tables,
    mo,
    t11_exp,
    t11_region,
    t1x_pertable,
):
    if interf_data is None or "val_em" not in interf_data:
        t11_tables_view = mo.md("_press **compute field profiles** above_")
    elif t11_exp.value is None:
        t11_tables_view = mo.md("_no experiment selected_")
    else:
        _ei, _dd = t11_exp.value; _rk = t11_region.value
        _rn = {"avgabs": "mask", "nabs": "!mask", "absum": "full"}.get(_rk, _rk)
        _en = f"{EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]} × {delta_labels.get(_dd, _dd)}"
        _tbl = t1x_pertable(_rk, _ei, _dd, rf"**{_en}** · region **{_rn}** — per-variable score $N_v=A_v/\max_\gamma A_v$ (rows) × schedule $\gamma$; **single score** $R=\sum_v N_v/\max_\gamma\sum_v N_v$; mean ± std over the $M={M}$ members; **bold** = max $\gamma$ per row.")
        t11_tables_view = mo.vstack([_tbl, interf_score_tables], align="start")
    t11_tables_view
    return


@app.cell(hide_code=True)
def _(mo):
    md_t12 = r"""
    ### T1.2: Spatial-deviation

    *How far does the guidance footprint's shape depart from the prescribed target mask, across variables and
    levels?* Same eval procedure as T1.1 — only the per-(variable, level) object changes.
    """
    mo.md(md_t12)
    return (md_t12,)


@app.cell(hide_code=True)
def md_t12_visual(mo):
    md_t12_visual = r"""
    ### Visual example

    The object T1.2 scores, shown directly: the spatially-normalized absolute guidance effect
    $\widetilde{G}_{v\ell} = |\Delta x^{\mathrm{GE}}_{v\ell}| / \sum_{ij}|\Delta x^{\mathrm{GE}}_{v\ell}|$ as a
    spatial footprint per schedule $\gamma$ (columns), one row per experiment; the first column is the
    normalized target mask $\widetilde{\pi}$. Use the field / region / member controls to browse variables and
    levels.
    """
    mo.md(md_t12_visual)
    return (md_t12_visual,)


@app.cell
def _(M, N, mo):
    realism_cmap_dropdown = mo.ui.dropdown(
        ["viridis", "magma", "inferno", "Reds", "hot", "warm (RdBu_r)"],
        value="inferno", label="realism cmap: ")
    realism_norm_dropdown = mo.ui.dropdown(
        ["common max", "experiment max"], value="experiment max", label="norm: ")
    realism_m_slider = mo.ui.slider(0, max(M - 1, 1), value=0, step=1, label="m: ", show_value=True)
    realism_n_slider = mo.ui.slider(1, max(N, 2), value=1, step=1, label="n: ", show_value=True)
    # field selector for the T4/T5 images (same combo as experiment_builder): pick which field's
    # footprint / spread is shown; level 0 = surface tick.
    field_var_dropdown = mo.ui.dropdown(
        ["geopotential", "u_component_of_wind", "v_component_of_wind", "temperature",
         "specific_humidity", "vertical_velocity", "mean_sea_level_pressure"],
        value="temperature", label="field: ")
    field_level_slider = mo.ui.slider(steps=[0, 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
                                      value=0, label="level: ", show_value=True)
    t12_region = mo.ui.dropdown(["mask", "globe", "!mask"], value="mask", label="T1.2 visual region: ")
    t12_zoom = mo.ui.slider(1, 8, value=1, step=1, label="T1.2 zoom: ", show_value=True)
    mo.vstack([
        mo.hstack([realism_m_slider, realism_n_slider, realism_cmap_dropdown, realism_norm_dropdown], justify="start"),
        mo.hstack([field_var_dropdown, field_level_slider, t12_region, t12_zoom], justify="start"),
    ], align="start")
    return (
        field_level_slider,
        field_var_dropdown,
        realism_cmap_dropdown,
        realism_m_slider,
        realism_n_slider,
        realism_norm_dropdown,
        t12_region,
        t12_zoom,
    )


@app.cell
def _(
    EXPS,
    WARM_CMAP,
    export_button,
    field_grid,
    flevel,
    fvar,
    gamma_key,
    gname,
    metrics,
    mo,
    np,
    plot_labels,
    plt,
    realism_cmap_dropdown,
    realism_m_slider,
    realism_n_slider,
    realism_norm_dropdown,
    region_crop,
    region_maskref,
    save_chart,
    t12_region,
    t12_zoom,
):
    # T1.2 visual example — normalized guidance footprint maps for the selected field, over the T1.2 eval
    # region (default globe: whole-grid footprint, matching the whole-grid D score). G~ = |Δ| / Σ|Δ|.
    if metrics is None or field_grid is None:
        t4_view = mo.md("_press **compute metrics**_")
    else:
        _m = int(realism_m_slider.value); _n = int(realism_n_slider.value) - 1
        _reg = t12_region.value
        _cmap = WARM_CMAP if realism_cmap_dropdown.value == "warm (RdBu_r)" else realism_cmap_dropdown.value
        _common = realism_norm_dropdown.value == "common max"
        _cols = ["mask"] + sorted(plot_labels, key=gamma_key)
        _mk = region_maskref(_reg, int(t12_zoom.value)); _mks = np.nansum(_mk); _mkp = _mk/_mks if _mks > 0 else _mk
        _grids = []; _exmax = []
        for _ei in range(len(EXPS)):
            _cell = field_grid.get((_ei, _m, _n), {})
            _r0 = []
            for _col in _cols:
                if _col == "mask":
                    _r0.append(_mkp)
                elif _col in _cell:
                    _gf = region_crop(_cell[_col]["gfield"], _reg, int(t12_zoom.value)); _gfs = np.nansum(_gf); _r0.append(_gf/_gfs if _gfs > 0 else _gf)
                else:
                    _r0.append(None)
            _grids.append(_r0)
            _exmax.append(max((float(np.nanmax(_p)) for _p in _r0 if _p is not None), default=1.0))
        _gmax = max(_exmax) if _exmax else 1.0
        _H, _W = _mkp.shape; _ncol = len(_cols); _nrow = len(_grids)
        _left, _right, _top, _bot = 0.14, 0.995, 0.90, 0.03
        _fw = 1.3 * _ncol + 1.9
        _cellw = (_right - _left) * _fw / _ncol
        _fh = _nrow * (_cellw * (_H / _W)) / (_top - _bot)
        _f, _axs = plt.subplots(_nrow, _ncol, figsize=(_fw, _fh), squeeze=False, dpi=120)
        for _ei, _r0 in enumerate(_grids):
            _vmax = _gmax if _common else _exmax[_ei]
            for _ci in range(_ncol):
                _ax = _axs[_ei][_ci]
                if _r0[_ci] is not None:
                    _ax.imshow(_r0[_ci], cmap=_cmap, vmin=0.0, vmax=_vmax)
                _ax.set_xticks([]); _ax.set_yticks([])
                if _ei == 0:
                    _ax.set_title("mask" if _cols[_ci] == "mask" else gname(_cols[_ci]), fontsize=8)
                if _ci == 0:
                    _ax.set_ylabel(EXPS[_ei].split(chr(47))[-1].split(chr(95))[0], fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
        _f.subplots_adjust(left=_left, right=_right, top=_top, bottom=_bot, wspace=0.06, hspace=0.08)
        if export_button.value:
            save_chart(_f, "T1.2a")
        t4_view = mo.vstack([
            mo.md(rf"_Selected field **{fvar}**@{flevel}, T1.2 eval region **{_reg}**, zoom **{int(t12_zoom.value)}×**, member $m={_m}$, step $n={_n + 1}$; colour scale **{realism_norm_dropdown.value}**. Normalized footprint $\widetilde{{G}}=|\Delta x^{{\mathrm{{GE}}}}|/\sum|\Delta x^{{\mathrm{{GE}}}}|$ per schedule $\gamma$ (columns), one row per experiment; first column: the normalized mask $\widetilde{{\pi}}$._"),
            mo.as_html(_f)], align="start")
    t4_view
    return


@app.cell(hide_code=True)
def md_t12_def(mo):
    md_t12_def = r"""
    ### Definition and per-variable profiles

    **Object.** For each experiment $e=(\text{startdate},\rho)$, member $m$, profile $\gamma$, variable $v$ and
    level $\ell$ we compare the normalized footprint to the normalized mask **over the mask region** ($\pi>0$)
    and take the **total-variation distance** (half the $L_1$ distance between the two mask-normalized distributions, so $D\in[0,1]$):
    $$D_{e,m,\gamma,v\ell} \;=\; \tfrac{1}{2}\sum_{ij\in\pi} \big| \widetilde{G}_{e,m,\gamma,v\ell,ij} - \widetilde{\pi}_{ij} \big|,\qquad
    \widetilde{G} = \frac{|\Delta x^{\mathrm{GE}}_{v\ell}|}{\sum_{ij\in\pi}|\Delta x^{\mathrm{GE}}_{v\ell,ij}|},\quad
    \widetilde{\pi} = \frac{\pi}{\sum_{ij}\pi_{ij}},$$
    with both the sum and the footprint normalization restricted to the mask. $D=0$ means the response is shaped
    exactly like the mask inside the target ($D=1$ = disjoint from it); larger $D$ = it concentrates differently
    from the mask within the region.

    **Chart** (one per experiment, selectable): per variable, $x=$ channels (surface→top), one line per
    $\gamma$ of $D_{v\ell}$; the solid line is member $m{=}0$ and the band spans $[\min_m,\max_m]$ over the $M$
    members.
    """
    mo.md(md_t12_def)
    return (md_t12_def,)


@app.cell(hide_code=True)
def t12_chart(M, interf_data, mo, t12_exp, t1x_chart):
    # T1.2 — per-experiment spatial-deviation profile chart (object D) for the selected experiment.
    if interf_data is None or "chan_em" not in interf_data or "tvd" not in interf_data["chan_em"]:
        t12_chart_view = mo.md("_press **compute field profiles** above_")
    elif t12_exp.value is None:
        t12_chart_view = mo.md("_no experiment selected_")
    else:
        t12_chart_view = t1x_chart("tvd", t12_exp.value[0], t12_exp.value[1],
            rf"$D_{{v\ell}}=\sum_{{ij}}|\widetilde{{G}}_{{v\ell}}-\widetilde{{\pi}}|$ per channel (surface→top); one line per $\gamma$, solid $=m{{=}}0$, band $=[\min_m,\max_m]$ over the $M={M}$ members.", _savename="T1.2")
    mo.vstack([t12_exp, t12_chart_view], align="start")
    return


@app.cell(hide_code=True)
def md_t12_scores(mo):
    md_t12_scores = r"""
    ### Scores

    Identical construction to T1.1, with $D$ in place of $g$: per-variable value
    $A_{e,m,\gamma,v} = \sum_{\ell} D_{e,m,\gamma,v\ell}$; per-variable score
    $N_{e,m,\gamma,v} = A_{e,m,\gamma,v} / \max_{\gamma'} A_{e,m,\gamma',v}$ (per member); single score
    $R_{e,m,\gamma} = \big(\sum_v N_{e,m,\gamma,v}\big) / \max_{\gamma'} \sum_v N_{e,m,\gamma',v}$. The
    per-experiment table (rows = variables + single-score row) and the two paired summary views ($V_1$
    startdates × $\gamma$; $V_2$ variables × $\gamma$; per $\rho$ and across $\rho$) follow the same pooling as
    T1.1. All spreads are population std; **bold** marks the max $\gamma$ per row.
    """
    mo.md(md_t12_scores)
    return (md_t12_scores,)


@app.cell(hide_code=True)
def t12_tables(
    EXPS,
    M,
    delta_labels,
    interf_data,
    interf_score_tables_t12,
    mo,
    t12_exp,
    t1x_pertable,
):
    if interf_data is None or "val_em" not in interf_data or "tvd" not in interf_data["val_em"]:
        t12_tables_view = mo.md("_press **compute field profiles** above_")
    elif t12_exp.value is None:
        t12_tables_view = mo.md("_no experiment selected_")
    else:
        _ei, _dd = t12_exp.value
        _en = f"{EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]} × {delta_labels.get(_dd, _dd)}"
        _tbl = t1x_pertable("tvd", _ei, _dd, rf"**{_en}** — per-variable score $N_v=A_v/\max_\gamma A_v$ (rows, $A_v=\sum_\ell D_{{v\ell}}$) × schedule $\gamma$; **single score** $R=\sum_v N_v/\max_\gamma\sum_v N_v$; mean ± std over the $M={M}$ members; **bold** = max $\gamma$ per row.")
        t12_tables_view = mo.vstack([_tbl, interf_score_tables_t12], align="start")
    t12_tables_view
    return


@app.cell(hide_code=True)
def md_t12_whymask(mo):
    md_t12_whymask = r"""
    ### Why the mask region

    $D$ is measured over the **mask region** (the target support $\pi>0$), not the whole grid. Both the
    footprint $\widetilde{G}$ and the mask $\widetilde{\pi}$ are normalized to sum to one over the mask, so $D$
    is a proper total-variation distance between two distributions *inside the target* — it asks whether the
    guidance concentrates where the mask prescribes, a controllability check on the footprint shape within the
    region we actually steer. Where the effect leaks **outside** the mask is a separate question, read off the
    outside-mask mass in T1.1.
    """
    mo.md(md_t12_whymask)
    return (md_t12_whymask,)


@app.cell(hide_code=True)
def _(mo):
    md_t13 = r"""
    ### T1.3: Ensemble diversity

    *How much does the guidance effect still depend on the stochastic realization, across variables and levels?*
    Same eval procedure as T1.1/T1.2 — the object is the ensemble std of the (signed) guidance effect, so the
    member axis is consumed and the tables/plots carry no per-member spread.
    """
    mo.md(md_t13)
    return (md_t13,)


@app.cell(hide_code=True)
def md_t13_visual(mo):
    md_t13_visual = r"""
    ### Visual example

    The **spatial footprint** of the object T1.3 scores: the per-pixel ensemble standard deviation of the
    **signed** guidance effect, sum-normalized to a unit footprint $\widetilde{S}_{v\ell} =
    \operatorname{std}_m(\Delta x^{\mathrm{GE}}_{v\ell}) / \sum_{ij}\operatorname{std}_m(\Delta x^{\mathrm{GE}}_{v\ell})$,
    as a spatial map per schedule $\gamma$ (columns), one row per experiment; the first column is the normalized
    target mask $\widetilde{\pi}$. This shows **where** the member-to-member response disagrees, not its
    magnitude — the magnitude is what the score $s=\mathcal{M}_\pi(\operatorname{std}_m(\Delta x^{\mathrm{GE}}))$
    below reports. Use the field / region controls to browse variables and levels.
    """
    mo.md(md_t13_visual)
    return (md_t13_visual,)


@app.cell(hide_code=True)
def _(
    field_level_slider,
    field_var_dropdown,
    mo,
    realism_cmap_dropdown,
    realism_n_slider,
    realism_norm_dropdown,
):
    # T1.3 controls: field selector + T1.3 eval region (default mask) + view opts
    t13_region = mo.ui.dropdown(["mask", "globe", "!mask"], value="mask", label="T1.3 eval region: ")
    mo.hstack([field_var_dropdown, field_level_slider, t13_region,
               realism_n_slider, realism_cmap_dropdown, realism_norm_dropdown], justify="start")
    return (t13_region,)


@app.cell(hide_code=True)
def _(
    EXPS,
    M,
    WARM_CMAP,
    export_button,
    field_grid,
    flevel,
    fvar,
    gamma_key,
    gname,
    metrics,
    mo,
    np,
    plot_labels,
    plt,
    realism_cmap_dropdown,
    realism_n_slider,
    realism_norm_dropdown,
    region_crop,
    region_maskref,
    save_chart,
    t13_region,
):
    # T1.3 visual example — per-pixel ensemble std of the SIGNED guidance effect std_m(x_gui - x_ung) for the
    # selected field, over the T1.3 eval region (default mask, matching the M_pi mask-domain score).
    if metrics is None or field_grid is None:
        t5_spread_view = mo.md("_press **compute metrics**_")
    else:
        _n = int(realism_n_slider.value) - 1
        _reg = t13_region.value
        _cmap = WARM_CMAP if realism_cmap_dropdown.value == "warm (RdBu_r)" else realism_cmap_dropdown.value
        _common = realism_norm_dropdown.value == "common max"
        _cols = ["mask"] + sorted(plot_labels, key=gamma_key)
        _mk = region_maskref(_reg); _mks = np.nansum(_mk); _mkp = _mk/_mks if _mks > 0 else _mk
        def _std_over_m(_ei, _lb):
            _stack = [np.asarray(field_grid[(_ei, _mm, _n)][_lb]["gfield_signed"], float)
                      for _mm in range(M) if (_ei, _mm, _n) in field_grid and _lb in field_grid[(_ei, _mm, _n)]]
            if len(_stack) < 2:
                return None
            return region_crop(np.std(np.stack(_stack, 0), axis=0), _reg)
        _grids = []; _exmax = []
        for _ei in range(len(EXPS)):
            _r0 = []
            for _col in _cols:
                if _col == "mask":
                    _r0.append(_mkp)
                else:
                    _sf = _std_over_m(_ei, _col)
                    _r0.append(_sf/np.nansum(_sf) if (_sf is not None and np.nansum(_sf) > 0) else _sf)
            _grids.append(_r0)
            _exmax.append(max((float(np.nanmax(_p)) for _p in _r0 if _p is not None), default=1.0))
        _gmax = max(_exmax) if _exmax else 1.0
        _H, _W = _mkp.shape; _ncol = len(_cols); _nrow = len(_grids)
        _left, _right, _top, _bot = 0.14, 0.995, 0.90, 0.03
        _fw = 1.3 * _ncol + 1.9
        _cellw = (_right - _left) * _fw / _ncol
        _fh = _nrow * (_cellw * (_H / _W)) / (_top - _bot)
        _f, _axs = plt.subplots(_nrow, _ncol, figsize=(_fw, _fh), squeeze=False, dpi=120)
        for _ei, _r0 in enumerate(_grids):
            _vmax = _gmax if _common else _exmax[_ei]
            for _ci in range(_ncol):
                _ax = _axs[_ei][_ci]
                if _r0[_ci] is not None:
                    _ax.imshow(_r0[_ci], cmap=_cmap, vmin=0.0, vmax=_vmax)
                _ax.set_xticks([]); _ax.set_yticks([])
                if _ei == 0:
                    _ax.set_title("mask" if _cols[_ci] == "mask" else gname(_cols[_ci]), fontsize=8)
                if _ci == 0:
                    _ax.set_ylabel(EXPS[_ei].split(chr(47))[-1].split(chr(95))[0], fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
        _f.subplots_adjust(left=_left, right=_right, top=_top, bottom=_bot, wspace=0.06, hspace=0.08)
        if export_button.value:
            save_chart(_f, "T1.3a")
        t5_spread_view = mo.vstack([
            mo.md(rf"_Selected field **{fvar}**@{flevel}, T1.3 eval region **{_reg}**, step $n={_n + 1}$; colour scale **{realism_norm_dropdown.value}**. Per-pixel std across members $m$ of the **signed** guidance effect $\operatorname{{std}}_m(\Delta x^{{\mathrm{{GE}}}})$ per schedule $\gamma$ (columns), one row per experiment; first column: the normalized mask $\widetilde{{\pi}}$; each panel sum-normalized to a unit footprint (shape, not magnitude)._"),
            mo.as_html(_f)])
    t5_spread_view
    return


@app.cell(hide_code=True)
def md_t13_def(mo):
    md_t13_def = r"""
    ### Definition and per-variable profiles

    **Object.** For each experiment $e=(\text{startdate},\rho)$, profile $\gamma$, variable $v$ and level $\ell$,
    take the **per-pixel ensemble std of the signed guidance effect** and reduce it by the area-weighted masked
    mean over the target region (mask domain, as motivated for the diversity criterion):
    $$s_{e,\gamma,v\ell} \;=\; \mathcal{M}_\pi\!\big(\operatorname{std}_m(\Delta x^{\mathrm{GE}}_{v\ell})\big),\qquad
    \Delta x^{\mathrm{GE}} = x^{\mathrm{gui}} - x^{\mathrm{ung}\mid\mathrm{gui}}.$$
    The std over the $M$ members **consumes the member axis**, so there is one value per $(e,\gamma,v,\ell)$ — no
    per-member spread. Higher $s$ = the guidance effect varies more across the stochastic ensemble.

    **Chart** (one per experiment, selectable): per variable, $x=$ channels (surface→top), one line per $\gamma$
    of $s_{v\ell}$ — a single line each, no min–max band (the std already consumed the members).
    """
    mo.md(md_t13_def)
    return (md_t13_def,)


@app.cell(hide_code=True)
def t13_chart_cell(mo, t13_chan, t13_chart, t13_exp):
    # T1.3 — per-experiment ensemble-diversity profile chart for the selected experiment.
    if t13_chan is None:
        t13_chart_view = mo.md("_press **compute field profiles** above_")
    elif t13_exp.value is None:
        t13_chart_view = mo.md("_no experiment selected_")
    else:
        t13_chart_view = t13_chart(t13_exp.value[0], t13_exp.value[1],
            rf"$s_{{v\ell}}=\mathcal{{M}}_\pi(\operatorname{{std}}_m(\Delta x^{{\mathrm{{GE}}}}))$ per channel (surface→top); one line per $\gamma$ (no member band — the std consumes $M$).", _savename="T1.3")
    mo.vstack([t13_exp, t13_chart_view], align="start")
    return


@app.cell(hide_code=True)
def md_t13_scores(mo):
    md_t13_scores = r"""
    ### Scores

    Same construction as T1.1/T1.2, with $s$ as the object and **no per-member spread**: per-variable value
    $A_{e,\gamma,v} = \sum_\ell s_{e,\gamma,v\ell}$; per-variable score
    $N_{e,\gamma,v} = A_{e,\gamma,v} / \max_{\gamma'} A_{e,\gamma',v}$; single score
    $R_{e,\gamma} = \big(\sum_v N_{e,\gamma,v}\big) / \max_{\gamma'} \sum_v N_{e,\gamma',v}$. The per-experiment
    table has a single value per cell. In the two paired views the spread is over the **remaining** axes only:
    $V_1$ (startdates × $\gamma$) shows single values per $\rho$ and mean ± std over $\rho$ across intensities;
    $V_2$ (variables × $\gamma$) shows mean ± std over startdates (per $\rho$) or startdates × $\rho$ (across).
    **Bold** marks the max $\gamma$ per row.
    """
    mo.md(md_t13_scores)
    return (md_t13_scores,)


@app.cell(hide_code=True)
def t13_tables(
    EXPS,
    delta_labels,
    interf_score_tables_t13,
    mo,
    t13_chan,
    t13_exp,
    t13_pertable,
):
    if t13_chan is None:
        t13_tables_view = mo.md("_press **compute field profiles** above_")
    elif t13_exp.value is None:
        t13_tables_view = mo.md("_no experiment selected_")
    else:
        _ei, _dd = t13_exp.value
        _en = f"{EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]} × {delta_labels.get(_dd, _dd)}"
        _tbl = t13_pertable(_ei, _dd, rf"**{_en}** — per-variable score $N_v=A_v/\max_\gamma A_v$ (rows, $A_v=\sum_\ell s_{{v\ell}}$) × schedule $\gamma$; **single score** $R=\sum_v N_v/\max_\gamma\sum_v N_v$; one value per cell (no member spread); **bold** = max $\gamma$ per row.")
        t13_tables_view = mo.vstack([_tbl, interf_score_tables_t13], align="start")
    t13_tables_view
    return


@app.cell(hide_code=True)
def _(mo, t13_chan, t13_views):
    # T1.3 — paired summary views (V1 single score, V2 per-variable) over the signed ensemble-std object.
    report_diversity = []
    if t13_chan is None:
        interf_score_tables_t13 = mo.md("_press **compute interference profiles** above_")
    else:
        interf_score_tables_t13 = t13_views(report_diversity, "diversity")
    return interf_score_tables_t13, report_diversity


@app.cell
def hdr_t14(mo):
    md_t14 = r"""
    ### T1.4: Leaderboard

    A single cross-eval summary. For each schedule $\gamma$ we take the **single score $R$** from each
    temporal-localization eval — **T1.1** multivariate propagation, **T1.2** spatial deviation from the mask,
    **T1.3** ensemble diversity — pooled over startdates × $\rho$ × members, and add an **Overall** column (the
    mean of the three). Each $R$ is normalized within its own eval ($\max_\gamma R = 1$), so the columns are
    comparable; higher = the schedule that maximizes that eval. Rows are sorted by Overall; **bold** = column max.
    """
    mo.md(md_t14)
    return (md_t14,)


@app.cell(hide_code=True)
def t14_leaderboard(
    EXPS,
    M,
    copyable_table,
    delta_order,
    gamma_key,
    gname,
    interf_data,
    label_delta,
    mo,
    np,
    t13_NR,
    t13_chan,
    t1x_NR,
):
    # T1.4 — cross-eval leaderboard: one row per schedule γ, columns = the three T1 single scores R
    # (T1.1 avgabs propagation, T1.2 tvd spatial deviation, T1.3 signed-std diversity) + Overall (mean), pooled
    # over startdates × ρ × members. Reuses t1x_NR / t13_NR. Higher = the schedule maximizing that eval.
    report_t14 = []
    if interf_data is None or "val_em" not in interf_data or "avgabs" not in interf_data.get("val_em", {}) or "tvd" not in interf_data.get("val_em", {}) or t13_chan is None:
        t14_view = mo.md("_press **compute field profiles** above (T1.4 needs the T1.1–T1.3 data)_")
    else:
        _labels = list(interf_data["labels"])
        _by_g = {}
        for _lb in _labels:
            _by_g.setdefault(gname(_lb), {})[label_delta.get(_lb)] = _lb
        _gammas = sorted(_by_g, key=lambda g: gamma_key(next(iter(_by_g[g].values()))))
        _deltas = [d for d in delta_order if any(label_delta.get(l) == d for l in _labels)]

        def _acc_t1x(_obj):
            _a = {g: [] for g in _gammas}
            for _dd in _deltas:
                for _ei in range(len(EXPS)):
                    _, _R = t1x_NR(_obj, _ei, _dd)
                    for _mm in range(M):
                        for _lb, _rv in _R[_mm].items():
                            if np.isfinite(_rv):
                                _a[gname(_lb)].append(_rv)
            return _a

        def _acc_t13():
            _a = {g: [] for g in _gammas}
            for _dd in _deltas:
                for _ei in range(len(EXPS)):
                    _, _R = t13_NR(_ei, _dd)
                    for _lb, _rv in _R.items():
                        if np.isfinite(_rv):
                            _a[gname(_lb)].append(_rv)
            return _a

        _A = _acc_t1x("avgabs"); _B = _acc_t1x("tvd"); _C = _acc_t13()
        _mean = lambda _xs: float(np.mean(_xs)) if _xs else float("nan")
        _std = lambda _xs: float(np.std(_xs)) if _xs else float("nan")
        _rd = {}
        for _g in _gammas:
            _m11, _m12, _m13 = _mean(_A[_g]), _mean(_B[_g]), _mean(_C[_g])
            _ov = _mean([x for x in (_m11, _m12, _m13) if np.isfinite(x)])
            _rd[_g] = (_m11, _std(_A[_g]), _m12, _std(_B[_g]), _m13, _std(_C[_g]), _ov)
        _og = sorted(_gammas, key=lambda g: -(_rd[g][6] if np.isfinite(_rd[g][6]) else -1.0))
        _cmax = {}
        for _key, _idx in (("t11", 0), ("t12", 2), ("t13", 4), ("ov", 6)):
            _vals = [_rd[g][_idx] for g in _gammas if np.isfinite(_rd[g][_idx])]
            _cmax[_key] = max(_vals) if _vals else None
        def _cell(_mu, _sd, _key):
            if not np.isfinite(_mu):
                return "—"
            _t = f"{_mu:.3f}±{_sd:.3f}" if (_sd is not None and np.isfinite(_sd)) else f"{_mu:.3f}"
            return f"**{_t}**" if (_cmax[_key] is not None and _mu == _cmax[_key]) else _t
        _lines = ["| schedule γ | T1.1 propagation | T1.2 spatial dev. | T1.3 diversity | **Overall** |",
                  "|---|---|---|---|---|"]
        for _g in _og:
            _d = _rd[_g]
            _lines.append(f"| {_g} | {_cell(_d[0], _d[1], 't11')} | {_cell(_d[2], _d[3], 't12')} | {_cell(_d[4], _d[5], 't13')} | {_cell(_d[6], None, 'ov')} |")
        _cap = (f"**T1.4 leaderboard** — single score $R$ per schedule $\\gamma$ for each eval "
                f"(T1.1 propagation, T1.2 spatial deviation, T1.3 diversity); T1.1/T1.2 = mean ± std over "
                f"startdates × ρ × members ({len(EXPS)}×{len(_deltas)}×{M}), T1.3 = mean ± std over startdates × ρ "
                f"(no member axis); **Overall** = mean of the three. **Bold** = column max; sorted by Overall.")
        t14_view = copyable_table("\n".join(_lines), _cap, into=report_t14)
    t14_view
    return (report_t14,)


@app.cell(hide_code=True)
def hdr_support(mo):
    md_support = r"""
    ## Support

    Supporting diagnostics that are not part of the core evaluation set: the realized-target table, the
    leaderboard, the guidance-kick decomposition (T2b–d), the latent-space closeness (PCA path & ping-pong),
    and the prediction-noise check.
    """
    support_checkbox = mo.ui.checkbox(value=False, label="compute support diagnostics (PCA closeness + prediction noise) — off keeps the analysis fast")
    mo.vstack([mo.md(md_support), support_checkbox])
    return md_support, support_checkbox


@app.cell(hide_code=True)
def hdr_r1(mo):
    md_target_real = r"""
    ### Target realization

    - **final gap** $\xi_n=\mathcal{M}_\pi(x_n^{\mathrm{gui}})-y_n^\star$ — the realized terminal target gap
      (Ch. 4), mean $\pm$ std over the pool.
    - **reached** = pooled runs within a $1\%$ relative miss,
      $|\xi_n|\,/\,|y_n^\star-\mathcal{M}_\pi(x_n^{\mathrm{ung}})|\le 0.01$.
    - **target reached** = 🎯 if every pooled run is within $1\%$, else ❌.
    """
    mo.md(md_target_real)
    return (md_target_real,)


@app.cell
def _(
    copyable_table,
    gamma_key,
    gname,
    intensity_dropdown,
    label_delta,
    metrics,
    mo,
):
    report_reliability = []
    # ---- Reliability: target-diff + within-1% target-reached (no plots) ----
    if metrics is None:
        t1_view = mo.md("_press **compute metrics**_")
    else:
        _rows = ["| γ | pool | final gap $\\xi_n$ (mean +/- std) | reached | target reached |",
                 "|---|---|---|---|---|"]
        for _lb in sorted((_l for _l in metrics if label_delta.get(_l) == intensity_dropdown.value), key=gamma_key):
            _r = metrics[_lb]
            _v = _r["eps_final"]; _td = f"{_v[0]:+.4f} +/- {_v[1]:.4f}"
            _rc = _r["reached_count"]; _np = _r["n_pool"]
            _rows.append(f"| {gname(_lb)} | {_np} | {_td} | {_rc}/{_np} | {chr(0x1F3AF) if _rc == _np else chr(0x274C)} |")
        t1_view = copyable_table("\n".join(_rows), r"**Target realization** — final gap $\xi_n$, reached ratio and target-reached per schedule $\gamma$ (selected intensity).", into=report_reliability)
    t1_view
    return (report_reliability,)


@app.cell(hide_code=True)
def hdr_leaderboard(mo):
    md_leaderboard = r"""
    ### Leaderboard

    Aggregate ranking of the schedules across the pooled metrics.
    """
    mo.md(md_leaderboard)
    return (md_leaderboard,)


@app.cell
def _(
    copyable_table,
    delta_labels,
    delta_order,
    gamma_key,
    gname,
    interf_data,
    label_delta,
    metrics,
    mo,
    np,
    realism_data,
):
    report_leaderboard = []
    if metrics is None:
        leaderboard = mo.md("## Leaderboard\n\n_press **compute metrics**_")
    else:
        _has_i = interf_data is not None
        _rd = realism_data  # T4 all-fields + T5 target/all (None until 'compute field profiles')
        def _push_m(_lb):
            return interf_data["avg_score_ms"].get(_lb) if _has_i else None
        def _push_g(_lb):
            return interf_data["absum_score_ms"].get(_lb) if _has_i else None
        def _reached_ms(_r):
            _n = _r["n_pool"]; _p = (_r["reached_count"] / _n) if _n else float("nan")
            _sd = float(np.sqrt(max(_p * (1.0 - _p), 0.0) / _n)) if _n else float("nan")
            return (_p, _sd)
        def _dist_ms(_r):
            _mu = (_r["dpc1"][0], _r["dpc2"][0], _r["dpc3"][0]); _sd = (_r["dpc1"][1], _r["dpc2"][1], _r["dpc3"][1])
            _d = float(np.sqrt(sum(_x * _x for _x in _mu)))
            _ds = float(np.sqrt(sum((_x * _y) ** 2 for _x, _y in zip(_mu, _sd))) / _d) if _d > 0 else 0.0
            return (_d, _ds)
        def _f(_ms, _is_best, signed=False):
            if _ms is None or not np.isfinite(_ms[0]):
                return "—"
            _s = f"{_ms[0]:+.3f} ± {_ms[1]:.3f}" if signed else f"{_ms[0]:.3f} ± {_ms[1]:.3f}"
            return f"**{_s}**" if _is_best else _s
        def _agg_ms(_lst):
            _v = [x for x in _lst if x is not None and np.isfinite(x[0])]
            if not _v:
                return (float("nan"), float("nan"))
            return (float(np.mean([x[0] for x in _v])), float(np.mean([x[1] for x in _v])))

        def _board(_entries, _title=None):
            # _entries: (name, r, pm, pg, rdl); r carries n_pool/reached_count/dpc1-3/realism_tv
            _ta = lambda rdl: rdl["tv_all"] if rdl else None
            _tt = lambda rdl: rdl["t5_target"] if rdl else None
            _tl = lambda rdl: rdl["t5_all"] if rdl else None
            _bre = max((_reached_ms(r)[0] for _, r, _, _, _ in _entries if np.isfinite(_reached_ms(r)[0])), default=None)
            _bpm = max((p[0] for _, _, p, _, _ in _entries if p is not None and np.isfinite(p[0])), default=None)
            _bpg = max((p[0] for _, _, _, p, _ in _entries if p is not None and np.isfinite(p[0])), default=None)
            _bt = max(r["realism_tv"][0] for _, r, _, _, _ in _entries)
            _bta = max((_ta(rdl)[0] for _, _, _, _, rdl in _entries if _ta(rdl) is not None and np.isfinite(_ta(rdl)[0])), default=None)
            _bst = max((_tt(rdl)[0] for _, _, _, _, rdl in _entries if _tt(rdl) is not None and np.isfinite(_tt(rdl)[0])), default=None)
            _bsa = max((_tl(rdl)[0] for _, _, _, _, rdl in _entries if _tl(rdl) is not None and np.isfinite(_tl(rdl)[0])), default=None)
            _rows = ["| γ | reached ratio | push score (mask) | push score (global) | realism TV (target) | realism TV (all) | fp-spread (target) | fp-spread (all) |",
                     "|---|---|---|---|---|---|---|---|"]
            for _name, r, pm, pg, rdl in _entries:
                _rc = _reached_ms(r)
                _rows.append(
                    f"| {_name} "
                    f"| {_f(_rc, _bre is not None and _rc[0] == _bre)} "
                    f"| {_f(pm, pm is not None and _bpm is not None and pm[0] == _bpm)} "
                    f"| {_f(pg, pg is not None and _bpg is not None and pg[0] == _bpg)} "
                    f"| {_f(r['realism_tv'], r['realism_tv'][0] == _bt)} "
                    f"| {_f(_ta(rdl), _bta is not None and _ta(rdl) is not None and _ta(rdl)[0] == _bta)} "
                    f"| {_f(_tt(rdl), _bst is not None and _tt(rdl) is not None and _tt(rdl)[0] == _bst)} "
                    f"| {_f(_tl(rdl), _bsa is not None and _tl(rdl) is not None and _tl(rdl)[0] == _bsa)} |")
            return copyable_table("\n".join(_rows), _title, into=report_leaderboard)

        _by_delta = {}
        for _lb in metrics:
            _by_delta.setdefault(label_delta.get(_lb, 0), []).append(_lb)
        _multi = len(_by_delta) > 1
        _desc = mo.md("mean ± std over the pool (3 dp); **bold** = winner per column (within each block). "
                      "**reached ratio** = fraction of pooled runs within 1% of the target (± binomial std); winners "
                      "are **max** everywhere. **push score (mask)** = "
                      "masked-mean push; **push score (global)** = whole-field |Δ| push; both need the **interference "
                      "profiles** compute. **realism TV (all)** / **fp-spread** need **compute field profiles** (else —).")
        _blocks = [mo.md("## Leaderboard"), _desc]
        if _multi:
            for _di in delta_order:
                if _di not in _by_delta:
                    continue
                _ents = [(gname(_lb), metrics[_lb], _push_m(_lb), _push_g(_lb), (_rd.get(_lb) if _rd else None)) for _lb in sorted(_by_delta[_di], key=gamma_key)]
                _blocks.append(_board(_ents, f"### Intensity {delta_labels[_di]}"))
            def _skey(_lb):
                return _lb.rsplit(" δ#", 1)[0]
            _by_sched = {}
            for _lb in metrics:
                _by_sched.setdefault(_skey(_lb), []).append(_lb)
            _sents = []
            for _sk, _lbs in sorted(_by_sched.items(), key=lambda _kv: gamma_key(_kv[0])):
                _rs = [metrics[_l] for _l in _lbs]
                _rag = {"n_pool": sum(_r["n_pool"] for _r in _rs),
                        "reached_count": sum(_r["reached_count"] for _r in _rs),
                        "dpc1": _agg_ms([_r["dpc1"] for _r in _rs]),
                        "dpc2": _agg_ms([_r["dpc2"] for _r in _rs]),
                        "dpc3": _agg_ms([_r["dpc3"] for _r in _rs]),
                        "realism_tv": _agg_ms([_r["realism_tv"] for _r in _rs])}
                _pam = _agg_ms([_push_m(_l) for _l in _lbs])
                _pgg = _agg_ms([_push_g(_l) for _l in _lbs])
                _rdag = None
                if _rd:
                    _xs = [_rd.get(_l) for _l in _lbs]
                    if any(_x is not None for _x in _xs):
                        _rdag = {"tv_all": _agg_ms([_x["tv_all"] for _x in _xs if _x]),
                                 "t5_target": _agg_ms([_x["t5_target"] for _x in _xs if _x]),
                                 "t5_all": _agg_ms([_x["t5_all"] for _x in _xs if _x])}
                _sents.append((gname(_sk), _rag, _pam, _pgg, _rdag))
            _blocks.append(_board(_sents, "### Summary across intensities  \n_each metric meaned over the intensity levels per schedule; reached ratio re-pooled over levels._"))
        else:
            _ents = [(gname(_lb), metrics[_lb], _push_m(_lb), _push_g(_lb), (_rd.get(_lb) if _rd else None)) for _lb in sorted(metrics, key=gamma_key)]
            _blocks.append(_board(_ents))
        leaderboard = mo.vstack(_blocks)
    leaderboard
    return (report_leaderboard,)


@app.cell(hide_code=True)
def _(mo):
    md_kick = r"""
    ### Guidance kick decomposition (T2b–d)

    Per variable and level, the anatomy of the applied guidance kick:

    - **T2b** — the raw guidance-gradient norm $\lVert\nabla_{z}\mathcal{L}\rVert$.
    - **T2c** — the actually applied kick $\lambda_{n,t}\,\lVert\nabla_{z}\mathcal{L}\rVert$.
    - **T2d** — the applied kick, normalized to surface temperature (2mT).

    Ranked; ★ = best-$k$.
    """
    mo.md(md_kick)
    return (md_kick,)


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
    md_closeness = r"""
    ### Closeness — PCA path & ping-pong

    *How far does the guided flow stray from the unguided one?* Measured in the mask-region PCA latent space
    (fit on the 12:00 climatology). Both views share the eval-region and 3-D view controls.

    - **PCA path grid** (one per experiment): the guided state trajectory $x_{n,t}$ (solid) together with the
      independent unguided rollout $x^{\mathrm{ung}}$, the same-seed twin $x^{\mathrm{ung|gui}}$, the ERA5
      cloud, and the ERA5 reference.
    - **Ping-pong**: the same guided path $x_{n,t}$ with a per-step **guidance jump** whisker — the
      intervention $-\lambda_{n,t}\,\nabla_{z_{n,t}}\mathcal{L}$ injected at each flow step.

    Table (per schedule): **path length** of the clean-prediction trajectory in 3-PC space; per-flow-step
    **guidance / pushback** norms; and the signed endpoint deviation from the twin along each PC
    ($\Delta\mathrm{PC}_k$) with its norm (**dist to** $x^{\mathrm{ung|gui}}$).
    """
    mo.md(md_closeness)
    return (md_closeness,)


@app.cell
def _(mo):
    # PCA region: fit/project the latent space on the whole globe, the mask footprint, or its
    # complement (see what guidance does inside vs. outside the target)
    pca_region_dropdown = mo.ui.dropdown(["mask", "globe", "!mask"], value="mask", label="eval region: ")
    pca_region_dropdown
    return (pca_region_dropdown,)


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
def _(mo):
    # --- T3 subsection 1 (PCA path grid): view controls, independent of subsection 2 ---
    elev_slider = mo.ui.slider(0, 90, value=0, step=5, label="elev: ", show_value=True)
    azim_slider = mo.ui.slider(-180, 180, value=0, step=5, label="azim: ", show_value=True)
    zoom_traj_checkbox = mo.ui.checkbox(label="zoom to trajectories", value=True)
    zoom_pad_slider = mo.ui.slider(-0.4, 2.0, step=0.05, value=0, label="zoom pad: ", show_value=True)
    mo.vstack([mo.md("### 3.1 PCA path grid"),
               mo.hstack([elev_slider, azim_slider, zoom_traj_checkbox, zoom_pad_slider], justify="start", align="center")])
    return azim_slider, elev_slider, zoom_pad_slider, zoom_traj_checkbox


@app.cell
def _(
    EXPS,
    M,
    azim_slider,
    basis,
    cloud_proj,
    copyable_table,
    elev_slider,
    gamma_key,
    gname,
    gt_proj,
    guiung_traj,
    intensity_dropdown,
    label_delta,
    metrics,
    mo,
    n_slider,
    np,
    plot_labels,
    plt,
    ref_metrics,
    sched_colors,
    support_checkbox,
    traj_grid,
    ung_traj,
    zoom_pad_slider,
    zoom_traj_checkbox,
):
    report_closeness = []
    # ---- Closeness: full PCA path grid (ported), rows=exp, cols=m, with zoom-to-trajectories ----
    if not support_checkbox.value or metrics is None or traj_grid is None or cloud_proj is None:
        t3_view = mo.md("_support diagnostics off — tick **compute support diagnostics** above_")
    else:
        _n = int(n_slider.value) - 1; _labels = sorted(plot_labels, key=gamma_key)
        _items = []
        for _ei in range(len(EXPS)):
            _items.append(mo.md(f"**{EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]}**  (n={_n + 1}, EVR={basis.evr.sum():.0%})"))
            _f, _axs = plt.subplots(1, M, figsize=(4.6*M, 4.4), subplot_kw={"projection": "3d"}, squeeze=False, dpi=110)
            for _m in range(M):
                _ax = _axs[0][_m]; _grp = []
                _ax.scatter(cloud_proj[:, 0], cloud_proj[:, 1], cloud_proj[:, 2], color="#BBBBBB", s=8, alpha=0.45, depthshade=False, label=f"ERA5 cloud ({cloud_proj.shape[0]})")
                _cell = traj_grid.get((_ei, _m, _n), {})
                for _lb in _labels:
                    if _lb not in _cell: continue
                    _pg = np.asarray(_cell[_lb]["pgx"]); _c = sched_colors[_lb]
                    _ax.plot(_pg[:, 0], _pg[:, 1], _pg[:, 2], "-", color=_c, linewidth=1.4, alpha=0.9, label=gname(_lb))
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
                    _ax.plot(_pu[:, 0], _pu[:, 1], _pu[:, 2], "--", color="#888888", linewidth=1.1, alpha=0.8, label="ung|gui")
                    _ax.scatter(_pu[-1, 0], _pu[-1, 1], _pu[-1, 2], marker="o", s=34, color="#888888", depthshade=False)
                    _grp.append(_pu)
                if _cell:
                    _sp = np.asarray(next(iter(_cell.values()))["pgx"])[0]
                    _ax.scatter(_sp[0], _sp[1], _sp[2], marker="o", s=45, facecolors="#FFD700", edgecolors="black", linewidths=1.2, depthshade=False, label="start")
                    pass
                if zoom_traj_checkbox.value and _grp:
                    _pts = np.vstack(_grp); _pts = _pts[np.isfinite(_pts).all(axis=1)]   # drop NaN (pending members)
                    if _pts.shape[0]:
                        _lo, _hi = _pts.min(0), _pts.max(0)
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
        _sweep = sorted([_l for _l in metrics if label_delta.get(_l) == intensity_dropdown.value], key=gamma_key)
        _bg = min(metrics[_l]["guidance_steps"][0] for _l in _sweep)
        _bp = min(metrics[_l]["pushback_steps"][0] for _l in _sweep)
        _bd = max(_dist(metrics[_l]) for _l in _sweep)  # highest divergence from the PCs (bold)
        _bd1 = max(abs(metrics[_l]["dpc1"][0]) for _l in _sweep)
        _bd2 = max(abs(metrics[_l]["dpc2"][0]) for _l in _sweep)
        _bd3 = max(abs(metrics[_l]["dpc3"][0]) for _l in _sweep)
        _rows = ["| row | pool | path length | guidance steps | pushback steps | \u0394PC1 | \u0394PC2 | \u0394PC3 | dist to ung\\|gui |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for _lb in _sweep:
            _r = metrics[_lb]; _npv = _r["n_pool"]; _Lc = _ms(_r["L"])
            _gs = _r["guidance_steps"]; _gc = _ms(_gs); _gc = f"**{_gc}**" if _gs[0] == _bg else _gc
            _ps = _r["pushback_steps"]; _pc = _ms(_ps); _pc = f"**{_pc}**" if _ps[0] == _bp else _pc
            _dv = _dist(_r); _dc = f"{_dv:.3g}"; _dc = f"**{_dc}**" if _dv == _bd else _dc
            _d1 = _mss(_r["dpc1"]); _d1 = f"**{_d1}**" if abs(_r["dpc1"][0]) == _bd1 else _d1
            _d2 = _mss(_r["dpc2"]); _d2 = f"**{_d2}**" if abs(_r["dpc2"][0]) == _bd2 else _d2
            _d3 = _mss(_r["dpc3"]); _d3 = f"**{_d3}**" if abs(_r["dpc3"][0]) == _bd3 else _d3
            _rows.append(f"| {gname(_lb)} | {_npv} | {_Lc} | {_gc} | {_pc} | {_d1} | {_d2} | {_d3} | {_dc} |")
        for _rlb in ("gui_ung", "ung"):
            if ref_metrics and _rlb in ref_metrics:
                _r = ref_metrics[_rlb]; _npv = _r["n_pool"]; _Lc = _ms(_r["L"])
                _d1 = _mss(_r["dpc1"]); _d2 = _mss(_r["dpc2"]); _d3 = _mss(_r["dpc3"]); _dc = f"{_dist(_r):.3g}"
                _rows.append(f"| _{'ung\\|gui' if _rlb == 'gui_ung' else _rlb}_ | {_npv} | {_Lc} | 0 | 0 | {_d1} | {_d2} | {_d3} | {_dc} |")
        t3_view = mo.vstack(_items + [copyable_table("\n".join(_rows), r"**Closeness (PCA)** — path length, guidance/pushback steps and endpoint PC deviation per schedule $\gamma$ (selected intensity).", into=report_closeness),
            mo.md(r"Grid = actual-state $x_t$ trajectories (guided solid). Length = PCA path length of the clean-pred trajectory. $\Delta$PC$_k$ = signed "
                  r"endpoint deviation from ung|gui along PC$_k$. Guidance/pushback in full bbox "
                  r"space (0 for the unguided ung / ung|gui reference rows).")])
    t3_view
    return (report_closeness,)


@app.cell(hide_code=True)
def _(mo):
    # --- T3 subsection 2 (ping-pong): its own view controls, independent of subsection 1 ---
    elev_slider2 = mo.ui.slider(0, 90, value=0, step=5, label="elev: ", show_value=True)
    azim_slider2 = mo.ui.slider(-180, 180, value=0, step=5, label="azim: ", show_value=True)
    zoom_traj_checkbox2 = mo.ui.checkbox(label="zoom to trajectories", value=True)
    zoom_pad_slider2 = mo.ui.slider(-0.4, 2.0, step=0.05, value=0, label="zoom pad: ", show_value=True)
    mo.vstack([mo.md("### 3.2 Ping-pong (guidance jumps)"),
               mo.hstack([elev_slider2, azim_slider2, zoom_traj_checkbox2, zoom_pad_slider2], justify="start", align="center")])
    return azim_slider2, elev_slider2, zoom_pad_slider2, zoom_traj_checkbox2


@app.cell
def _(
    EXPS,
    M,
    azim_slider2,
    basis,
    cloud_proj,
    elev_slider2,
    gamma_key,
    gname,
    gt_proj,
    guiung_traj,
    metrics,
    mo,
    n_slider,
    np,
    plot_labels,
    plt,
    sched_colors,
    support_checkbox,
    traj_grid,
    ung_traj,
    zoom_pad_slider2,
    zoom_traj_checkbox2,
):
    # ---- Closeness (ping-pong): full x_t trajectory + per-step guidance jumps, rows=exp, cols=m ----
    # Same PCA grid as T3 above: each schedule's FULL guided x_t trajectory (solid, as in T3) with a
    # per-step JUMP whisker (dotted) = the kick guidance injected at that step (pgx_t - pkx_t =
    # c*s_t*gui_vec_t; zero where guidance is off). ung / gui_ung drawn as full trajectories.
    if not support_checkbox.value or metrics is None or traj_grid is None or cloud_proj is None:
        t3pp_view = mo.md("_support diagnostics off — tick **compute support diagnostics** above_")
    else:
        _n = int(n_slider.value) - 1; _labels = sorted(plot_labels, key=gamma_key)
        _items = [mo.md(r"### Ping-pong — full guided $x_{n,t}$ trajectory (solid, as in T3) with a per-step guidance **jump** (dotted whisker = the kick injected at that step); **ung** / **ung|gui** shown as full trajectories, coloured as in T3.")]
        for _ei in range(len(EXPS)):
            _items.append(mo.md(f"**{EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]}**  (n={_n + 1}, EVR={basis.evr.sum():.0%})"))
            _f, _axs = plt.subplots(1, M, figsize=(4.6*M, 4.4), subplot_kw={"projection": "3d"}, squeeze=False, dpi=110)
            for _m in range(M):
                _ax = _axs[0][_m]; _grp = []
                _ax.scatter(cloud_proj[:, 0], cloud_proj[:, 1], cloud_proj[:, 2], color="#BBBBBB", s=8, alpha=0.45, depthshade=False, label=f"ERA5 cloud ({cloud_proj.shape[0]})")
                _cell = traj_grid.get((_ei, _m, _n), {})
                for _lb in _labels:
                    if _lb not in _cell or "pkx" not in _cell[_lb]: continue
                    _bp = np.asarray(_cell[_lb]["pgx"]); _kp = np.asarray(_cell[_lb]["pkx"]); _c = sched_colors[_lb]
                    # full guided x_t trajectory (continuous, as in the T3 path grid)
                    _ax.plot(_bp[:, 0], _bp[:, 1], _bp[:, 2], "-", color=_c, linewidth=1.4, alpha=0.9,
                             label=gname(_lb))
                    # per-step guidance JUMP whisker: x_t -> x_t-without-this-kick (== injected kick
                    # c*s_t*gui_vec_t). A spike schedule puts one huge kick early; cap the DRAWN length to
                    # _WCAP (0.5 x trajectory span) so it stays a visible spur instead of shooting off-frame
                    # (direction kept, magnitude clipped). Capped tips join _grp so pad=0 still contains them.
                    _tspan = float(np.linalg.norm(np.nanmax(_bp, 0) - np.nanmin(_bp, 0)))
                    _WCAP = 0.5 * _tspan if _tspan > 0 else 1.0
                    for _t in range(len(_bp)):
                        _vv = _kp[_t] - _bp[_t]; _nv = float(np.linalg.norm(_vv))
                        if _nv < 1e-9: continue
                        _tip = _bp[_t] + _vv * (min(_nv, _WCAP) / _nv)
                        _seg = np.stack([_bp[_t], _tip])
                        _ax.plot(_seg[:, 0], _seg[:, 1], _seg[:, 2], ":", color=_c, linewidth=1.1, alpha=0.7, label="_nolegend_")
                        _grp.append(_tip[None, :])
                    _ax.scatter(_bp[-1, 0], _bp[-1, 1], _bp[-1, 2], marker="o", s=42, color=_c, depthshade=False)
                    _grp.append(_bp)
                if (_ei, _n) in gt_proj:
                    _tp = np.asarray(gt_proj[(_ei, _n)])
                    _ax.scatter(_tp[0], _tp[1], _tp[2], marker="*", s=70, color="#FFD700", edgecolors="black", linewidths=1.0, depthshade=False, label="gt")
                    _grp.append(_tp[None, :])
                if (_ei, _m, _n) in ung_traj:
                    _pn = np.asarray(ung_traj[(_ei, _m, _n)])
                    _ax.plot(_pn[:, 0], _pn[:, 1], _pn[:, 2], "-", color="#111111", linewidth=1.2, alpha=0.85, label="ung")
                    _ax.scatter(_pn[-1, 0], _pn[-1, 1], _pn[-1, 2], marker="o", s=30, color="#111111", depthshade=False)
                    _grp.append(_pn)
                if _n > 0 and (_ei, _m, _n) in guiung_traj:
                    _pu = np.asarray(guiung_traj[(_ei, _m, _n)])
                    _ax.plot(_pu[:, 0], _pu[:, 1], _pu[:, 2], "--", color="#888888", linewidth=1.0, alpha=0.7, label="ung|gui")
                    _grp.append(_pu)
                if _cell:
                    _sp = np.asarray(next(iter(_cell.values()))["pgx"])[0]
                    _ax.scatter(_sp[0], _sp[1], _sp[2], marker="o", s=45, facecolors="#FFD700", edgecolors="black", linewidths=1.2, depthshade=False, label="start")
                if zoom_traj_checkbox2.value and _grp:
                    _pts = np.vstack(_grp); _pts = _pts[np.isfinite(_pts).all(axis=1)]   # drop NaN (pending members)
                    if _pts.shape[0]:
                        _lo, _hi = _pts.min(0), _pts.max(0)
                        _ctr = 0.5 * (_lo + _hi)                                          # centre the view on the content
                        _half = max(0.5 * float((_hi - _lo).max()) * (1.0 + zoom_pad_slider2.value), 1e-6)  # cubic half-extent
                        _ax.set_xlim(_ctr[0]-_half, _ctr[0]+_half); _ax.set_ylim(_ctr[1]-_half, _ctr[1]+_half); _ax.set_zlim(_ctr[2]-_half, _ctr[2]+_half)
                _ax.set_xlabel("PC1", fontsize=7); _ax.set_ylabel("PC2", fontsize=7); _ax.set_zlabel("PC3", fontsize=7)
                _ax.view_init(elev=elev_slider2.value, azim=azim_slider2.value)
                _ax.set_title(f"m={_m}", fontsize=9)
                if _m == M - 1 and _cell:
                    _ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
            _f.tight_layout(pad=0.3)
            _items.append(mo.as_html(_f))
        t3pp_view = mo.vstack(_items)
    t3pp_view
    return


@app.cell
def _(mo):
    md_prednoise = r"""
    ### Prediction noise

    *How noisy is the clean estimate the guidance gradient is taken on?* At flow step $t$ the model's clean
    estimate of the final latent is $\hat{z}_t = z_t + \tfrac{s_t}{h_t}\,(z_{t+1}-z_t)$ -- for the **unguided**
    flow the plain Euler step makes the model velocity the exact finite difference of the stored latents
    $z_0\dots z_T$. We report, over the eval region, how far this estimate is from the produced final latent
    $z_T$ at each flow step $t$ (for the selected weather step $n$):

    $$\mathrm{RMSE}(t)=\sqrt{\big\langle (\hat{z}_t-z_T)^2 \big\rangle_{\pi}}\quad(\text{latent }z\text{-space}).$$

    It is large early (the endpoint is still uncertain, so the guidance gradient is correspondingly noisy) and
    decays to $0$ at $t=T$ by construction. Shaded band = mean $\pm$ std over members $M$; markers = per-$t$ mean.
    Being an unguided quantity it does not depend on the guidance schedule or intensity -- **one curve per
    experiment**.
    """
    mo.md(md_prednoise)
    return (md_prednoise,)


@app.cell
def _(EXPS, mo, n_slider, np, plt, ung_pred_rmse):
    if ung_pred_rmse is None:
        ung_noise_view = mo.md("_press **compute metrics**_")
    elif not ung_pred_rmse:
        ung_noise_view = mo.md("_no unguided latent (`ung_res`) in the selected rollouts_")
    else:
        _n = int(n_slider.value) - 1
        _eis = sorted(ung_pred_rmse.keys())
        _cmap = plt.get_cmap("turbo")
        _f, _ax = plt.subplots(1, 1, figsize=(7.2, 3.3), dpi=120)
        for _k, _ei in enumerate(_eis):
            _r = ung_pred_rmse[_ei][:, _n, :]                          # (M, T)
            _x = np.arange(_r.shape[1])
            _mean = _r.mean(0); _sd = _r.std(0)
            _col = _cmap((_k + 0.5) / max(len(_eis), 1))
            _ax.fill_between(_x, _mean - _sd, _mean + _sd, color=_col, alpha=0.18, linewidth=0)
            _ax.plot(_x, _mean, "-o", color=_col, ms=3, linewidth=1.6, label=EXPS[_ei].split(chr(47))[-1].split(chr(95))[0])
        _ax.set_xlabel("flow step $t$")
        _ax.set_ylabel(r"$\mathrm{RMSE}(\hat{z}_t,\,z_T)$")
        _ax.grid(True, alpha=0.25); _ax.margins(x=0.01)
        _ax.legend(fontsize=8, frameon=False, loc="upper right")
        _f.tight_layout(pad=0.3)
        ung_noise_view = mo.as_html(_f)
    ung_noise_view
    return


@app.cell
def _(EXPS, copyable_table, mo, n_slider, np, ung_pred_rmse):
    report_prednoise = []
    if not ung_pred_rmse:
        t6_view = (mo.md("_press **compute metrics**_") if ung_pred_rmse is None
                   else mo.md("_no unguided latent (`ung_res`) in the selected rollouts_"))
    else:
        _n = int(n_slider.value) - 1
        _eis = sorted(ung_pred_rmse.keys())
        _T = ung_pred_rmse[_eis[0]].shape[2]
        _means = {_ei: [float(ung_pred_rmse[_ei][:, _n, _t].mean()) for _t in range(_T)] for _ei in _eis}
        _amax = {_ei: int(np.argmax(_means[_ei])) for _ei in _eis}
        _hdr = "| flow step $t$ | " + " | ".join(EXPS[_ei].split(chr(47))[-1].split(chr(95))[0] for _ei in _eis) + " |"
        _rows = [_hdr, "|" + "---|" * (len(_eis) + 1)]
        for _t in range(_T):
            _cells = []
            for _ei in _eis:
                _r = ung_pred_rmse[_ei][:, _n, _t]
                _c = f"{_r.mean():.4f}±{_r.std():.4f}"
                _cells.append(f"**{_c}**" if _t == _amax[_ei] else _c)
            _rows.append(f"| {_t} | " + " | ".join(_cells) + " |")
        _pool = {_t: np.concatenate([ung_pred_rmse[_ei][:, _n, _t] for _ei in _eis]) for _t in range(_T)}
        _smax = int(np.argmax([_pool[_t].mean() for _t in range(_T)]))
        _srows = ["| flow step $t$ | mean | std (over exp × members) |", "|---|---|---|"]
        for _t in range(_T):
            _c = f"{_pool[_t].mean():.4f}"
            _srows.append(f"| {_t} | {('**'+_c+'**') if _t == _smax else _c} | {_pool[_t].std():.4f} |")
        t6_view = mo.vstack([
            copyable_table(chr(10).join(_rows), rf"**Per experiment** — $\mathrm{{RMSE}}(\hat{{z}}_t,\,z_T)$ per flow step $t$ (rows) x experiment (cols); mean ± std over members $M$ (weather step $n={_n + 1}$); **bold** = peak (max) per experiment.", into=report_prednoise),
            copyable_table(chr(10).join(_srows), r"**Summary** — mean ± std over all experiments × members, per flow step $t$; **bold** = peak (max).", into=report_prednoise),
        ], align="start")
    t6_view
    return (report_prednoise,)


@app.cell
def _(
    BBOX,
    EXPS,
    LEVEL,
    N,
    PARTITION,
    PCA_REGION,
    VAR,
    base_pin,
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
    open_unguided_state,
    pathlength,
    pinned_records,
    plt,
    project,
    region_latent,
    residual_scaler,
    select_point,
    selected_point_labels,
    sweep_points,
):
    # ---- compute: pooled metrics + per-(exp, m, n) grid trajectories ----
    metrics = None
    traj_grid = None
    gt_proj = None
    sched_colors = None
    ref_metrics = None
    if True:  # auto-compute
        _labels = list(selected_point_labels)
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
            _sp = sweep_points(_sv, pinned_records(_recs, base_pin))
            _pts = {lb: _sp[lb] for lb in _labels if lb in _sp}
            try:
                _ung_final = np.asarray(channel(open_unguided_state(_dir, "ung", VAR), _cfg).isel(t=-1), float)
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
                _tw = channel(select_point(open_unguided_state(_dir, "gui_ung", VAR), _sel), _cfg)
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
                        _xt_lat = _g - _vfs_lat; _xk_lat = _kick - _vfs_lat        # x_t (clean-pred - c*vfs, ends x_{T-1})
                        # append the true converged endpoint x_T = gui so the guided x_t reaches x_T,
                        # matching the res-based gui_ung / ung trajectories (now length T+1, reaching x_T)
                        _gui_lat = region_latent(_guif, PCA_REGION)[None]
                        _xt_lat = np.concatenate([_xt_lat, _gui_lat], axis=0)
                        _xk_lat = np.concatenate([_xk_lat, _gui_lat], axis=0)
                        _pgx = project(basis, _xt_lat)                             # actual state x_t
                        _pkx = project(basis, _xk_lat)                             # x_t kick branch
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
                _ung_da = channel(open_unguided_state(_dir, "ung", VAR), _cfg)
                _giu_da = channel(select_point(open_unguided_state(_dir, "gui_ung", VAR), _sel0), _cfg)
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
        metrics,
        ref_metrics,
        sched_colors,
        traj_grid,
        ung_traj,
    )


@app.cell(hide_code=True)
def _(
    EVAL_W,
    EXPS,
    INTERF_LEVEL_VARS,
    INTERF_SURFACE_PAIR,
    INTERF_VARS,
    M,
    N,
    VAR,
    channel,
    ftt,
    interf_button,
    load_rollout,
    np,
    open_store,
    open_unguided_state,
    region_realism_tv,
    sched_colors,
    select_point,
    sweep_points,
    xr,
):
    # ---- heavy realism + footprint-spread across ALL fields (button-gated, reuses the T2 button) ----
    # Companions to the target-field T4 realism (metrics["realism_tv"]):
    #   realism TV (all fields): ftt.realism_tv(x_gui, x_gui_ung, mask) per channel, MEAN over all 6
    #                            level variables x levels (+ surface pair), over the EVAL REGION. -> T4 col.
    #   footprint spread  (T5) : per-pixel std over ensemble members m of the guidance footprint
    #                            |x_gui - x_gui_ung|, then masked-mean -> a scalar. Target field
    #                            (t5_target) and MEAN over all fields (t5_all). Pooled over (exp, guided n).
    # Also per-SINGLE-VARIABLE breakdowns (tv_var / t5_var), one scalar per INTERF_VARS variable: the
    # mean over that variable's channels (levels + its surface pair; mslp added as its own column, NOT
    # folded into tv_all/t5_all so those stay unchanged). Feeds the T1.2/T1.3 per-variable side tables.
    realism_data = None
    t13_chan = None
    if interf_button.value and sched_colors is not None:
        _labels = list(sched_colors)
        _wsum = float(EVAL_W.sum()) or 1.0
        _tv_all = {lb: [] for lb in _labels}   # per (exp,m,n): mean-over-channels realism TV
        _t5_tg = {lb: [] for lb in _labels}    # per (exp,n): target-field footprint spread
        _t5_all = {lb: [] for lb in _labels}   # per (exp,n): mean-over-channels footprint spread
        _tv_var = {lb: {v: [] for v in INTERF_VARS} for lb in _labels}   # per (exp,m,n): per-variable mean-over-channels TV
        _t5_var = {lb: {v: [] for v in INTERF_VARS} for lb in _labels}   # per (exp,n): per-variable mean-over-channels footprint spread
        _t13 = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}   # per (exp): signed ensemble-std object M_pi(std_m(x_gui-x_ung)) per channel
        for _ei, _rid in enumerate(EXPS):
            _dir, _cfg, _sv, _recs, _mask = load_rollout(_rid)
            _mask = np.asarray(_mask, float)
            _sp = sweep_points(_sv, _recs); _pts = {lb: _sp[lb] for lb in _labels if lb in _sp}
            _gui = xr.open_zarr(_dir / "gui.zarr")
            for _lb, _sel in _pts.items():
                _p = np.asarray(_sv["GUIDANCE_DELTA"][_sel["GUIDANCE_DELTA"]], float)[:N]
                _gns = [nn for nn in range(N) if _p[nn] != 0.0] or [0]
                _gt = channel(select_point(open_store(_dir, "gui", VAR), _sel), _cfg)      # target guided
                _tt = channel(select_point(open_unguided_state(_dir, "gui_ung", VAR), _sel), _cfg)  # target twin
                for _n in _gns:
                    # --- target-field footprint spread (T5 target): std over m of |gui-giu|, masked-mean ---
                    _tg_foot = [np.abs(np.asarray(_gt.isel(m=_m, n=_n), float)
                                       - np.asarray(_tt.isel(m=_m, n=_n).isel(t=-1), float)) for _m in range(M)]
                    if len(_tg_foot) >= 2:
                        _t5_tg[_lb].append(float((np.std(np.stack(_tg_foot, 0), axis=0) * EVAL_W).sum() / _wsum))
                    # --- all-fields realism TV + footprint spread (+ per-variable breakdown) ---
                    _foot_ch = {}   # (parent_var, channel) -> list over m of |gui - giu| footprint
                    _sfoot_ch = {}   # (parent_var, channel) -> list over m of SIGNED (gui - giu)
                    for _m in range(M):
                        _tv_ch_m = []
                        _tv_ch_var = {v: [] for v in INTERF_VARS}
                        for _vv in INTERF_LEVEL_VARS:
                            _levs = list(_gui[_vv]["level"].values)
                            _g3 = np.asarray(select_point(_gui[_vv], _sel).isel(m=_m, n=_n), float)   # (level,lat,lon)
                            _ud = select_point(open_unguided_state(_dir, "gui_ung", _vv), _sel).isel(m=_m, n=_n)
                            _u3 = np.asarray(_ud.isel(t=-1) if "t" in _ud.dims else _ud, float)        # (level,lat,lon)
                            for _li, _L in enumerate(_levs):
                                _tvv = region_realism_tv(_g3[_li], _u3[_li], _mask, EVAL_W)
                                _tv_ch_m.append(_tvv); _tv_ch_var[_vv].append(_tvv)
                                _foot_ch.setdefault((_vv, _L), []).append(np.abs(_g3[_li] - _u3[_li]))
                                _sfoot_ch.setdefault((_vv, _L), []).append(_g3[_li] - _u3[_li])
                            if _vv in INTERF_SURFACE_PAIR:
                                _svar = INTERF_SURFACE_PAIR[_vv]
                                _gs = np.asarray(select_point(_gui[_svar], _sel).isel(m=_m, n=_n), float)  # (lat,lon)
                                _usd = select_point(open_unguided_state(_dir, "gui_ung", _svar), _sel).isel(m=_m, n=_n)
                                _us = np.asarray(_usd.isel(t=-1) if "t" in _usd.dims else _usd, float)
                                _tvv = region_realism_tv(_gs, _us, _mask, EVAL_W)
                                _tv_ch_m.append(_tvv); _tv_ch_var[_vv].append(_tvv)   # surface pair folds into its parent variable
                                _foot_ch.setdefault((_vv, "sfc"), []).append(np.abs(_gs - _us))
                                _sfoot_ch.setdefault((_vv, "sfc"), []).append(_gs - _us)
                        # mslp: its own variable column (surface-only); kept OUT of tv_all/t5_all
                        try:
                            _mv = "mean_sea_level_pressure"
                            _gm = np.asarray(select_point(_gui[_mv], _sel).isel(m=_m, n=_n), float)
                            _umd = select_point(open_unguided_state(_dir, "gui_ung", _mv), _sel).isel(m=_m, n=_n)
                            _um = np.asarray(_umd.isel(t=-1) if "t" in _umd.dims else _umd, float)
                            _tv_ch_var[_mv].append(region_realism_tv(_gm, _um, _mask, EVAL_W))
                            _foot_ch.setdefault((_mv, "sfc"), []).append(np.abs(_gm - _um))
                            _sfoot_ch.setdefault((_mv, "sfc"), []).append(_gm - _um)
                        except Exception:
                            pass
                        _tv_all[_lb].append(float(np.nanmean(_tv_ch_m)))
                        for _v in INTERF_VARS:
                            if _tv_ch_var[_v]:
                                _tv_var[_lb][_v].append(float(np.nanmean(_tv_ch_var[_v])))
                    # per-channel footprint spread (std over m, masked-mean) -> group by parent variable
                    _by_var_sp = {v: [] for v in INTERF_VARS}
                    _t5_ch = []
                    for (_pvar, _ch), _fs in _foot_ch.items():
                        if len(_fs) >= 2:
                            _sp_val = float((np.std(np.stack(_fs, 0), axis=0) * EVAL_W).sum() / _wsum)
                            _by_var_sp[_pvar].append(_sp_val)
                            if _pvar != "mean_sea_level_pressure":
                                _t5_ch.append(_sp_val)
                    if _t5_ch:
                        _t5_all[_lb].append(float(np.mean(_t5_ch)))
                    for _v in INTERF_VARS:
                        if _by_var_sp[_v]:
                            _t5_var[_lb][_v].append(float(np.mean(_by_var_sp[_v])))
                    _sm2 = _mask / (_mask.sum() if _mask.sum() > 0 else 1.0)
                    for (_pv3, _ch3), _sfs in _sfoot_ch.items():
                        if len(_sfs) >= 2:
                            _sstd = np.std(np.stack(_sfs, 0), axis=0)
                            _t13[_lb][_pv3].setdefault(_ch3, {}).setdefault(_ei, []).append(float((_sstd * _sm2).sum()))
        def _ams(_xs):
            return ftt.aggregate_mean_std(_xs) if _xs else (float("nan"), float("nan"))
        realism_data = {lb: {"tv_all": ftt.aggregate_mean_std(_tv_all[lb]),
                             "t5_target": ftt.aggregate_mean_std(_t5_tg[lb]),
                             "t5_all": ftt.aggregate_mean_std(_t5_all[lb]),
                             "tv_var": {v: _ams(_tv_var[lb][v]) for v in INTERF_VARS},
                             "t5_var": {v: _ams(_t5_var[lb][v]) for v in INTERF_VARS},
                             "n_pool": len(_t5_tg[lb])} for lb in _labels}
        t13_chan = {lb: {v: {ch: {ei: float(np.mean(vals)) for ei, vals in byei.items()}
                             for ch, byei in _t13[lb][v].items()} for v in INTERF_VARS} for lb in _labels}
    return realism_data, t13_chan


@app.cell
def _(LEVELS_DICT):
    # constants for the interference subtests: the 6 level variables and their channel order
    # ('sfc' ALWAYS first, then levels bottom->top), like intensity_comparison's intensity_channels.
    INTERF_LEVEL_VARS = ["geopotential", "u_component_of_wind", "v_component_of_wind",
                         "temperature", "specific_humidity", "vertical_velocity"]
    INTERF_VARS = INTERF_LEVEL_VARS + ["mean_sea_level_pressure"]  # + surface-only mslp; T1 score uses V=7
    INTERF_SURFACE_PAIR = {"temperature": "2m_temperature",
                           "u_component_of_wind": "10m_u_component_of_wind",
                           "v_component_of_wind": "10m_v_component_of_wind"}
    INTERF_VSHORT = {"geopotential": "z", "u_component_of_wind": "u", "v_component_of_wind": "v",
                     "temperature": "T", "specific_humidity": "q", "vertical_velocity": "w", "mean_sea_level_pressure": "msl"}


    def interf_chan_order(_var):
        """Channel labels for one variable, surface->top. 'sfc' is ALWAYS the first channel (even
        for variables with no surface pair, e.g. geopotential) so every chart shares one aligned
        x-axis; variables without a surface value simply have a gap (NaN) at sfc."""
        return ["sfc"] + [f"L{_L}" for _L in reversed(LEVELS_DICT["level"])]

    return (
        INTERF_LEVEL_VARS,
        INTERF_SURFACE_PAIR,
        INTERF_VARS,
        interf_chan_order,
    )


@app.cell(hide_code=True)
def interf_display(mode_of):
    # variable display order (temperature first, then the two winds, then z, q, w, mslp) and the
    # short math labels used in every variable-grid title/column ($t$, $u$, ..., $\mathrm{mslp}$).
    VAR_ORDER = ["temperature", "u_component_of_wind", "v_component_of_wind", "geopotential",
                 "specific_humidity", "vertical_velocity", "mean_sea_level_pressure"]
    INTERF_SHORT = {"geopotential": "$z$", "u_component_of_wind": "$u$", "v_component_of_wind": "$v$",
                    "temperature": "$t$", "specific_humidity": "$q$", "vertical_velocity": "$w$",
                    "mean_sea_level_pressure": r"$\mathrm{mslp}$"}


    def pmode(_lb):
        # plain-text schedule name for selectors (spike@1 -> "spike@t=0")
        _b, _s, _r = mode_of(_lb).partition("@")
        if not _s:
            return mode_of(_lb)
        if "-" in _r:
            _a, _bb = _r.split("-", 1)
            return f"{_b}@{int(_a) - 1}-{int(_bb) - 1}" if _a.isdigit() and _bb.isdigit() else mode_of(_lb)
        return f"{_b}@{int(_r) - 1}" if _r.isdigit() else mode_of(_lb)

    INTERF_PLAIN = {"geopotential": "z", "u_component_of_wind": "u", "v_component_of_wind": "v",
                    "temperature": "t", "specific_humidity": "q", "vertical_velocity": "w",
                    "mean_sea_level_pressure": "mslp"}
    return INTERF_PLAIN, INTERF_SHORT, VAR_ORDER, pmode


@app.cell
def _(
    EXPS,
    INTERF_LEVEL_VARS,
    INTERF_SURFACE_PAIR,
    INTERF_VARS,
    M,
    N,
    datetime,
    find_era5_input,
    get_gt_rollout,
    get_rollout_dir,
    interf_button,
    load_rollout,
    np,
    open_unguided_state,
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
        _avg = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}
        _kick = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}
        _appk = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}
        _absg = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}   # M(x_gui)
        _absum = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}  # sum_grid |x_gui - x_gui_ung| (global, unmasked)
        _avgabs = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}  # M_mask(|x_gui - x_gui_ung|)  (mask absolute)
        _tvd = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}  # TV over the MASK region: sum|normspatial(|Δ|) - pi~|
        def _tvdist2d(_dfield, _mn, _sup):
            _a = np.abs(_dfield) * _sup; _s = float(_a.sum())   # footprint restricted to the mask region
            return 0.5 * float(np.abs(_a / (_s if _s > 0 else 1.0) - _mn).sum())   # 1/2 -> proper TV distance in [0,1]
        _pushg  = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}  # sum_grid(x_gui - x_gui_ung)   (global signed)
        _msig = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}  # Σ_mask (x_gui - x_gui_ung)
        _mabs = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}  # Σ_mask |x_gui - x_gui_ung|
        _nsig = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}  # Σ_!mask (x_gui - x_gui_ung)
        _nabs = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}  # Σ_!mask |x_gui - x_gui_ung|
        _absu = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}   # M(x_gui_ung)
        _absgt = {lb: {v: {} for v in INTERF_VARS} for lb in _labels}  # M(x_gt) @ valid time n+1
        _meta = {lb: [] for lb in _labels}   # (ei, m) per instance, aligned with the flat per-channel lists
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
            _mbool = (_mask >= 0.5 * _mask.max()).astype(float); _nbool = 1.0 - _mbool  # mask-core (half-max); mask + !mask = grid
            _mnorm = _mask / (_msum if _msum > 0 else 1.0); _msupp = (_mask > 0).astype(float)   # mask region (support) — T1.2 D restricted here
            _sp = sweep_points(_sv, _recs); _pts = {lb: _sp[lb] for lb in _labels if lb in _sp}
            _gui = xr.open_zarr(_dir / "gui.zarr"); _gr = xr.open_zarr(_dir / "grads.zarr")
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
                        _meta[_lb].append((_ei, _m))
                        _lam = _lam_raw_for(_recs, _sel, _m, _n)
                        _lam2 = None if _lam is None else (_lam ** 2)
                        for _var in INTERF_LEVEL_VARS:
                            _levs = list(_gui[_var]["level"].values)
                            _g = np.asarray(select_point(_gui[_var], _sel).isel(m=_m, n=_n), float)      # (level,lat,lon)
                            _ud = select_point(open_unguided_state(_dir, "gui_ung", _var), _sel).isel(m=_m, n=_n)
                            _u = np.asarray(_ud.isel(t=-1) if "t" in _ud.dims else _ud, float)            # (level,lat,lon)
                            _gg = np.asarray(select_point(_gr[_var], _sel).isel(m=_m, n=_n), float)       # (t,level,lat,lon)
                            _ga = (_g * _mask).sum((-2, -1)) / _msum                                       # (level,) M(x_gui)
                            _ua = (_u * _mask).sum((-2, -1)) / _msum                                       # (level,) M(x_gui_ung)
                            _pa = _ga - _ua                                                                # (level,) push
                            _pd = np.abs(_g - _u).sum((-2, -1))                                            # (level,) global sum|diff| over ALL grid points
                            _pdm = (np.abs(_g - _u) * _mask).sum((-2, -1)) / _msum                          # (level,) masked-mean |diff| (mask absolute)
                            _absmask = (np.abs(_g - _u) * _msupp[None]).sum((-2, -1)); _tvd_v = 0.5 * np.abs(np.abs(_g - _u) * _msupp[None] / np.clip(_absmask[:, None, None], 1e-30, None) - _mnorm[None]).sum((-2, -1))
                            _pgs = (_g - _u).sum((-2, -1))                                                  # (level,) global signed sum (global signed)
                            _msig_v = ((_g - _u) * _mbool).sum((-2, -1)); _mabs_v = (np.abs(_g - _u) * _mbool).sum((-2, -1))
                            _nsig_v = ((_g - _u) * _nbool).sum((-2, -1)); _nabs_v = (np.abs(_g - _u) * _nbool).sum((-2, -1))
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
                                _absum[_lb][_var].setdefault(_cl, []).append(float(_pd[_li]))
                                _avgabs[_lb][_var].setdefault(_cl, []).append(float(_pdm[_li]))
                                _tvd[_lb][_var].setdefault(_cl, []).append(float(_tvd_v[_li]))
                                _pushg[_lb][_var].setdefault(_cl, []).append(float(_pgs[_li]))
                                _msig[_lb][_var].setdefault(_cl, []).append(float(_msig_v[_li])); _mabs[_lb][_var].setdefault(_cl, []).append(float(_mabs_v[_li]))
                                _nsig[_lb][_var].setdefault(_cl, []).append(float(_nsig_v[_li])); _nabs[_lb][_var].setdefault(_cl, []).append(float(_nabs_v[_li]))
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
                                _usd = select_point(open_unguided_state(_dir, "gui_ung", _svar), _sel).isel(m=_m, n=_n)
                                _us = np.asarray(_usd.isel(t=-1) if "t" in _usd.dims else _usd, float)
                                _grs = np.asarray(select_point(_gr[_svar], _sel).isel(m=_m, n=_n), float)  # (t,lat,lon)
                                _pts = ((_grs ** 2) * _mask).sum((-2, -1))                                  # (t,)
                                _ga_s = float((_gs * _mask).sum() / _msum)
                                _ua_s = float((_us * _mask).sum() / _msum)
                                _avg[_lb][_var].setdefault("sfc", []).append(_ga_s - _ua_s)
                                _absum[_lb][_var].setdefault("sfc", []).append(float(np.abs(_gs - _us).sum()))
                                _avgabs[_lb][_var].setdefault("sfc", []).append(float((np.abs(_gs - _us) * _mask).sum() / _msum))
                                _tvd[_lb][_var].setdefault("sfc", []).append(_tvdist2d(_gs - _us, _mnorm, _msupp))
                                _pushg[_lb][_var].setdefault("sfc", []).append(float((_gs - _us).sum()))
                                _msig[_lb][_var].setdefault("sfc", []).append(float(((_gs - _us) * _mbool).sum())); _mabs[_lb][_var].setdefault("sfc", []).append(float((np.abs(_gs - _us) * _mbool).sum()))
                                _nsig[_lb][_var].setdefault("sfc", []).append(float(((_gs - _us) * _nbool).sum())); _nabs[_lb][_var].setdefault("sfc", []).append(float((np.abs(_gs - _us) * _nbool).sum()))
                                _absg[_lb][_var].setdefault("sfc", []).append(_ga_s)
                                _absu[_lb][_var].setdefault("sfc", []).append(_ua_s)
                                _kick[_lb][_var].setdefault("sfc", []).append(float(np.sqrt(_pts.sum())))
                                if _have_gt:
                                    _gts = np.asarray(_gtr[_svar].isel(time=_n + 1), float)                # (lat,lon)
                                    _absgt[_lb][_var].setdefault("sfc", []).append(float((_gts * _mask).sum() / _msum))
                                if _lam2 is not None:
                                    _appk[_lb][_var].setdefault("sfc", []).append(float(np.sqrt((_pts * _lam2).sum())))

                        # mslp: surface-only variable -> single 'sfc' channel (no vertical levels)
                        if "mean_sea_level_pressure" in INTERF_VARS:
                            _mv = "mean_sea_level_pressure"
                            try:
                                _gs = np.asarray(select_point(_gui[_mv], _sel).isel(m=_m, n=_n), float)
                                _msd = select_point(open_unguided_state(_dir, "gui_ung", _mv), _sel).isel(m=_m, n=_n)
                                _us = np.asarray(_msd.isel(t=-1) if "t" in _msd.dims else _msd, float)
                                _ga_s = float((_gs * _mask).sum() / _msum); _ua_s = float((_us * _mask).sum() / _msum)
                                _avg[_lb][_mv].setdefault("sfc", []).append(_ga_s - _ua_s)
                                _absum[_lb][_mv].setdefault("sfc", []).append(float(np.abs(_gs - _us).sum()))
                                _avgabs[_lb][_mv].setdefault("sfc", []).append(float((np.abs(_gs - _us) * _mask).sum() / _msum))
                                _tvd[_lb][_mv].setdefault("sfc", []).append(_tvdist2d(_gs - _us, _mnorm, _msupp))
                                _pushg[_lb][_mv].setdefault("sfc", []).append(float((_gs - _us).sum()))
                                _msig[_lb][_mv].setdefault("sfc", []).append(float(((_gs - _us) * _mbool).sum())); _mabs[_lb][_mv].setdefault("sfc", []).append(float((np.abs(_gs - _us) * _mbool).sum()))
                                _nsig[_lb][_mv].setdefault("sfc", []).append(float(((_gs - _us) * _nbool).sum())); _nabs[_lb][_mv].setdefault("sfc", []).append(float((np.abs(_gs - _us) * _nbool).sum()))
                                _absg[_lb][_mv].setdefault("sfc", []).append(_ga_s); _absu[_lb][_mv].setdefault("sfc", []).append(_ua_s)
                                try:
                                    _grs = np.asarray(select_point(_gr[_mv], _sel).isel(m=_m, n=_n), float)
                                    _pts = ((_grs ** 2) * _mask).sum((-2, -1))
                                    _kick[_lb][_mv].setdefault("sfc", []).append(float(np.sqrt(_pts.sum())))
                                    if _lam2 is not None:
                                        _appk[_lb][_mv].setdefault("sfc", []).append(float(np.sqrt((_pts * _lam2).sum())))
                                except Exception:
                                    pass
                                if _have_gt:
                                    _gts = np.asarray(_gtr[_mv].isel(time=_n + 1), float)
                                    _absgt[_lb][_mv].setdefault("sfc", []).append(float((_gts * _mask).sum() / _msum))
                            except Exception:
                                pass

        def _mean_map(_acc):
            return {lb: {v: {cl: float(np.mean(vs)) for cl, vs in _acc[lb][v].items() if vs} for v in INTERF_VARS} for lb in _labels}

        def _pvt(_D, _absv):
            return {lb: {v: float(sum((abs(x) if _absv else x) for x in _D[lb][v].values())) for v in INTERF_VARS} for lb in _labels}

        def _scores(_pv):
            _vmax = {v: (max(_pv[lb][v] for lb in _labels) or 1.0) for v in INTERF_VARS}
            return {lb: float(np.mean([_pv[lb][v] / _vmax[v] for v in INTERF_VARS])) for lb in _labels}

        _AVG = _mean_map(_avg); _KICK = _mean_map(_kick); _APPK = _mean_map(_appk)
        _ABSG = _mean_map(_absg); _ABSU = _mean_map(_absu); _ABSGT = _mean_map(_absgt)
        _ABSUM = _mean_map(_absum)
        _AVGABS = _mean_map(_avgabs); _PUSHG = _mean_map(_pushg)
        _MSIG = _mean_map(_msig); _MABS = _mean_map(_mabs); _NSIG = _mean_map(_nsig); _NABS = _mean_map(_nabs)
        _apv = _pvt(_AVG, True); _kpv = _pvt(_KICK, False); _cpv = _pvt(_APPK, False)
        _dpv = _pvt(_ABSUM, False)   # global push per variable (values already >= 0)
        _amv = _pvt(_AVGABS, False); _gpv = _pvt(_PUSHG, True)
        _msv = _pvt(_MSIG, True); _mav = _pvt(_MABS, False); _nsv = _pvt(_NSIG, True); _nav = _pvt(_NABS, False)
        _npool = max((len(vs) for v in INTERF_VARS for vs in _avg[_labels[0]][v].values()), default=0) if _labels else 0
        # per-sample T2a score + T2d collateral -> (mean, std) over the (exp x m x n) pool. Fixed
        # normalizers (overall vmax / mean 2mT) so the mean matches the T2a/T2d table values.
        _avg_score = _scores(_apv)
        def _nsamp(_l):
            return min((len(vs) for v in INTERF_VARS for vs in _avg[_l][v].values()), default=0)
        def _pv_sample(_acc, _l, _i, _absv):
            return {v: float(sum((abs(_acc[_l][v][cl][_i]) if _absv else _acc[_l][v][cl][_i]) for cl in _acc[_l][v])) for v in INTERF_VARS}
        _avmax = {v: (max(_apv[lb][v] for lb in _labels) or 1.0) for v in INTERF_VARS}
        _avg_score_ms, _collateral_ms = {}, {}
        for _l in _labels:
            _ns = _nsamp(_l)
            _si = [float(np.mean([_pv_sample(_avg, _l, _i, True)[v] / _avmax[v] for v in INTERF_VARS])) for _i in range(_ns)]
            _avg_score_ms[_l] = (_avg_score[_l], float(np.std(_si)) if _si else float("nan"))
            _sfc = _APPK[_l]["temperature"].get("sfc", float("nan"))
            _ci = [sum(_pv_sample(_appk, _l, _i, False).values()) / _sfc - 1.0 for _i in range(_ns)] if (np.isfinite(_sfc) and _sfc != 0.0) else []
            _collateral_ms[_l] = (float(np.mean(_ci)), float(np.std(_ci))) if _ci else (float("nan"), float("nan"))
        _absum_score = _scores(_dpv); _avgabs_score = _scores(_amv); _pushg_score = _scores(_gpv)
        _dvmax = {v: (max(_dpv[lb][v] for lb in _labels) or 1.0) for v in INTERF_VARS}
        _absum_score_ms = {}
        for _l in _labels:
            _dsi = [float(np.mean([_pv_sample(_absum, _l, _i, False)[v] / _dvmax[v] for v in INTERF_VARS])) for _i in range(_nsamp(_l))]
            _absum_score_ms[_l] = (_absum_score[_l], float(np.std(_dsi)) if _dsi else float("nan"))
        def _val_em(_acc, _absv):
            _out = {}
            for _l in _labels:
                _byem = {}
                for _i in range(len(_meta[_l])):
                    _byem.setdefault(_meta[_l][_i], []).append(_pv_sample(_acc, _l, _i, _absv))
                _out[_l] = {_em: {v: float(np.mean([_q[v] for _q in _pvs])) for v in INTERF_VARS}
                            for _em, _pvs in _byem.items()}
            return _out
        _val_em_map = {"absum": _val_em(_absum, False), "mabs": _val_em(_mabs, False),
                       "nabs": _val_em(_nabs, False), "pushg": _val_em(_pushg, True),
                       "msig": _val_em(_msig, True), "nsig": _val_em(_nsig, True),
                       "avgabs": _val_em(_avgabs, False), "tvd": _val_em(_tvd, False)}
        def _chan_em(_acc):
            # per-(ei,m) per-channel value for the T1.1 charts: group the flat per-channel lists by
            # _meta=(ei,m) and mean over the (several) guided steps n -> {lb:{var:{cl:{(ei,m):value}}}}.
            _out = {}
            for _l in _labels:
                _bv = {}
                for _v in INTERF_VARS:
                    _bv[_v] = {}
                    for _cl in _acc[_l][_v]:
                        _g = {}
                        for _i, _val in enumerate(_acc[_l][_v][_cl]):
                            _g.setdefault(_meta[_l][_i], []).append(_val)
                        _bv[_v][_cl] = {_em: float(np.mean(_vs)) for _em, _vs in _g.items()}
                _out[_l] = _bv
            return _out
        _chan_em_map = {"avgabs": _chan_em(_avgabs), "tvd": _chan_em(_tvd), "nabs": _chan_em(_nabs), "absum": _chan_em(_absum)}
        interf_data = {"val_em": _val_em_map, "chan_em": _chan_em_map, "avg": _AVG, "kick": _KICK, "appk": _APPK,
                       "abs_gui": _ABSG, "abs_ung": _ABSU, "abs_gt": _ABSGT, "have_gt": _any_gt,
                       "avg_pvt": _apv, "kick_pvt": _kpv, "appk_pvt": _cpv,
                       "avg_score": _avg_score, "kick_score": _scores(_kpv), "appk_score": _scores(_cpv),
                       "absum": _ABSUM, "absum_pvt": _dpv, "absum_score": _scores(_dpv),
                       "absum_score_ms": _absum_score_ms,
                       "avgabs": _AVGABS, "avgabs_pvt": _amv, "avgabs_score": _avgabs_score,
                       "pushg": _PUSHG, "pushg_pvt": _gpv, "pushg_score": _pushg_score,
                       "msig": _MSIG, "msig_pvt": _msv, "msig_score": _scores(_msv),
                       "mabs": _MABS, "mabs_pvt": _mav, "mabs_score": _scores(_mav),
                       "nsig": _NSIG, "nsig_pvt": _nsv, "nsig_score": _scores(_nsv),
                       "nabs": _NABS, "nabs_pvt": _nav, "nabs_score": _scores(_nav),
                       "avg_score_ms": _avg_score_ms, "collateral_ms": _collateral_ms,
                       "labels": _labels, "n_pool": _npool}
    return (interf_data,)


@app.cell
def _(
    INTERF_SHORT,
    VAR_ORDER,
    gamma_key,
    gname,
    intensity_dropdown,
    interf_chan_order,
    interf_data,
    interf_k_slider,
    label_delta,
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
        _head = [mo.md(_title_md)] if _title_md else []
        if interf_data is None:
            return mo.vstack([*_head, mo.md("_press **compute interference profiles** above_")])
        _D = interf_data[_key]
        _pvt = interf_data[_key + "_pvt"]
        _score = interf_data[_key + "_score"]
        _labels = [_l for _l in interf_data["labels"] if label_delta.get(_l) == intensity_dropdown.value]
        _k = min(int(interf_k_slider.value), len(_labels))
        # ---- ranking (winner first) ----
        if _normalize_by_2mT:
            # collateral per unit 2mT = total_kick/2mT - 1 ; MIN = most targeted / least side-effect
            def _coll(_l):
                _nrm = _D[_l]["temperature"].get("sfc", np.nan)
                if not (np.isfinite(_nrm) and _nrm != 0.0):
                    return float("-inf")   # missing target -> never wins under max
                return sum(_pvt[_l][_v] for _v in VAR_ORDER) / _nrm - 1.0
            _rankval = {_l: _coll(_l) for _l in _labels}
            _ranked = sorted(_labels, key=lambda _l: _rankval[_l], reverse=True)  # descending: max collateral first (higher=better)
        else:
            _rankval = _score
            _ranked = sorted(_labels, key=lambda _l: _score[_l], reverse=not _lower_better)
        _winner = _ranked[0]
        _top = _ranked[:_k]
        # ---- 6-chart grid ----
        _fig, _axs = plt.subplots(2, 4, figsize=(20, 7), dpi=120)
        _slots = [(_r, _c) for _r in range(2) for _c in range(4) if (_r, _c) != (0, 3)]
        for _si, _var in enumerate(VAR_ORDER):
            _ax = _axs[_slots[_si][0]][_slots[_si][1]]
            _cos = interf_chan_order(_var); _xs = list(range(len(_cos)))
            for _lb in sorted(_top, key=gamma_key):
                _ys = [_D[_lb][_var].get(_cl, np.nan) for _cl in _cos]
                if _normalize_by_2mT:
                    _nrm = _D[_lb]["temperature"].get("sfc", np.nan)
                    _ys = [(_y / _nrm) if (np.isfinite(_nrm) and _nrm != 0.0) else np.nan for _y in _ys]
                _ax.plot(_xs, _ys, "-o", ms=3, lw=1.7, color=sched_colors[_lb], label=gname(_lb))
            _ax.axhline(1.0 if _normalize_by_2mT else 0.0, color="#999999", lw=0.6,
                        ls="--" if _normalize_by_2mT else "-")
            _ax.set_xticks(_xs); _ax.set_xticklabels(_cos, fontsize=6, rotation=90)
            _ax.set_xlim(-0.5, len(_cos) - 0.5)   # half-slot pad: sfc is the first tick, off the y-axis
            _ax.set_title(INTERF_SHORT.get(_var, _var), fontsize=9, pad=4)
            _ax.set_axisbelow(True); _ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.7); _ax.tick_params(labelsize=7)
            for _s in ("top", "right"):
                _ax.spines[_s].set_visible(False)
        _lax = _axs[0][3]; _lax.axis("off")
        _h, _l = _axs[_slots[0][0]][_slots[0][1]].get_legend_handles_labels()
        _lax.legend(_h, _l, fontsize=8, loc="upper right", frameon=False)
        _fig.tight_layout()
        _grid = mo.vstack([mo.md(_ylabel), mo.as_html(_fig)], align="start"); plt.close(_fig)
        if not _show_table:
            return mo.vstack([*_head, mo.md(_desc_md), _grid], align="start")
        # ---- T2d collateral table (2mT-normalized) ----
        if _normalize_by_2mT:
            # per-variable kick per unit 2mT (temperature column = its LEVELS only, i.e. the collateral
            # WITHIN temperature; the 2mT reference is 1 by construction and omitted). MIN per column;
            # "collateral" = total non-target kick per unit 2mT; winner = min collateral.
            _cols = []
            for _v in VAR_ORDER:
                _cd = {}
                for _l in _labels:
                    _nrm = _D[_l]["temperature"].get("sfc", np.nan)
                    _raw = (_pvt[_l]["temperature"] - _D[_l]["temperature"].get("sfc", 0.0)) if _v == "temperature" else _pvt[_l][_v]
                    _cd[_l] = (_raw / _nrm) if (np.isfinite(_nrm) and _nrm != 0.0) else float("nan")
                _cols.append((INTERF_SHORT[_v], _cd, "max"))
            _cols.append(("collateral", {_l: _rankval[_l] for _l in _labels}, "max"))
            _best = {_h: max((_v for _v in _vv.values() if np.isfinite(_v)), default=float("nan")) for _h, _vv, _r in _cols}
            _rows = ["| schedule | " + " | ".join(_h for _h, _u1, _u2 in _cols) + " |",
                     "|" + "---|" * (len(_cols) + 1)]
            for _lb in _ranked:
                _cells = [(f"**{_v2[_lb]:.3g}**" if _v2[_lb] == _best[_h] else f"{_v2[_lb]:.3g}") for _h, _v2, _r in _cols]
                _star = "★ " if _lb in _top else ""
                _rows.append(f"| {_star}{gname(_lb)} | " + " | ".join(_cells) + " |")
            return mo.vstack([
                *_head, mo.md(_desc_md), _grid,
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
        for _v in VAR_ORDER:
            if _v == "temperature" and _split_T_sfc:
                _sfc = {_l: abs(_D[_l]["temperature"].get("sfc", float("nan"))) for _l in _labels}
                _tlev = {_l: _pvt[_l]["temperature"] - _sfc[_l] for _l in _labels}
                _cols.append(("$t$", _tlev, _defrule))
                _lead2mT = ("$t_{2m}$", _sfc, None)   # 2mT -> first column, no bold
            else:
                _cols.append((INTERF_SHORT[_v], {_l: _pvt[_l][_v] for _l in _labels}, _defrule))
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
            _rows.append(f"| {_star}{gname(_lb)} | " + " | ".join(_cells) + f" | {_sc} |")
        return mo.vstack([
            *_head,
            mo.md(_desc_md),
            _grid,
            mo.md(f"_best-{_k} schedules (★) drawn, ranked by the variable-normalized push score"
                  + (" (**lower = better**)" if _lower_better else "")
                  + f"; **bold** = per-column {'min' if _lower_better else 'max'}"
                  + (" ($t_{2m}$ = surface temperature, first column, unbolded; $t$ = temperature levels only)" if _split_T_sfc else "")
                  + "; **score** bold = the single winner (rank #1). "
                  f"Per-variable value = sum over that variable's channels; "
                  f"pooled over {interf_data['n_pool']} (exp×m×n) samples._"),
            mo.md("\n".join(_rows)),
        ], align="start")

    return (interf_render,)


@app.cell(hide_code=True)
def _(
    INTERF_SHORT,
    VAR_ORDER,
    export_button,
    gamma_key,
    gname,
    intensity_dropdown,
    interf_chan_order,
    interf_data,
    interf_k_slider,
    label_delta,
    mo,
    np,
    plt,
    save_chart,
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
        _labels = [_l for _l in interf_data["labels"] if label_delta.get(_l) == intensity_dropdown.value]; _score = interf_data["avg_score"]
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

        _fig, _axs = plt.subplots(2, 4, figsize=(20, 7), dpi=120)
        _slots = [(_r, _c) for _r in range(2) for _c in range(4) if (_r, _c) != (0, 3)]
        for _si, _var in enumerate(VAR_ORDER):
            _ax = _axs[_slots[_si][0]][_slots[_si][1]]
            _cos = interf_chan_order(_var); _xs = list(range(len(_cos)))
            _gtref = _refline(_absgt, _var, _cos) if _have_gt else [np.nan] * len(_cos)
            for _lb in sorted(_top, key=gamma_key):
                _ys = [_absg[_lb][_var].get(_cl, np.nan) for _cl in _cos]
                if _mode == "vsgt":
                    _ys = [(_y - _gt) if (np.isfinite(_y) and np.isfinite(_gt)) else np.nan
                           for _y, _gt in zip(_ys, _gtref)]
                _ax.plot(_xs, _ys, "-o", ms=3, lw=1.7, color=sched_colors[_lb], label=gname(_lb))
            # unguided-twin reference line
            _uref = _refline(_absu, _var, _cos)
            if _mode == "vsgt":
                _uref = [(_uu - _gt) if (np.isfinite(_uu) and np.isfinite(_gt)) else np.nan
                         for _uu, _gt in zip(_uref, _gtref)]
            _ax.plot(_xs, _uref, "--", color="#555555", lw=1.5, label="unguided (ung|gui)")
            # ground-truth reference
            if _mode == "abs":
                if _have_gt:
                    _ax.plot(_xs, _gtref, "--", color="#009E73", lw=1.5, label="ground truth (gt)")
            else:
                _ax.axhline(0.0, color="#009E73", lw=1.5, ls="--", label="ground truth (gt)")
            _ax.set_xticks(_xs); _ax.set_xticklabels(_cos, fontsize=6, rotation=90)
            _ax.set_xlim(-0.5, len(_cos) - 0.5)
            _ax.set_title(INTERF_SHORT.get(_var, _var), fontsize=9, pad=4)
            _ax.set_axisbelow(True); _ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.7); _ax.tick_params(labelsize=7)
            _ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            for _s in ("top", "right"):
                _ax.spines[_s].set_visible(False)
        _lax = _axs[0][3]; _lax.axis("off")
        _h, _l = _axs[_slots[0][0]][_slots[0][1]].get_legend_handles_labels()
        _lax.legend(_h, _l, fontsize=8, loc="upper right", frameon=False)
        _fig.tight_layout()
        if export_button.value:
            save_chart(_fig, "T04" if _mode == "abs" else "T05")
        _grid = mo.vstack([mo.md(_ylabel), mo.as_html(_fig)], align="start"); plt.close(_fig)
        return mo.vstack([mo.md(_title_md), mo.md(_desc_md), _grid], align="start")

    return (interf_profile_render,)


@app.cell
def _(EXPS, delta_labels, delta_order, mo):
    # T1.1 experiment selector + eval-region selector (drive the per-experiment chart + table + views).
    _t11_opts = {}
    for _ei in range(len(EXPS)):
        _date = EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]
        for _dd in delta_order:
            _t11_opts[f"{_date} × {delta_labels.get(_dd, _dd)}"] = (_ei, _dd)
    t11_exp = mo.ui.dropdown(options=_t11_opts, value=(next(iter(_t11_opts)) if _t11_opts else None),
                             label="experiment (startdate × ρ): ")
    t11_region = mo.ui.dropdown(options={"mask": "avgabs", "!mask": "nabs", "full": "absum"},
                                value="mask", label="eval region: ")
    return t11_exp, t11_region


@app.cell
def _(MASK0, pca_region_dropdown, region_bool):
    PCA_REGION = region_bool(MASK0, pca_region_dropdown.value)
    PCA_REGION_MODE = pca_region_dropdown.value
    return PCA_REGION, PCA_REGION_MODE


@app.cell(hide_code=True)
def _(BBOX, MASK0, PCA_REGION, PCA_REGION_MODE, np, region_bool):
    # eval region (the renamed selector) applied to T4/T5: image CROP window, a display BOOLEAN
    # (NaN pixels outside the region), a per-pixel metric WEIGHT, and the mask reference cropped to the
    # same window. 'mask' = current tight bbox + mask-weighting; 'globe' = whole grid, uniform; '!mask' =
    # whole grid with the mask footprint blanked, uniform over the complement.
    def region_realism_tv(_g, _u, _maskf, _w):
        # TV between the guidance footprint |gui-ung| and the mask, restricted to the eval region support
        _p = np.abs(np.asarray(_g, float) - np.asarray(_u, float))
        _supp = np.asarray(_w, float) > 0
        _pp = np.where(_supp, _p, 0.0); _qq = np.where(_supp, np.asarray(_maskf, float), 0.0)
        _ps = _pp.sum(); _qs = _qq.sum()
        if not (_ps > 0 and _qs > 0):
            return np.nan
        return float(0.5 * np.abs(_pp / _ps - _qq / _qs).sum())

    if PCA_REGION_MODE == "mask":
        EVAL_CROP = BBOX
        EVAL_IMG_BOOL = None
        EVAL_W = np.asarray(MASK0, float)
    elif PCA_REGION_MODE == "globe":
        EVAL_CROP = (slice(None), slice(None))
        EVAL_IMG_BOOL = None
        EVAL_W = np.ones_like(np.asarray(MASK0, float))
    else:  # !mask
        EVAL_CROP = (slice(None), slice(None))
        EVAL_IMG_BOOL = np.asarray(PCA_REGION, bool)
        EVAL_W = np.asarray(PCA_REGION, float)
    eval_mask_ref = np.asarray(MASK0, float)[EVAL_CROP[0], EVAL_CROP[1]]

    def _zoom_win(_img, _mode, _zoom):
        # crop the display image to a window (size = extent / zoom) centred on the mask
        if not _zoom or int(_zoom) <= 1:
            return _img
        if _mode == "mask":
            _rc, _cc = _img.shape[0] // 2, _img.shape[1] // 2
        else:
            _rows, _cols = np.where(np.asarray(MASK0, float) > 0)
            _rc = int((_rows.min() + _rows.max()) / 2); _cc = int((_cols.min() + _cols.max()) / 2)
        _hh = max(1, _img.shape[0] // (2 * int(_zoom))); _hw = max(1, _img.shape[1] // (2 * int(_zoom)))
        return _img[max(0, _rc - _hh):_rc + _hh, max(0, _cc - _hw):_cc + _hw]

    def region_crop(_full, _mode, _zoom=1):
        # crop + (for !mask) blank the mask region, then optionally zoom into the mask
        _crop = BBOX if _mode == "mask" else (slice(None), slice(None))
        _img = np.asarray(_full, float)[_crop[0], _crop[1]]
        if _mode == "!mask":
            _bl = np.asarray(region_bool(MASK0, "!mask"), bool)[_crop[0], _crop[1]]
            _img = np.where(_bl, _img, np.nan)
        return _zoom_win(_img, _mode, _zoom)

    def region_maskref(_mode, _zoom=1):
        _crop = BBOX if _mode == "mask" else (slice(None), slice(None))
        return _zoom_win(np.asarray(MASK0, float)[_crop[0], _crop[1]], _mode, _zoom)

    return EVAL_W, region_crop, region_maskref, region_realism_tv


@app.cell(hide_code=True)
def _(
    LEVEL,
    PCA_REGION,
    VAR,
    basis,
    project,
    region_cloud_sample,
    support_checkbox,
):
    # PCA-plot background: a projected subsample of the climatology cloud the basis was fit on
    cloud_proj = None if not support_checkbox.value else project(basis, region_cloud_sample(VAR, LEVEL, PCA_REGION, "era5", time_hour=12, max_points=400))
    return (cloud_proj,)


@app.cell(hide_code=True)
def _(INTERF_SURFACE_PAIR, field_level_slider, field_var_dropdown):
    # resolve the T4/T5 field selector (dropdown + level slider) -> (var, partition, level),
    # same mapping as experiment_builder.py (level 0 = surface tick; surface-paired vars map to
    # their 2m/10m twin; a level-only var at the surface tick falls back to its lowest level).
    def _resolve_field(_base, _lvl):
        if _lvl == 0:
            if _base in INTERF_SURFACE_PAIR:
                return INTERF_SURFACE_PAIR[_base], "surface", 0
            if _base == "mean_sea_level_pressure":
                return _base, "surface", 0
            return _base, "level", 50
        if _base == "mean_sea_level_pressure":
            return _base, "surface", 0
        return _base, "level", _lvl

    fvar, fpartition, flevel = _resolve_field(field_var_dropdown.value, int(field_level_slider.value))
    return flevel, fpartition, fvar


@app.cell(hide_code=True)
def _(
    EXPS,
    M,
    N,
    flevel,
    fpartition,
    fvar,
    load_rollout,
    metrics,
    np,
    open_store,
    open_unguided_state,
    sched_colors,
    select_point,
    sweep_points,
):
    # ---- selected-field footprint grid for the T1.2/T1.3 visual images (reactive to the field selector only) ----
    # Stores the FULL (uncropped) footprint per (exp, member, n, schedule); the T1.2/T1.3 visuals crop &
    # normalize to their own eval region independently. gfield = |x_gui - x_gui_ung|; gfield_signed = signed.
    field_grid = None
    if metrics is not None and sched_colors is not None:
        _labels = list(sched_colors)
        _lvl = flevel if fpartition == "level" else None
        field_grid = {}
        for _ei, _rid in enumerate(EXPS):
            _dir, _cfg, _sv, _recs, _mask = load_rollout(_rid)
            _sp = sweep_points(_sv, _recs); _pts = {lb: _sp[lb] for lb in _labels if lb in _sp}
            for _lb, _sel in _pts.items():
                _p = np.asarray(_sv["GUIDANCE_DELTA"][_sel["GUIDANCE_DELTA"]], float)[:N]
                _gd = select_point(open_store(_dir, "gui", fvar), _sel)
                _ud0 = select_point(open_unguided_state(_dir, "gui_ung", fvar), _sel)
                for _n in range(N):
                    if float(_p[_n]) == 0.0:
                        continue
                    for _m in range(M):
                        _g = _gd.isel(m=_m, n=_n)
                        _u = _ud0.isel(m=_m, n=_n); _u = _u.isel(t=-1) if "t" in _u.dims else _u
                        if _lvl is not None:
                            _g = _g.sel(level=_lvl); _u = _u.sel(level=_lvl)
                        _guif = np.asarray(_g, float); _ungf = np.asarray(_u, float)
                        field_grid.setdefault((_ei, _m, _n), {})[_lb] = {
                            "gfield": np.abs(_guif - _ungf), "gfield_signed": _guif - _ungf}
    return (field_grid,)


@app.cell(hide_code=True)
def _(
    EVAL_W,
    EXPS,
    LEVEL,
    PARTITION,
    VAR,
    get_rollout_dir,
    np,
    open_store,
    support_checkbox,
):
    # section 6 -- unguided clean-estimate prediction noise (guidance-independent).
    # zhat_T(t) = z_t + (s_t/h_t)(z_{t+1}-z_t) from ung_res (finite diff of the unmodified unguided Euler step);
    # RMSE vs z_T = res[-1] over the eval region (EVAL_W), per (m, n, t). One (M, N, T) cube per experiment.
    ung_pred_rmse = None
    if support_checkbox.value and EXPS:
        _w = np.asarray(EVAL_W, dtype=float)
        _wsum = float(_w.sum()) or 1.0
        ung_pred_rmse = {}
        for _ei, _rid in enumerate(EXPS):
            _dir = get_rollout_dir(_rid)
            _store = ("ung_res" if (_dir / "ung_res.zarr").exists()
                      else "gui_ung_res" if (_dir / "gui_ung_res.zarr").exists() else None)
            if _store is None:
                continue
            _z = open_store(_dir, _store, VAR)
            if PARTITION == "level" and "level" in _z.dims:
                _z = _z.sel(level=LEVEL)
            _z = _z.transpose("m", "n", "t", "latitude", "longitude")
            _za = np.asarray(_z, dtype=np.float32)                     # (M, N, T+1, lat, lon)
            _T = _za.shape[2] - 1
            _s = np.linspace(1000, 1, _T) / 1000
            _h = np.empty_like(_s); _h[:-1] = _s[:-1] - _s[1:]; _h[-1] = _s[-1]
            _ratio = (_s / _h).astype(np.float32)[None, None, :, None, None]
            _zhat = _za[:, :, :_T] + (_za[:, :, 1:] - _za[:, :, :_T]) * _ratio   # clean estimate zhat_T(t)
            _sq = (_zhat - _za[:, :, -1:]) ** 2
            ung_pred_rmse[_ei] = np.sqrt((_sq * _w[None, None, None]).sum(axis=(-2, -1)) / _wsum)   # (M, N, T)
    return (ung_pred_rmse,)


@app.cell(hide_code=True)
def _(mo):
    refresh_button = mo.ui.run_button(label="refresh")
    return (refresh_button,)


@app.cell(hide_code=True)
def _(M, mo):
    # reliability convergence controls: it now shows ALL 7 fields as a grid (surface, else level 1000),
    # so no field/level selector -- instead pick which member is drawn solid (default m=0) and whether the
    # member spread is drawn as shading.
    reliability_m_slider = mo.ui.slider(steps=list(range(M)), value=0, label="member m: ", show_value=True)
    reliability_shade_checkbox = mo.ui.checkbox(value=True, label="ensemble shading (min-max over m)")
    return reliability_m_slider, reliability_shade_checkbox


@app.cell(hide_code=True)
def wf_sel(gamma_key, mo, mode_of, plot_labels):
    # subsection-3 (landings) schedule selector: pick which schedule's convergence decomposition is drawn.
    def _wf_plain(_lb):
        _b, _s, _r = mode_of(_lb).partition("@")
        if not _s:
            return mode_of(_lb)
        if "-" in _r:
            _a, _bb = _r.split("-", 1)
            return f"{_b}@{int(_a) - 1}-{int(_bb) - 1}" if _a.isdigit() and _bb.isdigit() else mode_of(_lb)
        return f"{_b}@{int(_r) - 1}" if _r.isdigit() else mode_of(_lb)

    _wf_map = {_wf_plain(_lb): _lb for _lb in sorted(plot_labels, key=gamma_key)}
    reliability_wf_sched = mo.ui.dropdown(options=_wf_map, value=(next(iter(_wf_map)) if _wf_map else None), label="schedule γ: ")
    return (reliability_wf_sched,)


@app.cell(hide_code=True)
def effect_sel(INTERF_PLAIN, VAR_ORDER, gamma_key, mo, plot_labels, pmode):
    # §1.4 mode picker: which schedules' guidance-effect-over-t filmstrips to draw (default first two).
    _eopts = {pmode(_lb): _lb for _lb in sorted(plot_labels, key=gamma_key)}
    reliability_effect_modes = mo.ui.multiselect(options=_eopts, value=list(_eopts)[:2], label="modes: ")
    _vopts = {INTERF_PLAIN[_v]: _v for _v in VAR_ORDER}
    reliability_effect_vars = mo.ui.multiselect(options=_vopts, value=["t"], label="variables: ")
    return reliability_effect_modes, reliability_effect_vars


@app.cell(hide_code=True)
def _(
    EXPS,
    INTERF_SURFACE_PAIR,
    LEVEL,
    M,
    PARTITION,
    VAR,
    base_pin,
    convergence_state_line,
    get_masked_mean,
    load_rollout,
    metrics,
    n_slider,
    np,
    open_unguided_state,
    pinned_records,
    plot_labels,
    residual_scaler,
    select_point,
    sweep_points,
    traj_grid,
):
    # reliability convergence data for ALL 7 fields (surface value, else level 1000), keyed by
    # (ei, m, lb, var). Target field (guided VAR) -> reuse the precomputed gap xi=M(x_t)-y* from
    # traj_grid; every other field -> M(x_t^v)-M(twin^v_final). Also stores the schedule-independent
    # ung|gui twin convergence line reliability_twin[(ei, m, var)] (rel to the same panel reference).
    reliability_vars = ["geopotential", "u_component_of_wind", "v_component_of_wind", "temperature",
                        "specific_humidity", "vertical_velocity", "mean_sea_level_pressure"]

    def _rel_field2(_base):
        if _base in INTERF_SURFACE_PAIR:
            return INTERF_SURFACE_PAIR[_base], "surface", 0
        if _base == "mean_sea_level_pressure":
            return _base, "surface", 0
        return _base, "level", 1000

    def _rel_short(_base, _rp, _rl):
        _s = {"geopotential": "z", "u_component_of_wind": "u", "v_component_of_wind": "v",
              "temperature": "t", "specific_humidity": "q", "vertical_velocity": "w",
              "mean_sea_level_pressure": "msl"}.get(_base, _base)
        if _base == "mean_sea_level_pressure":
            return r"$\mathrm{mslp}$"
        if _rp == "surface":
            _h = "2m" if _base == "temperature" else "10m"
            return rf"${_s}_{{{_h}}}$"
        return rf"${_s}_{{{_rl}}}$"

    if metrics is None or traj_grid is None:
        reliability_states = None; reliability_twin = {}; reliability_land = {}; reliability_var_meta = {}
    else:
        _rn = int(n_slider.value) - 1
        reliability_states = {}; reliability_twin = {}; reliability_land = {}; reliability_var_meta = {}
        for _b in reliability_vars:
            _rv, _rp, _rl = _rel_field2(_b)
            _is_t = (_rv == VAR and _rp == PARTITION and (_rp == "surface" or _rl == LEVEL))
            reliability_var_meta[_b] = (_rel_short(_b, _rp, _rl), _is_t)
        for _ei, _rid in enumerate(EXPS):
            _rdir, _rcfg, _rsv, _rrecs, _rmask = load_rollout(_rid)
            _rsp = sweep_points(_rsv, pinned_records(_rrecs, base_pin))
            _sel0 = _rsp.get(plot_labels[0]) if plot_labels else None
            if _sel0 is None and _rsp:
                _sel0 = next(iter(_rsp.values()))
            for _b in reliability_vars:
                _rv, _rp, _rl = _rel_field2(_b); _lev = _rl if _rp == "level" else None
                _is_t = reliability_var_meta[_b][1]
                # twin trajectory (schedule-independent): masked-mean per flow step, per member
                _twmm = {}
                if _sel0 is not None:
                    try:
                        _rtw = open_unguided_state(_rdir, "gui_ung", _rv)
                        _rtw = _rtw.sel(level=_lev) if (_lev is not None and "level" in _rtw.dims) else _rtw
                        _rtwp0 = select_point(_rtw, _sel0)
                        for _m in range(M):
                            _arr = np.asarray(_rtwp0.isel(m=_m, n=_rn), float)
                            _twmm[_m] = np.array([get_masked_mean(_arr[_t], _rmask) for _t in range(_arr.shape[0])])
                    except Exception:
                        _twmm = {}
                if _is_t:
                    for (_e2, _m, _nn), _cell in traj_grid.items():
                        if _e2 != _ei or _nn != _rn:
                            continue
                        for _lb, _d in _cell.items():
                            if "states" in _d:
                                _st = np.asarray(_d["states"], float)
                                reliability_states[(_ei, _m, _lb, _b)] = _st
                                reliability_land[(_ei, _m, _lb, _b)] = np.asarray(_d.get("land_ung", []), float)
                                if _m in _twmm and (_ei, _m, _b) not in reliability_twin:
                                    reliability_twin[(_ei, _m, _b)] = _twmm[_m] - (_twmm[_m][0] - _st[0])
                    continue
                for _m in range(M):
                    if _m in _twmm:
                        reliability_twin[(_ei, _m, _b)] = _twmm[_m] - _twmm[_m][-1]
                _rc = residual_scaler(_rp, _rv, _rl)
                for _lb in plot_labels:
                    if _lb not in _rsp:
                        continue
                    _rsel = _rsp[_lb]
                    for _m in range(M):
                        if _m not in _twmm:
                            continue
                        try:
                            _rst, _rland = convergence_state_line(_rdir, _rsel, _m, _rn, _rv, _rc, _rmask, float(_twmm[_m][-1]), level=_lev)
                            reliability_states[(_ei, _m, _lb, _b)] = np.asarray(_rst, float)
                            reliability_land[(_ei, _m, _lb, _b)] = np.asarray(_rland, float)
                        except Exception:
                            continue
    return (
        reliability_land,
        reliability_states,
        reliability_twin,
        reliability_var_meta,
    )


@app.cell(hide_code=True)
def _(interf_data, mo, t11_region, t1x_views):
    report_propagation = []
    if interf_data is None or "val_em" not in interf_data:
        interf_score_tables = mo.md("_press **compute interference profiles** above_")
    else:
        _rn = {"avgabs": "mask", "nabs": "!mask", "absum": "full"}.get(t11_region.value, t11_region.value)
        interf_score_tables = t1x_views(t11_region.value, report_propagation, f"propagation · {_rn}")
    return interf_score_tables, report_propagation


@app.cell(hide_code=True)
def _():
    def mode_math(_mode):
        # gamma-mode display in flow-step notation: spike@1 -> $\mathrm{spike}@t_0$;
        # spread@1-3 -> $\mathrm{spread}@t_0-t_2$ (the @k index is the 1-based flow step -> t_{k-1}).
        _base, _sep, _rest = str(_mode).partition("@")
        if not _sep:
            return str(_mode)
        if "-" in _rest:
            _a, _b = _rest.split("-", 1)
            if _a.isdigit() and _b.isdigit():
                return rf"$\mathrm{{{_base}@{int(_a)-1}\text{{-}}{int(_b)-1}}}$"
            return str(_mode)
        return rf"$\mathrm{{{_base}@{int(_rest)-1}}}$" if _rest.isdigit() else str(_mode)


    def mode_of(_lb):
        """Extract the a_t_mode (gamma profile) value from a full sweep label -> 'spike@1' etc."""
        for _tok in str(_lb).split():
            if _tok.startswith("a_t_mode="):
                return _tok.split("=", 1)[1]
        return str(_lb)


    def gname(_lb):
        """Canonical schedule display name in report flow-step notation (spike@1 -> spike@t_0)."""
        return mode_math(mode_of(_lb))


    def gamma_key(_lb):
        """Canonical schedule order used throughout: other < spike < spread, then by @index."""
        _base, _sep, _rest = mode_of(_lb).partition("@")
        _cat = {"spike": 1, "spread": 2}.get(_base, 0)
        _first = _rest.split("-", 1)[0] if _sep else ""
        return (_cat, int(_first) if _first.isdigit() else 0, str(_lb))

    return gamma_key, gname, mode_of


@app.cell
def copy_helper(mo):
    # copy-to-clipboard helpers for the markdown tables. copyable_table(md) renders the table with a
    # small "copy markdown" button above it (an mo.iframe running execCommand/clipboard on click).
    def copyable(md_str, label="📋 copy markdown"):
        import html as _html
        _esc = _html.escape(str(md_str))
        _doc = (
            '<!doctype html><meta charset="utf-8"><body style="margin:0">'
            '<textarea id="s" readonly style="position:fixed;left:-9999px;top:0">' + _esc + '</textarea>'
            '<button id="b" style="font:12px system-ui,sans-serif;padding:2px 9px;cursor:pointer;'
            'border:1px solid #bbb;border-radius:5px;background:#f6f6f6">' + label + '</button>'
            '<script>var b=document.getElementById("b");b.onclick=function(){'
            'var t=document.getElementById("s");t.focus();t.select();t.setSelectionRange(0,999999);'
            'var ok=false;try{ok=document.execCommand("copy");}catch(e){}'
            'if(navigator.clipboard){navigator.clipboard.writeText(t.value).catch(function(){});}'
            'var o=b.textContent;b.textContent="✅ copied";setTimeout(function(){b.textContent=o;},1200);};'
            '</script></body>'
        )
        return mo.iframe(_doc, height="28px")

    def copyable_table(md_str, title=None, into=None):
        _t = str(title).strip() if title is not None else ""
        _full = (_t + "\n\n" + str(md_str)) if _t else str(md_str)
        if into is not None:
            into.append(_full)
        _parts = ([mo.md(_t)] if _t else []) + [copyable(_full), mo.md(str(md_str))]
        return mo.vstack(_parts, align="start")

    return copyable, copyable_table


@app.cell(hide_code=True)
def t1x_helpers(
    EXPS,
    INTERF_SHORT,
    INTERF_VARS,
    M,
    VAR_ORDER,
    copyable_table,
    delta_labels,
    delta_order,
    export_button,
    gamma_key,
    gname,
    interf_chan_order,
    interf_data,
    label_delta,
    mo,
    np,
    plt,
    save_chart,
    sched_colors,
):
    # Shared T1.1 / T1.2 renderers — identical eval procedure, parameterized by the per-channel object key
    # ("avgabs" for T1.1 propagation, "tvd" for T1.2 spatial deviation). A = Σ_levels object; N = A/max_γ A
    # (per member); R = Σ_v N / max_γ Σ_v N. All spreads are population std; tables bold the row-max γ.
    def t1x_glabels(dd):
        return sorted((l for l in interf_data["labels"] if label_delta.get(l) == dd), key=gamma_key)

    def t1x_chart(obj_key, ei, dd, ylabel_md, _savename=None):
        ce = interf_data["chan_em"][obj_key]
        labs = t1x_glabels(dd)
        en = f"{EXPS[ei].split(chr(47))[-1].split(chr(95))[0]} × {delta_labels.get(dd, dd)}"
        fig, axs = plt.subplots(2, 4, figsize=(20, 7), dpi=120)
        slots = [(r, c) for r in range(2) for c in range(4) if (r, c) != (0, 3)]
        for si, var in enumerate(VAR_ORDER):
            ax = axs[slots[si][0]][slots[si][1]]
            cos = interf_chan_order(var); xs = list(range(len(cos)))
            for lb in labs:
                y0, ylo, yhi = [], [], []
                for cl in cos:
                    d = ce.get(lb, {}).get(var, {}).get(cl, {})
                    vals = [d[(ei, mm)] for mm in range(M) if (ei, mm) in d]
                    y0.append(d.get((ei, 0), float("nan")))
                    ylo.append(min(vals) if vals else float("nan")); yhi.append(max(vals) if vals else float("nan"))
                col = sched_colors[lb]
                ax.fill_between(xs, ylo, yhi, color=col, alpha=0.15, linewidth=0)
                ax.plot(xs, y0, "-o", ms=3, lw=1.7, color=col, label=gname(lb))
            ax.set_xticks(xs); ax.set_xticklabels(cos, fontsize=6, rotation=90)
            ax.set_xlim(-0.5, len(cos) - 0.5)
            ax.set_title(INTERF_SHORT.get(var, var), fontsize=9, pad=4)
            ax.set_axisbelow(True); ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.7); ax.tick_params(labelsize=7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        lax = axs[0][3]; lax.axis("off")
        hh, ll = axs[slots[0][0]][slots[0][1]].get_legend_handles_labels()
        lax.legend(hh, ll, fontsize=8, loc="upper right", frameon=False)
        fig.tight_layout()
        out = mo.vstack([mo.md(f"**{en}** — {ylabel_md}"), mo.as_html(fig)], align="start")
        if export_button.value and _savename:
            save_chart(fig, _savename)
        plt.close(fig)
        return out

    def t1x_NR(obj_key, ei, dd):
        ve = interf_data["val_em"][obj_key]; labs = t1x_glabels(dd); Vv = list(INTERF_VARS)
        N = {}; R = {}
        for mm in range(M):
            A = {lb: {v: ve[lb].get((ei, mm), {}).get(v, float("nan")) for v in Vv} for lb in labs}
            vmx = {v: (max((A[lb][v] for lb in labs if np.isfinite(A[lb][v])), default=1.0) or 1.0) for v in Vv}
            N[mm] = {lb: {v: A[lb][v] / vmx[v] for v in Vv} for lb in labs}
            sums = {lb: sum(N[mm][lb][v] for v in Vv) for lb in labs}
            mx = max((s for s in sums.values() if np.isfinite(s)), default=1.0) or 1.0
            R[mm] = {lb: sums[lb] / mx for lb in labs}
        return N, R

    def t1x_pertable(obj_key, ei, dd, caption):
        labs = t1x_glabels(dd); Vv = list(INTERF_VARS)
        N, R = t1x_NR(obj_key, ei, dd)
        rows = ["| variable | " + " | ".join(gname(lb) for lb in labs) + " |", "|" + "---|" * (len(labs) + 1)]
        for v in [x for x in VAR_ORDER if x in Vv]:
            mus = {lb: float(np.mean([N[mm][lb][v] for mm in range(M)])) for lb in labs}
            best = max(mus.values(), default=None)
            cells = []
            for lb in labs:
                sd = float(np.std([N[mm][lb][v] for mm in range(M)]))
                t = f"{mus[lb]:.3f}±{sd:.3f}"
                cells.append(f"**{t}**" if (best is not None and mus[lb] == best) else t)
            rows.append(f"| {INTERF_SHORT[v]} | " + " | ".join(cells) + " |")
        smus = {lb: float(np.mean([R[mm][lb] for mm in range(M)])) for lb in labs}
        sbest = max(smus.values(), default=None)
        scells = []
        for lb in labs:
            sd = float(np.std([R[mm][lb] for mm in range(M)]))
            t = f"{smus[lb]:.3f}±{sd:.3f}"
            scells.append(f"**{t}**" if (sbest is not None and smus[lb] == sbest) else t)
        rows.append("| **single score** $R$ | " + " | ".join(scells) + " |")
        return copyable_table("\n".join(rows), caption)

    def t1x_views(obj_key, report_list, label):
        ve = interf_data["val_em"][obj_key]; Vv = list(INTERF_VARS); labels = list(interf_data["labels"]); nei = len(EXPS)
        by_g = {}
        for lb in labels:
            by_g.setdefault(gname(lb), {})[label_delta.get(lb)] = lb
        gammas = sorted(by_g, key=lambda g: gamma_key(next(iter(by_g[g].values()))))
        deltas = [d for d in delta_order if any(label_delta.get(l) == d for l in labels)]
        N_all = {}; R_all = {}
        for dd in deltas:
            dlabs = [l for l in labels if label_delta.get(l) == dd]
            for ei in range(nei):
                for mm in range(M):
                    A = {l: {v: ve[l].get((ei, mm), {}).get(v, float("nan")) for v in Vv} for l in dlabs}
                    vmx = {v: (max((A[l][v] for l in dlabs if np.isfinite(A[l][v])), default=1.0) or 1.0) for v in Vv}
                    Nn = {l: {v: A[l][v] / vmx[v] for v in Vv} for l in dlabs}
                    sums = {l: sum(Nn[l][v] for v in Vv) for l in dlabs}
                    mx = max((s for s in sums.values() if np.isfinite(s)), default=1.0) or 1.0
                    N_all[(ei, dd, mm)] = Nn
                    R_all[(ei, dd, mm)] = {l: sums[l] / mx for l in dlabs}
        def cellfmt(vals, best):
            vv = [x for x in vals if np.isfinite(x)]
            if not vv:
                return "—"
            mu = float(np.mean(vv)); sd = float(np.std(vv))
            t = f"{mu:.3f}±{sd:.3f}"
            return f"**{t}**" if (best is not None and mu == best) else t
        def colmean(vals):
            vv = [x for x in vals if np.isfinite(x)]
            return float(np.mean(vv)) if vv else None
        def v1(ddset, title):
            rows = ["| startdate | " + " | ".join(g for g in gammas) + " |", "|" + "---|" * (len(gammas) + 1)]
            for ei in range(nei):
                rv = {}
                for g in gammas:
                    acc = []
                    for dd in ddset:
                        lb = by_g[g].get(dd)
                        if lb is None:
                            continue
                        acc += [R_all[(ei, dd, mm)][lb] for mm in range(M) if lb in R_all[(ei, dd, mm)]]
                    rv[g] = acc
                best = max((m for m in (colmean(rv[g]) for g in gammas) if m is not None), default=None)
                sd = EXPS[ei].split(chr(47))[-1].split(chr(95))[0]
                rows.append(f"| {sd} | " + " | ".join(cellfmt(rv[g], best) for g in gammas) + " |")
            return copyable_table("\n".join(rows), title, into=report_list)
        def v2(ddset, title):
            rows = ["| variable | " + " | ".join(g for g in gammas) + " |", "|" + "---|" * (len(gammas) + 1)]
            for v in [x for x in VAR_ORDER if x in Vv]:
                rv = {}
                for g in gammas:
                    acc = []
                    for dd in ddset:
                        lb = by_g[g].get(dd)
                        if lb is None:
                            continue
                        for ei in range(nei):
                            acc += [N_all[(ei, dd, mm)][lb][v] for mm in range(M) if lb in N_all[(ei, dd, mm)]]
                    rv[g] = acc
                best = max((m for m in (colmean(rv[g]) for g in gammas) if m is not None), default=None)
                rows.append(f"| {INTERF_SHORT[v]} | " + " | ".join(cellfmt(rv[g], best) for g in gammas) + " |")
            return copyable_table("\n".join(rows), title, into=report_list)
        blocks = []
        for dd in deltas:
            rho = delta_labels.get(dd, f"δ{dd}")
            blocks.append(mo.hstack([
                v1([dd], f"**{rho} — V1 single score $R$ ({label})** per startdate (rows) × γ; mean ± std over the $M={M}$ members."),
                v2([dd], f"**{rho} — V2 per-variable $N_v$ ({label})** (rows) × γ; mean ± std over startdates × members ({nei}×{M})."),
            ], justify="start", align="start"))
        blocks.append(mo.hstack([
            v1(deltas, f"**across ρ — V1 single score $R$ ({label})** per startdate (rows) × γ; mean ± std over ρ × members ({len(deltas)}×{M})."),
            v2(deltas, f"**across ρ — V2 per-variable $N_v$ ({label})** (rows) × γ; mean ± std over startdates × ρ × members ({nei}×{len(deltas)}×{M})."),
        ], justify="start", align="start"))
        return mo.vstack(blocks, align="start")

    return t1x_NR, t1x_chart, t1x_pertable, t1x_views


@app.cell(hide_code=True)
def t12_exp_cell(EXPS, delta_labels, delta_order, mo):
    # T1.2 experiment selector (independent of T1.1's): one of the E = startdates × ρ experiments.
    _t12_opts = {}
    for _ei in range(len(EXPS)):
        _date = EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]
        for _dd in delta_order:
            _t12_opts[f"{_date} × {delta_labels.get(_dd, _dd)}"] = (_ei, _dd)
    t12_exp = mo.ui.dropdown(options=_t12_opts, value=(next(iter(_t12_opts)) if _t12_opts else None),
                             label="experiment (startdate × ρ): ")
    return (t12_exp,)


@app.cell(hide_code=True)
def t12_views(interf_data, mo, t1x_views):
    # T1.2 — paired summary views (V1 single score, V2 per-variable) over object D (tvd).
    report_realism = []
    if interf_data is None or "val_em" not in interf_data or "tvd" not in interf_data["val_em"]:
        interf_score_tables_t12 = mo.md("_press **compute interference profiles** above_")
    else:
        interf_score_tables_t12 = t1x_views("tvd", report_realism, "spatial deviation")
    return interf_score_tables_t12, report_realism


@app.cell(hide_code=True)
def t13_helpers(
    EXPS,
    INTERF_SHORT,
    INTERF_VARS,
    VAR_ORDER,
    copyable_table,
    delta_labels,
    delta_order,
    export_button,
    gamma_key,
    gname,
    interf_chan_order,
    interf_data,
    label_delta,
    mo,
    np,
    plt,
    save_chart,
    sched_colors,
    t13_chan,
):
    # T1.3 renderers — same eval procedure as T1.1/T1.2 but the object (signed ensemble std, t13_chan) has
    # NO member axis: charts have no min-max band and the per-experiment table has a single value per cell.
    def t13_glabels(dd):
        return sorted((l for l in interf_data["labels"] if label_delta.get(l) == dd), key=gamma_key)

    def t13_A(lb, v, ei):
        chs = t13_chan.get(lb, {}).get(v, {})
        vals = [chs[ch][ei] for ch in chs if ei in chs[ch]]
        return float(sum(vals)) if vals else float("nan")

    def t13_NR(ei, dd):
        labs = t13_glabels(dd); Vv = list(INTERF_VARS)
        A = {lb: {v: t13_A(lb, v, ei) for v in Vv} for lb in labs}
        vmx = {v: (max((A[lb][v] for lb in labs if np.isfinite(A[lb][v])), default=1.0) or 1.0) for v in Vv}
        N = {lb: {v: A[lb][v] / vmx[v] for v in Vv} for lb in labs}
        sums = {lb: sum(N[lb][v] for v in Vv) for lb in labs}
        mx = max((s for s in sums.values() if np.isfinite(s)), default=1.0) or 1.0
        R = {lb: sums[lb] / mx for lb in labs}
        return N, R

    def t13_chart(ei, dd, ylabel_md, _savename=None):
        labs = t13_glabels(dd)
        en = f"{EXPS[ei].split(chr(47))[-1].split(chr(95))[0]} × {delta_labels.get(dd, dd)}"
        fig, axs = plt.subplots(2, 4, figsize=(20, 7), dpi=120)
        slots = [(r, c) for r in range(2) for c in range(4) if (r, c) != (0, 3)]
        for si, var in enumerate(VAR_ORDER):
            ax = axs[slots[si][0]][slots[si][1]]
            cos = interf_chan_order(var); xs = list(range(len(cos)))
            for lb in labs:
                ys = [t13_chan.get(lb, {}).get(var, {}).get((cl if cl == "sfc" else int(cl[1:])), {}).get(ei, float("nan")) for cl in cos]
                ax.plot(xs, ys, "-o", ms=3, lw=1.7, color=sched_colors[lb], label=gname(lb))
            ax.set_xticks(xs); ax.set_xticklabels(cos, fontsize=6, rotation=90)
            ax.set_xlim(-0.5, len(cos) - 0.5)
            ax.set_title(INTERF_SHORT.get(var, var), fontsize=9, pad=4)
            ax.set_axisbelow(True); ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.7); ax.tick_params(labelsize=7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        lax = axs[0][3]; lax.axis("off")
        hh, ll = axs[slots[0][0]][slots[0][1]].get_legend_handles_labels()
        lax.legend(hh, ll, fontsize=8, loc="upper right", frameon=False)
        fig.tight_layout()
        out = mo.vstack([mo.md(f"**{en}** — {ylabel_md}"), mo.as_html(fig)], align="start")
        if export_button.value and _savename:
            save_chart(fig, _savename)
        plt.close(fig)
        return out

    def t13_pertable(ei, dd, caption):
        labs = t13_glabels(dd); Vv = list(INTERF_VARS)
        N, R = t13_NR(ei, dd)
        def fmt(x, best):
            if not np.isfinite(x):
                return "—"
            return f"**{x:.3f}**" if (best is not None and x == best) else f"{x:.3f}"
        rows = ["| variable | " + " | ".join(gname(lb) for lb in labs) + " |", "|" + "---|" * (len(labs) + 1)]
        for v in [x for x in VAR_ORDER if x in Vv]:
            vals = {lb: N[lb][v] for lb in labs}
            best = max((x for x in vals.values() if np.isfinite(x)), default=None)
            rows.append(f"| {INTERF_SHORT[v]} | " + " | ".join(fmt(vals[lb], best) for lb in labs) + " |")
        sv = {lb: R[lb] for lb in labs}
        sb = max((x for x in sv.values() if np.isfinite(x)), default=None)
        rows.append("| **single score** $R$ | " + " | ".join(fmt(sv[lb], sb) for lb in labs) + " |")
        return copyable_table("\n".join(rows), caption)

    def t13_views(report_list, label):
        Vv = list(INTERF_VARS); labels = list(interf_data["labels"]); nei = len(EXPS)
        by_g = {}
        for lb in labels:
            by_g.setdefault(gname(lb), {})[label_delta.get(lb)] = lb
        gammas = sorted(by_g, key=lambda g: gamma_key(next(iter(by_g[g].values()))))
        deltas = [d for d in delta_order if any(label_delta.get(l) == d for l in labels)]
        N_all = {}; R_all = {}
        for dd in deltas:
            for ei in range(nei):
                N, R = t13_NR(ei, dd)
                N_all[(ei, dd)] = N; R_all[(ei, dd)] = R
        def cellfmt(vals, best):
            vv = [x for x in vals if np.isfinite(x)]
            if not vv:
                return "—"
            mu = float(np.mean(vv))
            t = f"{mu:.3f}" if len(vv) == 1 else f"{mu:.3f}±{float(np.std(vv)):.3f}"
            return f"**{t}**" if (best is not None and mu == best) else t
        def colmean(vals):
            vv = [x for x in vals if np.isfinite(x)]
            return float(np.mean(vv)) if vv else None
        def v1(ddset, title):
            rows = ["| startdate | " + " | ".join(g for g in gammas) + " |", "|" + "---|" * (len(gammas) + 1)]
            for ei in range(nei):
                rv = {}
                for g in gammas:
                    acc = []
                    for dd in ddset:
                        lb = by_g[g].get(dd)
                        if lb is not None and lb in R_all.get((ei, dd), {}):
                            acc.append(R_all[(ei, dd)][lb])
                    rv[g] = acc
                best = max((m for m in (colmean(rv[g]) for g in gammas) if m is not None), default=None)
                sd = EXPS[ei].split(chr(47))[-1].split(chr(95))[0]
                rows.append(f"| {sd} | " + " | ".join(cellfmt(rv[g], best) for g in gammas) + " |")
            return copyable_table("\n".join(rows), title, into=report_list)
        def v2(ddset, title):
            rows = ["| variable | " + " | ".join(g for g in gammas) + " |", "|" + "---|" * (len(gammas) + 1)]
            for v in [x for x in VAR_ORDER if x in Vv]:
                rv = {}
                for g in gammas:
                    acc = []
                    for dd in ddset:
                        lb = by_g[g].get(dd)
                        if lb is None:
                            continue
                        for ei in range(nei):
                            val = N_all.get((ei, dd), {}).get(lb, {}).get(v)
                            if val is not None and np.isfinite(val):
                                acc.append(val)
                    rv[g] = acc
                best = max((m for m in (colmean(rv[g]) for g in gammas) if m is not None), default=None)
                rows.append(f"| {INTERF_SHORT[v]} | " + " | ".join(cellfmt(rv[g], best) for g in gammas) + " |")
            return copyable_table("\n".join(rows), title, into=report_list)
        blocks = []
        for dd in deltas:
            rho = delta_labels.get(dd, f"δ{dd}")
            blocks.append(mo.hstack([
                v1([dd], f"**{rho} — V1 single score $R$ ({label})** per startdate (rows) × γ (one value; std consumed by the ensemble std)."),
                v2([dd], f"**{rho} — V2 per-variable $N_v$ ({label})** (rows) × γ; mean ± std over startdates ({nei})."),
            ], justify="start", align="start"))
        blocks.append(mo.hstack([
            v1(deltas, f"**across ρ — V1 single score $R$ ({label})** per startdate (rows) × γ; mean ± std over ρ ({len(deltas)})."),
            v2(deltas, f"**across ρ — V2 per-variable $N_v$ ({label})** (rows) × γ; mean ± std over startdates × ρ ({nei}×{len(deltas)})."),
        ], justify="start", align="start"))
        return mo.vstack(blocks, align="start")

    return t13_NR, t13_chart, t13_pertable, t13_views


@app.cell(hide_code=True)
def t13_exp_cell(EXPS, delta_labels, delta_order, mo):
    # T1.3 experiment selector (independent): one of the E = startdates × ρ experiments.
    _t13_opts = {}
    for _ei in range(len(EXPS)):
        _date = EXPS[_ei].split(chr(47))[-1].split(chr(95))[0]
        for _dd in delta_order:
            _t13_opts[f"{_date} × {delta_labels.get(_dd, _dd)}"] = (_ei, _dd)
    t13_exp = mo.ui.dropdown(options=_t13_opts, value=(next(iter(_t13_opts)) if _t13_opts else None),
                             label="experiment (startdate × ρ): ")
    return (t13_exp,)


@app.cell(hide_code=True)
def t0_mapctl(
    MASK0,
    WARM_CMAP,
    get_mask_center,
    mcolors,
    mo,
    np,
    plt,
    save_chart,
    visualize_map,
):
    # T0 example-map machinery ported from guidance.py "Inspect states" (diff mode): the exact
    # visualize_map renderer + the zoom command (zoom_slider + mask-centred zoom) + RdBu colormaps.
    white_zero_cmap = plt.get_cmap("RdBu_r").copy(); white_zero_cmap.set_bad("white")
    cool_half_cmap = mcolors.LinearSegmentedColormap.from_list("rdbu_cool", plt.get_cmap("RdBu_r")(np.linspace(0.0, 0.5, 256)))
    zoom_centers = get_mask_center(np.asarray(MASK0, float))
    mask_region = np.asarray(MASK0, float) >= 0.5 * float(np.asarray(MASK0, float).max())
    zoom_slider = mo.ui.slider(1, 8, value=3, step=1, label="zoom: ", show_value=True)
    contour_checkbox_t02 = mo.ui.checkbox(label="contours", value=True)
    contour_levels_slider_t02 = mo.ui.slider(4, 30, step=2, value=24, label="levels: ", show_value=True, debounce=True)
    contour_color_dropdown_t02 = mo.ui.dropdown(["dimgray", "white", "black"], value="black", label="contour color: ")
    contour_checkbox_t03 = mo.ui.checkbox(label="contours", value=True)
    contour_levels_slider_t03 = mo.ui.slider(4, 30, step=2, value=24, label="levels: ", show_value=True, debounce=True)
    contour_color_dropdown_t03 = mo.ui.dropdown(["dimgray", "white", "black"], value="black", label="contour color: ")

    def viz_panel(_arr, _label, _is_diff, _ovmin=None, _ovmax=None, _ocmap=None, _ocenter="auto", _mask=None, _savename=None, _contour_on=True, _contour_levels=24, _contour_color="black", _do_save=False):
        # exactly guidance.py's diff-mode panel: white_zero (straddling) / warm|cool half (single-signed)
        # for absolute fields; symmetric white_zero for differences. + min/max/mean/std stamped over the mask.
        _z = np.asarray(_arr, float)
        _mk2 = np.asarray(MASK0, float) if _mask is None else np.asarray(_mask, float)
        _mreg = _mk2 >= 0.5 * float(_mk2.max()); _zc = get_mask_center(_mk2)   # panel mask = selected experiment's target region
        if _ocmap is not None:
            _cmap, _c, _vmin, _vmax = _ocmap, (0.0 if _ocenter == "auto" else _ocenter), _ovmin, _ovmax
        elif _is_diff:
            _am = (float(np.nanmax(np.abs(_z))) if np.isfinite(_z).any() else 0.0) or 1e-9
            _cmap, _c, _vmin, _vmax = white_zero_cmap, 0.0, -_am, _am
        else:
            _vmin = float(np.nanmin(_z)); _vmax = float(np.nanmax(_z))
            if _vmin < 0.0 < _vmax:
                _cmap, _c = white_zero_cmap, 0.0
            elif _vmin >= 0.0:
                _cmap, _c = WARM_CMAP, None
            else:
                _cmap, _c = cool_half_cmap, None
        _fig, _ax = visualize_map(_z, cmap=_cmap, mask_2d=_mk2, title=_label,
                                  vmin=_vmin, vmax=_vmax, center=_c, show_mask=True,
                                  contour_2d=(_z if _contour_on else None), contour_levels=int(_contour_levels), contour_color=_contour_color, contour_linewidth=0.5,   # iso-lines of the plotted field
                                  zoom=int(zoom_slider.value), zoom_center_lon=_zc[0], zoom_center_lat=_zc[1],
                                  dpi=90, figsize=(5.5, 3.6))
        _ax.tick_params(axis="both", labelsize=6)                       # match T06 tick sizing
        _ax.xaxis.label.set_size(7); _ax.yaxis.label.set_size(7)        # Longitude/Latitude labels
        for _cx in _fig.axes:
            if _cx is not _ax:
                _cx.tick_params(labelsize=6)                            # colorbar numbers
        _ax.set_title(_label, fontsize=9)
        _sa = np.where(_mreg, _z, np.nan)
        if np.isfinite(_sa).any():
            _ax.set_title(f"min = {np.nanmin(_sa):.3g} | max = {np.nanmax(_sa):.3g}", loc="left", fontsize=7)
            _ax.set_title(f"mean = {np.nanmean(_sa):.3g} | std = {np.nanstd(_sa):.3g}", loc="right", fontsize=7)
        if _do_save and _savename:
            save_chart(_fig, _savename)
        _out = mo.as_html(_fig); plt.close(_fig)
        return _out

    return (
        contour_checkbox_t02,
        contour_checkbox_t03,
        contour_color_dropdown_t02,
        contour_color_dropdown_t03,
        contour_levels_slider_t02,
        contour_levels_slider_t03,
        cool_half_cmap,
        viz_panel,
        white_zero_cmap,
        zoom_slider,
    )


if __name__ == "__main__":
    app.run()
