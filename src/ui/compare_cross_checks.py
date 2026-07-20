import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    from src.normalization import XarrayNormalizer
    from src.ui.comparison import (
        clean_pred_trajectory,
        load_rollout,
        masked_mean,
        residual_scaler,
        select_point,
        sweep_points,
    )
    from src.utils import get_rollout_ids

    return (
        XarrayNormalizer,
        clean_pred_trajectory,
        get_rollout_ids,
        load_rollout,
        masked_mean,
        mo,
        np,
        plt,
        residual_scaler,
        select_point,
        sweep_points,
        xr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Where did the guidance go?

    Per-channel effect of the guidance at one forecast step $n$: the mask-averaged
    **gui − gui_ung** move of every channel, in its own ERA5 z-units — so channels of any
    magnitude are comparable. Grouped bars compare the selected runs; the guided channel
    is highlighted. The leaderboard ranks every run by how well it hit its target.
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
    GUIDED_CH = f"{VAR} L{LEVEL}" if config["PARTITION"] == "level" else VAR
    return (
        GUIDED_CH,
        LEVEL,
        VAR,
        config,
        mask,
        points,
        records,
        rollout_dir,
        sweep_values,
    )


@app.cell
def _(config, mo, points):
    m_slider = mo.ui.slider(1, config["M"], step=1, value=1, label="m: ", show_value=True, debounce=True)
    n_slider = mo.ui.slider(1, config["N"], step=1, value=min(2, config["N"]), label="n: ", show_value=True, debounce=True)
    topk_slider = mo.ui.slider(5, 30, step=5, value=10, label="top K channels: ", show_value=True, debounce=True)
    region_dropdown = mo.ui.dropdown(["inside mask", "outside mask"], value="inside mask", label="region: ")

    # sweep axes that actually vary across this rollout's runs -> the compare choices;
    # "n" compares across forecast steps instead
    _axis_vals = {}
    for _sel in points.values():
        for _k, _v in _sel.items():
            _axis_vals.setdefault(_k, set()).add(str(_v))
    varying_axes = [_k for _k, _vs in _axis_vals.items() if len(_vs) > 1]
    compare_dropdown = mo.ui.dropdown(
        (varying_axes + ["n"]) or ["n"],
        value=("eta" if "eta" in varying_axes else ((varying_axes + ["n"])[0])),
        label="compare: ",
    )
    return (
        compare_dropdown,
        m_slider,
        n_slider,
        region_dropdown,
        topk_slider,
        varying_axes,
    )


@app.cell
def _(
    GUIDED_CH,
    LEVEL,
    VAR,
    XarrayNormalizer,
    clean_pred_trajectory,
    config,
    m_slider,
    mask,
    masked_mean,
    np,
    points,
    records,
    residual_scaler,
    rollout_dir,
    select_point,
    sweep_values,
    xr,
):
    # ===== one fused pass per run =====
    # per region (inside = gaussian mask weights, outside = complement of the half-max
    # footprint): weighted MEAN of the normalized difference per channel (bars, coherence)
    # and weighted spatial l2 of the normalized difference per channel (profiles).
    # Physical (K) guided-channel series (inside mask) for the leaderboard.
    xnorm = XarrayNormalizer()
    _m = m_slider.value - 1
    RESID_SCALER = residual_scaler(config["PARTITION"], VAR, LEVEL)
    _w_in = np.asarray(mask, dtype=float)
    _w_out = (np.asarray(mask) < 0.5 * float(np.asarray(mask).max())).astype(float)
    REGIONS = {"inside mask": _w_in, "outside mask": _w_out}

    def _wda(ds, w_np):
        return xr.DataArray(w_np, dims=("latitude", "longitude"),
                            coords={"latitude": ds.latitude, "longitude": ds.longitude})

    def _wmean(ds, w_np):
        return (ds * _wda(ds, w_np)).sum(("latitude", "longitude")) / float(w_np.sum())

    def _wl2(ds, w_np):
        return np.sqrt(((ds ** 2) * _wda(ds, w_np)).sum(("latitude", "longitude")) / float(w_np.sum()))

    def _flat(mm_ds):
        """Dataset (n, [level]) -> {channel label: (N,)}; levels become 'var L{p}'."""
        out = {}
        for v in mm_ds.data_vars:
            da = mm_ds[v]
            if "level" in da.dims:
                for lv in da["level"].values:
                    out[f"{v} L{int(lv)}"] = np.asarray(da.sel(level=lv), dtype=float)
            else:
                out[v] = np.asarray(da, dtype=float)
        return out

    run_data = {}
    for _label, _sel in points.items():
        _gui = select_point(xr.open_zarr(rollout_dir / "gui.zarr"), _sel).isel(m=_m)
        _twin = select_point(xr.open_zarr(rollout_dir / "gui_ung.zarr"), _sel).isel(m=_m)
        _twin = _twin.isel(t=-1) if "t" in _twin.dims else _twin
        _diff_field = xnorm.normalize(_gui) - xnorm.normalize(_twin)  # lazy, z-units
        _diffs, _l2s = {}, {}
        for _rname, _w in REGIONS.items():
            _diffs[_rname] = _flat(_wmean(_diff_field, _w).compute())
            _l2s[_rname] = _flat(_wl2(_diff_field, _w).compute())
        _g = _flat(_wmean(_gui, _w_in).compute())
        _t = _flat(_wmean(_twin, _w_in).compute())
        _delta = np.asarray(sweep_values["GUIDANCE_DELTA"][_sel["GUIDANCE_DELTA"]], dtype=float)[: config["N"]]
        # total pushback: over the flow steps t, every move of the masked mean AGAINST
        # the push direction (clean-pred trajectory stepping "downwards"), summed in K
        _pushback = 0.0
        for _ni in range(config["N"]):
            if _delta[_ni] == 0.0:
                continue
            _traj = clean_pred_trajectory(rollout_dir, records, _sel, _m, _ni, VAR, RESID_SCALER, level=LEVEL)
            if _traj is None or not np.isfinite(_traj).any():
                continue
            _mmt = masked_mean(_traj, mask)
            _pushback += float(np.maximum(0.0, -np.sign(_delta[_ni]) * np.diff(_mmt)).sum())
        run_data[_label] = {
            "diff_norm": _diffs,                                # region -> channel -> (N,), z-units
            "l2_norm": _l2s,                                    # region -> channel -> (N,), z-units
            "pushback": _pushback,
            "gap": _g[GUIDED_CH] - (1.0 + _delta) * _t[GUIDED_CH],
            "push": _g[GUIDED_CH] - _t[GUIDED_CH],
            "asked": _delta * _t[GUIDED_CH],
            "delta": _delta,
        }
    return (run_data,)


@app.cell
def _(
    GUIDED_CH,
    compare_dropdown,
    m_slider,
    mo,
    n_slider,
    np,
    plt,
    points,
    region_dropdown,
    run_data,
    topk_slider,
    varying_axes,
):
    # ===== where did the guidance go? =====
    # bars within a chart = the values of the COMPARE axis ("n" = across forecast steps);
    # one chart per combination of the remaining varying axes. All runs always displayed.
    _cmp = compare_dropdown.value
    _region = region_dropdown.value

    def _fmt_val(_k, _v):
        return f"δ#{_v}" if _k == "GUIDANCE_DELTA" else f"{_k}={_v}"

    def _sort_key(_v):
        try:
            return (0, float(_v), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(_v))

    _gcmap = plt.get_cmap("Greens")
    def _fade(_i, _total):
        return _gcmap(0.7) if _total <= 1 else _gcmap(0.25 + 0.72 * _i / (_total - 1))

    # charts: (title, [(bar label, color, {channel: value})])
    _charts = []
    if _cmp == "n":
        for _label in sorted(run_data):
            _rd = run_data[_label]
            _dn = _rd["diff_norm"][_region]
            _ns = [_i for _i in range(len(_rd["delta"])) if _rd["delta"][_i] != 0]
            _bars = [
                (f"n={_i + 1}", _fade(_j, len(_ns)), {_ch: float(_v[_i]) for _ch, _v in _dn.items()})
                for _j, _i in enumerate(_ns)
            ]
            _charts.append((_label, _bars))
    else:
        _n = n_slider.value - 1
        _others = [_k for _k in varying_axes if _k != _cmp]
        _cvals = sorted({points[_l].get(_cmp) for _l in run_data}, key=_sort_key)
        _groups = {}
        for _label in run_data:
            _gkey = ", ".join(_fmt_val(_k, points[_label][_k]) for _k in _others) or "all runs"
            _groups.setdefault(_gkey, []).append(_label)
        for _gkey in sorted(_groups):
            _labels = sorted(_groups[_gkey], key=lambda l: _sort_key(points[l].get(_cmp)))
            _bars = [
                (
                    _fmt_val(_cmp, points[_label].get(_cmp)),
                    _fade(_cvals.index(points[_label].get(_cmp)), len(_cvals)),
                    {_ch: float(_v[_n]) for _ch, _v in run_data[_label]["diff_norm"][_region].items()},
                )
                for _label in _labels
            ]
            _charts.append((f"{_gkey}  (n={n_slider.value})", _bars))

    # channels ranked by the largest |delta| across ALL bars; SAME rows in every chart
    _score = {}
    for _t, _bars in _charts:
        for _bl, _c, _vals in _bars:
            for _ch, _val in _vals.items():
                _a = abs(_val)
                if np.isfinite(_a) and _a > _score.get(_ch, 0.0):
                    _score[_ch] = _a
    _chs = [_ch for _ch, _s in sorted(_score.items(), key=lambda kv: -kv[1])[: topk_slider.value]][::-1]

    _figs = []
    for _title, _bars in _charts:
        _k = len(_bars)
        if _k == 0 or not _chs:
            continue
        _bh = 0.8 / _k
        _y = np.arange(len(_chs))
        with plt.rc_context({"font.size": 9, "axes.titlesize": 11, "legend.fontsize": 8}):
            _fig, _ax = plt.subplots(figsize=(6.5, 0.42 * len(_chs) + 1.4), dpi=110)
            for _i, (_bl, _c, _vals) in enumerate(_bars):
                _ax.barh(_y + (_i - (_k - 1) / 2) * _bh, [_vals.get(_ch, np.nan) for _ch in _chs],
                         height=_bh, color=_c, alpha=0.9, label=_bl)
            _ax.set_yticks(_y)
            _ax.set_yticklabels(_chs)
            for _tick, _ch in zip(_ax.get_yticklabels(), _chs):
                if _ch == GUIDED_CH:
                    _tick.set_fontweight("bold")
                    _tick.set_color("#D55E00")
            _ax.axvline(0, color="#888888", linewidth=0.8)
            _ax.set_xlabel(r"$\Delta$ = gui − gui_ung, region-averaged  (z-units)")
            _ax.set_title(f"{_title}  ({_region}, m={m_slider.value})", loc="left", fontweight="bold")
            for _sp in ("top", "right"):
                _ax.spines[_sp].set_visible(False)
            _ax.legend(loc="lower right", frameon=False)
            _fig.tight_layout()
        _figs.append(_fig)
    mo.vstack(
        [
            mo.hstack([compare_dropdown, region_dropdown, m_slider, n_slider, topk_slider], justify="start", align="start"),
            mo.hstack(_figs, justify="start", align="start", wrap=True) if _figs else mo.md("_no data_"),
        ],
        align="start",
    )
    return


@app.cell
def _(compare_dropdown, m_slider, mo, np, points, run_data):
    # ===== final gap across the comparison =====
    # one number per compared value: the final-step gap to the (1+delta)*twin target,
    # averaged over everything else (other axes, delta trajectories / forecast steps)
    _cmp = compare_dropdown.value

    def _fmt_val2(_k, _v):
        return f"δ#{_v}" if _k == "GUIDANCE_DELTA" else f"{_k}={_v}"

    def _sort_key2(_s):
        _t = str(_s).split("=")[-1].lstrip("δ#")
        try:
            return (0, float(_t), "")
        except ValueError:
            return (1, 0.0, str(_s))

    _vals = {}
    for _label in run_data:
        _rd = run_data[_label]
        _act = _rd["delta"] != 0
        if not _act.any() or not np.isfinite(_rd["gap"][_act]).any():
            continue
        if _cmp == "n":
            for _i in np.where(_act)[0]:
                _vals.setdefault(f"n={_i + 1}", []).append(float(_rd["gap"][_i]))
        else:
            _vals.setdefault(_fmt_val2(_cmp, points[_label].get(_cmp)), []).append(float(_rd["gap"][_act][-1]))

    _rows = sorted(((float(np.nanmean(_v)), _k, len(_v)) for _k, _v in _vals.items()), key=lambda r: _sort_key2(r[1]))
    _best = min((abs(_g) for _g, _k, _c in _rows), default=None)
    _md = ["| " + " | ".join([compare_dropdown.value, "final gap (mK)", "runs pooled"]) + " |", "|---|---|---|"]
    for _g, _k, _c in _rows:
        _s = f"{1000.0 * _g:.3f}"
        if abs(_g) == _best:
            _s = f"**{_s}**"
        _md.append(f"| {_k} | {_s} | {_c} |")
    mo.vstack(
        [
            mo.md(
                "## Final gap\n\nOne number per compared value (m={}): the final-step gap to the "
                "$(1+\\delta)\\cdot$twin target, in mK, averaged over everything not compared. "
                "Smallest \\|gap\\| is **bold**.".format(m_slider.value)
            ),
            mo.md("\n".join(_md)) if _rows else mo.md("_no completed guided runs_"),
        ],
        align="start",
    )
    return


@app.cell(hide_code=True)
def _(compare_dropdown, mo, np, plt, points, region_dropdown, run_data):
    # ===== difference profiles across pressure levels =====
    # Fig. 3 of "Guided Diffusion Sampling for Precipitation Forecast Interventions"
    # (arXiv:2605.14317), with the DIFFERENCE gui - gui_ung instead of the perturbation:
    # per pressure level, the region-weighted spatial l2 of the normalized difference,
    # one panel per compared value, one line per level variable, averaged over guided n
    # and the axes not compared.
    _cmp = compare_dropdown.value
    _region = region_dropdown.value

    def _fmt_val3(_k, _v):
        return f"δ#{_v}" if _k == "GUIDANCE_DELTA" else f"{_k}={_v}"

    def _sort_key3(_s):
        _t = str(_s).split("=")[-1].lstrip("δ#")
        try:
            return (0, float(_t), "")
        except ValueError:
            return (1, 0.0, str(_s))

    _acc = {}  # panel -> var -> level -> [values]
    for _label in run_data:
        _rd = run_data[_label]
        _l2 = _rd["l2_norm"][_region]
        _ns = [_i for _i in range(len(_rd["delta"])) if _rd["delta"][_i] != 0]
        for _ch, _vv in _l2.items():
            if " L" not in _ch:
                continue
            _v, _lv = _ch.rsplit(" L", 1)
            for _i in _ns:
                _pk = f"n={_i + 1}" if _cmp == "n" else _fmt_val3(_cmp, points[_label].get(_cmp))
                _val = float(_vv[_i])
                if np.isfinite(_val):
                    _acc.setdefault(_pk, {}).setdefault(_v, {}).setdefault(int(_lv), []).append(_val)

    _panels = sorted(_acc, key=_sort_key3)
    _var_colors = {_v: plt.get_cmap("tab10").colors[_i % 10]
                   for _i, _v in enumerate(sorted({_v for _p in _acc.values() for _v in _p}))}
    if _panels:
        _xmax = max(np.mean(_vals) for _p in _acc.values() for _lvls in _p.values() for _vals in _lvls.values())
        with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8}):
            _fig, _axs = plt.subplots(1, len(_panels), figsize=(2.9 * len(_panels) + 1.6, 4.8),
                                      dpi=110, sharey=True)
            _axs = np.atleast_1d(_axs)
            for _ax, _pk in zip(_axs, _panels):
                for _v in sorted(_acc[_pk]):
                    _lvls = sorted(_acc[_pk][_v])
                    _prof = [float(np.mean(_acc[_pk][_v][_l])) for _l in _lvls]
                    _ax.plot(_prof, _lvls, "-o", markersize=2.5, linewidth=1.4,
                             color=_var_colors[_v], label=_v)
                _ax.invert_yaxis()  # 1000 hPa at the bottom, upper atmosphere on top
                _ax.set_xlim(0, 1.05 * _xmax)
                _ax.set_title(_pk, loc="left", fontweight="bold")
                _ax.set_xlabel(r"spatial $\ell_2$ of $\Delta$  (z-units)")
                _ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.6)
                for _sp in ("top", "right"):
                    _ax.spines[_sp].set_visible(False)
            _axs[0].set_ylabel("pressure level (hPa)")
            _axs[-1].legend(loc="lower right", frameon=False)
            _fig.tight_layout()
        _out = _fig
    else:
        _out = mo.md("_no data_")
    mo.vstack(
        [
            mo.md(
                "## Difference profiles across pressure levels\n\nPer level: the region-weighted spatial "
                "$\\ell_2$ of the normalized difference gui − gui_ung ({}), one line per level variable, "
                "averaged over guided $n$ and the axes not compared "
                "(after Fig. 3 of arXiv:2605.14317, with the difference instead of the perturbation)."
                .format(region_dropdown.value)
            ),
            _out,
        ],
        align="start",
    )
    return


@app.cell(hide_code=True)
def _(m_slider, mo, np, region_dropdown, run_data):
    # ===== leaderboard: level coherence =====
    # how coherent is the response ACROSS PRESSURE LEVELS? For every level variable and
    # guided n, take the vertical profile of the region-averaged difference (z-units) and
    # score its lag-1 autocorrelation across levels (1 = smooth vertical structure,
    # 0 = incoherent, negative = alternating). Profiles are weighted by their RMS so
    # strong responses dominate; averaged across the delta trajectories.
    _region = region_dropdown.value

    def _coherence(_label):
        _rd = run_data[_label]
        _dn = _rd["diff_norm"][_region]
        _ns = [_i for _i in range(len(_rd["delta"])) if _rd["delta"][_i] != 0]
        _profs = {}
        for _ch in _dn:
            if " L" in _ch:
                _v, _lv = _ch.rsplit(" L", 1)
                _profs.setdefault(_v, []).append((int(_lv), _ch))
        _wsum = _rsum = 0.0
        for _v, _items in _profs.items():
            _items.sort()
            for _i in _ns:
                _p = np.array([float(_dn[_ch][_i]) for _lv, _ch in _items])
                if not np.isfinite(_p).all():
                    continue
                _w = float(np.sqrt(np.mean(_p ** 2)))
                _a, _b = _p[:-1], _p[1:]
                if _w < 1e-9 or _a.std() < 1e-12 or _b.std() < 1e-12:
                    continue
                _rsum += _w * float(np.corrcoef(_a, _b)[0, 1])
                _wsum += _w
        return _rsum / _wsum if _wsum > 0 else np.nan

    _scores = {}
    for _label in run_data:
        _c = _coherence(_label)
        if np.isfinite(_c):
            _g = " ".join(_p for _p in _label.split() if not _p.startswith("δ#"))
            _scores.setdefault(_g, []).append(_c)
    _rows = sorted(((float(np.mean(_cs)), _g, len(_cs)) for _g, _cs in _scores.items()), key=lambda r: -r[0])
    _best = max((_c for _c, _g, _n_ in _rows), default=None)
    _md = ["| run | δ's | level coherence ↑ |", "|---|---|---|"]
    for _c, _g, _n_ in _rows:
        _s = f"{_c:.3f}"
        if _c == _best:
            _s = f"**{_s}**"
        _md.append(f"| {_g} | {_n_} | {_s} |")
    mo.vstack(
        [
            mo.md(
                "## Level coherence\n\nRMS-weighted lag-1 autocorrelation of every level variable's "
                "vertical $\\Delta$ profile ({}, m={}), pooled over variables and guided $n$, averaged "
                "across the $\\delta$ trajectories. Higher = the guidance moves neighbouring pressure "
                "levels together (physically coherent); the winner is **bold**.".format(region_dropdown.value, m_slider.value)
            ),
            mo.md("\n".join(_md)) if _rows else mo.md("_no data_"),
        ],
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
