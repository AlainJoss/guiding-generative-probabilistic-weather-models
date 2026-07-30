import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import sys
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from pathlib import Path

    try:
        _base = Path(mo.notebook_dir())
    except Exception:
        _base = Path(__file__).parent
    _repo = _base.parents[1]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))

    from src.utils import (
        get_rollout_ids, get_rollout, get_config, get_sweep_dict,
        sweep_coord_label, get_slices, get_gt_rollout, get_guidance_schedule,
    )
    from src.run import iter_sweeps
    from src.mask import get_mask_2d, get_masked_mean
    from src.dimensions import LEVELS_DICT
    from src.ui.map import visualize_map, to_display_units
    import src.utils as _u_mod
    _u_mod.ROLLOUTS = _repo / "local_data" / "rollouts"  # pin: 2026-07-27 lives here (data was moved off switchdrive)

    # experiment + first sweep combo
    rid = "2026-07-27_06:00:00"  # pinned: the complete rollout (avoid auto-picking a still-generating one)
    cfg = get_config(rid)
    _spd = get_sweep_dict(rid)
    sp = {k: sweep_coord_label(k, v, _spd)
          for k, v in next(iter(iter_sweeps(_spd))).items()}

    # states at this combo; guided residual = x_n^gui - x_n^gui_ung (same noise seed)
    gui = get_rollout("gui", rid).sel(sp)
    guf = get_rollout("gui_ung", rid).sel(sp).isel(t=-1)
    _ung = get_rollout("ung", rid).isel(t=-1)
    mask = get_mask_2d(sp["MASK_MODE"], cfg.MASK_CORNERS, sigma_div=float(sp["sigma_div"]))

    # map zoom (x3) centered on the mask
    _lon_c = 0.5 * (cfg.MASK_CORNERS[0] + cfg.MASK_CORNERS[1])
    _lat_c = 0.5 * (cfg.MASK_CORNERS[2] + cfg.MASK_CORNERS[3])
    if _lon_c > 180.0:
        _lon_c -= 360.0

    def _resid(partition, var, level, n):
        a = get_slices(gui, partition, var, level)[0][n]
        b = get_slices(guf, partition, var, level)[0][n]
        d, _u = to_display_units(a - b, var, is_difference=True)
        return d

    def resid_map(partition, var, level, n, title):
        d = _resid(partition, var, level, n)
        M = float(np.nanmax(np.abs(d))) or 1.0
        res = visualize_map(
            d, cmap="RdBu_r", mask_2d=mask, show_mask=True, title=title,
            vmin=-M, vmax=M, center=0.0, figsize=(11, 5.5), dpi=130,
            interactive=False,
            zoom=3, zoom_center_lon=_lon_c, zoom_center_lat=_lat_c,
            contour_2d=d, contour_levels=15, contour_color="#222222",
            contour_linewidth=0.3,
        )
        return res[0] if isinstance(res, tuple) else res

    # x-axis: surface (2 m), then pressure levels bottom -> top
    _LEVELS = [0] + list(reversed(LEVELS_DICT["level"]))
    _core = np.asarray(mask) >= 0.5 * float(np.asarray(mask).max())

    def _tvals(states, level, n):
        p, v, lv = ("surface", "2m_temperature", 0) if level == 0 else ("level", "temperature", level)
        vv, _u = to_display_units(np.asarray(get_slices(states, p, v, lv)[0][n], dtype=float), v)
        vv = vv[_core]
        return vv[np.isfinite(vv)]

    def _tcore(level, n):
        p, v, lv = ("surface", "2m_temperature", 0) if level == 0 else ("level", "temperature", level)
        vv, _u = to_display_units(np.asarray(get_slices(gui, p, v, lv)[0][n], dtype=float), v)
        vv = vv[_core]
        return vv[np.isfinite(vv)]

    # fixed temperature axis: global min/max over ALL levels and ALL forecast steps
    _tall = np.concatenate([_tcore(L, n) for L in _LEVELS for n in range(int(gui.sizes["n"]))])
    _TMIN, _TMAX = float(np.min(_tall)), float(np.max(_tall))


    def _dT(level, n):
        p, v, lv = ("surface", "2m_temperature", 0) if level == 0 else ("level", "temperature", level)
        return float(get_masked_mean(_resid(p, v, lv, n)[None, None], mask)[0, 0])

    def level_profile():
        xs = list(range(len(_LEVELS)))
        fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=130)
        for n, c in ((0, "#0072B2"), (1, "#D55E00")):
            ys = [_dT(L, n) for L in _LEVELS]
            ax.plot(xs, ys, "-o", color=c, lw=2.0, ms=4, label=f"step n={n}")
        ax.axhline(0.0, color="#bbbbbb", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(["sfc" if L == 0 else str(L) for L in _LEVELS])
        ax.set_xlabel("pressure level [hPa]  (surface → top)")
        ax.set_ylabel("mask-mean ΔT (guided − unguided) [°C]")
        ax.grid(True, axis="y", color="#eeeeee")
        ax.legend(frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def _T_abs(level, n):
        p, v, lv = ("surface", "2m_temperature", 0) if level == 0 else ("level", "temperature", level)
        a = get_slices(gui, p, v, lv)[0][n]
        a, _u = to_display_units(a, v, is_difference=False)
        return float(get_masked_mean(a[None, None], mask)[0, 0])

    def level_profile_abs():
        xs = list(range(len(_LEVELS)))
        fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=130)
        for n, c in ((0, "#0072B2"), (1, "#D55E00")):
            ys = [_T_abs(L, n) for L in _LEVELS]
            ax.plot(xs, ys, "-o", color=c, lw=2.0, ms=4, label=f"step n={n}")
        ax.set_xticks(xs)
        ax.set_xticklabels(["sfc" if L == 0 else str(L) for L in _LEVELS])
        ax.set_xlabel("pressure level [hPa]  (surface → top)")
        ax.set_ylabel("mask-mean temperature (guided) [°C]")
        ax.grid(True, axis="y", color="#eeeeee")
        ax.legend(frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.close(fig)
        return fig

    n_levels = len(_LEVELS)

    def hist_levels_fig(i):
        fig, ax = plt.subplots(figsize=(10.0, 4.5), dpi=130)
        _bins = np.linspace(_TMIN, _TMAX, 41)
        picks = [(int(i), 1.0, "#0072B2")]
        if int(i) + 1 < len(_LEVELS):
            picks.append((int(i) + 1, 0.5, "#D55E00"))
        for j, alpha, color in picks:
            L = _LEVELS[j]
            vals = _tcore(L, 0)
            if vals.size == 0:
                continue
            lbl = ("surface" if L == 0 else f"{L} hPa") + ("" if alpha == 1.0 else " (next)")
            ax.hist(vals, bins=_bins, density=True, histtype="step",
                    color=color, linewidth=2.0, alpha=alpha, label=lbl)
        ax.set_xlim(_TMIN, _TMAX)
        ax.set_xlabel("temperature, guided [°C]")
        ax.set_ylabel("density")
        ax.legend(frameon=False, loc="upper left")
        ax.yaxis.grid(True, color="#eeeeee")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def hist_over_n(i):
        L = _LEVELS[int(i)]
        tag = "surface" if L == 0 else f"{L} hPa"
        N = int(gui.sizes["n"])
        per = [_tcore(L, n) for n in range(N)]
        # tight axis: min/max over the forecast steps at THIS level
        _fin = [p for p in per if p.size]
        lo = min(float(p.min()) for p in _fin) if _fin else _TMIN
        hi = max(float(p.max()) for p in _fin) if _fin else _TMAX
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 41)
        ks = list(range(1, N)) or [0]  # one panel per later step (N-1 panels)
        fig, axes = plt.subplots(len(ks), 1, figsize=(10.0, 2.5 * len(ks) + 0.5),
                                 dpi=130, sharex=True, squeeze=False)
        for row, k in enumerate(ks):
            ax = axes[row, 0]
            ax.hist(per[0], bins=bins, density=True, histtype="step",
                    color="#999999", linewidth=1.8, label="n = 0")
            ax.hist(per[k], bins=bins, density=True, histtype="step",
                    color="#0072B2", linewidth=2.4, label=f"n = {k}")
            ax.set_xlim(lo, hi)
            ax.set_ylabel("density")
            ax.legend(frameon=False, loc="upper left", fontsize=9)
            ax.yaxis.grid(True, color="#eeeeee")
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[-1, 0].set_xlabel(f"temperature @ {tag}, guided [°C]")
        fig.tight_layout()
        plt.close(fig)
        return fig

    n_steps = int(gui.sizes["n"])

    def ridgeline_over_levels(nn):
        bins = np.linspace(_TMIN, _TMAX, 51)
        centers = 0.5 * (bins[:-1] + bins[1:])
        srcs = [("gui", gui, "#0072B2"), ("gui_ung", guf, "#800080"), ("ung", _ung, "#8B5A2B")]
        fig, ax = plt.subplots(figsize=(10.0, 1.6 + 0.6 * len(_LEVELS)), dpi=130)
        step = 0.9
        for idx, L in enumerate(_LEVELS):
            base = idx * step
            dts = [(np.histogram(_tvals(states, L, int(nn)), bins=bins, density=True)[0], col, name)
                   for name, states, col in srcs]
            pk = max((float(d.max()) for d, _c, _n in dts), default=1.0) or 1.0
            for d, col, name in dts:
                ax.plot(centers, base + d / pk, color=col, lw=1.3, drawstyle="steps-mid",
                        zorder=len(_LEVELS) - idx, label=name if idx == 0 else "_nolegend_")
        ax.set_xlim(_TMIN, _TMAX)
        ax.set_yticks([idx * step for idx in range(len(_LEVELS))])
        ax.set_yticklabels(["surface" if L == 0 else str(L) for L in _LEVELS], fontsize=8.5)
        ax.set_ylabel("pressure level [hPa]")
        ax.set_xlabel("temperature [°C]")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.close(fig)
        return fig


    # --- comparison across guidance intensities (GUIDANCE_DELTA profiles) ---
    _DELTAS = _spd["GUIDANCE_DELTA"]
    _sp_noD = {k: v for k, v in sp.items() if k != "GUIDANCE_DELTA"}
    _gui_D = get_rollout("gui", rid).sel(_sp_noD)
    _guf_D = get_rollout("gui_ung", rid).sel(_sp_noD).isel(t=-1)
    _gt = get_gt_rollout(cfg.N + 1, cfg.START_TS)
    _grads_D = get_rollout("grads", rid).sel(_sp_noD)
    _maskda = xr.DataArray(np.asarray(mask), dims=("latitude", "longitude"),
                           coords={"latitude": _grads_D.latitude, "longitude": _grads_D.longitude})
    _det_D = get_rollout("gui_det", rid).sel(_sp_noD)
    try:
        _undet_D = get_rollout("ung_det", rid).sel(_sp_noD)
    except (FileNotFoundError, KeyError):
        _undet_D = _det_D  # ung_det not saved -> deterministic core (= gui_det at n=0)
    _ICOLORS = ["#a50f15", "#de2d26", "#fc9272", "#fcbba1"]

    def _rD(i, L, n):
        p, v, lv = ("surface", "2m_temperature", 0) if L == 0 else ("level", "temperature", L)
        a = get_slices(_gui_D.isel(GUIDANCE_DELTA=i), p, v, lv)[0][n]
        b = get_slices(_guf_D.isel(GUIDANCE_DELTA=i), p, v, lv)[0][n]
        d, _u = to_display_units(a - b, v, is_difference=True)
        return d

    def _abs_mm(states, L, n):
        p, v, lv = ("surface", "2m_temperature", 0) if L == 0 else ("level", "temperature", L)
        a = np.asarray(get_slices(states, p, v, lv))
        a = a[0][n] if a.ndim == 4 else a[n + 1]  # gui/ung: (m,n,..); gt: (time,..) IC at 0
        a, _u = to_display_units(a, v, is_difference=False)
        return float(get_masked_mean(a[None, None], mask)[0, 0])

    def _det_ung_mm(L, n):
        p, v, lv = ("surface", "2m_temperature", 0) if L == 0 else ("level", "temperature", L)
        a = get_slices(_det_D.isel(GUIDANCE_DELTA=0), p, v, lv)[0][n]
        b = get_slices(_guf_D.isel(GUIDANCE_DELTA=0), p, v, lv)[0][n]
        d, _u = to_display_units(a - b, v, is_difference=True)
        return float(get_masked_mean(d[None, None], mask)[0, 0])

    def intensity_profile():
        fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=130)
        xs = list(range(len(_LEVELS)))
        for i, prof in enumerate(_DELTAS):
            ys = [float(get_masked_mean(_rD(i, L, 0)[None, None], mask)[0, 0]) for L in _LEVELS]
            ax.plot(xs, ys, "-o", color=_ICOLORS[i % len(_ICOLORS)], lw=2.0, ms=4,
                    label=f"δ₀ = {prof[0] * 100:.3g}%")
        ax.axhline(0.0, color="#bbbbbb", lw=0.8)
        ax.plot(xs, [_det_ung_mm(L, 0) for L in _LEVELS], "--", color="#333333",
                lw=1.8, label="gui_det − ung")
        ax.set_xticks(xs)
        ax.set_xticklabels(["sfc" if L == 0 else str(L) for L in _LEVELS])
        ax.set_xlabel("pressure level [hPa]  (surface → top)")
        ax.set_ylabel("mask-mean ΔT (guided − unguided) [°C]")
        ax.grid(True, axis="y", color="#eeeeee")
        ax.legend(frameon=False, title="intensity")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def intensity_profile_abs():
        fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=130)
        xs = list(range(len(_LEVELS)))
        for i, prof in enumerate(_DELTAS):
            ys = [_abs_mm(_gui_D.isel(GUIDANCE_DELTA=i), L, 0) for L in _LEVELS]
            ax.plot(xs, ys, "-o", color=_ICOLORS[i % len(_ICOLORS)], lw=2.0, ms=4,
                    label=f"δ₀ = {prof[0] * 100:.3g}%")
        ax.plot(xs, [_abs_mm(_guf_D.isel(GUIDANCE_DELTA=0), L, 0) for L in _LEVELS],
                "--", color="#555555", lw=1.8, label="unguided (ung)")
        ax.plot(xs, [_abs_mm(_gt, L, 0) for L in _LEVELS],
                "--", color="#009E73", lw=1.8, label="ground truth (gt)")
        ax.set_xticks(xs)
        ax.set_xticklabels(["sfc" if L == 0 else str(L) for L in _LEVELS])
        ax.set_xlabel("pressure level [hPa]  (surface → top)")
        ax.set_ylabel("mask-mean temperature [°C]")
        ax.grid(True, axis="y", color="#eeeeee")
        ax.legend(frameon=False, fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def intensity_profile_vs_gt():
        fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=130)
        xs = list(range(len(_LEVELS)))
        gt = [_abs_mm(_gt, L, 0) for L in _LEVELS]
        def _dev(states):
            return [_abs_mm(states, L, 0) - gt[k] for k, L in enumerate(_LEVELS)]
        for i, prof in enumerate(_DELTAS):
            ax.plot(xs, _dev(_gui_D.isel(GUIDANCE_DELTA=i)), "-o",
                    color=_ICOLORS[i % len(_ICOLORS)], lw=2.0, ms=4,
                    label=f"δ₀ = {prof[0] * 100:.3g}%")
        for lab, states, ls, col in [
            ("ung", _ung, "--", "#8B5A2B"),
            ("gui_ung", _guf_D.isel(GUIDANCE_DELTA=0), "--", "#800080"),
            ("gui_det", _det_D.isel(GUIDANCE_DELTA=0), "-.", "#555555"),
            ("ung_det", _undet_D.isel(GUIDANCE_DELTA=0), ":", "#17becf"),
        ]:
            ax.plot(xs, _dev(states), ls, color=col, lw=1.6, label=lab)
        ax.axhline(0.0, color="#009E73", lw=1.8, ls="--", label="ground truth (gt)")
        ax.set_xticks(xs)
        ax.set_xticklabels(["sfc" if L == 0 else str(L) for L in _LEVELS])
        ax.set_xlabel("pressure level [hPa]  (surface → top)")
        ax.set_ylabel("mask-mean temperature − gt [°C]")
        ax.grid(True, axis="y", color="#eeeeee")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def _kick_profile(i):
        gd = _grads_D.isel(GUIDANCE_DELTA=i)
        s = gd["2m_temperature"].isel(m=0, n=0)
        e0 = float(((s ** 2) * _maskda).sum(("latitude", "longitude")).sum("t").compute())
        tl = gd["temperature"].isel(m=0, n=0)
        el = ((tl ** 2) * _maskda).sum(("latitude", "longitude")).sum("t").compute()
        return [np.sqrt(e0)] + [float(np.sqrt(el.sel(level=L))) for L in _LEVELS[1:]]

    def intensity_kick_profile():
        fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=130)
        xs = list(range(len(_LEVELS)))
        for i, prof in enumerate(_DELTAS):
            ax.plot(xs, _kick_profile(i), "-o", color=_ICOLORS[i % len(_ICOLORS)],
                    lw=2.0, ms=4, label=f"δ₀ = {prof[0] * 100:.3g}%")
        ax.set_xticks(xs)
        ax.set_xticklabels(["sfc" if L == 0 else str(L) for L in _LEVELS])
        ax.set_xlabel("pressure level [hPa]  (surface → top)")
        ax.set_ylabel(r"guidance perturbation  $\|\nabla\mathcal{L}\|$  (mask, all steps)")
        ax.grid(True, axis="y", color="#eeeeee")
        ax.legend(frameon=False, title="intensity")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def intensity_maps(n=0):
        ds = [_rD(i, 0, n) for i in range(len(_DELTAS))]
        M = max((float(np.nanmax(np.abs(d))) for d in ds), default=1.0) or 1.0
        figs = []
        for i in range(len(_DELTAS)):
            res = visualize_map(
                ds[i], cmap="RdBu_r", mask_2d=mask, show_mask=True,
                title=f"δ₀ = {_DELTAS[i][0] * 100:.3g}%", vmin=-M, vmax=M, center=0.0,
                figsize=(7.5, 4.2), dpi=120, interactive=False,
                zoom=3, zoom_center_lon=_lon_c, zoom_center_lat=_lat_c,
                contour_2d=ds[i], contour_levels=15, contour_color="#222222",
                contour_linewidth=0.3,
            )
            figs.append(res[0] if isinstance(res, tuple) else res)
        return figs


    # --- per-level maps + adjacent-level difference (prev - current) ---
    def _residL(L, n):
        p, v, lv = ("surface", "2m_temperature", 0) if L == 0 else ("level", "temperature", L)
        return _resid(p, v, lv, n)

    def _map_from(d, title, figsize=(8.5, 4.5), vmin=None, vmax=None):
        if vmin is None or vmax is None:
            M = float(np.nanmax(np.abs(d))) or 1.0
            vmin, vmax = -M, M
        res = visualize_map(
            d, cmap="RdBu_r", mask_2d=mask, show_mask=True, title=title,
            vmin=vmin, vmax=vmax, center=0.0, figsize=figsize, dpi=120, interactive=False,
            zoom=3, zoom_center_lon=_lon_c, zoom_center_lat=_lat_c,
            contour_2d=d, contour_levels=15, contour_color="#222222", contour_linewidth=0.3,
        )
        return res[0] if isinstance(res, tuple) else res

    # temperature-levels section: shared colour scales across the shown levels / diffs
    _TLEVELS = [0, 1000, 925, 850, 700]
    _diff_pairs = [(0, 1000), (1000, 925), (925, 850), (850, 700)]
    _lev_ds = [_residL(L, 0) for L in _TLEVELS]
    _LMIN = min(min(float(np.nanmin(d)) for d in _lev_ds), -1e-9)
    _LMAX = max(max(float(np.nanmax(d)) for d in _lev_ds), 1e-9)
    _diff_ds = [_residL(pp, 0) - _residL(cc, 0) for pp, cc in _diff_pairs]
    _DMIN = min(min(float(np.nanmin(d)) for d in _diff_ds), -1e-9)
    _DMAX = max(max(float(np.nanmax(d)) for d in _diff_ds), 1e-9)

    def level_map(L, n=0):
        lab = "2 m temperature" if L == 0 else f"temperature {L} hPa"
        return _map_from(_residL(L, n), f"{lab}  [°C]", vmin=_LMIN, vmax=_LMAX)

    def level_diff_map(prev_L, cur_L, n=0):
        plab = "2 m" if prev_L == 0 else f"{prev_L} hPa"
        clab = "2 m" if cur_L == 0 else f"{cur_L} hPa"
        return _map_from(_residL(prev_L, n) - _residL(cur_L, n), f"{plab} − {clab}  [°C]",
                         vmin=_DMIN, vmax=_DMAX)

    def level_stats():
        core = np.asarray(mask) >= 0.5 * float(np.asarray(mask).max())
        rows = []
        for L in _TLEVELS:
            d = np.asarray(_residL(L, 0))[core]
            d = d[np.isfinite(d)]
            lab = "surface (2 m)" if L == 0 else f"{L} hPa"
            rows.append((lab, float(d.min()), float(d.max()), float(d.mean()), float(d.std())))
        return rows


    def delta_w_scatter():
        recs = get_guidance_schedule(rid)
        D = _spd["GUIDANCE_DELTA"]
        xs = np.array([D[r["sweep"]["GUIDANCE_DELTA"]][r["n"]] * 100.0 for r in recs])
        ys = np.array([float(np.mean(r["w_t"])) for r in recs])
        ns = np.array([int(r["n"]) for r in recs])
        _cols = ["#0072B2", "#D55E00", "#009E73", "#B7950B"]
        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=130)
        for n in sorted(set(ns)):
            m = ns == n
            ax.scatter(xs[m], ys[m], s=45, alpha=0.85, edgecolors="white", lw=0.5,
                       color=_cols[n % len(_cols)], label=f"n = {n}")
        ax.set_xlabel(r"target $\delta_n$ per step [%]")
        ax.set_ylabel(r"guidance weight $w$ (recorded)")
        ax.grid(True, color="#eeeeee")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, title="forecast step")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.close(fig)
        return fig


    def level_stats_gui():
        core = np.asarray(mask) >= 0.5 * float(np.asarray(mask).max())
        rows = []
        for L in _TLEVELS:
            p, v, lv = ("surface", "2m_temperature", 0) if L == 0 else ("level", "temperature", L)
            a, _u = to_display_units(np.asarray(get_slices(gui, p, v, lv)[0][0], dtype=float), v)
            d = a[core]
            d = d[np.isfinite(d)]
            lab = "surface (2 m)" if L == 0 else f"{L} hPa"
            rows.append((lab, float(d.min()), float(d.max()), float(d.mean()), float(d.std())))
        return rows


    def intensity_profile_vs_ung():
        fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=130)
        xs = list(range(len(_LEVELS)))
        ung = [_abs_mm(_guf_D.isel(GUIDANCE_DELTA=0), L, 0) for L in _LEVELS]
        for i, prof in enumerate(_DELTAS):
            ys = [_abs_mm(_gui_D.isel(GUIDANCE_DELTA=i), L, 0) - ung[k] for k, L in enumerate(_LEVELS)]
            ax.plot(xs, ys, "-o", color=_ICOLORS[i % len(_ICOLORS)], lw=2.0, ms=4,
                    label=f"δ₀ = {prof[0] * 100:.3g}%")
        gtn = [_abs_mm(_gt, L, 0) - ung[k] for k, L in enumerate(_LEVELS)]
        ax.plot(xs, gtn, "--", color="#009E73", lw=1.8, label="ground truth (gt)")
        ax.axhline(0.0, color="#555555", lw=1.8, ls="--", label="unguided (ung)")
        ax.set_xticks(xs)
        ax.set_xticklabels(["sfc" if L == 0 else str(L) for L in _LEVELS])
        ax.set_xlabel("pressure level [hPa]  (surface → top)")
        ax.set_ylabel("mask-mean temperature − ung [°C]")
        ax.grid(True, axis="y", color="#eeeeee")
        ax.legend(frameon=False, fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def _guiL(L, n):
        p, v, lv = ("surface", "2m_temperature", 0) if L == 0 else ("level", "temperature", L)
        a, _u = to_display_units(np.asarray(get_slices(gui, p, v, lv)[0][n], dtype=float), v)
        return a

    # guided-state section: shared colour scales across the shown levels / diffs
    _gui_ds = [_guiL(L, 0) for L in _TLEVELS]
    _GMIN = min(min(float(np.nanmin(d)) for d in _gui_ds), -1e-9)
    _GMAX = max(max(float(np.nanmax(d)) for d in _gui_ds), 1e-9)
    _gui_diff_ds = [_guiL(pp, 0) - _guiL(cc, 0) for pp, cc in _diff_pairs]
    _GDMIN = min(min(float(np.nanmin(d)) for d in _gui_diff_ds), -1e-9)
    _GDMAX = max(max(float(np.nanmax(d)) for d in _gui_diff_ds), 1e-9)

    def level_map_gui(L, n=0):
        lab = "2 m temperature" if L == 0 else f"temperature {L} hPa"
        return _map_from(_guiL(L, n), f"{lab} (guided)  [°C]", vmin=_GMIN, vmax=_GMAX)

    def level_diff_map_gui(prev_L, cur_L, n=0):
        plab = "2 m" if prev_L == 0 else f"{prev_L} hPa"
        clab = "2 m" if cur_L == 0 else f"{cur_L} hPa"
        return _map_from(_guiL(prev_L, n) - _guiL(cur_L, n), f"{plab} − {clab}  [°C]",
                         vmin=_GDMIN, vmax=_GDMAX)


    return (
        delta_w_scatter,
        hist_levels_fig,
        hist_over_n,
        intensity_kick_profile,
        intensity_maps,
        intensity_profile_abs,
        intensity_profile_vs_gt,
        level_diff_map,
        level_diff_map_gui,
        level_map,
        level_map_gui,
        level_profile,
        level_profile_abs,
        level_stats,
        level_stats_gui,
        mo,
        n_levels,
        n_steps,
        resid_map,
        ridgeline_over_levels,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Guided − unguided residuals

    "
        f"First sweep combo: {sp['GUIDANCE_MODE']}, {sp['MASK_MODE']}, "
        f"a_t={sp['a_t_mode']}, σ_div={sp['sigma_div']}, w={sp['fgwnolr_w_init']:g}. "
        "Same noise seed, so outside the mask the two states are identical.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Masked-mean temperature by level
    """)
    return


@app.cell(hide_code=True)
def _(level_profile, mo):
    mo.vstack([
        mo.md(
            "**Temperature averaged over the mask, by level.** Strongest guided "
            "warming near the surface; it fades — and flips sign — higher up."
        ),
        level_profile(),
    ])
    return


@app.cell(hide_code=True)
def _(level_profile_abs, mo):
    mo.vstack([
        mo.md(
            "**Same, but the guided temperature itself (no difference).** Mask-region "
            "mean by level — the profile the guided run lands on."
        ),
        level_profile_abs(),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Distributions
    """)
    return


@app.cell(hide_code=True)
def _(mo, n_levels, n_steps):
    hist_level_slider = mo.ui.slider(
        0, n_levels - 1, value=0, show_value=True,
        label="level (0 = surface → top): ",
    )
    hist_n_slider = mo.ui.slider(
        0, n_steps - 1, value=0, show_value=True,
        label="forecast step n: ",
    )
    return hist_level_slider, hist_n_slider


@app.cell(hide_code=True)
def _(hist_level_slider, hist_levels_fig, mo):
    mo.vstack([
        mo.md(
            "**Guided temperature distribution, by level.** Selected level (solid) and "
            "the next one up (half-opacity)."
        ),
        hist_level_slider,
        hist_levels_fig(hist_level_slider.value),
    ])
    return


@app.cell(hide_code=True)
def _(hist_level_slider, hist_over_n, mo):
    mo.vstack([
        mo.md(
            "**Distribution over forecast steps, at the selected level.** One panel per "
            "later step, each overlaid on n = 0 (grey) for a direct comparison."
        ),
        hist_over_n(hist_level_slider.value),
    ])
    return


@app.cell(hide_code=True)
def _(hist_n_slider, mo, ridgeline_over_levels):
    mo.vstack([
        mo.md(
            "**Distribution across all levels, at the selected step.** Per level, the guided "
            "(gui), unguided-twin (gui_ung) and unguided (ung) temperature distributions "
            "overlaid as hist lines; ridges march colder with height."
        ),
        hist_n_slider,
        ridgeline_over_levels(hist_n_slider.value),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Temperature by level — residual (guided − unguided)
    """)
    return


@app.cell(hide_code=True)
def _(level_stats, mo):
    mo.vstack([
        mo.md(
            "**Guided − unguided residual by level** — min / max / avg / std over the mask "
            "region (n = 0) [°C]. The level maps below share one colour scale across levels."
        ),
        mo.md(
            "| level | min | max | avg | std |\n|---|---:|---:|---:|---:|\n"
            + "".join(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} |\n"
                      for r in level_stats())
        ),
    ])
    return


@app.cell(hide_code=True)
def _(level_map, mo):
    mo.vstack([
        mo.md(
            "**2 m temperature.** Guidance acts inside the mask (bleeds a little past "
            "it); land warms, and the cooler patches are pushed hardest — a fill-in."
        ),
        level_map(0),
    ])
    return


@app.cell(hide_code=True)
def _(level_diff_map, level_map, mo):
    mo.vstack([
        mo.md("**Temperature, 1000 hPa.**  Right: change from the level below (2 m − 1000)."),
        mo.hstack([level_map(1000), level_diff_map(0, 1000)], widths="equal"),
    ])
    return


@app.cell(hide_code=True)
def _(level_diff_map, level_map, mo):
    mo.vstack([
        mo.md("**Temperature, 925 hPa.** The signal disperses more over land with height.  Right: 1000 − 925."),
        mo.hstack([level_map(925), level_diff_map(1000, 925)], widths="equal"),
    ])
    return


@app.cell(hide_code=True)
def _(level_diff_map, level_map, mo):
    mo.vstack([
        mo.md("**Temperature, 850 hPa.**  Right: 925 − 850."),
        mo.hstack([level_map(850), level_diff_map(925, 850)], widths="equal"),
    ])
    return


@app.cell(hide_code=True)
def _(level_diff_map, level_map, mo):
    mo.vstack([
        mo.md("**Temperature, 700 hPa.**  Right: 850 − 700."),
        mo.hstack([level_map(700), level_diff_map(850, 700)], widths="equal"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Temperature by level — guided state
    """)
    return


@app.cell(hide_code=True)
def _(level_stats_gui, mo):
    mo.vstack([
        mo.md(
            "**Guided state $x^{gui}_n$ by level** — min / max / avg / std over the mask "
            "region (n = 0) [°C]."
        ),
        mo.md(
            "| level | min | max | avg | std |\n|---|---:|---:|---:|---:|\n"
            + "".join(f"| {r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} |\n"
                      for r in level_stats_gui())
        ),
    ])
    return


@app.cell(hide_code=True)
def _(level_diff_map_gui, level_map_gui, mo):
    mo.vstack([
        mo.md(
            "**Guided state $x^{gui}$ by level.** Left: the absolute guided temperature at each "
            "level (shared colour scale). Right: change from the level below (shared scale)."
        ),
        level_map_gui(0),
        mo.hstack([level_map_gui(1000), level_diff_map_gui(0, 1000)], widths="equal"),
        mo.hstack([level_map_gui(925), level_diff_map_gui(1000, 925)], widths="equal"),
        mo.hstack([level_map_gui(850), level_diff_map_gui(925, 850)], widths="equal"),
        mo.hstack([level_map_gui(700), level_diff_map_gui(850, 700)], widths="equal"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Other variables
    """)
    return


@app.cell(hide_code=True)
def _(mo, resid_map):
    mo.vstack([
        mo.md(
            "**Specific humidity, 1000 hPa.** Up on the sea side, down over land; the "
            "humid band right of the mask lines up with the temperature dip there."
        ),
        resid_map("level", "specific_humidity", 1000, 0, "specific humidity 1000 hPa  [kg/kg]"),
    ])
    return


@app.cell(hide_code=True)
def _(mo, resid_map):
    mo.vstack([
        mo.md("**10 m wind (u, v).** The wind change lifts temperature even over the sea."),
        mo.hstack(
            [resid_map("surface", "10m_u_component_of_wind", 0, 0, "10 m u-wind  [m/s]"),
             resid_map("surface", "10m_v_component_of_wind", 0, 0, "10 m v-wind  [m/s]")],
            widths="equal",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Guidance intensity
    """)
    return


@app.cell(hide_code=True)
def _(intensity_profile_abs, mo):
    mo.vstack([
        mo.md(
            "**Absolute temperature vs guidance intensity.** Mask-mean temperature by level "
            "— guided per intensity, with the unguided and ground-truth baselines."
        ),
        intensity_profile_abs(),
    ])
    return


@app.cell(hide_code=True)
def _(intensity_profile_vs_gt, mo):
    mo.vstack([
        mo.md(
            "**Temperature relative to ground truth** (each line minus gt, so gt is flat at "
            "0). Guided per intensity, plus the baselines ung, gui_ung, gui_det and ung_det. "
            "(ung_det isn\'t stored for this run, so it falls back to the deterministic core — "
            "it coincides with gui_det at n = 0.)"
        ),
        intensity_profile_vs_gt(),
    ])
    return


@app.cell(hide_code=True)
def _(intensity_kick_profile, mo):
    mo.vstack([
        mo.md(
            "**Perturbation profile — guidance kick ‖∇ℒ‖ by level.** Mask-region norm of the "
            "guidance gradient (the applied perturbation) per level, one line per intensity "
            "(summed over flow steps, n = 0): concentrated near the surface and scaling with δ."
        ),
        intensity_kick_profile(),
    ])
    return


@app.cell(hide_code=True)
def _(intensity_maps, mo):
    mo.vstack([
        mo.md(
            "**2 m temperature residual at each intensity** — shared colour scale (n = 0), "
            "strongest δ on the left."
        ),
        mo.hstack(intensity_maps(), widths="equal"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Guidance weight vs target
    """)
    return


@app.cell(hide_code=True)
def _(delta_w_scatter, mo):
    mo.vstack([
        mo.md(
            "**Guidance weight vs target, over all steps and members.** Each point is one "
            "(sweep, m, n) run: the recorded guidance weight w against the per-step target δ_n."
        ),
        delta_w_scatter(),
    ])
    return


if __name__ == "__main__":
    app.run()
