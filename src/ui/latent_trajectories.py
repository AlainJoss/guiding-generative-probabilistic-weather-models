import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from datetime import datetime, timedelta
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import xarray as xr

    from geoarches.paths import STATS_PATH
    from src.mask import get_mask_2d
    from src.paths import ROLLOUTS
    from src.ui.plot_trajectory import plot_trajectory
    from src.utils import get_gt_rollout, get_var_idx

    return (
        Path,
        ROLLOUTS,
        STATS_PATH,
        datetime,
        get_gt_rollout,
        get_mask_2d,
        get_var_idx,
        json,
        mo,
        np,
        plot_trajectory,
        plt,
        timedelta,
        torch,
        xr,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Latent trajectories (PCA) — prototyping

    3D PCA latent-trajectory analysis (arXiv:2605.14317-style) on a fixed rollout and
    FGWNOLR $\eta$ sweep. The production version (single sweep point, reactive to the
    sweep widgets) lives in the guidance notebook; this playground keeps the multi-$\eta$
    comparison and the thesis-figure export.

    **Solid curves**: guided clean-pred trajectories $\hat{x}_t$ (mask-bbox pixels,
    flattened) per $\eta$; marker size grows with $t$. **Dashed curves**: each run's own
    `ung_gui` twin — same seed *and* conditioning, so guided and twin coincide exactly at
    $t=0$ and fan out. **Gray cloud**: daily ERA5 states in the same bbox — the fixed PCA
    frame ("real weather here"). **○** = the independent unguided rollout's final state;
    **★** = GT at the valid day.
    """)
    return


@app.cell
def _(ROLLOUTS, datetime, get_mask_2d, json, xr):
    # ===== rollout access =====
    ROLLOUT_ID = "2026-07-06_11:30:21"
    ROLLOUT_DIR = ROLLOUTS / ROLLOUT_ID
    rollout_config = json.load(open(ROLLOUT_DIR / "config.json"))
    sweep_values = json.load(open(ROLLOUT_DIR / "sweep_params.json"))
    schedule_records = json.load(open(ROLLOUT_DIR / "guidance_schedule.json"))
    ROLLOUT_START = datetime.fromisoformat(rollout_config["START_TS"])

    # rollout stores are dense sweep hypercubes; pin the first FGWNOLR sweep point
    FIRST_SWEEP = {k: v[0] for k, v in sweep_values.items() if k != "GUIDANCE_DELTA"} | {
        "GUIDANCE_MODE": "FGWNOLR",
        "GUIDANCE_DELTA": 0,
    }

    def load_temperature(store, **overrides):
        """2m_temperature from a rollout store at FIRST_SWEEP (coords overridable)."""
        da = xr.open_zarr(ROLLOUT_DIR / store)["2m_temperature"]
        sel = FIRST_SWEEP | overrides
        return da.sel({k: v for k, v in sel.items() if k in da.dims})

    demo_mask = get_mask_2d(sweep_values["MASK_MODE"][0], rollout_config["MASK_CORNERS"])
    return (
        ROLLOUT_ID,
        ROLLOUT_START,
        demo_mask,
        load_temperature,
        rollout_config,
        schedule_records,
    )


@app.cell
def _(
    ROLLOUT_START,
    STATS_PATH,
    demo_mask,
    get_gt_rollout,
    get_var_idx,
    load_temperature,
    np,
    rollout_config,
    schedule_records,
    timedelta,
    torch,
):
    # ===== latent construction =====
    PCA_N = 1                    # forecast step under inspection (0-based)
    PCA_ETAS = [0.1, 0.5, 0.9]   # FGWNOLR eta sweep to draw
    FLOW_T = rollout_config["T"]
    RESID_SCALER = float(
        torch.load(STATS_PATH / "deltapred24_aws_denorm.pt", weights_only=False)["surface"][
            get_var_idx("surface", "2m_temperature")
        ].squeeze()
    )

    def mask_bbox(mask, rel_threshold=0.5):
        """Row/col slices of the mask footprint at rel_threshold * max."""
        rows, cols = np.where(np.asarray(mask) >= rel_threshold * float(np.asarray(mask).max()))
        return slice(int(rows.min()), int(rows.max()) + 1), slice(int(cols.min()), int(cols.max()) + 1)

    BBOX = mask_bbox(demo_mask)

    def bbox_latent(field):
        """Flatten the bbox region of (..., lat, lon) fields into feature vectors."""
        field = np.asarray(field, dtype=float)
        return field[..., BBOX[0], BBOX[1]].reshape(*field.shape[:-2], -1)

    def gt_reference_latents(days_back):
        """Daily GT states ending just before the rollout start (shrinks if the store is short)."""
        err = None
        for days in (days_back, days_back // 2, days_back // 4, 7):
            try:
                ds = get_gt_rollout(days, ROLLOUT_START - timedelta(days=days + 2))
                return bbox_latent(ds["2m_temperature"].values)
            except Exception as e:
                err = e
        raise RuntimeError(f"no GT cloud loadable: {err}")

    def flow_grids(T):
        """Noise levels s_t and Euler step sizes h_t of the sampling grid."""
        s = np.linspace(1000, 1, T) / 1000
        h = np.empty_like(s)
        h[:-1] = s[:-1] - s[1:]
        h[-1] = s[-1]
        return s, h

    def applied_lambda(method, eta, n):
        """lambda_t = w_t * a_t recorded in the schedule sidecar for one (method, eta, n)."""
        recs = [
            r for r in schedule_records
            if r["method"] == method and r["m"] == 0 and r["n"] == n
            and r["sweep"]["fgwnolr_eta"] == eta and r["sweep"]["GUIDANCE_DELTA"] == 0
        ]
        return np.asarray(recs[0]["w_t"], float) * np.asarray(recs[0]["a_t"], float) if recs else None

    def clean_pred_latents(eta, n):
        """Guided clean-pred trajectory over t (bbox latents), reconstructed from raw traces:
        gui_vfs = vfs - lam*grads*s;  z_T = res[-1] + gui_vfs[-1]*(h/s)[-1];
        x_hat_t = gui_final + ((res_t + vfs_t) - z_T) * c."""
        lam = applied_lambda("FGWNOLR", eta, n)
        if lam is None:
            return None
        res, vfs, grads, gui = (
            bbox_latent(load_temperature(store, fgwnolr_eta=eta).isel(m=0, n=n).values)
            for store in ("res.zarr", "vfs.zarr", "grads.zarr", "gui.zarr")
        )
        s, h = flow_grids(FLOW_T)
        z_final = res[-1] + (vfs[-1] - lam[-1] * grads[-1] * s[-1]) * (h[-1] / s[-1])
        return gui[None, :] + ((res + vfs) - z_final[None, :]) * RESID_SCALER

    pca_cloud = gt_reference_latents(60)
    pca_target = bbox_latent(
        get_gt_rollout(PCA_N + 2, ROLLOUT_START)["2m_temperature"].isel(time=PCA_N + 1).values
    ).ravel()
    pca_ung_final = bbox_latent(load_temperature("ung.zarr").isel(m=0, n=PCA_N).values).ravel()

    pca_trajs, pca_ung_gui = {}, {}
    for _eta in PCA_ETAS:
        _traj = clean_pred_latents(_eta, PCA_N)
        if _traj is not None:
            pca_trajs[f"FGWNOLR η={_eta}"] = _traj
            pca_ung_gui[f"FGWNOLR η={_eta}"] = bbox_latent(
                load_temperature("ung_gui.zarr", fgwnolr_eta=_eta).isel(m=0, n=PCA_N).values
            )
    print(f"cloud: {pca_cloud.shape} | trajectories: {list(pca_trajs)}")
    return PCA_N, pca_cloud, pca_target, pca_trajs, pca_ung_final, pca_ung_gui


@app.cell
def _(np, pca_cloud, pca_target, pca_trajs, pca_ung_final, pca_ung_gui):
    # ===== PCA frame fit on the reference cloud =====
    def pca_frame(reference, n_components=3):
        """PCA basis of `reference` rows; returns (project, explained_variance_ratio)."""
        mu = reference.mean(axis=0)
        _, sv, vt = np.linalg.svd(reference - mu, full_matrices=False)
        basis = vt[:n_components].T
        return (lambda x: (x - mu) @ basis), (sv ** 2 / (sv ** 2).sum())[:n_components]

    pca_project, pca_evr = pca_frame(pca_cloud)
    pca_cloud_proj = pca_project(pca_cloud)
    pca_traj_proj = {k: pca_project(v) for k, v in pca_trajs.items()}
    pca_ung_gui_proj = {k: pca_project(v) for k, v in pca_ung_gui.items()}
    pca_target_proj = pca_project(pca_target)
    pca_ung_proj = pca_project(pca_ung_final)
    print("explained variance (3 PCs):", np.round(pca_evr, 3), "| total:", round(float(pca_evr.sum()), 3))
    return (
        pca_cloud_proj,
        pca_evr,
        pca_target_proj,
        pca_traj_proj,
        pca_ung_gui_proj,
        pca_ung_proj,
    )


@app.cell
def _(mo):
    elev_slider = mo.ui.slider(0, 90, step=5, value=25, label="elev: ", show_value=True, debounce=True)
    azim_slider = mo.ui.slider(-180, 180, step=5, value=-60, label="azim: ", show_value=True, debounce=True)
    zoom_traj_checkbox = mo.ui.checkbox(label="zoom to trajectories")
    return azim_slider, elev_slider, zoom_traj_checkbox


@app.cell
def _(
    PCA_N,
    azim_slider,
    elev_slider,
    mo,
    np,
    pca_cloud_proj,
    pca_evr,
    pca_target_proj,
    pca_traj_proj,
    pca_ung_gui_proj,
    pca_ung_proj,
    plt,
    zoom_traj_checkbox,
):
    TRAJ_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

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
    for (_key, _pts), _color in zip(pca_traj_proj.items(), TRAJ_COLORS):
        _draw_trajectory(_ax3, _pts, _color, _key)
        _draw_twin(_ax3, pca_ung_gui_proj[_key], _color, f"ung_gui {_key}")
    _draw_marker(_ax3, pca_ung_proj, "ung", "#111111", marker="o", s=130, facecolors="none",
                 edgecolors="#111111", linewidths=2.0, label="ung rollout (final state)")
    _draw_marker(_ax3, pca_target_proj, "GT", "black", marker="*", s=320, color="black",
                 label="GT (valid day)")
    if zoom_traj_checkbox.value:
        _zoom_to(_ax3, list(pca_traj_proj.values()) + list(pca_ung_gui_proj.values())
                 + [pca_target_proj[None, :], pca_ung_proj[None, :]])
    _ax3.set_xlabel("PC1"); _ax3.set_ylabel("PC2"); _ax3.set_zlabel("PC3")
    _ax3.set_title(
        f"Clean-pred trajectories in the ERA5 PCA frame  (n={PCA_N+1}, bbox latents, EVR={pca_evr.sum():.0%})",
        loc="left",
    )
    _ax3.view_init(elev=elev_slider.value, azim=azim_slider.value)
    _ax3.legend(loc="upper left", fontsize=8)
    mo.vstack([mo.hstack([elev_slider, azim_slider, zoom_traj_checkbox], justify="start"), pca_fig], align="start")
    return (pca_fig,)


@app.cell
def _(PCA_N, Path, ROLLOUT_ID, mo, pca_fig, save_fig_button):
    # ===== thesis-figure export (high-dpi PNG + vector PDF) =====
    _msg = "export the 3D figure at the current view"
    if save_fig_button.value:
        _out_dir = Path("figures")
        _out_dir.mkdir(exist_ok=True)
        _stem = _out_dir / f"pca_trajectories_{ROLLOUT_ID.replace(':', '-')}_n{PCA_N + 1}"
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
def _(mo, np, pca_target, pca_trajs, pca_ung_gui, plot_trajectory):
    # ===== Fig. 5 analog: per-step deviation vs the GT latent (in-region) =====
    def rmse_to_target(traj):
        return np.sqrt(np.mean((traj - pca_target[None, :]) ** 2, axis=1))

    def deviation_alignment(guided, ung_gui):
        """cos(guidance displacement, direction from the ung_gui twin to GT), per flow step."""
        def cos(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        return [cos(g - u, pca_target - u) for g, u in zip(guided, ung_gui)]

    _rmse = {k: rmse_to_target(v) for k, v in pca_trajs.items()}
    _rmse |= {f"ung_gui {k}": rmse_to_target(v) for k, v in pca_ung_gui.items()}
    _alignment = {k: deviation_alignment(pca_trajs[k], pca_ung_gui[k]) for k in pca_trajs}
    mo.vstack(
        [
            mo.md(r"""
    ### Reading the two plots below

    **RMSE to GT latent over $t$** — for each trajectory, the root-mean-square difference (Kelvin,
    over the bbox pixels) between the clean prediction $\hat{x}_t$ and the ground-truth state at the
    step's valid day. The `ung_gui` curves (each run's unguided twin) show how far the plain forecast sits from reality. The guided
    curves are *expected* to end higher: the guidance target is $(1+\delta)\times$ the unguided
    reference — a deliberate counterfactual, not the truth. What matters is the **shape**: a smooth,
    monotone departure means the flow integrates the push coherently; spikes or late jumps would
    indicate pasted-on, off-manifold corrections.

    **Deviation alignment over $t$** — the cosine between the guidance-induced displacement
    $\hat{x}^{gui}_t - \hat{x}^{ung\_gui}_t$ and the true-error direction $x^{gt} - \hat{x}^{ung\_gui}_t$,
    per flow step. It asks: *does the pattern the guidance adds resemble the pattern by which the
    unguided forecast actually differs from real weather?* $+1$: the push looks like a correction
    toward reality; $0$: orthogonal to the true error (pure fabrication w.r.t. reality); negative:
    against it. Since our $\delta>0$ target warms the region, alignment tells whether the warming is
    realized with a physically plausible spatial pattern or a mask-shaped artifact.
    """),
            mo.hstack(
                [
                    plot_trajectory(_rmse, title="RMSE to GT latent over $t$",
                                    subtitle="bbox clean-pred vs GT state at valid day",
                                    xlabel="$t$", figsize=(11, 5), prepend_zero=False, start_index=1),
                    plot_trajectory(_alignment, title="Deviation alignment over $t$",
                                    subtitle=r"$\cos(\hat{x}^{gui}_t - \hat{x}^{ung\_gui}_t,\; x^{gt} - \hat{x}^{ung\_gui}_t)$",
                                    xlabel="$t$", figsize=(11, 5), prepend_zero=False, start_index=1),
                ],
                justify="start",
            ),
        ],
        align="start",
    )
    return


if __name__ == "__main__":
    app.run()
