"""Pure metric functions for the flow-time tests (docs/report/chapters/5-evaluation-framework.tex):

    T1 Reliability -- the gap to the guidance target is driven to zero; mean/std of the FINAL gap.
    T2 Stability   -- the trajectory is free of oscillation; pushback + overshoot/oscillation.
    T3 Closeness   -- guided vs unguided (gui_ung) deviation; total guidance, PCA pathlength,
                      and per-step cosine vs the gui_ung twin.

No marimo. Inputs are numpy arrays produced by the loaders in ``src/ui/comparison.py`` and
the reduction cubes in ``src/reductions.py``. A field masked mean here is ``S(x) = sum(mask*x)``
with the sum-1 normalized mask from ``get_mask_2d`` (identical to ``get_masked_mean``).

t = flow step (Euler steps within one ``sample()`` call).
"""
import numpy as np


# --- T1 -------------------------------------------------------------------------------
def target_A(base, p_n):
    """Absolute guidance target ``A = (1 + p_n) * base`` (masked-mean baseline)."""
    return (1.0 + np.asarray(p_n, dtype=float)) * np.asarray(base, dtype=float)


def eps_trajectory(s_xhat_t, A):
    """Gap over flow steps and its normalized form.

    ``s_xhat_t`` (T,): masked-mean of the clean prediction per flow step; ``A``: scalar target.
    Returns ``(eps_t, eps_rel_t)`` with ``eps_rel = eps/eps_0`` (NaN if the initial gap ~ 0)."""
    eps = np.asarray(s_xhat_t, dtype=float) - float(A)
    eps0 = eps[0]
    eps_rel = eps / eps0 if abs(eps0) > 1e-12 else np.full_like(eps, np.nan)
    return eps, eps_rel


def final_gap(s_xhat_final, A):
    """Final gap ``S(x_final) - A`` (physical units)."""
    return float(s_xhat_final) - float(A)


# --- T2 -------------------------------------------------------------------------------
def overshoot_oscillation(eps):
    """T2 scalars from the gap curve. ``eps`` (T,): gap over flow steps (``eps[0]`` = initial gap).

    overshoot = how far the RELATIVE gap crosses past target = ``max(0, -min_t eps/eps_0)``;
    oscillation_count = number of sign changes of the per-step gap change ``diff(eps)``
    (0 = monotone convergence)."""
    eps = np.asarray(eps, dtype=float)
    eps = eps[np.isfinite(eps)]
    if eps.size < 2:
        return 0.0, 0
    eps0 = eps[0]
    eps_rel = eps / eps0 if abs(eps0) > 1e-12 else eps
    overshoot = float(max(0.0, -np.min(eps_rel)))
    sgn = np.sign(np.diff(eps))
    sgn = sgn[sgn != 0]
    osc = int(np.sum(sgn[1:] != sgn[:-1])) if sgn.size > 1 else 0
    return overshoot, osc


def pushback_trajectory(vfs, gui_vfs, mask):
    """T2 pushback masked-mean over flow steps: ``dvf_t = vfs_t - gui_vfs_{t-1}`` (``dvf_0=0``),
    reduced by ``S(dvf_t) = sum(mask*dvf_t)``. ``vfs``/``gui_vfs`` (T,lat,lon). Returns (T,)."""
    vfs = np.asarray(vfs, dtype=float)
    gv = np.asarray(gui_vfs, dtype=float)
    prev = np.concatenate([np.zeros_like(gv[:1]), gv[:-1]], axis=0)   # gui_vfs.shift(t=1), fill 0
    dvf = vfs - prev
    return (dvf * np.asarray(mask)).sum(axis=(-2, -1))


def step_reversal_frac(gui_vfs, mask):
    """T2 oscillation proxy: fraction of flow steps whose guided velocity REVERSES vs the
    previous step (mask-weighted cosine < 0, i.e. deflection > 90 deg)."""
    a = np.asarray(gui_vfs, dtype=float)
    b = np.concatenate([a[:1], a[:-1]], axis=0)                       # previous step
    m = np.asarray(mask, dtype=float)
    dot = (m * a * b).sum(axis=(-2, -1))
    na = np.sqrt((m * a * a).sum(axis=(-2, -1)))
    nb = np.sqrt((m * b * b).sum(axis=(-2, -1)))
    denom = na * nb
    cos = dot / np.where(denom > 1e-12, denom, np.nan)
    cos = cos[1:]                                                     # step 0 has no predecessor
    cos = cos[np.isfinite(cos)]
    return float(np.mean(cos < 0.0)) if cos.size else np.nan


def kick_per_step(gui_vec, h):
    """T3(a) per-step SQUARED kick norm (full-field L2): ``sum_space (gui_vec_t * h_t)^2``.
    ``gui_vec`` (T,lat,lon), ``h`` (T,). Returns (T,) -- feed to ``total_guidance``."""
    gv = np.asarray(gui_vec, dtype=float)
    hh = np.asarray(h, dtype=float)[:, None, None]
    return ((gv * hh) ** 2).sum(axis=(-2, -1))


# --- T3 -------------------------------------------------------------------------------
def total_guidance_normal(gui_vec, h, c):
    """T3(a) total applied guidance in NORMAL (physical field) space, summed over flow steps:

        G = sum_t || c * h_t * gui_vec_t ||_2   (full-field L2)

    i.e. the per-step physical clean-pred displacement produced by the applied guidance
    velocity ``gui_vec_t`` (mapped from latent to physical by the residual scaler ``c``),
    integrated over the flow. Method-agnostic (identical for spike and continuous schedules)
    and convention-robust (``gui_vec`` reconstructed from stored primitives). PCA space is
    used only for path lengths, never for this magnitude. ``gui_vec`` (T,lat,lon), ``h`` (T,)."""
    gv = np.asarray(gui_vec, dtype=float)
    hh = np.asarray(h, dtype=float)[:, None, None]
    per_t = np.sqrt(((float(c) * hh * gv) ** 2).sum(axis=(-2, -1)))
    return float(per_t.sum())


def total_guidance(kick_lat_l2_t):
    """(legacy, latent units) total applied guidance ``= sum_t ||kick_t||`` where
    ``kick_lat_l2_t`` is the per-step SQUARED kick norm. Prefer ``total_guidance_normal``."""
    k = np.asarray(kick_lat_l2_t, dtype=float)
    return float(np.nansum(np.sqrt(np.clip(k, 0.0, None))))


def total_deviance(gui_vec, s, c, bbox):
    """T3 total deviance from the UNGUIDED STEPS inside the guided flow: the sum over flow
    steps of the per-pixel RMS norm (over the mask-region ``bbox``) of the clean-pred guidance
    contribution ``c * s_t * gui_vec_t`` -- i.e. how far each guided step lands from what the
    *unguided* step would give FROM THE SAME STATE (``base - kick`` in comparison.py).

    Unlike comparing to a separate same-seed unguided twin (which diverges cumulatively and
    conflates WHEN guidance acts with HOW MUCH), this sums the ACTUAL per-step interventions,
    so it is applicable to any schedule -- spike or continuous. ``gui_vec`` (T, lat, lon)
    is the applied guidance velocity ((vfs - gui_vfs)/s); ``s`` (T,) the noise levels; ``c``
    the residual scaler."""
    gv = np.asarray(gui_vec, dtype=float)[..., bbox[0], bbox[1]]   # (T, bh, bw)
    ss = np.asarray(s, dtype=float)[:, None, None]
    contrib = float(c) * ss * gv                                   # per-step clean-pred guidance kick
    F = gv.shape[-1] * gv.shape[-2]
    return float(np.sqrt((contrib ** 2).sum(axis=(-2, -1)) / F).sum())


def step_sums(base, kick):
    """T3 total guidance-step and pushback-step lengths (per-pixel RMS over F), summed over
    flow steps -- exactly the two moves visualized in intensity_comparison's pingpong anatomy:

        guidance step_t = ||kick_t - base_t||        (base_t -> kick_t: the applied kick)
        pushback step_t = ||base_{t+1} - kick_t||    (kick_t -> base_{t+1}: the flow pull-back)

    ``base`` (gui_ung_over_t) and ``kick`` (guided) are the (T, F) bbox-latent branch pair from
    ``clean_pred_branches_primitive``. Returns ``(guidance_sum, pushback_sum)``. Applicable to
    any schedule (the sum of the actual per-step moves inside the guided flow)."""
    base = np.asarray(base, dtype=float)
    kick = np.asarray(kick, dtype=float)
    f = np.sqrt(base.shape[1])
    guidance = np.linalg.norm(kick - base, axis=1) / f
    pushback = np.linalg.norm(base[1:] - kick[:-1], axis=1) / f
    return float(guidance.sum()), float(pushback.sum())


def step_cosine(g, u):
    """T3(c) per-step cosine between the guided step and the ``gui_ung`` twin step, in full
    masked-region field space. ``g``, ``u`` (T, F): bbox-latent trajectories.
    Returns ``(cos_t (T-1,), mean_cos)``."""
    dg = np.diff(np.asarray(g, dtype=float), axis=0)
    du = np.diff(np.asarray(u, dtype=float), axis=0)
    denom = np.linalg.norm(dg, axis=1) * np.linalg.norm(du, axis=1)
    cos = np.einsum("tf,tf->t", dg, du) / np.where(denom > 1e-12, denom, np.nan)
    mean_cos = float(np.nanmean(cos)) if np.isfinite(cos).any() else np.nan
    return cos, mean_cos


# --- Interference (measured on the guidance-convergence plot, average mask space) -----
def interference_from_convergence(states, land_ung):
    """All interference metrics read off the guidance-convergence plot for the TARGET
    variable, in average (masked-mean) space. ``states`` (T+1,) = masked mean - target
    (converging to 0, anchored at noise); ``land_ung`` (T,) = the raw-flow Euler landing.

        flow_move_t = land_ung_t - state_t     (the flow's own move; red candle)
        gui_move_t  = state_{t+1} - land_ung_t  (the guidance move; blue candle)

    Returns a dict:
      total_guidance   = sum_t |gui_move_t|                      (total guidance applied)
      pushback_count   = # steps whose flow_move pushes AWAY from target (same sign as state)
      pushback_amount  = sum of |flow_move_t| over those steps
      overshoot_count  = # steps where the state is PAST the target (opposite sign to the
                         initial gap state_0)
      overshoot_amount = total |state_t| past the target over those steps
    """
    s = np.asarray(states, dtype=float)
    lu = np.asarray(land_ung, dtype=float)
    T = lu.shape[0]
    flow = lu - s[:T]
    gui = s[1:] - lu
    total_guid = float(np.nansum(np.abs(gui)))
    pb = (flow * s[:T]) > 0                         # flow move increases |state| -> pushback
    pushback_count = int(np.nansum(pb))
    pushback_amount = float(np.nansum(np.abs(flow[pb])))
    s0 = s[0]
    excursion = (-np.sign(s0) * s) if s0 != 0 else np.zeros_like(s)   # >0 = past target
    ov = excursion > 0
    overshoot_count = int(np.nansum(ov))
    overshoot_amount = float(np.nansum(excursion[ov]))
    return {"total_guidance": total_guid,
            "pushback_count": pushback_count, "pushback_amount": pushback_amount,
            "overshoot_count": overshoot_count, "overshoot_amount": overshoot_amount}


# --- Realism ---------------------------------------------------------------------------
def realism_tv(gui_field, ung_field, mask):
    """Realism = total-variation distance between the normalized guidance-change field
    ``|gui - ung|`` and the (normalized) ``mask``, over the full field:

        p = |gui - ung| / sum|gui - ung|,   q = mask / sum(mask),   TV = 0.5 * sum|p - q|.

    0 = the guidance change is distributed exactly like the mask footprint; larger = the
    guidance leaks beyond / concentrates differently from the mask (less realistic control).
    ``gui_field``/``ung_field``/``mask`` are (lat, lon) for the target channel."""
    g = np.abs(np.asarray(gui_field, dtype=float) - np.asarray(ung_field, dtype=float))
    gs = g.sum()
    if not np.isfinite(gs) or gs <= 0:
        return np.nan
    p = g / gs
    q = np.asarray(mask, dtype=float)
    q = q / q.sum()
    return float(0.5 * np.abs(p - q).sum())


# --- aggregation ----------------------------------------------------------------------
def aggregate_mean_std(values):
    """``(mean, std, n_pool)`` over a pool of scalars, ignoring NaN (population std)."""
    v = np.asarray(list(values), dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std()), int(v.size)
