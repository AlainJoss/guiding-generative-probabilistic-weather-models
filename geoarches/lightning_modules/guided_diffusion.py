import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from hydra.utils import instantiate
from tensordict.tensordict import TensorDict
from tqdm.auto import tqdm

from geoarches.backbones.dit import TimestepEmbedder
from geoarches.lightning_modules import BaseLightningModule
from geoarches.lightning_modules.base_module import AvgModule, load_module
from geoarches.utils.tensordict_utils import tensordict_apply, tensordict_cat

from geoarches.paths import STATS_PATH

A_T_MODES = [
    "gaussian", "linear", "logistic", "exponential", "gap-closing", "spike",
    "gaussian-spec", "linear-spec", "logistic-spec", "exponential-spec", "gap-closing-spec", "spike-spec",
    # spike@k: guide ONLY at the k-th flow step (1-based; eta is ignored). Offered up to
    # T-1 for the max T=25 flow steps (the last tick T is never guided); larger k on a
    # shorter run is clipped to the last tick.
    *[f"spike@{k}" for k in range(1, 25)],
    # spread@{start}-{end}: linear ramp 1 -> 0 over the flow-step window [start, end]
    # (1-based), ZERO outside (eta ignored). Parsed dynamically by alpha_t_profile -- NOT
    # enumerated here; the guided_rollout UI builds one string per selected range.
    # Distributed analogue of spike@k for the spike-vs-spread comparison.
]


def alpha_t_profile(mode: str, eta: float, T: int, floor: float = 0.01):
    """Guidance profile a_t in [floor, 1] over t = 0..T-1.

    eta in (0, 1] means, per mode:
      gaussian    end level E = floor + eta*(1-floor); peak 1 mid-flow
                  (eta->0 deep bell with tails at the floor, eta->1 flat)
      linear      end level: 1 -> F with F = floor + eta*(1-floor)
                  (eta->0 drops to the floor, eta->1 stays flat at 1)
      logistic    same endpoints as linear (1 -> F), S-shaped in between
      exponential same endpoints as linear (1 -> F), geometric decay F^x
                  (convex: front-loaded drop, flattening toward F)
      gap-closing the remaining-gap schedule (1-eta)^(t+1)
      spike       guide at ONE step: index round(eta*(T-1)) (eta->0 first step,
                  eta=1 last tick, which the sampler never guides anyway);
                  ZERO elsewhere (floor not applied)
      spike@k     guide at ONE step: the k-th flow step (1-based, so spike@1 is
                  the first step); eta is IGNORED, ZERO elsewhere; k past the
                  last tick is clipped to it (never guided by the sampler)
      spread@a-b  linear ramp 1 -> 0 over the flow-step window [a, b] (1-based); ZERO
                  outside; eta is IGNORED; a==b -> spike at a. Distributed analogue of
                  spike@k for the spike-vs-spread comparison.
    "-spec" suffix: time reversal (monotone modes) / vertical flip (gaussian).
    """
    specular = mode.endswith("-spec")
    base = mode[:-5] if specular else mode
    eta = float(np.clip(eta, 1e-6, 1.0))
    floor = float(np.clip(floor, 0.0, 0.99))
    x = np.linspace(0.0, 1.0, T)

    if base.startswith("spike@"):
        k = int(base.split("@", 1)[1])
        a = np.zeros(T)
        a[int(np.clip(k - 1, 0, T - 1))] = 1.0
        return a[::-1] if specular else a

    if base.startswith("spread@"):
        _lo, _hi = base.split("@", 1)[1].split("-")
        s0 = int(np.clip(int(_lo) - 1, 0, T - 1))   # 1-based -> 0-based flow index
        e0 = int(np.clip(int(_hi) - 1, 0, T - 1))
        if e0 < s0:
            s0, e0 = e0, s0
        a = np.zeros(T)
        if e0 == s0:
            a[s0] = 1.0                               # degenerate window -> spike at start
        else:
            w = np.arange(s0, e0 + 1)
            a[w] = 1.0 - (w - s0) / (e0 - s0)         # linear ramp 1 -> 0 over the window
        return a[::-1] if specular else a            # 0 outside the window; eta unused

    if base == "spike":
        a = np.zeros(T)
        a[int(round(eta * (T - 1)))] = 1.0
        return a[::-1] if specular else a

    if base == "gaussian":
        end = min(floor + eta * (1.0 - floor), 0.999)
        sigma = 0.5 / np.sqrt(2.0 * np.log(1.0 / end))
        a = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
        if specular:
            a = 1.0 + end - a  # vertical flip: ends at 1, valley at `end`
        return np.clip(a, floor, 1.0)
    if base == "linear":
        finish = floor + eta * (1.0 - floor)
        a = 1.0 + (finish - 1.0) * x
    elif base == "logistic":
        # same endpoints as linear; affinely normalized so the raw logistic
        # hits them exactly despite its asymptotic tails
        finish = floor + eta * (1.0 - floor)
        raw = 1.0 / (1.0 + np.exp(-10.0 * (0.5 - x)))
        raw = (raw - raw[-1]) / (raw[0] - raw[-1])
        a = finish + (1.0 - finish) * raw
    elif base == "exponential":
        # geometric decay hitting the linear/logistic endpoints exactly
        finish = floor + eta * (1.0 - floor)
        a = finish ** x
    elif base == "gap-closing":
        a = (1.0 - eta) ** (np.arange(T) + 1)
    else:
        raise ValueError(f"unknown a_t mode {mode!r}")

    if specular:
        a = a[::-1]
    return np.clip(a, floor, 1.0)


class GuidedFlow(BaseLightningModule):
    """
    Deterministic + generative-residual forecasting with guided flow sampling.

    FGWNOLR uses the clean-prediction loss gradient with a constant strength w
    and a flow-time profile a_t: u_guided = u - w * a_t * grad(L).
    A secant search chooses w to close the signed terminal target gap.

    The trace stores the gradient, velocity, and latent trajectory for analysis.
    The last flow step is unguided: it snaps the latent onto the predicted
    residual, leaving no subsequent model step to process a guidance kick.
    """

    # ---------------------------------------------------------------- init ---

    def __init__(
        self,
        cfg,
        name="diffusion",
        cond_dim=256,
        num_train_timesteps=1000,
        scheduler="flow",  # only available option
        prediction_type="sample",  # or velocity
        beta_schedule="squaredcos_cap_v2",
        beta_start=0.0001,
        beta_end=0.012,
        loss_weighting_strategy=None,
        conditional="",  # things that the model is conditioned
        load_deterministic_model=False,
        loss_delta_normalization=False,
        state_normalization=False,
        pow=2,
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-5,
        num_warmup_steps=1000,
        num_training_steps=300000,
        num_cycles=0.5,
        learn_residual=False,
        sd3_timestep_sampling=True,
        **kwargs,
    ):
        super().__init__()
        self.__dict__.update(locals())
        print("initializing GuidedFlow")

        self.scale_input_noise = 1.05  # input-noise inflation (not sigma)
        self.num_train_timesteps = 1000
        self.cond_dim = 256
        self.T = 25  # flow / sampling steps (overridden per rollout)

        self.cfg = cfg
        self.backbone = instantiate(cfg.backbone)
        self.embedder = instantiate(cfg.embedder)

        if isinstance(load_deterministic_model, str):
            self.det_model, _ = load_module(load_deterministic_model)
        else:
            self.det_model = AvgModule(load_deterministic_model)

        self.month_embedder = TimestepEmbedder(self.cond_dim)
        self.hour_embedder = TimestepEmbedder(self.cond_dim)
        self.timestep_embedder = TimestepEmbedder(self.cond_dim)

        # residual -> pangu-normalized scale, and state (de)normalization stats
        pangu_stats = torch.load(
            STATS_PATH / "pangu_norm_stats2_with_w.pt", weights_only=True
        )
        pangu_scaler = TensorDict(
            level=pangu_stats["level_std"], surface=pangu_stats["surface_std"]
        )
        scaler = TensorDict(
            **torch.load(
                STATS_PATH / "deltapred24_aws_denorm.pt", weights_only=False
            )
        )
        scaler["level"][-1] *= 3  # de-emphasize vertical velocity
        self.residual_to_pangu_scale = scaler / pangu_scaler
        self.data_mean = TensorDict(
            surface=pangu_stats["surface_mean"],
            level=pangu_stats["level_mean"],
        )
        self.data_std = TensorDict(
            surface=pangu_stats["surface_std"],
            level=pangu_stats["level_std"],
        )

    def move_objects_to_device(self):
        # device is assigned by the loading pipeline, after __init__
        self.residual_to_pangu_scale = self.residual_to_pangu_scale.to(self.device)
        self.data_mean = self.data_mean.to(self.device)
        self.data_std = self.data_std.to(self.device)
        self.generator = torch.Generator(self.device)
        print("initialized GuidedFlow")

    # ------------------------------------------------------- normalization ---

    def denormalize(self, batch):
        # a bare state TensorDict has "surface" at the top level; a batch dict
        # holds states under "*state*" keys
        if "surface" in batch:
            return batch * self.data_std + self.data_mean
        return {
            k: (v * self.data_std + self.data_mean if "state" in k else v)
            for k, v in batch.items()
        }

    def normalize(self, batch):
        if "surface" in batch:
            return (batch - self.data_mean) / self.data_std
        return {
            k: ((v - self.data_mean) / self.data_std if "state" in k else v)
            for k, v in batch.items()
        }

    # ------------------------------------------------------ flow primitives ---

    def embed_time(self, batch, t):
        times = pd.to_datetime(
            batch["timestamp"].detach().cpu().numpy(), unit="s"
        ).tz_localize(None)
        month_emb = self.month_embedder(torch.tensor(times.month).to(self.device))
        hour_emb = self.hour_embedder(torch.tensor(times.hour).to(self.device))
        timestep_emb = self.timestep_embedder(torch.tensor([t]).to(self.device))
        return month_emb + hour_emb + timestep_emb

    def velocity_input(self, z, batch):
        # [pred_state || prev_state || z]
        assert "pred_state" in batch
        x = tensordict_cat([batch["prev_state"], z], dim=1)
        return tensordict_cat([batch["pred_state"], x], dim=1)

    def velocity(self, batch, time_embedding, input_state, z_t, s_t):
        # residual estimate r_t, then u_t = (r_t - z_t) / s_t
        x = self.embedder.encode(batch["state"], input_state)
        x = self.backbone(x, time_embedding)
        r_t = self.embedder.decode(x)
        return (r_t - z_t).apply(lambda x: x / s_t)

    def guided_velocity(self, u_t, gui_vec, lambda_):
        return tensordict_apply(lambda u, g: u - lambda_ * g, u_t, gui_vec)

    def euler_step(self, z_t, u_t, h):
        return tensordict_apply(lambda z, u: z + h * u, z_t, u_t)

    def init_noise(self, x_cond, seed=None):
        if seed is not None:
            self.generator.manual_seed(seed)
        z_t = x_cond["state"].apply(
            lambda x: torch.empty_like(x).normal_(generator=self.generator)
        )
        return z_t * self.scale_input_noise

    def flow_timesteps(self):
        return torch.linspace(
            self.num_train_timesteps, 1, self.T, device=self.device
        )

    def step_factors(self, i, timesteps):
        # noise level s_t and Euler step size h (last step integrates to 0: h_T~0.001, h_t~0.046)
        t = timesteps[i]
        s_t = t / self.num_train_timesteps
        if i < len(timesteps) - 1:
            h = s_t - timesteps[i + 1] / self.num_train_timesteps
        else:
            h = s_t
        return t, s_t, h

    def clean_prediction(self, det_pred, z_t, u_t, s_t):
        # x_hat_t = x_det + sigma_r * (z_t + s_t * u_t), denormalized
        x_hat_norm = det_pred + (
            z_t + tensordict_apply(torch.mul, s_t, u_t)
        ) * self.residual_to_pangu_scale
        return self.denormalize(x_hat_norm)

    def final_prediction(self, det_pred, z_t):
        x_hat_norm = det_pred + tensordict_apply(
            torch.mul, z_t, self.residual_to_pangu_scale
        )
        return self.denormalize(x_hat_norm)

    # --------------------------------------- masked objective + autograd ---

    def masked_residual(self, x_hat_t, x_ref, delta_t, mask):
        # signed gap r = S(x_hat) - (1 + delta) * S(x_ref); masked_loss = r^2
        pred = sum((mask[k] * x_hat_t[k]).sum() for k in x_hat_t.keys())
        target = sum((mask[k] * x_ref[k]).sum() for k in x_ref.keys())
        return pred - (1 + delta_t) * target

    def masked_loss(self, x_hat_t, x_ref, delta_t, mask):
        return self.masked_residual(x_hat_t, x_ref, delta_t, mask) ** 2

    def guidance_gradient(self, x_hat_t, x_ref, delta_t, mask, z_t):
        # g_t = dL/dz of the masked loss at the clean prediction
        loss_ = self.masked_loss(x_hat_t, x_ref, delta_t, mask)
        return self.grad_wrt_z(loss_, z_t)

    def grad_wrt_z(self, loss_, z_t, create_graph: bool = False):
        keys = list(z_t.keys())
        grads = torch.autograd.grad(
            loss_,
            [z_t[k] for k in keys],
            retain_graph=create_graph,
            create_graph=create_graph,
            allow_unused=True,
        )
        return z_t.__class__(
            {
                k: torch.zeros_like(z_t[k]) if g is None else g
                for k, g in zip(keys, grads)
            },
            batch_size=z_t.batch_size,
            device=z_t.device,
        )

    # ------------------------------------------------------ sampling entry ---

    def sample(
        self,
        guidance_flag: bool,
        guidance_type: str,
        x_cond: dict,
        delta_n: torch.Tensor | None = None,
        mask: TensorDict | None = None,
        x_ref: TensorDict | None = None,
        seed: int | None = None,
        guidance_kwargs: dict | None = None,
    ):
        with torch.no_grad():
            det_pred = self.det_model(x_cond)
            x_cond["pred_state"] = det_pred
        x_cond = {k: v for k, v in x_cond.items() if "next" not in k}

        if not guidance_flag:
            z, sampling_trace = self.unguided_flow(x_cond, det_pred, seed)
        else:
            if guidance_type != "FGWNOLR":
                raise ValueError(f"Unknown guidance_type: {guidance_type}; expected FGWNOLR")

            z, sampling_trace = self._fgwnolr_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_n,
                mask=mask,
                x_ref=x_ref,
                seed=seed,
                **(guidance_kwargs or {}),
            )

        # deterministic core of this step -- persisted as gui_det / ung_det
        # (rollout.py pops it before stacking the per-t traces)
        sampling_trace["det_pred"] = det_pred.detach()

        # append the converged endpoint latent z_T onto the `res` trace so it holds the FULL
        # trajectory z_0..z_T (length T+1): the loop only appended the pre-step z_0..z_{T-1}, and z
        # is the converged endpoint. The reconstructed state x_t = x_det + sigma_r*z_t then reaches
        # the true final x_T at res[-1] with no separate z_T store. (vfs/grads stay length T.)
        sampling_trace["res"].append(z.detach().cpu())

        x_hat_norm = det_pred + tensordict_apply(
            torch.mul, z, self.residual_to_pangu_scale
        )
        return x_hat_norm, sampling_trace

    # ------------------------------------------------------- unguided pass ---

    def unguided_flow(self, x_cond, det_pred, seed=None):
        # traces the per-step clean-state estimate AND the raw latent z_t (res). The latent
        # lets the analysis reconstruct the ACTUAL unguided state x_t = x_det + sigma_res*z_t
        # (mirrors the guided pass, which traces "res" the same way in _guided_flow).
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.flow_timesteps()
        sampling_trace = defaultdict(list)

        for i in tqdm(range(len(timesteps))):
            t, s_t, h = self.step_factors(i, timesteps)
            with torch.no_grad():
                time_embedding = self.embed_time(x_cond, t)
                input_state = self.velocity_input(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                x_hat_t = self.clean_prediction(det_pred, z_t, u_t, s_t)
                sampling_trace["clean_preds"].append(x_hat_t.detach().cpu())
                sampling_trace["res"].append(z_t.detach().cpu())  # pre-step latent z_t (== guided "res")
                z_t = self.euler_step(z_t, u_t, h)

        return z_t, sampling_trace

    # ---------------------------------------------------- guided pass core ---

    def _guided_flow(
        self,
        guidance_name: str,
        x_cond: dict,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_ref: TensorDict,
        w_schedule: list,
        a_schedule: list[float],
        seed: int | None = None,
        record: bool = True,
    ):
        # Apply the raw loss-gradient kick w*a_t*g.
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.flow_timesteps()
        sampling_trace = defaultdict(list)

        for i in tqdm(range(len(timesteps)), desc=f"{guidance_name} sampling", disable=not record):
            t, s_t, h = self.step_factors(i, timesteps)
            z_t = z_t.apply(lambda x: x.detach().requires_grad_(True))

            with torch.enable_grad():
                time_embedding = self.embed_time(x_cond, t)
                input_state = self.velocity_input(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                x_hat_t_norm = det_pred + self.euler_step(
                    z_t, u_t, s_t
                ) * self.residual_to_pangu_scale
                x_hat_t = self.denormalize(x_hat_t_norm)
                grad_vec = self.guidance_gradient(x_hat_t, x_ref, delta_t, mask, z_t)

            g_norm = float(torch.sqrt(
                sum((grad_vec[k].detach() ** 2).sum() for k in grad_vec.keys())
            ).clamp_min(1e-30))
            if i == len(timesteps) - 1:
                # i == T-1 is the 0-based index of thesis flow step T (the final Euler step):
                # it snaps z onto the predicted residual (h = s), so a kick there is state
                # surgery no model step can re-process -> NEVER guided
                a_t, c_t = 0.0, 0.0
            else:
                a_t = float(a_schedule[i])
                c_t = g_norm
            lam_raw = float(w_schedule[i]) * a_t * c_t / g_norm
            gui_step = grad_vec.apply(lambda g: g * lam_raw)
            if record:
                # Per-step traces are only needed for the final rollout;
                # skip them during the strength search (record=False).
                sampling_trace["grads"].append(grad_vec.detach().cpu())
                sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())
                sampling_trace["res"].append(z_t.detach().cpu())

            u_t = self.guided_velocity(u_t=u_t, gui_vec=gui_step, lambda_=1.0)
            with torch.no_grad():
                z_t = self.euler_step(z_t, u_t, h)

        return z_t, sampling_trace

    def _final_guided_pass(
        self, guidance_name, w_schedule, a_schedule,
        x_cond, det_pred, delta_t, mask, x_ref, seed,
    ):
        # final pass with the optimized schedule, yielding the standard traces
        z_t, sampling_trace = self._guided_flow(
            guidance_name=guidance_name,
            x_cond=x_cond,
            det_pred=det_pred,
            delta_t=delta_t,
            mask=mask,
            x_ref=x_ref,
            w_schedule=w_schedule,
            a_schedule=a_schedule,
            seed=seed,
        )
        return z_t, sampling_trace

    # -------------------------- FGWNOLR: secant on a constant strength w ---

    def _fgwnolr_flow(
        self,
        x_cond,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_ref: TensorDict,
        seed: int,
        fgwnolr_w_init: float = 250.0,
        eta: float = 0.5,
        a_t_mode: str = "gap-closing",
    ):
        # Target realization by secant ROOT FINDING on the signed terminal gap xi(w).
        # For the fixed flow-time profile a_t, lam_t = w * a_t; each evaluation runs the
        # ACTUAL guided flow (record=False) with the SAME initial noise and returns the
        # signed endpoint gap xi(w) = M(x_gui(w)) - y*. Secant searches xi(w) = 0.
        # Safeguards: w >= 0,
        # step capped at 3x the last interval, eval cap, and the evaluated w with the
        # smallest |xi| wins if no root is reached.
        GAP_THRESHOLD = 1e-3   # |xi| <= 1e-3 (was L = xi^2 <= 1e-6); in the target's units
        MAX_EVALS = 30

        timesteps = self.flow_timesteps()
        T = len(timesteps)
        alpha_sched = alpha_t_profile(a_t_mode, eta, T).tolist()

        def evaluate(w):
            # run the actual guided flow at strength w and read off the signed terminal gap
            z_T, _ = self._guided_flow(
                "FGWNOLR", x_cond, det_pred, delta_t, mask, x_ref,
                w_schedule=[w] * T, a_schedule=alpha_sched, seed=seed,
                record=False,
            )
            gap = self.masked_residual(
                self.final_prediction(det_pred, z_T), x_ref, delta_t, mask
            )
            return float(gap.detach().cpu())

        history = []  # (w, gap)
        # w = 0 is one secant point: it reproduces the matched gui_ung endpoint, gap = xi(0).
        w_prev = 0.0
        gap_prev = evaluate(w_prev)
        history.append((w_prev, gap_prev))
        print(f"FGWNOLR eval: w={w_prev:.4f} gap={gap_prev:.6f}", flush=True)

        w_curr = max(float(fgwnolr_w_init), 0.0)
        # Search only when needed: skip if the unguided endpoint (w=0) already realizes the
        # target, or if w_init==0 collapses the two secant points (secant needs two distinct w).
        while abs(gap_prev) > GAP_THRESHOLD and w_curr != w_prev and len(history) < MAX_EVALS:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gap_curr = evaluate(w_curr)
            history.append((w_curr, gap_curr))
            print(f"FGWNOLR eval: w={w_curr:.4f} gap={gap_curr:.6f}", flush=True)

            if abs(gap_curr) <= GAP_THRESHOLD:
                break
            # secant toward the root of xi:  w <- w - xi * dw/d(xi)
            d_gap = gap_curr - gap_prev
            if not math.isfinite(d_gap) or abs(d_gap) < 1e-20 or w_curr == w_prev:
                break  # flat gap or duplicated point -> secant undefined

            step = -gap_curr * (w_curr - w_prev) / d_gap
            max_step = 3.0 * max(abs(w_curr - w_prev), 1e-6)
            step = max(-max_step, min(step, max_step))

            w_prev, gap_prev = w_curr, gap_curr
            w_curr = max(w_curr + step, 0.0)
            if abs(step) < 1e-3:
                break  # converged in w

        w_star, best_gap = min(history, key=lambda item: abs(item[1]))
        print(
            f"FGWNOLR w*={w_star:.4f} best|gap|={abs(best_gap):.6f} "
            f"history={[(round(w_, 3), round(g_, 6)) for w_, g_ in history]}",
            flush=True,
        )

        return self._final_guided_pass(
            "FGWNOLR", [w_star] * T, alpha_sched,
            x_cond, det_pred, delta_t, mask, x_ref, seed,
        )
