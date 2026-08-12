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

    Guidance methods (dispatched by `sample`):
      FGWNOLR - secant search on a constant strength w; lambda_t = w * a_t
      FGWNOGAP - exact per-step closure of the masked gap along a schedule
      FGWFREE - Adam optimization of the full lambda trajectory with a
                kick-energy regularizer (no w / a_t split)
      UG - backward Universal Guidance: closed-form clean-space shift, a_t-gated

    Unified schedule convention: every method applies the UNIT gradient,
    u~ = u - (w * a_t * c_t) * g/||g||, so lambda_hat = w*a_t*c_t is the kick
    norm. w = scalar scale, a_t = dimensionless O(1) profile, c_t = measured
    magnitude carrier (||g|| for NOLR, Newton magnitude for NOGAP, ||g||/h for
    FREE). The sidecar records
    {w_t, a_t, c_t, g_norm_t}; the raw-gradient multiplier is w*a*c/||g||.
    The LAST flow step is never guided in any method: it snaps z onto the
    predicted residual (h = s), so a kick there would be raw state surgery
    with no model step left to re-process it.
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

    # TODO: not sure
    def _vjp(self, outputs_td, inputs_td, v_td, create_graph: bool = False):
        # J^T v with J = d(outputs)/d(inputs), per TensorDict key
        keys = list(inputs_td.keys())
        grads = torch.autograd.grad(
            [outputs_td[k] for k in keys],
            [inputs_td[k] for k in keys],
            grad_outputs=[v_td[k] for k in keys],
            retain_graph=create_graph,
            create_graph=create_graph,
            allow_unused=True,
        )
        return inputs_td.__class__(
            {
                k: torch.zeros_like(inputs_td[k]) if g is None else g
                for k, g in zip(keys, grads)
            },
            batch_size=inputs_td.batch_size,
            device=inputs_td.device,
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
            flows = {
                "FGWNOLR": self._fgwnolr_flow,
                "FGWNOGAP": self._fgwnogap_flow,
                "FGWFREE": self._fgwfree_flow,
                "UG": self._ug_flow,
            }
            if guidance_type not in flows:
                raise ValueError(f"Unknown guidance_type: {guidance_type}")
            
            z, sampling_trace = flows[guidance_type](
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

        x_hat_norm = det_pred + tensordict_apply(
            torch.mul, z, self.residual_to_pangu_scale
        )
        return x_hat_norm, sampling_trace

    # ------------------------------------------------------- unguided pass ---

    def unguided_flow(self, x_cond, det_pred, seed=None):
        # traces the per-step clean-state estimate -> full-t gui_ung trajectory
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
        c_fn=None,
    ):
        # guided sampling on the UNIT gradient: u~ = u - (w*a_t*c_t) * g/||g||,
        # so lambda_hat = w*a_t*c_t IS the applied kick norm. c_fn(i, u_t, g_t,
        # g_norm, h) supplies the magnitude carrier c_t; the default c_t = ||g_t||
        # reproduces the raw-gradient kick w*a*g exactly (FGWNOLR). Records the
        # raw trace primitives (grads = dL/dz, vfs = s_t*u_t, res = z_t) and the
        # {w_t, a_t, c_t, g_norm_t} sidecar; the raw-gradient multiplier used by
        # reconstructions is w*a*c / g_norm.
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.flow_timesteps()
        sampling_trace = defaultdict(list)
        w_t_trace, a_t_trace, c_t_trace, g_norm_trace = [], [], [], []

        for i in tqdm(range(len(timesteps)), desc=f"{guidance_name} sampling"):
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
                # the last Euler step snaps z onto the predicted residual (h = s);
                # a kick there is state surgery no model step can re-process ->
                # NEVER guided (uniform across methods)
                a_t, c_t = 0.0, 0.0
            else:
                a_t = float(a_schedule[i])
                c_t = float(c_fn(i, u_t, grad_vec, g_norm, float(h))) if c_fn is not None else g_norm
            lam_raw = float(w_schedule[i]) * a_t * c_t / g_norm
            gui_step = grad_vec.apply(lambda g: g * lam_raw)
            w_t_trace.append(float(w_schedule[i]))
            a_t_trace.append(a_t)
            c_t_trace.append(c_t)
            g_norm_trace.append(g_norm)

            sampling_trace["grads"].append(grad_vec.detach().cpu())
            sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())
            sampling_trace["res"].append(z_t.detach().cpu())

            u_t = self.guided_velocity(u_t=u_t, gui_vec=gui_step, lambda_=1.0)
            with torch.no_grad():
                z_t = self.euler_step(z_t, u_t, h)

        sampling_trace["guidance_schedule"] = {
            "w_t": w_t_trace, "a_t": a_t_trace,
            "c_t": c_t_trace, "g_norm_t": g_norm_trace,
        }
        return z_t, sampling_trace

    def _final_guided_pass(
        self, guidance_name, w_schedule, a_schedule,
        x_cond, det_pred, delta_t, mask, x_ref, seed, w_star=None,
        c_fn=None,
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
            c_fn=c_fn,
        )
        if w_star is not None:
            sampling_trace["w_star"] = float(w_star)  # -> w_star.json sidecar
        return z_t, sampling_trace

    # ------------------------ UG: Universal-Guidance backward shift --------

    def _ug_flow(
        self,
        x_cond,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_ref: TensorDict,
        seed: int,
        eta: float = 0.5,
        a_t_mode: str = "spike@1",
    ):
        # Backward universal guidance (Bansal et al., arXiv 2302.07121, Eq. 7/9)
        # adapted to the residual flow: at a_t-gated steps, apply the CLOSED-FORM
        # loss-minimizing clean-space shift instead of a gradient kick. For the
        # quadratic masked loss L = (S(x_hat) - A)^2 with linear S, the paper's
        # m-step GD from Delta=0 converges to the minimum-norm exact solution
        #   Delta_x = -e * M / ||M||^2   (e = signed residual, M = mask weights,
        # denormalized units), which zeroes the residual of the implied clean
        # prediction. The paper's epsilon substitution (Eq. 9) becomes a velocity
        # correction: x_hat = denorm(det + (z + s*u) * rtps)  =>
        #   Delta_u = Delta_x / (data_std * rtps * s_t),   u <- u + a_t * Delta_u.
        # No outer optimization, no w; the remaining (contractive) flow acts on
        # the shift untouched -- the contrast to FGWNOLR's optimized kick.
        # Schedule semantics: the kick is along the MASK PATTERN, not the unit
        # gradient; c_t = ||Delta_u|| so lambda_hat = w*a_t*c_t is the applied
        # kick norm (w_t = 1 throughout). No w_star sidecar (nothing optimized).
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.flow_timesteps()
        a_sched = alpha_t_profile(a_t_mode, eta, len(timesteps)).tolist()
        m2 = float(sum((mask[k] ** 2).sum() for k in mask.keys()))
        sampling_trace = defaultdict(list)
        w_t_trace, a_t_trace, c_t_trace, g_norm_trace = [], [], [], []

        for i in tqdm(range(len(timesteps)), desc="UG sampling"):
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
                # recorded for the traces only; the UG kick needs no gradients
                grad_vec = self.guidance_gradient(x_hat_t, x_ref, delta_t, mask, z_t)

            g_norm = float(torch.sqrt(
                sum((grad_vec[k].detach() ** 2).sum() for k in grad_vec.keys())
            ).clamp_min(1e-30))
            # the last Euler step snaps z onto the predicted residual (h = s);
            # NEVER guided (uniform convention across methods)
            a_t = 0.0 if i == len(timesteps) - 1 else float(a_sched[i])

            e = float(self.masked_residual(x_hat_t.detach(), x_ref, delta_t, mask))
            delta_x = mask.apply(lambda m: (-e / m2) * m)  # exact minimum-norm shift
            delta_u = (delta_x / self.data_std / self.residual_to_pangu_scale).apply(
                lambda x: x / float(s_t)
            )
            c_t = float(torch.sqrt(
                sum((delta_u[k].detach() ** 2).sum() for k in delta_u.keys())
            ))

            w_t_trace.append(1.0)
            a_t_trace.append(a_t)
            c_t_trace.append(c_t)
            g_norm_trace.append(g_norm)
            sampling_trace["grads"].append(grad_vec.detach().cpu())
            sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())
            sampling_trace["res"].append(z_t.detach().cpu())

            if a_t != 0.0:
                # guided_velocity subtracts gui_vec, so pass -a_t * Delta_u
                gui_step = delta_u.apply(lambda x: (-a_t) * x)
                u_t = self.guided_velocity(u_t=u_t, gui_vec=gui_step, lambda_=1.0)
            with torch.no_grad():
                z_t = self.euler_step(z_t, u_t, h)

        sampling_trace["guidance_schedule"] = {
            "w_t": w_t_trace, "a_t": a_t_trace,
            "c_t": c_t_trace, "g_norm_t": g_norm_trace,
        }
        return z_t, sampling_trace

    # --------------------------------------- shared lambda-gradient eval ---

    def _flow_loss_and_lambda_grads(
        self, x_cond, det_pred, delta_t, mask, x_ref, seed, lam_sched,
    ):
        """
        Run one guided pass with per-step multipliers lam_sched (a list of T
        floats; step i kicks u~_i = u_i - lam_i * g_i) and return
        (gap_loss, dL_dlam, g_norm2, resid):
          gap_loss    final masked loss L(z_T) = resid^2
          resid       the SIGNED final masked residual e (for Gauss-Newton fits)
          dL_dlam[i]  exact dL/dlam_i under the frozen-g first-order scheme
          g_norm2[i]  ||g_i||^2  (for regularizers)

        Scheme: freeze g_i, so the step map has dz_{i+1}/dz_i = I + h_i du/dz.
        The forward pass caches the leaf z_i, detached g_i and h_i; a backward
        adjoint loop then gives dL/dlam_i = -h_i <adj_{i+1}, g_i> with the adjoint
        adj = dL/dz propagated by adj <- adj + h_i (du/dz)^T adj (one VJP/step).
        """
        timesteps = self.flow_timesteps()
        T = len(timesteps)
        z_t = self.init_noise(x_cond, seed)
        z_cache, g_cache, h_cache = [], [], []          # per-step leaves for the adjoint
        g_norm2 = torch.zeros(T, device=self.device)

        # ---- forward: integrate the guided flow, caching what the adjoint needs
        for i in range(T):
            t, s_t, h = self.step_factors(i, timesteps)
            z_t = z_t.detach().apply(lambda x: x.requires_grad_(True))

            with torch.enable_grad():
                time_embedding = self.embed_time(x_cond, t)
                input_state = self.velocity_input(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                # clean-state estimate x_hat = denorm(det + (z + s*u) * rtps)
                x_hat_t = self.denormalize(
                    det_pred + self.euler_step(z_t, u_t, s_t) * self.residual_to_pangu_scale
                )
                g_t = self.guidance_gradient(x_hat_t, x_ref, delta_t, mask, z_t)

            z_cache.append(z_t.detach())
            g_cache.append(g_t.detach())
            h_cache.append(float(h))
            g_norm2[i] = sum((g_t[k].detach() ** 2).sum() for k in g_t.keys())

            # last step is never guided (see _guided_flow); otherwise kick by lam_i
            lam_i = 0.0 if i == T - 1 else float(lam_sched[i])
            gui_step = g_t.apply(lambda g: g * lam_i)
            u_t = self.guided_velocity(u_t=u_t, gui_vec=gui_step, lambda_=1.0)
            z_t = self.euler_step(z_t, u_t, h).detach()

        # ---- final loss and adjoint seed  adj = dL/dz_T
        z_T = z_t.detach().apply(lambda x: x.requires_grad_(True))
        with torch.enable_grad():
            x_hat_gui = self.final_prediction(det_pred, z_T)
            residual = self.masked_residual(x_hat_gui, x_ref, delta_t, mask)
            gap_loss = residual ** 2
        adj = self.grad_wrt_z(gap_loss, z_T)

        # ---- backward: dL/dlam_i, then push the adjoint back through step i
        dL_dlam = torch.zeros(T, device=self.device)
        for i in range(T - 1, -1, -1):
            t, s_t, h = self.step_factors(i, timesteps)
            dL_dlam[i] = -h_cache[i] * sum(
                (adj[k] * g_cache[i][k]).sum() for k in adj.keys()
            )
            z_i = z_cache[i].detach().apply(lambda x: x.requires_grad_(True))
            with torch.enable_grad():
                time_embedding = self.embed_time(x_cond, t)
                input_state = self.velocity_input(z_i, x_cond)
                u_i = self.velocity(x_cond, time_embedding, input_state, z_i, s_t)
            vjp = self._vjp(u_i, z_i, adj)
            adj = tensordict_apply(lambda a, v: a + h * v, adj, vjp)

        dL_dlam[-1] = 0.0  # lambda_{T-1} is never applied -> derivative is exactly 0

        return (
            float(gap_loss.detach().cpu()), dL_dlam.detach(),
            g_norm2.detach(), float(residual.detach().cpu()),
        )

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
        # dL/dw is an exact scalar derivative -> solve dL/dw = 0 by secant (no
        # learning rate, no iteration hyper). Safeguards:
        # w >= 0, step growth capped at 3x, eval cap, best-loss w wins.
        LOSS_THRESHOLD = 1e-6
        MAX_EVALS = 30

        timesteps = self.flow_timesteps()
        alpha_sched = alpha_t_profile(a_t_mode, eta, len(timesteps)).tolist()

        def evaluate(w):
            # loss and the exact scalar slope dL/dw at strength w. lam_t = w*a_t,
            # so by the chain rule dL/dw = sum_t a_t * dL/dlam_t.
            gap, dL_dlam, _, _ = self._flow_loss_and_lambda_grads(
                x_cond, det_pred, delta_t, mask, x_ref, seed,
                lam_sched=[w * a for a in alpha_sched],
            )
            slope = float(sum(a * d for a, d in zip(alpha_sched, dL_dlam.tolist())))
            return gap, slope

        history = []  # (w, loss, slope=dL/dw)
        w_prev = max(float(fgwnolr_w_init), 0.0)
        loss_prev, slope_prev = evaluate(w_prev)
        history.append((w_prev, loss_prev, slope_prev))
        print(f"FGWNOLR eval: w={w_prev:.4f} loss={loss_prev:.6f} dL/dw={slope_prev:.3e}", flush=True)

        # bootstrap the secant with a second point: a 10% move against the slope
        step0 = 0.1 * max(abs(w_prev), 1.0)
        w_curr = max(w_prev - math.copysign(step0, slope_prev), 0.0)

        while loss_prev > LOSS_THRESHOLD and len(history) < MAX_EVALS:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            loss_curr, slope_curr = evaluate(w_curr)
            history.append((w_curr, loss_curr, slope_curr))
            print(f"FGWNOLR eval: w={w_curr:.4f} loss={loss_curr:.6f} dL/dw={slope_curr:.3e}", flush=True)

            if loss_curr <= LOSS_THRESHOLD:
                break
            # secant toward the root of the slope:  w <- w - slope * dw/d(slope)
            d_slope = slope_curr - slope_prev
            if not math.isfinite(d_slope) or abs(d_slope) < 1e-20 or w_curr == w_prev:
                break  # flat derivative or duplicated point -> secant undefined

            step = -slope_curr * (w_curr - w_prev) / d_slope
            max_step = 3.0 * max(abs(w_curr - w_prev), 1e-6)
            step = max(-max_step, min(step, max_step))

            w_prev, slope_prev, loss_prev = w_curr, slope_curr, loss_curr
            w_curr = max(w_curr + step, 0.0)
            if abs(step) < 1e-3:
                break  # converged in w

        w_star, best_loss, _ = min(history, key=lambda item: item[1])
        print(
            f"FGWNOLR w*={w_star:.4f} best_loss={best_loss:.6f} "
            f"history={[(round(w_, 3), round(l_, 6)) for w_, l_, _ in history]}",
            flush=True,
        )

        return self._final_guided_pass(
            "FGWNOLR", [w_star] * len(timesteps), alpha_sched,
            x_cond, det_pred, delta_t, mask, x_ref, seed, w_star=w_star,
        )

    # ------------------------- FGWNOGAP: exact per-step gap closure ---

    def _fgwnogap_flow(
        self,
        x_cond,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_ref: TensorDict,
        seed: int,
        eta: float = 1.0,
        a_t_mode: str = "gap-closing",
    ):
        # Anchored gap closure: aim the signed residual at the deterministic
        # remaining-gap PATH r_target(t) = a_t * r_0 (a_t = a_t_profile(a_t_mode,
        # eta); the classic behavior is a_t_mode="gap-closing" -> (1-eta)^(t+1))
        # with a Newton step along g = 2 r dS/dz:
        #   gui_step = 2 r (r - r_target) g / (h ||g||^2).
        # Drift and overshoot are corrected against the prescribed path. Non-
        # monotone or non-closing profiles are allowed: the gap then FOLLOWS that
        # path (it may reopen or stay open at the end, by design).
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.flow_timesteps()
        a_sched = alpha_t_profile(a_t_mode, eta, len(timesteps))
        sampling_trace = defaultdict(list)
        scale_trace, g_norm_trace = [], []

        for i in tqdm(range(len(timesteps)), desc="FGWNOGAP sampling"):
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
                resid = self.masked_residual(x_hat_t, x_ref, delta_t, mask)
                loss_sq = resid ** 2
                gui_vec = self.grad_wrt_z(loss_sq, z_t)

            resid_val = float(resid.detach())
            if i == 0:
                r_0 = resid_val                        # initial gap r_0
            r_target = float(a_sched[i]) * r_0         # target gap on the prescribed path

            g_norm2 = sum((gui_vec[k] ** 2).sum() for k in gui_vec.keys())
            if i == len(timesteps) - 1:
                scale = torch.zeros(())  # the last flow step is never guided
            else:
                # Newton multiplier: lambda_t = 2 r (r - r_target) / (h ||g||^2)
                scale = (2.0 * resid_val * (resid_val - r_target)) / (h * g_norm2.clamp_min(1e-30))
            gui_step = gui_vec.apply(lambda g: g * scale)
            scale_trace.append(float(scale))
            g_norm_trace.append(float(torch.sqrt(g_norm2).clamp_min(1e-30)))

            print(
                f"FGWNOGAP step: t={i} r={resid_val:.6f} r_target={r_target:.6f} "
                f"scale(=lambda_t)={float(scale):.4f}",
                flush=True,
            )

            sampling_trace["grads"].append(gui_vec.detach().cpu())
            sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())
            sampling_trace["res"].append(z_t.detach().cpu())

            u_t = self.guided_velocity(u_t=u_t, gui_vec=gui_step, lambda_=1.0)
            with torch.no_grad():
                z_t = self.euler_step(z_t, u_t, h)

        # unit-gradient convention: lambda_hat = w*a*c = ||kick||; `scale` was the
        # raw-gradient multiplier, so c = scale * ||g|| / a (w = 1: no scale knob)
        a_theory = [float(a_sched[i]) for i in range(len(scale_trace))]
        c_impl = [
            _sc * _gn / _a if _a > 0 else float("nan")
            for _sc, _gn, _a in zip(scale_trace, g_norm_trace, a_theory)
        ]
        sampling_trace["guidance_schedule"] = {
            "w_t": [1.0] * len(scale_trace), "a_t": a_theory,
            "c_t": c_impl, "g_norm_t": g_norm_trace,
        }

        return z_t, sampling_trace

    # ------------- FGWFREE: full lambda trajectory, kick-regularized ---

    def _fgwfree_flow(
        self,
        x_cond,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_ref: TensorDict,
        seed: int,
        phi: float = 0.1,
    ):
        # Optimize the per-step guidance kick k_t = h_t * lambda_t jointly (no
        # w / a_t split) with Adam on
        #   J(k) = L_gap(z_T) + phi * sum_t ||k_t g_t||^2.
        # Kick space keeps the coordinates comparable and the phi-curvature
        # uniform: the last flow step has h ~ 0, so lambda_t = k_t / h_t is NOT
        # comparable across steps and would ill-condition a shared lr.
        # Each Adam step costs one forward + one frozen-g adjoint (same memory
        # profile as an FGWNOLR evaluation; no autograd through the unrolled
        # flow). Stops on closed gap, objective plateau, or the eval cap; the
        # best-J kick runs the final traced pass with a_t profile = kick shape.
        LOSS_THRESHOLD = 1e-6   # gap part considered closed
        REL_TOL = 1e-3          # relative objective plateau
        MAX_EVALS = 30
        INIT_DAMP = 0.5         # first-order init overshoots -> take half the jump
        LR_FRAC = 0.05          # Adam lr as a fraction of the init scale
        LR_MIN = 1.0

        timesteps = self.flow_timesteps()
        T = len(timesteps)
        # step sizes h_t, used to convert kick <-> lambda (lambda_t = kick_t / h_t)
        h_vec = torch.tensor(
            [float(self.step_factors(i, timesteps)[2]) for i in range(T)],
            device=self.device,
        )

        # probe at kick = 0 (unguided path): the linearization is linear (gap) +
        # quadratic (reg), minimized in closed form at -dJ/dkick / (2 phi ||g||^2)
        # -- the phi-aware regularized Newton step. Start there (damped) instead
        # of crawling from zero, and match Adam's lr to the discovered scale.
        gap0, dL_dlam0, g_norm2_0, resid0 = self._flow_loss_and_lambda_grads(
            x_cond, det_pred, delta_t, mask, x_ref, seed, lam_sched=[0.0] * T,
        )
        if phi > 0 and abs(resid0) > 1e-12:
            dJ_dkick0 = dL_dlam0 / h_vec                          # chain rule: /h
            reg_curv = (2.0 * phi * g_norm2_0).clamp_min(1e-30)   # regularizer curvature
            # Gauss-Newton: the SIGNED gap e is modeled linear and L = e^2 stays
            # quadratic, so the shrinkage bounds the jump by "close the gap
            # exactly" as phi -> 0 (the plain linear-loss model promises gap
            # reductions far beyond zero and detonates the flow at small phi);
            # for large phi it reduces to the regularized-Newton formula.
            shrink = 1.0 / (1.0 + float((dJ_dkick0 ** 2 / (4.0 * resid0 ** 2 * reg_curv)).sum()))
            kick = (INIT_DAMP * shrink * (-dJ_dkick0 / reg_curv)).clamp_(min=0.0)
            lr = max(LR_MIN, LR_FRAC * float(kick.max()))
        else:
            kick = torch.zeros(T, device=self.device)  # gap closed or unregularized
            lr = 50.0
        print(
            f"FGWFREE init: gap(0)={gap0:.6f} |kick0|max={float(kick.max()):.3f} lr={lr:.3f}",
            flush=True,
        )
        optimizer = torch.optim.Adam([kick], lr=lr)

        # kick = 0 is a legitimate candidate (J = gap0, reg = 0): a bad init jump
        # can never worsen the final schedule
        best_J, best_kick = gap0, torch.zeros(T, device=self.device)
        J_prev = None
        for k in range(MAX_EVALS - 1):  # the probe consumed one evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # guided pass at lambda_t = kick_t / h_t; objective J = gap + phi*reg
            gap_loss, dL_dlam, g_norm2, _ = self._flow_loss_and_lambda_grads(
                x_cond, det_pred, delta_t, mask, x_ref, seed,
                lam_sched=(kick.detach() / h_vec).tolist(),
            )
            reg = float((kick.detach() ** 2 * g_norm2).sum())
            J = gap_loss + phi * reg
            print(
                f"FGWFREE eval {k}: gap={gap_loss:.6f} reg={reg:.6f} J={J:.6f} "
                f"|kick|max={float(kick.detach().abs().max()):.3f}",
                flush=True,
            )

            if J < best_J:
                best_J, best_kick = J, kick.detach().clone()
            if gap_loss > max(100.0 * gap0, 1e-6):
                # out of the trust region: gradients on a blown-up path are garbage;
                # pull kick halfway back toward the best-known schedule instead
                with torch.no_grad():
                    kick.data = 0.5 * (kick.data + best_kick)
                print(f"FGWFREE backtrack at eval {k}: gap={gap_loss:.3g}", flush=True)
                continue
            if gap_loss <= LOSS_THRESHOLD:
                break
            if J_prev is not None and abs(J_prev - J) <= REL_TOL * max(abs(J_prev), 1e-12):
                break  # objective plateau
            J_prev = J

            # exact frozen-g gradient in kick space:
            #   dJ/dkick = dL/dlam / h + 2 phi ||g||^2 kick   (adjoint + reg terms)
            kick.grad = dL_dlam / h_vec + 2.0 * phi * kick.detach() * g_norm2
            optimizer.step()
            with torch.no_grad():
                kick.clamp_(min=0.0)  # same >= 0 projection as w in FGWNOLR

        print(
            f"FGWFREE best J={best_J:.6f} "
            f"kick={[round(float(v), 3) for v in best_kick]}",
            flush=True,
        )

        # final traced pass: w = kick scale, a_t = learned O(1) kick profile, and
        # c = ||g||/h so the raw-gradient multiplier stays exactly kick_t / h_t
        kick_max = float(best_kick.max().clamp_min(1e-30))
        return self._final_guided_pass(
            "FGWFREE", [kick_max] * T, (best_kick / kick_max).tolist(),
            x_cond, det_pred, delta_t, mask, x_ref, seed,
            c_fn=lambda i, u, g, gn, h: gn / h,
        )