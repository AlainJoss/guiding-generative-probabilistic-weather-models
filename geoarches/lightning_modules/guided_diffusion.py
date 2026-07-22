import math
from collections import defaultdict

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


class GuidedFlow(BaseLightningModule):
    """
    Deterministic + generative-residual forecasting with guided flow sampling.

    Guidance methods (dispatched by `sample`):
      FGWNOLR - secant search on a constant strength w; lambda_t = w * a_t
      FGWRHO - secant search on the guidance-to-flow ratio w; per-step kick
               normalized to the unguided vf norm on the GUIDED CHANNEL
               (lambda_t = w * ||u_t||_c / ||g_t||_c)
      FGWNOGAP - exact per-step closure of the masked gap along a schedule
      FGWFREE - Adam optimization of the full lambda trajectory with a
                kick-energy regularizer (no w / a_t split)
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
                "FGWRHO": self._fgwrho_flow,
                "FGWNOGAP": self._fgwnogap_flow,
                "FGWFREE": self._fgwfree_flow,
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
        kick_norm_ratio: bool = False,
    ):
        # guided sampling with lambda_t = w_schedule[t] * a_schedule[t]; records
        # the raw trace primitives (grads = dL/dz, vfs = s_t*u_t, res = z_t) and
        # the applied {w_t, a_t} sidecar -- everything else is reconstructed in
        # the UI from these. kick_norm_ratio folds ||u_t||/||g_t|| into a_t so the
        # kick magnitude is w*a*||u_t|| (g as pure direction, FGWRHO).
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.flow_timesteps()
        sampling_trace = defaultdict(list)
        w_t_trace, a_t_trace = [], []

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

            a_t = a_schedule[i]
            if kick_norm_ratio:
                a_t = a_t * self._kick_ratio(u_t, grad_vec, mask)
            gui_step = grad_vec.apply(lambda g: g * (w_schedule[i] * a_t))
            w_t_trace.append(float(w_schedule[i]))
            a_t_trace.append(float(a_t))

            sampling_trace["grads"].append(grad_vec.detach().cpu())
            sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())
            sampling_trace["res"].append(z_t.detach().cpu())

            u_t = self.guided_velocity(u_t=u_t, gui_vec=gui_step, lambda_=1.0)
            with torch.no_grad():
                z_t = self.euler_step(z_t, u_t, h)

        sampling_trace["guidance_schedule"] = {"w_t": w_t_trace, "a_t": a_t_trace}
        return z_t, sampling_trace

    def _final_guided_pass(
        self, guidance_name, w_schedule, a_schedule,
        x_cond, det_pred, delta_t, mask, x_ref, seed, w_star=None,
        kick_norm_ratio: bool = False,
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
            kick_norm_ratio=kick_norm_ratio,
        )
        if w_star is not None:
            sampling_trace["w_star"] = float(w_star)  # -> w_star.json sidecar
        return z_t, sampling_trace

    def _kick_ratio(self, u_t, g_t, mask):
        # ||u_t|| / ||g_t|| restricted to the GUIDED CHANNEL (the variable/level
        # the mask lives on): scaling g by this makes the kick as strong as the
        # flow where guidance acts. Global norms would hand the kick the whole
        # 82-channel flow budget, concentrated onto one channel -> state wrecked.
        u2 = torch.zeros((), device=self.device)
        g2 = torch.zeros((), device=self.device)
        for k in mask.keys():
            sup = (mask[k] != 0).any(dim=-1, keepdim=True).any(dim=-2, keepdim=True)
            sup = sup.to(u_t[k].dtype)
            u2 = u2 + (u_t[k].detach() ** 2 * sup).sum()
            g2 = g2 + (g_t[k].detach() ** 2 * sup).sum()
        return float(torch.sqrt(u2) / torch.sqrt(g2).clamp_min(1e-30))

    # --------------------------------------- shared lambda-gradient eval ---

    def _flow_loss_and_lambda_grads(
        self, x_cond, det_pred, delta_t, mask, x_ref, seed, lambdas,
        need_grads: bool = True,
    ):
        """
        One guided pass with per-step multipliers lambda_i, returning
        (gap_loss, dlam, g_norm2, lam_used, resid):
          gap_loss  final masked loss L(z_T) = resid^2
          resid     the SIGNED final masked residual e (for Gauss-Newton models)
          dlam[i]   exact dL/dlambda_i under the frozen-g first-order scheme
                    (forward caches leaf z_i and detached g_i; backward is the
                    adjoint loop with one VJP per step); None if need_grads=False
                    (skips the whole backward loop, ~2x cheaper)
          g_norm2[i] ||g_i||^2 (for regularizers)
          lam_used  the realized multipliers

        `lambdas` is a list of floats, or a callable (i, u_t, g_t) -> float
        evaluated per step on the realized path (path-dependent rules, FGWRHO).
        """
        timesteps = self.flow_timesteps()
        T = len(timesteps)
        z_t = self.init_noise(x_cond, seed)
        z_cache, g_cache, h_cache, lam_used = [], [], [], []
        g_norm2 = torch.zeros(T, device=self.device)

        for i in range(T):
            t, s_t, h = self.step_factors(i, timesteps)
            z_t = z_t.detach().apply(lambda x: x.requires_grad_(True))

            with torch.enable_grad():
                time_embedding = self.embed_time(x_cond, t)
                input_state = self.velocity_input(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                x_hat_t_norm = det_pred + self.euler_step(
                    z_t, u_t, s_t
                ) * self.residual_to_pangu_scale
                x_hat_t = self.denormalize(x_hat_t_norm)
                g_t = self.guidance_gradient(x_hat_t, x_ref, delta_t, mask, z_t)

            z_cache.append(z_t.detach())
            g_cache.append(g_t.detach())
            h_cache.append(float(h))
            g_norm2[i] = sum((g_t[k].detach() ** 2).sum() for k in g_t.keys())

            lam_i = float(lambdas(i, u_t, g_t)) if callable(lambdas) else float(lambdas[i])
            lam_used.append(lam_i)
            gui_step = g_t.apply(lambda g: g * lam_i)
            u_t = self.guided_velocity(u_t=u_t, gui_vec=gui_step, lambda_=1.0)
            z_t = self.euler_step(z_t, u_t, h).detach()

        # adjoint a = dL/dz_T
        z_T = z_t.detach().apply(lambda x: x.requires_grad_(True))
        with torch.enable_grad():
            x_hat_gui = self.final_prediction(det_pred, z_T)
            residual = self.masked_residual(x_hat_gui, x_ref, delta_t, mask)
            gap_loss = residual ** 2

        if not need_grads:
            return (
                float(gap_loss.detach().cpu()), None, g_norm2.detach(),
                lam_used, float(residual.detach().cpu()),
            )

        a = self.grad_wrt_z(gap_loss, z_T)

        # backward adjoint loop (g frozen -> dz_{i+1}/dz_i = I + h du/dz)
        dlam = torch.zeros(T, device=self.device)
        for i in range(T - 1, -1, -1):
            t, s_t, h = self.step_factors(i, timesteps)
            # per-step contribution BEFORE propagating a across step i
            dlam[i] = -h_cache[i] * sum(
                (a[k] * g_cache[i][k]).sum() for k in a.keys()
            )
            z_i = z_cache[i].detach().apply(lambda x: x.requires_grad_(True))
            with torch.enable_grad():
                time_embedding = self.embed_time(x_cond, t)
                input_state = self.velocity_input(z_i, x_cond)
                u_i = self.velocity(x_cond, time_embedding, input_state, z_i, s_t)
            vjp = self._vjp(u_i, z_i, a)
            a = tensordict_apply(lambda an, v: an + h * v, a, vjp)

        return (
            float(gap_loss.detach().cpu()), dlam.detach(), g_norm2.detach(),
            lam_used, float(residual.detach().cpu()),
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
    ):
        # dL/dw is an exact scalar derivative -> solve dL/dw = 0 by secant (no
        # learning rate, no iteration hyper). a_t = (1-eta)^(t+1) is the NOGAP
        # closure schedule, used in evaluations AND the final pass. Safeguards:
        # w >= 0, step growth capped at 3x, eval cap, best-loss w wins.
        LOSS_THRESHOLD = 1e-6
        MAX_EVALS = 30

        timesteps = self.flow_timesteps()
        a_sched = [(1.0 - eta) ** (i + 1) for i in range(len(timesteps))]

        def evaluate(w):
            gap, dlam, _, _, _ = self._flow_loss_and_lambda_grads(
                x_cond, det_pred, delta_t, mask, x_ref, seed,
                lambdas=[w * a for a in a_sched],
            )
            dw = float(sum(a * d for a, d in zip(a_sched, dlam.tolist())))
            return gap, dw

        history = []  # (w, loss, dL/dw)
        w_prev = max(float(fgwnolr_w_init), 0.0)
        loss_prev, g_prev = evaluate(w_prev)
        history.append((w_prev, loss_prev, g_prev))
        print(f"FGWNOLR eval: w={w_prev:.4f} loss={loss_prev:.6f} dL/dw={g_prev:.3e}", flush=True)

        # bootstrap second point: a 10% move against the gradient sign
        step0 = 0.1 * max(abs(w_prev), 1.0)
        w_curr = max(w_prev - math.copysign(step0, g_prev), 0.0)

        while loss_prev > LOSS_THRESHOLD and len(history) < MAX_EVALS:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            loss_curr, g_curr = evaluate(w_curr)
            history.append((w_curr, loss_curr, g_curr))
            print(f"FGWNOLR eval: w={w_curr:.4f} loss={loss_curr:.6f} dL/dw={g_curr:.3e}", flush=True)

            if loss_curr <= LOSS_THRESHOLD:
                break
            denom = g_curr - g_prev
            if not math.isfinite(denom) or abs(denom) < 1e-20 or w_curr == w_prev:
                break  # flat derivative or duplicated point -> secant undefined

            step = -g_curr * (w_curr - w_prev) / denom
            max_step = 3.0 * max(abs(w_curr - w_prev), 1e-6)
            step = max(-max_step, min(step, max_step))

            w_prev, g_prev, loss_prev = w_curr, g_curr, loss_curr
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
            "FGWNOLR", [w_star] * len(timesteps), a_sched,
            x_cond, det_pred, delta_t, mask, x_ref, seed, w_star=w_star,
        )

    # ---------------- FGWRHO: secant on the guidance-to-flow ratio w ---

    def _fgwrho_flow(
        self,
        x_cond,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_ref: TensorDict,
        seed: int,
        fgwrho_w_init: float = 1.0,
    ):
        # Per-step normalized kick: lambda_t = w * ||u_t|| / ||g_t|| with norms on
        # the GUIDED CHANNEL, so g is a pure direction and ||kick|| / ||u_t|| = w
        # where guidance acts (cBottle's rho uses global norms, viable only when
        # the classifier gradient spans the state).
        # w is found by successive PARABOLIC INTERPOLATION on the measured loss:
        # the frozen-ratio analytic dL/dw is biased near the optimum (observed
        # positive on both sides of the loss minimum), so a secant on it never
        # settles -- the measured losses are the trustworthy signal. Derivatives
        # are not needed, so evaluations skip the adjoint pass (~2x cheaper).
        LOSS_THRESHOLD = 1e-6
        MAX_EVALS = 30
        W_TOL = 1e-3  # relative convergence in w
        W_MIN = 1e-6

        T = len(self.flow_timesteps())

        history = []  # (w, loss)

        def probe(w):
            w = max(float(w), W_MIN)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            loss, _, _, _, _ = self._flow_loss_and_lambda_grads(
                x_cond, det_pred, delta_t, mask, x_ref, seed,
                lambdas=lambda i, u_t, g_t: w * self._kick_ratio(u_t, g_t, mask),
                need_grads=False,
            )
            history.append((w, loss))
            print(f"FGWRHO eval: w={w:.4f} loss={loss:.6f}", flush=True)
            return loss

        probe(max(float(fgwrho_w_init), W_MIN))
        probe(max(0.9 * float(fgwrho_w_init), W_MIN))

        while len(history) < MAX_EVALS:
            if min(l for _, l in history) <= LOSS_THRESHOLD:
                break
            # three lowest-loss points with pairwise-distinct w
            pts = []
            for w_, l_ in sorted(history, key=lambda p: p[1]):
                if all(abs(w_ - pw) > 1e-9 for pw, _ in pts):
                    pts.append((w_, l_))
                if len(pts) == 3:
                    break
            if len(pts) < 3:
                w_next = 1.1 * pts[0][0]
            else:
                (w1, l1), (w2, l2), (w3, l3) = pts
                _den = (w1 - w2) * (w1 - w3) * (w2 - w3)
                _A = (w3 * (l2 - l1) + w2 * (l1 - l3) + w1 * (l3 - l2)) / _den
                _B = (w3 * w3 * (l1 - l2) + w2 * w2 * (l3 - l1) + w1 * w1 * (l2 - l3)) / _den
                if not math.isfinite(_A) or _A <= 0:
                    w_next = 0.5 * (w1 + w2)  # non-convex fit -> bisect the two best
                else:
                    w_next = -_B / (2.0 * _A)  # parabola vertex
            # trust region: stay within the explored range +/- one range-width
            _lo = min(w_ for w_, _ in history)
            _hi = max(w_ for w_, _ in history)
            _span = max(_hi - _lo, W_TOL)
            w_next = min(max(w_next, max(W_MIN, _lo - _span)), _hi + _span)
            if any(abs(w_next - w_) <= W_TOL * max(1.0, abs(w_next)) for w_, _ in history):
                break  # vertex keeps landing on known points -> converged in w
            probe(w_next)

        w_star, best_loss = min(history, key=lambda item: item[1])
        print(
            f"FGWRHO w*={w_star:.4f} best_loss={best_loss:.6f} "
            f"history={[(round(w_, 4), round(l_, 6)) for w_, l_ in history]}",
            flush=True,
        )

        return self._final_guided_pass(
            "FGWRHO", [w_star] * T, [1.0] * T,
            x_cond, det_pred, delta_t, mask, x_ref, seed,
            w_star=w_star, kick_norm_ratio=True,
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
    ):
        # Anchored gap closure: aim the signed residual at the deterministic
        # schedule r_target(t) = (1-eta)^(t+1) * r_0 with a Newton step along
        # g = 2 r dS/dz:  gui_step = 2 r (r - r_target) g / (h ||g||^2).
        # Drift and overshoot are corrected against the prescribed path; on-path
        # this reduces to the relative form 2*eta*L/(h*||g||^2). No w and no
        # learned schedule: the sidecar records a_t = (1-eta)^(t+1) (theoretical)
        # and w_t = lambda_t / a_t.
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.flow_timesteps()
        sampling_trace = defaultdict(list)
        scale_trace = []

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
                loss_ = self.gap_loss(x_hat_t, x_ref, delta_t, mask)
                gui_vec = self.grad_wrt_z(loss_, z_t)

            r_det = float(r_.detach())
            if i == 0:
                r_0 = r_det
            r_target = ((1.0 - eta) ** (i + 1)) * r_0

            g_norm2 = sum((gui_vec[k] ** 2).sum() for k in gui_vec.keys())
            scale = (2.0 * r_det * (r_det - r_target)) / (h * g_norm2.clamp_min(1e-30))
            gui_step = gui_vec.apply(lambda g: g * scale)
            scale_trace.append(float(scale))

            print(
                f"FGWNOGAP step: t={i} r={r_det:.6f} r_target={r_target:.6f} "
                f"scale(=lambda_t)={float(scale):.4f}",
                flush=True,
            )

            sampling_trace["grads"].append(gui_vec.detach().cpu())
            sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())
            sampling_trace["res"].append(z_t.detach().cpu())

            u_t = self.guided_velocity(u_t=u_t, gui_vec=gui_step, lambda_=1.0)
            with torch.no_grad():
                z_t = self.euler_step(z_t, u_t, h)

        a_theory = [(1.0 - eta) ** (i + 1) for i in range(len(scale_trace))]
        w_impl = [
            _sc / _a if _a > 0 else float("nan")
            for _sc, _a in zip(scale_trace, a_theory)
        ]
        sampling_trace["guidance_schedule"] = {"w_t": w_impl, "a_t": a_theory}

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
        # Optimize the per-step guidance KICK kappa_t = h_t * lambda_t jointly
        # (no w / a_t split) with Adam on
        #   J(kappa) = L_gap(z_T) + phi * sum_t ||kappa_t g_t||^2.
        # Kick space keeps the coordinates comparable and the phi-curvature
        # uniform: the last flow step has h ~ 0, so lambda_t = kappa_t / h_t is
        # NOT comparable across steps and would ill-condition a shared lr.
        # Each Adam step costs one forward + one frozen-g adjoint (same memory
        # profile as an FGWNOLR evaluation; no autograd through the unrolled
        # flow). Stops on closed gap, objective plateau, or the eval cap; the
        # best-J kick runs the final traced pass with a_t = 1, w_t = lambda_t.
        LOSS_THRESHOLD = 1e-6   # gap part considered closed
        REL_TOL = 1e-3          # relative objective plateau
        MAX_EVALS = 30
        INIT_DAMP = 0.5         # first-order init overshoots -> take half the jump
        LR_FRAC = 0.05          # Adam lr as a fraction of the init scale
        LR_MIN = 1.0

        timesteps = self.flow_timesteps()
        T = len(timesteps)
        h_vec = torch.tensor(
            [float(self.step_factors(i, timesteps)[2]) for i in range(T)],
            device=self.device,
        )

        # probe at kappa = 0 (unguided path): its linearization is linear (gap)
        # + quadratic (reg), minimized in closed form at -dkap / (2 phi ||g||^2)
        # -- the phi-aware regularized Newton step. Start there (damped) instead
        # of crawling from zero, and match Adam's lr to the discovered scale.
        gap0, dlam0, g_norm2_0, _, resid0 = self._flow_loss_and_lambda_grads(
            x_cond, det_pred, delta_t, mask, x_ref, seed, lambdas=[0.0] * T,
        )
        if phi > 0 and abs(resid0) > 1e-12:
            dkap0 = dlam0 / h_vec  # chain rule: dJ/dkappa = (dJ/dlambda) / h
            _denom = (2.0 * phi * g_norm2_0).clamp_min(1e-30)
            # Gauss-Newton: the SIGNED gap e is modeled linear and L = e^2 stays
            # quadratic, so the shrinkage bounds the jump by "close the gap
            # exactly" as phi -> 0 (the plain linear-loss model promises gap
            # reductions far beyond zero and detonates the flow at small phi);
            # for large phi it reduces to the regularized-Newton formula.
            _shrink = 1.0 / (1.0 + float((dkap0 ** 2 / (4.0 * resid0 ** 2 * _denom)).sum()))
            kap = (INIT_DAMP * _shrink * (-dkap0 / _denom)).clamp_(min=0.0)
            lr = max(LR_MIN, LR_FRAC * float(kap.max()))
        else:
            kap = torch.zeros(T, device=self.device)  # gap closed or unregularized
            lr = 50.0
        print(
            f"FGWFREE init: gap(0)={gap0:.6f} |kappa0|max={float(kap.max()):.3f} lr={lr:.3f}",
            flush=True,
        )
        optimizer = torch.optim.Adam([kap], lr=lr)

        # kappa = 0 is a legitimate candidate (J = gap0, reg = 0): a bad init
        # jump can never worsen the final schedule
        best_J, best_kap = gap0, torch.zeros(T, device=self.device)
        J_prev = None
        for k in range(MAX_EVALS - 1):  # the probe consumed one evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gap_loss, dlam, g_norm2, _, _ = self._flow_loss_and_lambda_grads(
                x_cond, det_pred, delta_t, mask, x_ref, seed,
                lambdas=(kap.detach() / h_vec).tolist(),
            )
            reg = float((kap.detach() ** 2 * g_norm2).sum())
            J = gap_loss + phi * reg
            print(
                f"FGWFREE eval {k}: gap={gap_loss:.6f} reg={reg:.6f} J={J:.6f} "
                f"|kappa|max={float(kap.detach().abs().max()):.3f}",
                flush=True,
            )

            if J < best_J:
                best_J, best_kap = J, kap.detach().clone()
            if gap_loss > max(100.0 * gap0, 1e-6):
                # out of the trust region: gradients on a blown-up path are garbage;
                # pull kappa halfway back toward the best-known schedule instead
                with torch.no_grad():
                    kap.data = 0.5 * (kap.data + best_kap)
                print(f"FGWFREE backtrack at eval {k}: gap={gap_loss:.3g}", flush=True)
                continue
            if gap_loss <= LOSS_THRESHOLD:
                break
            if J_prev is not None and abs(J_prev - J) <= REL_TOL * max(abs(J_prev), 1e-12):
                break  # objective plateau
            J_prev = J

            # exact frozen-g gradient in kick space: adjoint term + reg term
            kap.grad = dlam / h_vec + 2.0 * phi * kap.detach() * g_norm2
            optimizer.step()
            with torch.no_grad():
                kap.clamp_(min=0.0)  # same >= 0 projection as w in FGWNOLR

        best_lam = best_kap / h_vec  # applied multiplier (spikes where h ~ 0)
        print(
            f"FGWFREE best J={best_J:.6f} "
            f"kappa={[round(float(v), 3) for v in best_kap]}",
            flush=True,
        )

        return self._final_guided_pass(
            "FGWFREE", best_lam.tolist(), [1.0] * T,
            x_cond, det_pred, delta_t, mask, x_ref, seed,
        )