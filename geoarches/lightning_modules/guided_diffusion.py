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
    Rollout, sample, and guide predictions (det+gen).
    """
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
        
        # some constants that were floating hardcoded around the codebase
        # or are uselessly in the cfg (since they are fix anyway)
        self.scale_input_noise = 1.05  # NOTE: this is not sigma
        self.num_train_timesteps=1000
        self.cond_dim=256
        self.T = 25

        self.cfg = cfg
        self.backbone = instantiate(cfg.backbone)  # necessary to put it on device
        self.embedder = instantiate(cfg.embedder)

        # NOTE: we could change this to use multiple det models
        if isinstance(load_deterministic_model, str):
            self.det_model, _ = load_module(load_deterministic_model)
        else:
            self.det_model = AvgModule(load_deterministic_model)

        # TODO: can I put this in a pipeline too?
        self.month_embedder = TimestepEmbedder(self.cond_dim)
        self.hour_embedder = TimestepEmbedder(self.cond_dim)
        self.timestep_embedder = TimestepEmbedder(self.cond_dim)

        # sigma scaling factor (shape like usual states)
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
        scaler["level"][-1] *= 3  # we don't care too much about vertical velocity
        self.residual_to_pangu_scale = scaler / pangu_scaler  # inverse because we divide by state_scaler
        # taken from era5.py
        self.data_mean = TensorDict(
            surface=pangu_stats["surface_mean"],
            level=pangu_stats["level_mean"],
        )
        self.data_std = TensorDict(
            surface=pangu_stats["surface_std"],
            level=pangu_stats["level_std"],
        )


    ### utils ###

    def move_objects_to_device(self):
        # device comes from initialization pipeline, not visible in init
        self.residual_to_pangu_scale = self.residual_to_pangu_scale.to(self.device)
        self.data_mean = self.data_mean.to(self.device)
        self.data_std = self.data_std.to(self.device)
        self.generator = torch.Generator(self.device)
        print("initialized GuidedFlow")


    def denormalize(self, batch):
        # TODO: not sure why the presence of surface should enable this 
        if "surface" in batch:
            # we can denormalize directly
            return batch * self.data_std + self.data_mean

        out = {k: (v * self.data_std + self.data_mean if "state" in k else v) for k, v in batch.items()}
        return out
    
    
    def normalize(self, batch):
        if "surface" in batch:
            return (batch - self.data_mean) / self.data_std

        out = {
            k: ((v - self.data_mean) / self.data_std if "state" in k else v)
            for k, v in batch.items()
        }
        return out
    

    ### flow funcs ###

    def embedd_time(self, batch, t):
        times = pd.to_datetime(
            batch["timestamp"].detach().cpu().numpy(),
            unit="s",
        ).tz_localize(None)
        month = torch.tensor(times.month).to(self.device)
        month_emb = self.month_embedder(month)
        hour = torch.tensor(times.hour).to(self.device)
        hour_emb = self.hour_embedder(hour)
        timestep_emb = self.timestep_embedder(torch.tensor([t]).to(self.device))
        # print(f"embedding time for gen model - month:{int(month)}, hour:{int(hour)}")

        time_embedding = month_emb + hour_emb + timestep_emb
        return time_embedding
    

    def get_velocity_input_state(self, z, batch):
        # only init of concat (we need z later as is)
        assert "pred_state" in batch
        pred_state = batch["pred_state"]
        prev_state = batch["prev_state"]
        x = tensordict_cat([prev_state, z], dim=1)
        x = tensordict_cat([pred_state, x], dim=1)
        return x
    

    def velocity(self, batch, time_embedding, input_state, z_t, s_t):
        ##### compute residual #####

        # here we embedd prev_state (input_state[0]), current_state (batch["state"]), noisy_state (input_state[1])
        x = self.embedder.encode(batch["state"], input_state)
        x = self.backbone(x, time_embedding)
        r_t = self.embedder.decode(x)  # we get tdict

        ##### compute velocity from residual
        # u_t = r_t - eps_t := (r_t - z_t) / s_t
        u_t = (r_t - z_t).apply(lambda x: x / s_t)
        return u_t
    
    
    def guided_velocity(self, u_t, gui_vec, lambda_):
        return tensordict_apply(
            lambda u, g: u - lambda_ * g,
            u_t,
            gui_vec,
        )


    def euler_step(self, z_t, u_t, h):
        # z_new = z_t + h * u_t
        return tensordict_apply(lambda z, u: z + h * u, z_t, u_t)
    

    def init_noise(self, x_cond, seed=None):
        if seed is not None:
            self.generator.manual_seed(seed)

        z_t = x_cond["state"].apply(
            lambda x: torch.empty_like(x).normal_(generator=self.generator)
        )
        return z_t * self.scale_input_noise


    def get_flow_timesteps(self):
        return torch.linspace(
            self.num_train_timesteps,
            1,
            self.T,
            device=self.device,
        )
    

    def get_step_factors(self, i, timesteps):
        # works perfectly with different Ts
        t = timesteps[i]
        s_t = t / self.num_train_timesteps
        if i < len(timesteps) - 1:
            s_next = timesteps[i + 1] / self.num_train_timesteps
            h = s_t - s_next
        else:
            h = s_t
        return t, s_t, h
 

    def clean_prediction(self, det_pred, z_t, u_t, s_t):
        x_hat_t_norm = det_pred + (
            z_t + tensordict_apply(  # hat(r_T)_t
                torch.mul, 
                s_t, 
                u_t
            )
        ) * self.residual_to_pangu_scale

        return self.denormalize(x_hat_t_norm)
        

    def final_prediction(self, det_pred, z_t):
        x_hat_norm = det_pred + tensordict_apply( # r_T
            torch.mul,
            z_t,
            self.residual_to_pangu_scale,
        )
        return self.denormalize(x_hat_norm)

            
    def masked_loss_grad(self, loss_, z_t, create_graph: bool = False):
        keys = list(z_t.keys())
        tensors = [z_t[k] for k in keys]

        grads = torch.autograd.grad(
            loss_,
            tensors,
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
    

    def masked_residual(self, x_hat_t, x_ref, delta_t, mask):
        # signed residual r = S(x_hat) - (1 + delta) * S(x_ref); masked_loss = r^2
        pred = sum((mask[k] * x_hat_t[k]).sum() for k in x_hat_t.keys())
        target = sum((mask[k] * x_ref[k]).sum() for k in x_ref.keys())
        return pred - (1 + delta_t) * target

    def masked_loss(self, x_hat_t, x_ref, delta_t, mask):
        # masked-average match: guide the masked mean toward (1+delta) * reference mean
        pred = sum((mask[k] * x_hat_t[k]).sum() for k in x_hat_t.keys())
        target = sum((mask[k] * x_ref[k]).sum() for k in x_ref.keys())
        target = (1 + delta_t) * target
        return (pred - target) ** 2


    ### used outside to wire guidance type ###
    # TODO: make the guidance methods protocols such that I can just select the guidance method from the name and pass the parameters once
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

        # TODO: check that next is actually there in my case
        x_cond = {k: v for k, v in x_cond.items() if "next" not in k}

        if not guidance_flag:
            z, sampling_trace = self.unguided_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                seed=seed,
            )

        elif guidance_type == "FGWNOLR":
            z, sampling_trace = self._fgwnolr_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_n,
                mask=mask,
                x_ref=x_ref,
                seed=seed,
                **guidance_kwargs,
            )
        elif guidance_type == "FGWNOGAP":
            z, sampling_trace = self._fgwnogap_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_n,
                mask=mask,
                x_ref=x_ref,
                seed=seed,
                **guidance_kwargs,
            )
        else:
            raise ValueError(f"Unknown guidance_type: {guidance_type}")

        # x_det + r*sigma_r
        x_hat_norm = det_pred + tensordict_apply(
            torch.mul,
            z,
            self.residual_to_pangu_scale,
        )

        return x_hat_norm, sampling_trace

    ### base flow ###
    
    def unguided_flow(
        self,
        x_cond: dict,
        det_pred: TensorDict,
        seed: int | None = None,
    ):
        # trace: record the per-flow-step clean-state estimate (same quantity as the
        # guided clean_preds), so the unguided pass yields a full-t trajectory -> gui_ung.
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_flow_timesteps()
        sampling_trace = defaultdict(list)

        for i in tqdm(range(len(timesteps))):
            t, s_t, h = self.get_step_factors(i, timesteps)

            with torch.no_grad():
                time_embedding = self.embedd_time(x_cond, t)
                input_state = self.get_velocity_input_state(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                x_hat_t = self.clean_prediction(det_pred, z_t, u_t, s_t)
                sampling_trace["clean_preds"].append(x_hat_t.detach().cpu())
                z_t = self.euler_step(z_t, u_t, h)

        return z_t, sampling_trace


    def _gradient_flow(
        self,
        guidance_name: str,
        guidance_fn,
        x_cond: dict,
        det_pred: TensorDict,
        delta_n: torch.Tensor,
        mask: TensorDict,
        x_ref: TensorDict,
        w_schedule: list[torch.Tensor],
        a_schedule: list[float],
        seed: int | None = None,
    ):
        # w_schedule : per-step w_t; the applied multiplier is lambda_t = w_t * a_t
        # a_schedule : per-step a_t, e.g. FGWNOLR's deterministic closure schedule
        #              (1-eta)^(t+1)
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_flow_timesteps()
        sampling_trace = defaultdict(list)
        w_t_trace, a_t_trace = [], []  # applied factorization: lambda_t = w_t * a_t

        for i in tqdm(range(len(timesteps)), desc=f"{guidance_name} sampling"):
            t, s_t, h = self.get_step_factors(i, timesteps)

            z_t = z_t.apply(lambda x: x.detach().requires_grad_(True))

            with torch.enable_grad():
                time_embedding = self.embedd_time(x_cond, t)
                input_state = self.get_velocity_input_state(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)

                x_hat_t_norm = det_pred + self.euler_step(
                    z_t,
                    u_t,
                    s_t,
                ) * self.residual_to_pangu_scale

                x_hat_t = self.denormalize(x_hat_t_norm)

                a_t = a_schedule[i]

                grad_vec = guidance_fn(
                    x_hat_t=x_hat_t,
                    x_hat_t_norm=x_hat_t_norm,
                    x_ref=x_ref,
                    delta_t=delta_n,
                    mask=mask,
                    z_t=z_t,
                    a_t=a_t,  # flow-time scaling
                )

            gui_step = grad_vec.apply(
                lambda g: g * (w_schedule[i] * a_t)
            )
            w_t_trace.append(float(w_schedule[i]))
            a_t_trace.append(float(a_t))

            # raw primitives only: gui_vec (= lambda_t * grads), gui_vf, gui_res and
            # clean_preds are reconstructed in-memory in the UI from these plus the
            # guidance_schedule sidecar
            sampling_trace["grads"].append(grad_vec.detach().cpu())  # raw dL/dz
            sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())
            sampling_trace["res"].append(z_t.detach().cpu())         # noisy state z_t

            u_t = self.guided_velocity(u_t=u_t, gui_vec=gui_step, lambda_=1.0)

            with torch.no_grad():
                z_t = self.euler_step(z_t, u_t, h)

        # applied weight schedule sidecar (rollout.py pops this before stacking and
        # persists it to guidance_schedule.json; lambda_t = w_t * a_t)
        sampling_trace["guidance_schedule"] = {"w_t": w_t_trace, "a_t": a_t_trace}

        return z_t, sampling_trace


    # point-estimate loss-based guidance direction: dL/dz of the masked loss at the
    # clean prediction (the g_t that FGWNOLR scales by w * a_t)
    def lbg_guidance(
        self,
        *,
        x_hat_t,
        x_hat_t_norm,
        x_ref,
        delta_t,
        mask,
        z_t,
        a_t
    ):
        loss_ = self.masked_loss(x_hat_t, x_ref, delta_t, mask)
        return self.masked_loss_grad(loss_, z_t, create_graph=False)

    ### flow variant 6 — FGWNOLR: secant on the scalar guidance strength w ###

    def _fgw_final_pass(self, guidance_name, w_star, x_cond, det_pred, delta_t, mask, x_ref, seed,
                        a_schedule):
        # final pass through the loss-gradient flow with the optimized constant w:
        # identical application scheme, and yields the standard stackable traces
        # (clean_preds/grads/vfs/gui_vfs) that the rollout zarr saving expects.
        z_t, sampling_trace = self._gradient_flow(
            guidance_name=guidance_name,
            guidance_fn=self.lbg_guidance,
            x_cond=x_cond,
            det_pred=det_pred,
            delta_n=delta_t,
            mask=mask,
            x_ref=x_ref,
            w_schedule=[w_star] * self.T,
            seed=seed,
            a_schedule=a_schedule,
        )
        # scalar sidecar for analysis (rollout.py pops this before stacking the traces
        # and persists it to w_star.json; see src/utils.append_w_star)
        sampling_trace["w_star"] = float(w_star)
        return z_t, sampling_trace

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
        # LR-free FGW: dL/dw is an exact SCALAR derivative, so instead of gradient
        # descent solve dL/dw = 0 with the secant method -- no learning rate and no
        # iteration-count hyper: it optimizes UNTIL the loss (squared masked-sum
        # residual) drops under LOSS_THRESHOLD.
        # a_t is the DETERMINISTIC NOGAP closure schedule (remaining-gap fraction):
        #   a_t = (1 - eta)^(t+1),
        # used consistently in the secant evaluations AND the final pass, so the
        # optimized w* means the same thing in both. Safeguards: w >= 0 projection,
        # per-iteration step growth capped at 3x the previous step (secant can shoot
        # off when the derivative flattens), a hard evaluation cap against plateaus,
        # and the best-LOSS w wins (secant targets a stationary point, which need not
        # be the best point visited).
        LOSS_THRESHOLD = 1e-6   # stop when loss is essentially closed
        MAX_EVALS = 30          # plateau safety net (never a tuning knob)

        timesteps = self.get_flow_timesteps()
        a_sched = [(1.0 - eta) ** (i + 1) for i in range(len(timesteps))]
        history = []  # (w, loss, dL/dw)

        w_prev = max(float(fgwnolr_w_init), 0.0)
        loss_prev, g_prev = self._fgw_loss_and_grad(
            x_cond, det_pred, delta_t, mask, x_ref, seed, w_prev, timesteps,
            a_schedule=a_sched,
        )
        history.append((w_prev, loss_prev, g_prev))
        print(f"FGWNOLR eval: w={w_prev:.4f} loss={loss_prev:.6f} dL/dw={g_prev:.3e}", flush=True)

        # bootstrap second point: a 10% move against the gradient sign
        step0 = 0.1 * max(abs(w_prev), 1.0)
        w_curr = max(w_prev - math.copysign(step0, g_prev), 0.0)

        while loss_prev > LOSS_THRESHOLD and len(history) < MAX_EVALS:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            loss_curr, g_curr = self._fgw_loss_and_grad(
                x_cond, det_pred, delta_t, mask, x_ref, seed, w_curr, timesteps,
                a_schedule=a_sched,
            )
            history.append((w_curr, loss_curr, g_curr))
            print(f"FGWNOLR eval: w={w_curr:.4f} loss={loss_curr:.6f} dL/dw={g_curr:.3e}", flush=True)

            if loss_curr <= LOSS_THRESHOLD:
                break  # under threshold: done

            denom = g_curr - g_prev
            if not math.isfinite(denom) or abs(denom) < 1e-20 or w_curr == w_prev:
                break  # derivative flat or duplicated point -> secant undefined

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

        # final pass with the optimized w*: the full closure schedule a_t is applied
        # over every flow step (a_sched spans all T; the last step is guided like the
        # rest), matching the schedule used in the secant evaluations.
        return self._fgw_final_pass(
            "FGWNOLR", w_star, x_cond, det_pred, delta_t, mask, x_ref, seed,
            a_schedule=a_sched,
        )

    def _fgw_loss_and_grad(self, x_cond, det_pred, delta_t, mask, x_ref, seed, w, timesteps,
                           a_schedule):
        # One evaluation of the final masked loss L(w) and its EXACT scalar derivative
        # dL/dw under the loss-gradient application scheme (gui_step = w * a_t * g_t).
        # Detached Euler forward caching the leaf state z_i, the (detached) guidance
        # direction g_i and (h_i, a_t_i); then the backward adjoint loop (first-order
        # Jacobian trick: g_t frozen, one VJP per step).
        T = len(timesteps)
        z_t = self.init_noise(x_cond, seed)
        z_cache, g_cache, fac_cache = [], [], []

        for i in range(T):
            t, s_t, h = self.get_step_factors(i, timesteps)
            a_t = a_schedule[i]
            z_t = z_t.detach().apply(lambda x: x.requires_grad_(True))

            with torch.enable_grad():
                time_embedding = self.embedd_time(x_cond, t)
                input_state = self.get_velocity_input_state(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                x_hat_t_norm = det_pred + self.euler_step(
                    z_t, u_t, s_t
                ) * self.residual_to_pangu_scale
                x_hat_t = self.denormalize(x_hat_t_norm)
                g_t = self.lbg_guidance(
                    x_hat_t=x_hat_t,
                    x_hat_t_norm=x_hat_t_norm,
                    x_ref=x_ref,
                    delta_t=delta_t,
                    mask=mask,
                    z_t=z_t,
                    a_t=a_t,
                )

            z_cache.append(z_t.detach())
            g_cache.append(g_t.detach())
            fac_cache.append((h, a_t))

            gui_step = g_t.apply(lambda g: g * (w * a_t))
            u_t = self.guided_velocity(u_t=u_t, gui_vec=gui_step, lambda_=1.0)
            z_t = self.euler_step(z_t, u_t, h).detach()

        # adjoint a_T = d(target_loss)/d(z_T)
        z_T = z_t.detach().apply(lambda x: x.requires_grad_(True))
        with torch.enable_grad():
            x_hat_gui = self.final_prediction(det_pred, z_T)
            target_loss = self.masked_loss(x_hat_gui, x_ref, delta_t, mask)
        a = self.masked_loss_grad(target_loss, z_T)

        # backward adjoint loop (g frozen -> dz_{i+1}/dz_i = I + h du/dz)
        dw = torch.zeros((), device=self.device)
        for i in range(T - 1, -1, -1):
            t, s_t, h = self.get_step_factors(i, timesteps)
            h_i, a_t_i = fac_cache[i]
            # dL/dw contribution of step i (uses a = a_{i+1}, BEFORE propagation)
            dw = dw - h_i * a_t_i * sum((a[k] * g_cache[i][k]).sum() for k in a.keys())

            z_i = z_cache[i].detach().apply(lambda x: x.requires_grad_(True))
            with torch.enable_grad():
                time_embedding = self.embedd_time(x_cond, t)
                input_state = self.get_velocity_input_state(z_i, x_cond)
                u_i = self.velocity(x_cond, time_embedding, input_state, z_i, s_t)

            vjp = self._vjp(u_i, z_i, a)
            a = tensordict_apply(lambda an, v: an + h * v, a, vjp)

        return float(target_loss.detach().cpu()), float(dw.detach().cpu())

    ### flow variant 7 — exact per-step gap closure (FGWNOGAP) ###

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
        # Anchored per-step gap closure: track the DETERMINISTIC residual schedule
        #   r_target(t) = (1 - eta)^(t+1) * r_0
        # (closed form of a_{k+1} = a_k + eta*(1 - a_k)). Each step computes the Newton
        # move from the CURRENTLY measured signed residual r_t to r_target(t), so
        # over/under-achievement, model drift and overshoot are corrected against the
        # prescribed path. On-path this reduces to the relative form 2*eta*L/(h*||g||^2).
        # No w, no learned schedule: the recorded factorization is a_t = (1-eta)^(t+1)
        # (theoretical) and w_t = lambda_t / a_t with lambda_t = the applied scale.
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_flow_timesteps()
        sampling_trace = defaultdict(list)
        scale_trace = []

        for i in tqdm(range(len(timesteps)), desc="FGWNOGAP sampling"):
            t, s_t, h = self.get_step_factors(i, timesteps)
            z_t = z_t.apply(lambda x: x.detach().requires_grad_(True))

            with torch.enable_grad():
                time_embedding = self.embedd_time(x_cond, t)
                input_state = self.get_velocity_input_state(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)

                x_hat_t_norm = det_pred + self.euler_step(
                    z_t,
                    u_t,
                    s_t,
                ) * self.residual_to_pangu_scale

                x_hat_t = self.denormalize(x_hat_t_norm)

                r_ = self.masked_residual(x_hat_t, x_ref, delta_t, mask)
                loss_ = r_ ** 2
                gui_vec = self.masked_loss_grad(loss_, z_t, create_graph=False)

            # anchored trajectory tracking: aim the SIGNED residual at the
            # deterministic schedule r_target = (1 - eta)^(t+2) * r_0, computed from
            # the CURRENT state -- over/under-achievement, drift and overshoot are all
            # corrected against the prescribed path at the next step. On-path this
            # reduces exactly to the relative form 2*eta*L/(h*||g||^2).
            # Newton step along g = 2 r dS/dz: <dS, dz> = r_target - r_t with
            # dz = -h * gui_step  =>  gui_step = 2 r_t (r_t - r_target) g / (h ||g||^2).
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

            # raw primitives only, matching _gradient_flow: gui_vec/gui_vf/gui_res and
            # clean_preds are reconstructed in the UI (lambda_t = scale is recorded in
            # the guidance_schedule sidecar with a_t = 1)
            sampling_trace["grads"].append(gui_vec.detach().cpu())  # raw dL/dz
            sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())
            sampling_trace["res"].append(z_t.detach().cpu())        # noisy state z_t

            u_t = self.guided_velocity(u_t=u_t, gui_vec=gui_step, lambda_=1.0)

            with torch.no_grad():
                z_t = self.euler_step(z_t, u_t, h)

        # deterministic factorization: a_t is the THEORETICAL remaining-gap schedule
        # (1 - eta)^(t+1); w_t = lambda_t / a_t so that lambda_t = w_t * a_t = scale.
        # Kept in sync with r_target above -- the UI draws r_0 * a_t as the reference.
        a_theory = [(1.0 - eta) ** (i + 1) for i in range(len(scale_trace))]
        w_impl = [
            _sc / _a if _a > 0 else float("nan")
            for _sc, _a in zip(scale_trace, a_theory)
        ]
        sampling_trace["guidance_schedule"] = {"w_t": w_impl, "a_t": a_theory}

        return z_t, sampling_trace

    ### Jacobian-trick adjoint helper ###

    def _vjp(self, outputs_td, inputs_td, v_td, create_graph: bool = False):
        # Vector-Jacobian product Jᵀv where J = d(outputs)/d(inputs), evaluated per
        # TensorDict key. Same None->zeros contract as grad_loss; this is the single
        # backward pass that the FGWNOLR adjoint loop reuses ("two backwards" via vjp).
        keys = list(inputs_td.keys())
        inputs = [inputs_td[k] for k in keys]
        outputs = [outputs_td[k] for k in keys]
        grad_outputs = [v_td[k] for k in keys]

        grads = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_outputs,
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

