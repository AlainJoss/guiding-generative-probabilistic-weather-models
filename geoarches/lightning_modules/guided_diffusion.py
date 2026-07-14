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

        # LBG-MC (Monte-Carlo loss-based guidance) hyperparameters (tunable)
        # n_mc: number of MC samples; R: spread magnitude of q(x0|xt)=N(x_hat_t, sigma_t^2 I)
        # with sigma_t = R * a_t, drawn in normalized space. n_mc=1, R=0 recovers LBG.

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
        z = tensordict_cat([prev_state, z], dim=1)
        z = tensordict_cat([pred_state, z], dim=1)
        return z
    

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
    
    
    def a_t(self, s_t, lambda0=1.0, c=5.0):
        # -ds shift: a_t is EXACTLY 0 at the final flow step by design -- the last
        # step applies no guidance (the state is already decided; late corrections
        # would be pasted on rather than integrated).
        ds = 1.0 / self.num_train_timesteps

        s_eff = (s_t - ds).clamp(
            min=0.0,
            max=1.0 - ds,
        )

        a = s_eff / (1.0 - s_eff)

        # Bounded version: approaches lambda0 but never explodes
        return lambda0 * a / (a + c)


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
        w_schedule: list[torch.Tensor] | None = None,
        seed: int | None = None,
        guidance_kwargs: dict | None = None,
    ):
        with torch.no_grad():
            det_pred = self.det_model(x_cond)
            x_cond["pred_state"] = det_pred
        
        # TODO: check that next is actually there in my case
        x_cond = {k: v for k, v in x_cond.items() if "next" not in k}

        gradient_guidance = {
            "LBG": self.lbg_guidance,
            "LBG-MC": self.lbgmc_guidance,
        }

        if not guidance_flag:
            z, sampling_trace = self.unguided_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                seed=seed,
            )

        elif guidance_type in gradient_guidance:
            z, sampling_trace = self._gradient_flow(
                guidance_name=guidance_type,
                guidance_fn=gradient_guidance[guidance_type],
                x_cond=x_cond,
                det_pred=det_pred,
                delta_n=delta_n,
                mask=mask,
                x_ref=x_ref,
                w_schedule=w_schedule,
                seed=seed,
                **guidance_kwargs,
            )
        elif guidance_type == "UG":
            z, sampling_trace = self._ug_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_n,
                mask=mask,
                x_ref=x_ref,
                w_schedule=w_schedule,
                seed=seed,
                **guidance_kwargs,
            )
        elif guidance_type == "FG":
            z, sampling_trace = self._flowgrad_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_n,
                mask=mask,
                x_ref=x_ref,
                seed=seed,
                **guidance_kwargs,
            )
        elif guidance_type == "FGF":
            z, sampling_trace = self._flowgrad_free_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_n,
                mask=mask,
                x_ref=x_ref,
                seed=seed,
                **guidance_kwargs,
            )
        elif guidance_type == "FGW":
            z, sampling_trace = self._fgw_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_n,
                mask=mask,
                x_ref=x_ref,
                seed=seed,
                **guidance_kwargs,
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
        # guided clean_preds), so the unguided pass yields a full-t trajectory -> ung_gui.
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
        seed: int | None = None,
        a_schedule: list[float] | None = None,
        # TODO: not sure whether I actually need these -> maybe flowgradn needs them
        differentiable_flag: bool = False,
        **guidance_kwargs,
    ):
        # w_schedule   : per-step w_t; the applied multiplier is lambda_t = w_t * a_t
        # a_schedule   : optional per-step a_t override (default: self.a_t(s_t));
        #                e.g. FGWNOLR's deterministic closure schedule (1-eta)^(t+1)
        # create_graph : keep higher-order graph through the guidance gradient
        # differentiable: keep the whole trajectory attached so the outer
        #                 objective stays differentiable w.r.t. w_schedule (FLOWGRAD)
        # trace         : record the per-step sampling_trace (skipped for FLOWGRAD inner passes)
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_flow_timesteps()
        sampling_trace = defaultdict(list)
        w_t_trace, a_t_trace = [], []  # applied factorization: lambda_t = w_t * a_t

        if differentiable_flag:
            z_t = z_t.apply(lambda x: x.detach().requires_grad_(True))

        for i in tqdm(range(len(timesteps)), desc=f"{guidance_name} sampling"):
            t, s_t, h = self.get_step_factors(i, timesteps)

            if not differentiable_flag:
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

                a_t = a_schedule[i] if a_schedule is not None else self.a_t(s_t)

                grad_vec = guidance_fn(
                    x_hat_t=x_hat_t,
                    x_hat_t_norm=x_hat_t_norm,
                    x_ref=x_ref,
                    delta_t=delta_n,
                    mask=mask,
                    z_t=z_t,
                    a_t=a_t,  # flow-time scaling; LBG-MC sizes its MC cloud as R * a_t (LBG ignores)
                    **guidance_kwargs,
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

            if differentiable_flag:
                z_t = self.euler_step(z_t, u_t, h)
            else:
                with torch.no_grad():
                    z_t = self.euler_step(z_t, u_t, h)

        # applied weight schedule sidecar (rollout.py pops this before stacking and
        # persists it to guidance_schedule.json; lambda_t = w_t * a_t)
        sampling_trace["guidance_schedule"] = {"w_t": w_t_trace, "a_t": a_t_trace}

        return z_t, sampling_trace


    # TODO: should be a protocol, so I do not inadvertedly break something
    # LBG: point-estimate loss-based guidance (no MC cloud).
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

    # LBG-MC: Monte-Carlo loss-based guidance.
    def lbgmc_guidance(
        self,
        *,
        x_hat_t,
        x_hat_t_norm,
        x_ref,
        delta_t,
        mask,
        z_t,
        a_t,
        n_mc,
        r,
    ):
        # TODO: how to set a good sigma_t -> derivation and hyperparam search
        sigma_t = r * a_t
        per_sample_losses = []

        for _ in range(n_mc):
            x_i_norm = x_hat_t_norm.apply(
                lambda x: x + sigma_t * torch.empty_like(x).normal_(generator=self.generator)
            )
            x_i = self.denormalize(x_i_norm)
            loss_i = self.masked_loss(x_i, x_ref, delta_t, mask)
            per_sample_losses.append(loss_i)

        losses = torch.stack(per_sample_losses)
        loss_ = math.log(n_mc) - torch.logsumexp(-losses, dim=0)
        return self.masked_loss_grad(loss_, z_t, create_graph=False)

    ### flow variant 3 ###

    def _ug_flow(
        self,
        x_cond,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_hat_ung: TensorDict,
        x_ref: TensorDict,
        w_schedule: list[torch.Tensor],
        seed: int | None = None,
        ug_s: int = 3,
        ug_k: int = 5,
        ug_lr: float = 1e-1,
        shift_init: float = 0.0,
    ):
        # Backward universal guidance (arXiv:2302.07121): per step, optimize a LATENT shift
        # dz that drives the masked-mean objective, *through the model* (the dz->loss
        # Jacobian runs through `velocity`, so dz is a full-field, model-consistent shift,
        # not a mask blob). Apply z_t <- z_t + lambda*dz, then take the normal Euler step.
        # Self-recurrence (S, renoise) refines the correction at each noise level.
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_flow_timesteps()
        sampling_trace = defaultdict(list)

        for i in tqdm(range(len(timesteps)), desc="UG sampling"):
            t, s_t, h = self.get_step_factors(i, timesteps)
            s_next = s_t - h  # h == s_t on the final step -> s_next == 0
            a_t = self.a_t(s_t)  # flow-time scaling, shift applied as w * a_t * dz

            # detached: it's a constant w.r.t. the dz optimization and is reused across
            # the m inner backward passes (otherwise its graph is freed after the first).
            time_embedding = self.embedd_time(x_cond, t).detach()

            # self recurrence at time t
            for k in range(ug_s):
                delta_z = self.UG_latent_shift(
                    z_t=z_t,
                    x_cond=x_cond,
                    det_pred=det_pred,
                    x_ref=x_ref,
                    x_ung=x_hat_ung,
                    delta_t=delta_t,
                    mask=mask,
                    time_embedding=time_embedding,
                    s_t=s_t,
                    optimize_k=ug_k,
                    optimize_lr=ug_lr,
                    shift_init=shift_init
                )
                # keep the inner SGD's own magnitude (NOT unit-normalized): the converged
                # dz shrinks as the masked-mean approaches the target, so the correction
                # self-brakes instead of overshooting. w_schedule is now a gain on
                # that true-magnitude shift (set alpha=0, w=1 for a flat gain of 1).

                with torch.no_grad():
                    # unguided field at the current latent (trace baseline)
                    u_pre = self.velocity(
                        x_cond, time_embedding,
                        self.get_velocity_input_state(z_t, x_cond), z_t, s_t,
                    )

                    # apply the backward shift in latent space (w * a_t scales it)
                    z_t = tensordict_apply(
                        lambda z, d: z + w_schedule[i] * a_t * d, z_t, delta_z
                    )

                    # field / clean prediction at the shifted latent, then step
                    u_t = self.velocity(
                        x_cond, time_embedding,
                        self.get_velocity_input_state(z_t, x_cond), z_t, s_t,
                    )
                    x_hat_t = self.clean_prediction(det_pred, z_t, u_t, s_t)
                    z_next = self.euler_step(z_t, u_t, h)

                    if k < ug_s - 1:
                        z_t = self.renoise(z_next, s_t, s_next)

                if k == ug_s - 1:
                    sampling_trace["clean_preds"].append(x_hat_t.detach().cpu())
                    sampling_trace["vfs"].append(u_pre.apply(lambda x: x * s_t).detach().cpu())
                    sampling_trace["gui_vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())
                    # for UG, "grads" carries the backward latent shift dz (not a loss grad)
                    sampling_trace["grads"].append(delta_z.detach().cpu())

            z_t = z_next

        return z_t, sampling_trace

    def UG_latent_shift(
        self,
        z_t: TensorDict,
        x_cond,
        det_pred: TensorDict,
        x_ref: TensorDict,
        x_ung: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        time_embedding,
        s_t: torch.Tensor,
        optimize_k: int = 5,
        optimize_lr: float = 1e-1,
        shift_init: float = 0.0,
        loss_type: str = "MASKED_AVERAGE",
        reg_type="ID",
        beta_reg: float = 1e-4,
    ):
        # Backward guidance: dz = argmin_dz loss(masked_mean(clean_pred(z_t + dz))), via
        # optimize_k-step GD from dz=shift_init. The gradient flows through `velocity` (the
        # backbone), so the network Jacobian d x_hat / d z_t spreads dz across the whole
        # field -- NOT confined to the mask. No projection of dz onto the mask.
        delta_z = z_t.apply(lambda x: torch.full_like(x, shift_init).requires_grad_(True))
        optimizer = torch.optim.SGD([delta_z[k] for k in delta_z.keys()], lr=optimize_lr)

        z_base = z_t.detach()
        x_ung = x_ung.detach()
        x_ref = x_ref.detach()

        for _ in range(optimize_k):
            optimizer.zero_grad()

            z_shift = tensordict_apply(lambda z, d: z + d, z_base, delta_z)

            with torch.enable_grad():
                input_state = self.get_velocity_input_state(z_shift, x_cond)
                u = self.velocity(x_cond, time_embedding, input_state, z_shift, s_t)
                x_hat_t = self.clean_prediction(det_pred, z_shift, u, s_t)
                loss = self.masked_loss(x_hat_t, x_ref, delta_t, mask)

            loss.backward()
            optimizer.step()

        return delta_z.detach()


    def renoise(self, z, s_high, s_low):
        # Self-recurrence renoise: bring a sample at noise level s_low back up to s_high.
        # Rectified-flow translation of the DDPM renoise under alpha_bar ~ (1 - s)^2.
        ratio = (1 - s_high) / (1 - s_low)
        eps = z.apply(lambda x: torch.empty_like(x).normal_(generator=self.generator))
        eps = eps * self.scale_input_noise
        coeff = (1 - ratio ** 2) ** 0.5
        return tensordict_apply(lambda zz, ee: ratio * zz + coeff * ee, z, eps)
    

    ### flow variant 4 ###

    def _flowgrad_flow(
        self,
        x_cond,
        det_pred,
        delta_t,
        mask,
        x_hat_ung,
        x_ref,
        seed: int,
        fg_k: int = 10,
        fg_lr: float = 1e-2,
        fg_gamma: float = 1e-3,
        fg_lambda_init: float = 0.0,
        fg_n_windows: int = 4,
        loss_type: str = "MASKED_AVERAGE",
        reg_type="ID",
        beta_reg: float = 1e-4,
    ):
        # FlowGrad-style optimization of a COARSE lambda schedule weighting the LBG
        # direction g_t (velocity is u_t - lambda_t g_t). The gradient w.r.t. lambda
        # is obtained with the Jacobian trick: one cached *detached* forward, then a
        # backward loop of per-step vector-Jacobian products (no full-trajectory
        # graph). g_t's Jacobian is frozen (first-order), so each step costs one VJP.
        windows, win_of = self._coarse_windows(fg_n_windows)
        K = len(windows)
        win_of_t = torch.tensor(win_of, device=self.device)
        lengths = torch.tensor(
            [length for (_, length) in windows], device=self.device, dtype=mask.dtype
        )

        # lambda is parametrized DIRECTLY (init_lambda IS the starting lambda, default 0
        # -> guidance starts off). Non-negativity is enforced by projecting (clamping)
        # after each optimizer step, which -- unlike a softplus/relu reparametrization --
        # leaves the gradient intact at lambda=0, so optimization can move off zero.
        raw_lambda = torch.nn.Parameter(
            torch.full((K,), fg_lambda_init, device=self.device, dtype=mask.dtype)
        )
        optimizer = torch.optim.Adam([raw_lambda], lr=fg_lr)
        trace = defaultdict(list)

        timesteps = self.get_flow_timesteps()

        def expand(lambda_win):
            # coarse (K,) -> fine (T,) by piecewise-constant window assignment
            return lambda_win[win_of_t]

        def run_flow(w_schedule, *, create_graph, differentiable, do_trace):
            return self._gradient_flow(
                guidance_name="FG",
                guidance_fn=self.lbg_guidance,
                x_cond=x_cond,
                det_pred=det_pred,
                delta_n=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                x_ref=x_ref,
                w_schedule=w_schedule,
                seed=seed,
                create_graph_flag=create_graph,
                differentiable_flag=differentiable,
                loss_type=loss_type,
                reg_type=reg_type,
                beta_reg=beta_reg,
            )

        def cached_forward(lambda_fine):
            # Detached Euler forward with the guided velocity; cache the leaf state
            # z_i and the (detached) guidance direction g_i entering each step.
            z_t = self.init_noise(x_cond, seed)
            z_cache, g_cache = [], []

            for i in range(len(timesteps)):
                t, s_t, h = self.get_step_factors(i, timesteps)
                z_t = z_t.detach().apply(lambda x: x.requires_grad_(True))

                with torch.enable_grad():
                    time_embedding = self.embedd_time(x_cond, t)
                    input_state = self.get_velocity_input_state(z_t, x_cond)
                    u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                    x_hat_t = self.clean_prediction(det_pred, z_t, u_t, s_t)
                    g_t = self.lbg_guidance(
                        x_hat_t=x_hat_t,
                        x_ref=x_ref,
                        x_ung=x_hat_ung,
                        delta_n=delta_t,
                        mask=mask,
                        z_t=z_t,
                    )

                z_cache.append(z_t.detach())
                g_cache.append(g_t.detach())

                w_t = self.guided_velocity(u_t, g_t, lambda_=lambda_fine[i])
                z_t = self.euler_step(z_t, w_t, h).detach()

            return z_t, z_cache, g_cache

        z_cache = g_cache = None
        for _ in tqdm(range(fg_k), desc="FG lambda optimization"):
            # release the previous iteration's caches BEFORE cached_forward allocates the
            # next ones; otherwise both cache sets (each 2*T detached states) coexist for
            # the duration of the forward and can OOM the GPU.
            z_cache = g_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            lambda_win = raw_lambda.detach().clamp_min(0.0)
            lambda_fine = expand(lambda_win)

            z_T, z_cache, g_cache = cached_forward(lambda_fine)

            # adjoint a_T = d(target_loss)/d(z_T)
            z_T = z_T.detach().apply(lambda x: x.requires_grad_(True))
            with torch.enable_grad():
                x_hat_gui = self.final_prediction(det_pred, z_T)
                target_loss = self.masked_loss(x_hat_gui, x_ref, delta_t, mask)
            a = self.masked_loss_grad(target_loss, z_T)

            # backward adjoint loop (Jacobian trick, g frozen -> dz_{i+1}/dz_i = I + h du/dz)
            dlam_win = torch.zeros(K, device=self.device, dtype=raw_lambda.dtype)
            for i in range(len(timesteps) - 1, -1, -1):
                t, s_t, h = self.get_step_factors(i, timesteps)
                z_i = z_cache[i].detach().apply(lambda x: x.requires_grad_(True))

                with torch.enable_grad():
                    time_embedding = self.embedd_time(x_cond, t)
                    input_state = self.get_velocity_input_state(z_i, x_cond)
                    u_i = self.velocity(x_cond, time_embedding, input_state, z_i, s_t)

                # dL/dlambda_i = <a_{i+1}, dz_{i+1}/dlambda_i> = <a, -h g_i>
                dlam_win[win_of[i]] += -h * sum((a[k] * g_cache[i][k]).sum() for k in a.keys())

                vjp = self._vjp(u_i, z_i, a)
                a = tensordict_apply(lambda an, v: an + h * v, a, vjp)

            # lambda is the parameter directly (no softplus chain); add length-weighted
            # L2 reg gamma * sum_i lambda_fine_i^2, step, then project to lambda >= 0.
            reg_grad = 2.0 * fg_gamma * lengths * lambda_win
            raw_lambda.grad = dlam_win + reg_grad

            optimizer.step()
            with torch.no_grad():
                raw_lambda.clamp_(min=0.0)  # projected gradient descent: keep lambda >= 0

            with torch.no_grad():
                reg_loss = fg_gamma * (lengths * lambda_win ** 2).sum()
            trace["loss"].append((target_loss.detach() + reg_loss).cpu())
            trace["target_loss"].append(target_loss.detach().cpu())
            trace["reg_loss"].append(reg_loss.cpu())
            trace["lambda_schedule"].append(lambda_fine.detach().cpu())

        lambda_star_fine = expand(raw_lambda.detach().clamp_min(0.0))

        z_t, sampling_trace = run_flow(
            lambda_star_fine,
            create_graph=False,
            differentiable=False,
            do_trace=True,
        )

        sampling_trace["flowgrad"] = trace
        sampling_trace["lambda_star"] = lambda_star_fine.detach().cpu()

        return z_t, sampling_trace


    ### flow variant 5 — true FlowGrad (free state-space controls, paper-faithful) ###

    def _flowgrad_free_flow(
        self,
        x_cond,
        det_pred,
        delta_t,
        mask,
        x_hat_ung,
        x_ref,
        seed: int,
        fgf_k: int = 10,
        fgf_lr: float = 1e-2,
        fgf_gamma: float = 1e-3,
        fgf_shift_init: float = 0.0,  # initial value of the free controls
        fgf_n_windows: int = 5,
        loss_type: str = "MASKED_AVERAGE",
        reg_type="ID",
        beta_reg: float = 1e-4,
    ):
        # Paper-faithful FlowGrad: optimize free additive state-space controls u_j at
        # a coarse set of timesteps (window starts), with NO guidance direction in the
        # loop. Forward is detached fine Euler; the gradient uses the paper's Jacobian
        # trick — per control point one VJP through the collapsed block map
        # func(x) = x + u_j + v(x + u_j, t_j) * Delta_j; since d func/d u_j == d func/d x,
        # u_j.grad equals the propagated adjoint.
        windows, _ = self._coarse_windows(fgf_n_windows)
        K = len(windows)
        timesteps = self.get_flow_timesteps()

        # per-window block step Delta_j (sum of fine h over the window), start time t_j, s_j
        block = []
        for (start, length) in windows:
            t_j, s_j, _ = self.get_step_factors(start, timesteps)
            delta_j = sum(
                self.get_step_factors(i, timesteps)[2] for i in range(start, start + length)
            )
            block.append((start, t_j, s_j, delta_j))

        # free controls, one TensorDict of leaf params per window (init at shift_init)
        z0 = self.init_noise(x_cond, seed)
        controls, params = [], []
        for _ in range(K):
            cj = {}
            for k in z0.keys():
                p = torch.full_like(z0[k], fgf_shift_init).requires_grad_(True)
                cj[k] = p
                params.append(p)
            controls.append(z0.__class__(cj, batch_size=z0.batch_size, device=z0.device))
        optimizer = torch.optim.Adam(params, lr=fgf_lr)
        trace = defaultdict(list)

        def cached_forward():
            # detached fine Euler; inject control (detached) at each window start;
            # cache the state entering each window for the backward block VJP.
            z_t = self.init_noise(x_cond, seed)
            state_at = [None] * K
            win = 0
            for i in range(len(timesteps)):
                t, s_t, h = self.get_step_factors(i, timesteps)
                if win < K and i == windows[win][0]:
                    state_at[win] = z_t.detach()
                    z_t = tensordict_apply(lambda zz, cc: zz + cc, z_t, controls[win].detach())
                    win += 1
                with torch.no_grad():
                    time_embedding = self.embedd_time(x_cond, t)
                    input_state = self.get_velocity_input_state(z_t, x_cond)
                    u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                z_t = self.euler_step(z_t, u_t, h).detach()
            return z_t, state_at

        def block_map(x, j):
            # collapsed single Euler step over window j with control u_j injected
            start, t_j, s_j, delta_j = block[j]
            xu = tensordict_apply(lambda xx, cc: xx + cc, x, controls[j].detach())
            time_embedding = self.embedd_time(x_cond, t_j)
            input_state = self.get_velocity_input_state(xu, x_cond)
            u = self.velocity(x_cond, time_embedding, input_state, xu, s_j)
            return self.euler_step(xu, u, delta_j)

        for _ in tqdm(range(fgf_k), desc="FGF control optimization"):
            z_T, state_at = cached_forward()

            z_T = z_T.detach().apply(lambda x: x.requires_grad_(True))
            with torch.enable_grad():
                x_hat_gui = self.final_prediction(det_pred, z_T)
                target_loss = self.masked_loss(x_hat_gui, x_ref, delta_t, mask)
            a = self.masked_loss_grad(target_loss, z_T)

            # backward over control points high->low; one VJP each (paper identity)
            for j in range(K - 1, -1, -1):
                x_in = state_at[j].detach().apply(lambda x: x.requires_grad_(True))
                with torch.enable_grad():
                    out = block_map(x_in, j)
                a = self._vjp(out, x_in, a)  # = J^T a ; J_x == J_{u_j}
                for k in controls[j].keys():
                    controls[j][k].grad = a[k] + 2.0 * fgf_gamma * controls[j][k].detach()

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                reg_loss = fgf_gamma * sum(
                    (controls[j][k] ** 2).sum() for j in range(K) for k in controls[j].keys()
                )
            trace["loss"].append((target_loss.detach() + reg_loss).cpu())
            trace["target_loss"].append(target_loss.detach().cpu())
            trace["reg_loss"].append(reg_loss.cpu())
            trace["control_norm"].append(
                torch.tensor([
                    float(sum((controls[j][k] ** 2).sum() for k in controls[j].keys()) ** 0.5)
                    for j in range(K)
                ])
            )

        z_t, state_at = cached_forward()
        sampling_trace = defaultdict(list)
        sampling_trace["flowgrad"] = trace
        sampling_trace["control_star"] = [
            controls[j].detach().cpu() for j in range(K)
        ]

        return z_t, sampling_trace

    ### flow variant 6 — FlowGrad on the scalar guidance strength w (FGW / FGWNOLR) ###

    def _fgw_final_pass(self, guidance_name, w_star, x_cond, det_pred, delta_t, mask, x_ref, seed,
                        a_schedule=None):
        # final pass through the standard LBG flow with the optimized constant w:
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

    def _fgw_flow(
        self,
        x_cond,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_ref: TensorDict,
        seed: int,
        fgw_k: int = 10,
        fgw_lr: float = 50.0,
        fgw_w_init: float = 250.0,
    ):
        # Naive FlowGrad: optimize the SINGLE scalar guidance strength w, applied exactly
        # like LBG (gui_step = w * a_t * g_t at every flow step), against the final masked
        # loss. Same first-order Jacobian trick as FG (g_t frozen, one VJP per step), but
        # the control space is one scalar:
        #   z_{i+1} = z_i + h_i u_i - h_i w a_t_i g_i
        #   dL/dw   = sum_i <a_{i+1}, -h_i a_t_i g_i>,  a = adjoint dL/dz.
        # Adam normalizes the gradient, so fgw_lr IS the step size in w-units per
        # iteration while the gradient sign is stable -> choose
        # fgw_lr ~ |w_init - w*_expected| / fgw_k; final resolution ~ fgw_lr.
        # w >= 0 is enforced by projection (clamping) after each optimizer step, like FG.
        timesteps = self.get_flow_timesteps()
        T = len(timesteps)

        raw_w = torch.nn.Parameter(torch.tensor(float(fgw_w_init), device=self.device))
        optimizer = torch.optim.Adam([raw_w], lr=fgw_lr)
        losses = []

        for _ in tqdm(range(fgw_k), desc="FGW w optimization"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            w = float(raw_w.detach().clamp_min(0.0))
            loss, dw = self._fgw_loss_and_grad(
                x_cond, det_pred, delta_t, mask, x_ref, seed, w, timesteps
            )
            raw_w.grad = torch.tensor(dw, device=self.device)
            optimizer.step()
            with torch.no_grad():
                raw_w.clamp_(min=0.0)  # projected gradient descent: keep w >= 0

            losses.append(loss)
            print(f"FGW iter: w={w:.4f} loss={loss:.6f} dL/dw={dw:.3e}", flush=True)

        w_star = float(raw_w.detach().clamp_min(0.0))
        print(f"FGW w*={w_star:.4f} (loss trajectory: {losses})", flush=True)

        return self._fgw_final_pass("FGW", w_star, x_cond, det_pred, delta_t, mask, x_ref, seed)

    def _fgw_loss_and_grad(self, x_cond, det_pred, delta_t, mask, x_ref, seed, w, timesteps,
                           a_schedule=None):
        # One evaluation of the final masked loss L(w) and its EXACT scalar derivative
        # dL/dw under the LBG application scheme (shared by FGW and FGWNOLR).
        # Detached Euler forward caching the leaf state z_i, the (detached) guidance
        # direction g_i and (h_i, a_t_i); then the backward adjoint loop (first-order
        # Jacobian trick as in FG: g_t frozen, one VJP per step).
        T = len(timesteps)
        z_t = self.init_noise(x_cond, seed)
        z_cache, g_cache, fac_cache = [], [], []

        for i in range(T):
            t, s_t, h = self.get_step_factors(i, timesteps)
            a_t = a_schedule[i] if a_schedule is not None else self.a_t(s_t)
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

    ### FlowGrad helpers (Jacobian-trick adjoint + coarse schedule) ###

    def _vjp(self, outputs_td, inputs_td, v_td, create_graph: bool = False):
        # Vector-Jacobian product Jᵀv where J = d(outputs)/d(inputs), evaluated per
        # TensorDict key. Same None->zeros contract as grad_loss; this is the single
        # backward pass that the FlowGrad adjoint loop reuses ("two backwards" via vjp).
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

    def _coarse_windows(self, K: int):
        # Partition the T fine steps into K contiguous, near-uniform windows.
        # Returns `windows` = [(start_i, length), ...] (len K) and `win_of` (len T)
        # mapping each fine step to its window index. Remainder is spread over the
        # first windows so lengths differ by at most 1.
        T = self.T
        K = max(1, min(K, T))
        base, rem = divmod(T, K)
        windows = []
        win_of = [0] * T
        start = 0
        for j in range(K):
            length = base + (1 if j < rem else 0)
            windows.append((start, length))
            for i in range(start, start + length):
                win_of[i] = j
            start += length
        return windows, win_of