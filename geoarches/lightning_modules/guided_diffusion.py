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

        # LBG (Monte-Carlo loss-based guidance) hyperparameters (tunable)
        # n_mc: number of MC samples; r_t: std of the Gaussian q(x0|xt)=N(x_hat_t, r_t^2 I)
        # drawn in normalized space. n_mc=1, r_t=0 makes LBG reduce to DPS.

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

        print("initialized GuidedFlow")


    ### utils ###

    def move_objects_to_device(self):
        # device comes from initialization pipeline, not visible in init
        self.residual_to_pangu_scale = self.residual_to_pangu_scale.to(self.device)
        self.data_mean = self.data_mean.to(self.device)
        self.data_std = self.data_std.to(self.device)
        self.generator = torch.Generator(self.device)


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
            z_t + tensordict_apply(torch.mul, s_t, u_t)
        ) * self.residual_to_pangu_scale

        return self.denormalize(x_hat_t_norm)
        

    def final_prediction(self, det_pred, z_t):
        x_hat_norm = det_pred + tensordict_apply(
            torch.mul,
            z_t,
            self.residual_to_pangu_scale,
        )
        return self.denormalize(x_hat_norm)

            
    def grad_loss(self, loss_, z_t, create_graph: bool = False):
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


    def _td_dot(self, a_td, b_td):
        # Flat inner product <a, b> summed over every element of every state key.
        return sum((a_td[k] * b_td[k]).sum() for k in a_td.keys())


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


    ### used outside to wire guidance type ###
        
    def sample(
        self,
        guidance_flag: bool,
        guidance_type: str,
        x_cond: dict,
        delta_t: torch.Tensor | None = None,
        mask: TensorDict | None = None,
        x_hat_ung: TensorDict | None = None,
        lambda_schedule: list[torch.Tensor] | None = None,
        seed: int | None = None,
        guidance_kwargs: dict | None = None,
    ):
        guidance_kwargs = guidance_kwargs or {}

        with torch.no_grad():
            det_pred = self.det_model(x_cond)
            x_cond["pred_state"] = det_pred

        x_cond = {k: v for k, v in x_cond.items() if "next" not in k}

        gradient_guidance = {
            "DPS": self.dps_guidance,
            "LBG": self.lbg_guidance,
        }

        if not guidance_flag:
            z, sampling_trace = self.unguided_flow(
                x_cond=x_cond,
                seed=seed,
            )
        elif guidance_type in gradient_guidance:
            z, sampling_trace = self._gradient_flow(
                guidance_name=guidance_type,
                guidance_fn=gradient_guidance[guidance_type],
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                lambda_schedule=lambda_schedule,
                seed=seed,
                **guidance_kwargs,
            )
        elif guidance_type == "UG":
            z, sampling_trace = self._ug_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                lambda_schedule=lambda_schedule,
                seed=seed,
                **guidance_kwargs,
            )
        elif guidance_type == "FLOWGRAD":
            z, sampling_trace = self._flowgrad_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                seed=seed,
                **guidance_kwargs,
            )
        elif guidance_type == "FLOWGRAD_FREE":
            z, sampling_trace = self._flowgrad_free_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                seed=seed,
                **guidance_kwargs,
            )
        else:
            raise ValueError(f"Unknown guidance_type: {guidance_type}")

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
        seed: int | None = None,
    ):
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_flow_timesteps()

        for i in tqdm(range(len(timesteps))):
            t, s_t, h = self.get_step_factors(i, timesteps)

            with torch.no_grad():
                time_embedding = self.embedd_time(x_cond, t)
                input_state = self.get_velocity_input_state(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                z_t = self.euler_step(z_t, u_t, h)

        return z_t, None

    def _gradient_flow(
        self,
        guidance_name: str,
        guidance_fn,
        x_cond: dict,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_hat_ung: TensorDict,
        lambda_schedule: list[torch.Tensor],
        seed: int | None = None,
        create_graph: bool = False,
        differentiable: bool = False,
        trace: bool = True,
        **guidance_kwargs,
    ):
        # create_graph : keep higher-order graph through the guidance gradient
        # differentiable: keep the whole trajectory attached so the outer
        #                 objective stays differentiable w.r.t. lambda_schedule (FLOWGRAD)
        # trace         : record the per-step sampling_trace (skipped for FLOWGRAD inner passes)
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_flow_timesteps()
        sampling_trace = defaultdict(list)

        if differentiable:
            z_t = z_t.apply(lambda x: x.detach().requires_grad_(True))

        for i in tqdm(range(len(timesteps)), desc=f"{guidance_name} sampling"):
            t, s_t, h = self.get_step_factors(i, timesteps)

            if not differentiable:
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

                gui_vec = guidance_fn(
                    x_hat_t=x_hat_t,
                    x_hat_t_norm=x_hat_t_norm,
                    x_hat_ung=x_hat_ung,
                    delta_t=delta_t,
                    mask=mask,
                    z_t=z_t,
                    create_graph=create_graph,
                    **guidance_kwargs,
                )

            if trace:
                sampling_trace["clean_preds"].append(x_hat_t.detach().cpu())
                sampling_trace["grads"].append(gui_vec.detach().cpu())
                sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())

            u_t = self.guided_velocity(
                u_t=u_t,
                gui_vec=gui_vec,
                lambda_=lambda_schedule[i],
            )

            if trace:
                sampling_trace["gui_vfs"].append(
                    u_t.apply(lambda x: x * s_t).detach().cpu()
                )

            if differentiable:
                z_t = self.euler_step(z_t, u_t, h)
            else:
                with torch.no_grad():
                    z_t = self.euler_step(z_t, u_t, h)

        return z_t, sampling_trace


    ### loss + guidance-vector helpers (shared by all guidance types) ###

    def masked_residual(self, x_hat_t, x_hat_ung, delta_t, mask):
        pred = sum((mask[k] * x_hat_t[k]).sum() for k in x_hat_t.keys())
        target = sum((mask[k] * x_hat_ung[k]).sum() for k in x_hat_ung.keys())
        target = (1 + delta_t) * target
        return pred - target


    def similarity_loss(self, x_hat_t, x_hat_ung):
        return sum(
            ((x_hat_t[k] - x_hat_ung[k]) ** 2).mean()
            for k in x_hat_t.keys()
        )


    def build_loss(self, x_hat_t, x_hat_ung, delta_t, mask, regularized=False, beta=1e-4):
        # single return contract for every guidance type: (loss, |residual|)
        residual = self.masked_residual(x_hat_t, x_hat_ung, delta_t, mask)
        loss = residual ** 2
        if regularized:
            loss = loss + beta * self.similarity_loss(x_hat_t, x_hat_ung)
        return loss, residual.abs()


    def normalize_grad(self, gui_vec, residual, eps=1e-8):
        # divide the guidance vector by the (detached) residual magnitude
        scale = residual.detach().clamp_min(eps)
        return gui_vec.apply(lambda g: g / scale)

    # TODO: should be a protocol, so I do not inadvertedly break something
    def dps_guidance(
        self,
        *,
        x_hat_t,
        x_hat_t_norm=None,
        x_hat_ung,
        delta_t,
        mask,
        z_t,
        regularized=False,
        beta=1e-4,
        normalize=True,
        eps=1e-8,
        create_graph=False,
        **extra,
    ):
        loss_, residual = self.build_loss(
            x_hat_t, x_hat_ung, delta_t, mask, regularized=regularized, beta=beta
        )
        gui_vec = self.grad_loss(loss_, z_t, create_graph=create_graph)

        if normalize:
            gui_vec = self.normalize_grad(gui_vec, residual, eps=eps)

        return gui_vec


    def lbg_guidance(
        self,
        *,
        x_hat_t,
        x_hat_t_norm,
        x_hat_ung,
        delta_t,
        mask,
        z_t,
        regularized=False,
        beta=1e-4,
        normalize=False,
        eps=1e-8,
        n_mc=4,
        r_t=1.0,
        create_graph=False,
        **extra,
    ):
        per_sample_losses = []

        for _ in range(n_mc):
            x_i_norm = x_hat_t_norm.apply(
                lambda x: x + r_t * torch.empty_like(x).normal_(generator=self.generator)
            )
            x_i = self.denormalize(x_i_norm)

            loss_i, _ = self.build_loss(
                x_i, x_hat_ung, delta_t, mask, regularized=regularized, beta=beta
            )
            per_sample_losses.append(loss_i)

        losses = torch.stack(per_sample_losses)
        loss_ = math.log(n_mc) - torch.logsumexp(-losses, dim=0)
        gui_vec = self.grad_loss(loss_, z_t, create_graph=create_graph)

        if normalize:
            _, residual = self.build_loss(x_hat_t, x_hat_ung, delta_t, mask)
            gui_vec = self.normalize_grad(gui_vec, residual, eps=eps)

        return gui_vec


    ### flow variant 3 ###

    def _ug_flow(
        self,
        x_cond,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_hat_ung: TensorDict,
        lambda_schedule: list[torch.Tensor],
        seed: int | None = None,
        S: int = 3,
        m: int = 5,
        delta_lr: float = 1e-1,
        regularized: bool = False,
        beta: float = 1e-4,
        normalize: bool = False,  # unused by UG (no velocity guidance vector); kept for parity
        eps: float = 1e-8,
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

            # detached: it's a constant w.r.t. the dz optimization and is reused across
            # the m inner backward passes (otherwise its graph is freed after the first).
            time_embedding = self.embedd_time(x_cond, t).detach()

            # self recurrence at time t
            for k in range(S):
                delta_z = self.UG_latent_shift(
                    z_t=z_t,
                    x_cond=x_cond,
                    det_pred=det_pred,
                    x_hat_ung=x_hat_ung,
                    delta_t=delta_t,
                    mask=mask,
                    time_embedding=time_embedding,
                    s_t=s_t,
                    m=m,
                    lr=delta_lr,
                    regularized=regularized,
                    beta=beta,
                )

                with torch.no_grad():
                    # unguided field at the current latent (trace baseline)
                    u_pre = self.velocity(
                        x_cond, time_embedding,
                        self.get_velocity_input_state(z_t, x_cond), z_t, s_t,
                    )

                    # apply the backward shift in latent space (lambda anneals it)
                    z_t = tensordict_apply(
                        lambda z, d: z + lambda_schedule[i] * d, z_t, delta_z
                    )

                    # field / clean prediction at the shifted latent, then step
                    u_t = self.velocity(
                        x_cond, time_embedding,
                        self.get_velocity_input_state(z_t, x_cond), z_t, s_t,
                    )
                    x_hat_t = self.clean_prediction(det_pred, z_t, u_t, s_t)
                    z_next = self.euler_step(z_t, u_t, h)

                    if k < S - 1:
                        z_t = self.renoise(z_next, s_t, s_next)

                if k == S - 1:
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
        x_hat_ung: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        time_embedding,
        s_t: torch.Tensor,
        m: int = 5,
        lr: float = 1e-1,
        regularized: bool = False,
        beta: float = 1e-4,
    ):
        # Backward guidance: dz = argmin_dz loss(masked_mean(clean_pred(z_t + dz))), via
        # m-step GD from dz=0. The gradient flows through `velocity` (the backbone), so the
        # network Jacobian d x_hat / d z_t spreads dz across the whole field -- NOT confined
        # to the mask. No projection of dz onto the mask.
        delta_z = z_t.apply(lambda x: torch.zeros_like(x).requires_grad_(True))
        optimizer = torch.optim.SGD([delta_z[k] for k in delta_z.keys()], lr=lr)

        z_base = z_t.detach()
        x_hat_ung = x_hat_ung.detach()

        for _ in range(m):
            optimizer.zero_grad()

            z_shift = tensordict_apply(lambda z, d: z + d, z_base, delta_z)

            with torch.enable_grad():
                input_state = self.get_velocity_input_state(z_shift, x_cond)
                u = self.velocity(x_cond, time_embedding, input_state, z_shift, s_t)
                x_hat_t = self.clean_prediction(det_pred, z_shift, u, s_t)
                loss, _ = self.build_loss(
                    x_hat_t, x_hat_ung, delta_t, mask, regularized=regularized, beta=beta
                )

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
        seed: int,
        n_opt: int = 10,
        lr: float = 1e-2,
        gamma: float = 1e-3,
        init_lambda: float = 0.0,
        n_lambda: int = 4,
        regularized: bool = True,
        beta: float = 1e-4,
        normalize: bool = False,
        eps: float = 1e-8,
    ):
        # FlowGrad-style optimization of a COARSE lambda schedule weighting the DPS
        # direction g_t (velocity is u_t - lambda_t g_t). The gradient w.r.t. lambda
        # is obtained with the Jacobian trick: one cached *detached* forward, then a
        # backward loop of per-step vector-Jacobian products (no full-trajectory
        # graph). g_t's Jacobian is frozen (first-order), so each step costs one VJP.
        windows, win_of = self._coarse_windows(n_lambda)
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
            torch.full((K,), init_lambda, device=self.device, dtype=mask.dtype)
        )
        optimizer = torch.optim.Adam([raw_lambda], lr=lr)
        trace = defaultdict(list)

        timesteps = self.get_flow_timesteps()

        def expand(lambda_win):
            # coarse (K,) -> fine (T,) by piecewise-constant window assignment
            return lambda_win[win_of_t]

        def run_flow(lambda_schedule, *, create_graph, differentiable, do_trace):
            return self._gradient_flow(
                guidance_name="FLOWGRAD",
                guidance_fn=self.dps_guidance,
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                lambda_schedule=lambda_schedule,
                seed=seed,
                create_graph=create_graph,
                differentiable=differentiable,
                trace=do_trace,
                regularized=regularized,
                beta=beta,
                normalize=normalize,
                eps=eps,
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
                    g_t = self.dps_guidance(
                        x_hat_t=x_hat_t,
                        x_hat_ung=x_hat_ung,
                        delta_t=delta_t,
                        mask=mask,
                        z_t=z_t,
                        regularized=regularized,
                        beta=beta,
                        normalize=normalize,
                        eps=eps,
                        create_graph=False,
                    )

                z_cache.append(z_t.detach())
                g_cache.append(g_t.detach())

                w_t = self.guided_velocity(u_t, g_t, lambda_=lambda_fine[i])
                z_t = self.euler_step(z_t, w_t, h).detach()

            return z_t, z_cache, g_cache

        z_cache = g_cache = None
        for _ in tqdm(range(n_opt), desc="FLOWGRAD lambda optimization"):
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
                target_loss, _ = self.build_loss(
                    x_hat_gui, x_hat_ung, delta_t, mask, regularized=regularized, beta=beta
                )
            a = self.grad_loss(target_loss, z_T)

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
                dlam_win[win_of[i]] += -h * self._td_dot(a, g_cache[i])

                vjp = self._vjp(u_i, z_i, a)
                a = tensordict_apply(lambda an, v: an + h * v, a, vjp)

            # lambda is the parameter directly (no softplus chain); add length-weighted
            # L2 reg gamma * sum_i lambda_fine_i^2, step, then project to lambda >= 0.
            reg_grad = 2.0 * gamma * lengths * lambda_win
            raw_lambda.grad = dlam_win + reg_grad

            optimizer.step()
            with torch.no_grad():
                raw_lambda.clamp_(min=0.0)  # projected gradient descent: keep lambda >= 0

            with torch.no_grad():
                reg_loss = gamma * (lengths * lambda_win ** 2).sum()
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
        seed: int,
        n_opt: int = 10,
        lr: float = 1e-2,
        gamma: float = 1e-3,
        init_lambda: float = 0.0,  # unused (controls init at 0); kept for kwarg parity
        n_lambda: int = 5,
        regularized: bool = True,
        beta: float = 1e-4,
        normalize: bool = False,
        eps: float = 1e-8,
    ):
        # Paper-faithful FlowGrad: optimize free additive state-space controls u_j at
        # a coarse set of timesteps (window starts), with NO guidance direction in the
        # loop. Forward is detached fine Euler; the gradient uses the paper's Jacobian
        # trick — per control point one VJP through the collapsed block map
        # func(x) = x + u_j + v(x + u_j, t_j) * Delta_j; since d func/d u_j == d func/d x,
        # u_j.grad equals the propagated adjoint.
        windows, _ = self._coarse_windows(n_lambda)
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

        # free controls, one TensorDict of leaf params per window (init 0)
        z0 = self.init_noise(x_cond, seed)
        controls, params = [], []
        for _ in range(K):
            cj = {}
            for k in z0.keys():
                p = torch.zeros_like(z0[k], requires_grad=True)
                cj[k] = p
                params.append(p)
            controls.append(z0.__class__(cj, batch_size=z0.batch_size, device=z0.device))
        optimizer = torch.optim.Adam(params, lr=lr)
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

        for _ in tqdm(range(n_opt), desc="FLOWGRAD_FREE control optimization"):
            z_T, state_at = cached_forward()

            z_T = z_T.detach().apply(lambda x: x.requires_grad_(True))
            with torch.enable_grad():
                x_hat_gui = self.final_prediction(det_pred, z_T)
                target_loss, _ = self.build_loss(
                    x_hat_gui, x_hat_ung, delta_t, mask, regularized=regularized, beta=beta
                )
            a = self.grad_loss(target_loss, z_T)

            # backward over control points high->low; one VJP each (paper identity)
            for j in range(K - 1, -1, -1):
                x_in = state_at[j].detach().apply(lambda x: x.requires_grad_(True))
                with torch.enable_grad():
                    out = block_map(x_in, j)
                a = self._vjp(out, x_in, a)  # = J^T a ; J_x == J_{u_j}
                for k in controls[j].keys():
                    controls[j][k].grad = a[k] + 2.0 * gamma * controls[j][k].detach()

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                reg_loss = gamma * sum(
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