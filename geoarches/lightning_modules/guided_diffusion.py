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


    def euler_step(self, z_t, u_t, dt):
        # z_new = z_t + h * u_t, where h = dt
        return tensordict_apply(lambda z, u: z + dt * u, z_t, u_t)
    

    def init_noise(self, x_cond, seed=None):
        if seed is not None:
            self.generator.manual_seed(seed)

        z_t = x_cond["state"].apply(
            lambda x: torch.empty_like(x).normal_(generator=self.generator)
        )
        return z_t * self.scale_input_noise


    def get_timesteps(self):
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
            dt = s_t - s_next
        else:
            dt = s_t

        return t, s_t, dt
 

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

            
    def guidance_grad(self, loss_, z_t, create_graph: bool = False):
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
        flowgrad_kwargs: dict | None = None,
    ):
        # det pred
        with torch.no_grad():
            det_pred = self.det_model(x_cond)
            x_cond["pred_state"] = det_pred

        # remove future state for decreasing memory 
        x_cond = {k: v for k, v in x_cond.items() if "next" not in k}

        if not guidance_flag:
            z, sampling_trace = self.unguided_flow(
                x_cond=x_cond,
                seed=seed,
            )

        elif guidance_type in ("DPS", "LBG"):
            z, sampling_trace = self.gradient_guided_flow(
                guidance_type=guidance_type,
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                lambda_schedule=lambda_schedule,
                seed=seed,
            )

        elif guidance_type == "UG":
            z, sampling_trace = self.UG_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                lambda_schedule=lambda_schedule,
                seed=seed,
            )

        elif guidance_type == "FLOWGRAD":
            z, sampling_trace = self.flowgrad_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                seed=seed,
                **(flowgrad_kwargs or {}),
            )

        else:
            raise ValueError(f"Unknown guidance_type: {guidance_type}")

        # pred gets denormalized to dataspace outside 
        x_hat_norm = det_pred + tensordict_apply(torch.mul, z, self.residual_to_pangu_scale)
        return x_hat_norm, sampling_trace
    

    ### base flow ###
    
    def unguided_flow(
        self,
        x_cond: dict,
        seed: int | None = None,
    ):
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_timesteps()

        for i in tqdm(range(len(timesteps))):
            t, s_t, dt = self.get_step_factors(i, timesteps)

            with torch.no_grad():
                time_embedding = self.embedd_time(x_cond, t)
                input_state = self.get_velocity_input_state(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                z_t = self.euler_step(z_t, u_t, dt)

        return z_t, None
    
    
    ### flow variants 1-2 ###
    
    def gradient_guided_flow(
        self,
        guidance_type: str,
        x_cond: dict,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_hat_ung: TensorDict,
        lambda_schedule: list[torch.Tensor],
        seed: int | None = None,
    ):
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_timesteps()
        sampling_trace = defaultdict(list)

        for i in tqdm(range(len(timesteps))):
            t, s_t, dt = self.get_step_factors(i, timesteps)

            # track computation graph for loss down the line
            # we take single step gradients here
            z_t = z_t.apply(lambda x: x.detach().requires_grad_(True))

            with torch.enable_grad():
                time_embedding = self.embedd_time(x_cond, t)
                input_state = self.get_velocity_input_state(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)

                # NOTE: we want to go back in one step to the denoised space
                x_hat_t_norm = det_pred + self.euler_step(z_t, u_t, 1) * self.residual_to_pangu_scale

                x_hat_t = self.denormalize(x_hat_t_norm)

                gui_vec = self.get_gradient_based_guidance(
                    guidance_type=guidance_type,
                    x_hat_t=x_hat_t,
                    x_hat_ung=x_hat_ung,
                    delta_t=delta_t,
                    mask=mask,
                    z_t=z_t,
                    dt=dt,
                    x_hat_t_norm=x_hat_t_norm,
                )

            sampling_trace["clean_preds"].append(x_hat_t.detach().cpu())
            sampling_trace["grads"].append(gui_vec.detach().cpu())
            sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())

            u_t = self.guided_velocity(u_t, gui_vec, lambda_schedule[i])

            sampling_trace["gui_vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())

            with torch.no_grad():
                z_t = self.euler_step(z_t, u_t, dt)

        return z_t, sampling_trace
    
    
    def get_gradient_based_guidance(self, guidance_type, x_hat_t, x_hat_ung, delta_t, mask, z_t, dt, x_hat_t_norm):
        match guidance_type:
            case "DPS":
                return self.DPS_guidance(x_hat_t, x_hat_ung, delta_t, mask, z_t)
            case "LBG":
                return self.LBG_guidance(x_hat_t_norm, x_hat_ung, delta_t, mask, z_t)
            case _:
                # "UG" is handled by its own sampler (UG_flow), not this dispatch
                raise ValueError(f"Invalid loss type {guidance_type} -.-")
                

    def DPS_guidance(self, x_hat_t, x_hat_ung, delta_t, mask, z_t):
        loss_ = self.masked_loss(x_hat_t, x_hat_ung, delta_t, mask)
        return self.guidance_grad(loss_, z_t)
        

    def masked_loss(self, x_hat_t, x_hat_ung, delta_t, mask):
        term_left = sum(
            (mask[k] * x_hat_t[k]).sum()
            for k in x_hat_t.keys()
        )

        term_right = sum(
            (mask[k] * x_hat_ung[k]).sum()
            for k in x_hat_ung.keys()
        ) * (1 + delta_t)

        return (term_left - term_right) ** 2


    def LBG_guidance(self, x_hat_t_norm, x_hat_ung, delta_t, mask, z_t):
        n_mc, r_t = 4, 1
        # target term: constant w.r.t. z_t (x_hat_ung is detached upstream)
        term_right = sum(
            (mask[k] * x_hat_ung[k]).sum()
            for k in x_hat_ung.keys()
        ) * (1 + delta_t)

        per_sample_losses = []
        for _ in range(n_mc):
            # perturb in normalized space (scalar r_t), then denormalize -> x^(i) (physical units).
            # noise is a fresh leaf (no grad), so the gradient flows only through x_hat_t_norm -> z_t.
            x_i_norm = x_hat_t_norm.apply(
                lambda x: x + r_t * torch.empty_like(x).normal_(generator=self.generator)
            )
            x_i = self.denormalize(x_i_norm)
            term_left = sum(
                (mask[k] * x_i[k]).sum()
                for k in x_i.keys()
            )
            per_sample_losses.append((term_left - term_right) ** 2)

        losses = torch.stack(per_sample_losses)  # (n_mc,)
        loss_ = math.log(n_mc) - torch.logsumexp(-losses, dim=0)

        grad_l = self.guidance_grad(loss_, z_t)

        return grad_l


    ### flow variant 3 ###

    def UG_flow(self,
        x_cond,
        det_pred: TensorDict,
        delta_t: torch.Tensor | None = None,
        mask: TensorDict | None = None,
        x_hat_ung: TensorDict | None = None,
        lambda_schedule: list[torch.Tensor] | None = None,
        seed: int | None = None,
        S: int = 4,
    ):
        # Backward universal guidance (Bansal et al.) translated to flow, with per-step
        # self-recurrence. The loss-minimizing clean-space correction is closed form (the masked
        # difference to the target), so this sampler is gradient-free -> runs entirely under no_grad.
        if seed is not None:
            self.generator.manual_seed(seed)

        z_t = x_cond["state"].apply(
            lambda x: torch.empty_like(x).normal_(generator=self.generator)
        )
        z_t = z_t * self.scale_input_noise

        sampling_trace = defaultdict(list)

        timesteps = torch.linspace(self.num_train_timesteps, 1, self.T).to(self.device)
        with torch.no_grad():
            for i in tqdm(range(len(timesteps))):
                t = timesteps[i]
                s_t = t / self.num_train_timesteps
                if i < len(timesteps) - 1:
                    s_next = timesteps[i + 1] / self.num_train_timesteps
                    dt = s_t - s_next
                else:
                    s_next = 0.0  # clean level (for the final renoise)
                    dt = s_t

                time_embedding = self.embedd_time(x_cond, t)

                for k in range(S):  # self-recurrence
                    input_state = self.get_velocity_input_state(z_t, x_cond)
                    u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                    x_hat_t_norm = det_pred + (z_t + tensordict_apply(torch.mul, s_t, u_t)) * self.residual_to_pangu_scale
                    x_hat_t = self.denormalize(x_hat_t_norm)

                    gui_vec = self.UG_guidance(x_hat_t, x_hat_ung, delta_t, mask, s_t)
                    u_t = tensordict_apply(lambda u, g: u - (lambda_schedule[i]) * g, u_t, gui_vec)

                    z_next = self.euler_step(z_t, u_t, dt)
                    if k < S - 1:
                        z_t = self.renoise(z_next, s_t, s_next)  # renoise back up to s_t, repeat

                # trace from the final inner iteration (keeps length T)
                sampling_trace["clean_preds"].append(x_hat_t.detach().cpu())
                sampling_trace["grads"].append(gui_vec.detach().cpu())
                sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())
                sampling_trace["gui_vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())

                z_t = z_next

                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return z_t, sampling_trace


    def renoise(self, z, s_high, s_low):
        # Self-recurrence renoise: bring a sample at noise level s_low back up to s_high.
        # Rectified-flow translation of the DDPM renoise under alpha_bar ~ (1 - s)^2.
        ratio = (1 - s_high) / (1 - s_low)
        eps = z.apply(lambda x: torch.empty_like(x).normal_(generator=self.generator))
        eps = eps * self.scale_input_noise
        coeff = (1 - ratio ** 2) ** 0.5
        return tensordict_apply(lambda zz, ee: ratio * zz + coeff * ee, z, eps)
    

    ### flow variant 4 ###

    def flowgrad_flow(
        self,
        x_cond: dict,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_hat_ung: TensorDict,
        seed: int | None = None,
        n_opt: int = 10,
        lr: float = 1e-2,
        gamma: float = 1e-3,
        init_lambda: float = -10.0,
    ):
        if seed is None:
            raise ValueError("FLOWGRAD requires a fixed seed so optimized lambda matches final sampling.")
        
        raw_lambda = torch.nn.Parameter(
            torch.full((self.T,), init_lambda, device=self.device)
        )
        optimizer = torch.optim.Adam([raw_lambda], lr=lr)
        trace = defaultdict(list)

        for _ in tqdm(range(n_opt), desc="FLOWGRAD lambda optimization"):
            optimizer.zero_grad()

            lambda_schedule = torch.nn.functional.softplus(raw_lambda)

            z_t = self.flowgrad_differentiable_flow(
                x_cond=x_cond,
                det_pred=det_pred,
                delta_t=delta_t,
                mask=mask,
                x_hat_ung=x_hat_ung,
                lambda_schedule=lambda_schedule,
                seed=seed,
            )

            x_hat_gui = self.final_prediction(det_pred, z_t)

            target_loss = self.masked_loss(x_hat_gui, x_hat_ung, delta_t, mask)

            reg_loss = gamma * (lambda_schedule ** 2).sum()
            loss = target_loss + reg_loss

            loss.backward()
            optimizer.step()

            trace["flowgrad_loss"].append(loss.detach().cpu())
            trace["flowgrad_target_loss"].append(target_loss.detach().cpu())
            trace["flowgrad_reg_loss"].append(reg_loss.detach().cpu())
            trace["lambda_schedule"].append(lambda_schedule.detach().cpu())

        lambda_star = torch.nn.functional.softplus(raw_lambda).detach()

        z_t, sampling_trace = self.flowgrad_guided_flow(
            x_cond=x_cond,
            det_pred=det_pred,
            delta_t=delta_t,
            mask=mask,
            x_hat_ung=x_hat_ung,
            lambda_schedule=lambda_star,
            seed=seed,
        )

        sampling_trace["flowgrad"] = trace
        sampling_trace["lambda_star"] = lambda_star.detach().cpu()

        return z_t, sampling_trace


    def flowgrad_differentiable_flow(
        self,
        x_cond: dict,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_hat_ung: TensorDict,
        lambda_schedule: torch.Tensor,
        seed: int | None = None,
    ):
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_timesteps()

        for i in range(len(timesteps)):
            t, s_t, dt = self.get_step_factors(i, timesteps)

            # do NOT detach here
            z_t = z_t.apply(lambda x: x.requires_grad_(True))

            time_embedding = self.embedd_time(x_cond, t)
            input_state = self.get_velocity_input_state(z_t, x_cond)
            u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)

            x_hat_t = self.clean_prediction(det_pred, z_t, u_t, s_t)

            step_loss = self.masked_loss(x_hat_t, x_hat_ung, delta_t, mask)

            gui_vec = self.guidance_grad(step_loss, z_t, create_graph=True)

            u_t = self.guided_velocity(
                u_t=u_t,
                gui_vec=gui_vec,
                lambda_=lambda_schedule[i],
            )

            # do NOT use no_grad here
            z_t = self.euler_step(z_t, u_t, dt)

        return z_t


    def flowgrad_guided_flow(
        self,
        x_cond: dict,
        det_pred: TensorDict,
        delta_t: torch.Tensor,
        mask: TensorDict,
        x_hat_ung: TensorDict,
        lambda_schedule: torch.Tensor,
        seed: int | None = None,
    ):
        z_t = self.init_noise(x_cond, seed)
        timesteps = self.get_timesteps()
        sampling_trace = defaultdict(list)

        for i in tqdm(range(len(timesteps)), desc="FLOWGRAD final sampling"):
            t, s_t, dt = self.get_step_factors(i, timesteps)

            z_t = z_t.apply(lambda x: x.detach().requires_grad_(True))

            with torch.enable_grad():
                time_embedding = self.embedd_time(x_cond, t)
                input_state = self.get_velocity_input_state(z_t, x_cond)
                u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)

                x_hat_t = self.clean_prediction(det_pred, z_t, u_t, s_t)

                step_loss = self.masked_loss(x_hat_t, x_hat_ung, delta_t, mask)

                gui_vec = self.guidance_grad(step_loss, z_t)

            sampling_trace["clean_preds"].append(x_hat_t.detach().cpu())
            sampling_trace["grads"].append(gui_vec.detach().cpu())
            sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())

            u_t = self.guided_velocity(
                u_t=u_t,
                gui_vec=gui_vec,
                lambda_=lambda_schedule[i],
            )

            sampling_trace["gui_vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())

            with torch.no_grad():
                z_t = self.euler_step(z_t, u_t, dt)

        return z_t, sampling_trace