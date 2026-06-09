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
        self.n_mc = 4
        self.r_t = 0.1

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

    def sample(self,
        guidance_flag: bool, # TODO implement
        guidance_type: str,
        x_cond: dict,  # timestamp, TensorDict for "state", "prev", etc.
        delta_t: torch.Tensor | None = None, 
        mask: TensorDict | None = None, 
        x_hat_ung: TensorDict | None = None,  
        lambda_schedule: list[torch.Tensor] | None = None,
        seed: int | None = None,
    ):  
        # NOTE: det prediction is in normalized space!
        with torch.no_grad():
            det_pred = self.det_model(x_cond)
            x_cond["pred_state"] = det_pred

        
        # remove next_state (save compute)
        x_cond = {k: v for k, v in x_cond.items() if "next" not in k} 
        z, sampling_trace = self.flow(
            guidance_flag, guidance_type,
            x_cond, det_pred, 
            delta_t, mask, x_hat_ung, lambda_schedule, 
            seed
        )
        # x_hat = x_det + r_hat (=sigma*z_T)
        x_hat = det_pred + tensordict_apply(torch.mul, z, self.residual_to_pangu_scale)
    
        return x_hat, sampling_trace
    
    def flow(self,
        guidance_flag,
        guidance_type,
        x_cond, 
        det_pred: TensorDict,
        delta_t: torch.Tensor | None = None,  # NOTE: used as guidance flag (None == no guidance)
        mask: TensorDict | None = None, 
        x_hat_ung: TensorDict | None = None,
        lambda_schedule: list[torch.Tensor] | None = None,
        seed: int | None = None
    ):
        ##### init #####
        # draw noise
        if seed is not None:
            self.generator.manual_seed(seed)
            
        z_t = x_cond["state"].apply(
            lambda x: torch.empty_like(x).normal_(generator=self.generator)
        )
        z_t = z_t * self.scale_input_noise
    
        ##### sample #####
        sampling_trace=defaultdict(list) if guidance_flag else None

        timesteps = torch.linspace(self.num_train_timesteps, 1, self.T).to(self.device)
        for i in tqdm(range(len(timesteps))):
            t = timesteps[i]

            # integration factor
            s_t = t / self.num_train_timesteps  # integration factor
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                s_next = t_next / self.num_train_timesteps
                dt = s_t - s_next
            else:
                dt = s_t

            # reset graph at each step and make z_t differentiable
            if guidance_flag:
                z_t = z_t.apply(lambda x: x.detach().requires_grad_(True))
            
            time_embedding = self.embedd_time(x_cond, t)      
            input_state = self.get_velocity_input_state(z_t, x_cond)

            # vector field 
            if guidance_flag:
                with torch.enable_grad():
                    u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)
                    x_hat_t_norm = det_pred + (z_t + tensordict_apply(torch.mul, s_t, u_t)) * self.residual_to_pangu_scale
                    x_hat_t = self.denormalize(x_hat_t_norm)

                    gui_vec = self.get_guidance(guidance_type, x_hat_t, x_hat_ung, delta_t, mask, z_t, dt, x_hat_t_norm)
            
                sampling_trace["clean_preds"].append(x_hat_t.detach().cpu())
                sampling_trace["grads"].append(gui_vec.detach().cpu())
                sampling_trace["vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())  # rescale to only have the raw diff in residual

                # guided vector field
                u_t = tensordict_apply(lambda u, g: u - (lambda_schedule[i]) * g, u_t, gui_vec)
                # sampling_trace["gui_vfs"].append(u_t.apply(lambda x: x * s_t).detach().cpu())  # rescale to only have the raw diff in residual
            else:
                with torch.no_grad():
                    u_t = self.velocity(x_cond, time_embedding, input_state, z_t, s_t)

            with torch.no_grad():
                z_t = self.euler_step(z_t, u_t, dt)
            
            # TODO: not sure this is in the right place
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.empty_cache()

        return z_t, sampling_trace
    
    
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
            
    def euler_step(self, z_t, u_t, dt):
        # z_new = z_t + h * u_t, where h = dt
        return tensordict_apply(lambda z, u: z + dt * u, z_t, u_t)
        
    def grad_loss(self, loss_, z_t):
        keys = list(z_t.keys())
        tensors = [z_t[k] for k in keys]

        grads = torch.autograd.grad(
            loss_,
            tensors,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        grad_l = z_t.__class__(
            {
                k: torch.zeros_like(z_t[k]) if g is None else g
                for k, g in zip(keys, grads)
            },
            batch_size=z_t.batch_size,
            device=z_t.device,
        )

        return grad_l
    
    def get_guidance(self, guidance_type, x_hat_t, x_hat_ung, delta_t, mask, z_t, dt, x_hat_t_norm):
        match guidance_type:
            case "DPS":
                return self.DPS_guidance(x_hat_t, x_hat_ung, delta_t, mask, z_t)
            case "UG":
                return self.UG_guidance(x_hat_ung, dt, x_hat_t_norm)
            case "LBG":
                return self.LBG_guidance(x_hat_t_norm, x_hat_ung, delta_t, mask, z_t)
            case _:
                raise ValueError(f"Invalid loss type {guidance_type} -.-")
            
    def DPS_guidance(self, x_hat_t, x_hat_ung, delta_t, mask, z_t):
        # L = \left( \sum_{ij} m_{ij} x^{\text{hat}}_{t,ij} - (1+\delta_t)\sum_{ij} m_{ij} x^{\text{hat,ung}}_{ij} \right)^2
        term_left = sum(
            (mask[k] * x_hat_t[k]).sum()
            for k in x_hat_t.keys()
        )
        term_right = sum(
            (mask[k] * x_hat_ung[k]).sum()
            for k in x_hat_ung.keys()
        ) * (1 + delta_t)
        loss_ = (term_left - term_right) ** 2

        grad_l = self.grad_loss(loss_, z_t)

        return grad_l


    def UG_guidance(self, x_hat_ung, dt, x_hat_t_norm):
        x_hat_ung_norm = self.normalize(x_hat_ung)
        Delta = x_hat_ung - x_hat_ung_norm
        eps = x_hat_ung.apply(
            lambda x: torch.empty_like(x).normal_(generator=self.generator)
        )
        return Delta + eps * dt  # TODO: not sure if h must be applied here or it's enough to have it later

    def LBG_guidance(self, x_hat_t_norm, x_hat_ung, delta_t, mask, z_t):
        # Monte-Carlo loss-based guidance (Eq. 9):
        #   MC_n(x_t, y) = grad_{x_t} log( (1/n) sum_i exp(-l_y(x^(i))) )
        # with x^(i) ~ N(x_hat_t, r_t^2 I) drawn in normalized space, l_y the DPS loss.
        # We differentiate the scalar
        #   L_MC = -log( (1/n) sum_i exp(-l_i) ) = log(n) - logsumexp_i(-l_i)
        # which reduces to DPS_guidance when n_mc=1, r_t=0.
        n_mc, r_t = self.n_mc, self.r_t

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

        grad_l = self.grad_loss(loss_, z_t)

        return grad_l