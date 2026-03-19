
import importlib
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from hydra.utils import instantiate
from tensordict.tensordict import TensorDict
from tqdm import tqdm

import geoarches.stats as geoarches_stats
from geoarches.backbones.dit import TimestepEmbedder
from geoarches.lightning_modules import BaseLightningModule
from geoarches.utils.tensordict_utils import tensordict_apply, tensordict_cat

from geoarches.lightning_modules.base_module import AvgModule, load_module

geoarches_stats_path = importlib.resources.files(geoarches_stats)


class GuidedFlow(BaseLightningModule):
    """
    Rollout, sample, and guide predictions (det+gen).
    """
    def __init__(
        self,
        cfg,
        load_deterministic_model=None
    ):
        super().__init__()
        self.__dict__.update(locals())
        
        # some constants that were floating hardcoded around the codebase 
        # or are uselessly in the cfg (since they are fix anyway)
        self.scale_input_noise = 1.05
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

        self.month_embedder = TimestepEmbedder(self.cond_dim)
        self.hour_embedder = TimestepEmbedder(self.cond_dim)
        self.timestep_embedder = TimestepEmbedder(self.cond_dim)

        # might get rid of this to simplify or speed up
        self.inference_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps
        )

        # sigma scaling factor (shape like usual states)
        pangu_stats = torch.load(
            geoarches_stats_path / "pangu_norm_stats2_with_w.pt", weights_only=True
        )
        pangu_scaler = TensorDict(
            level=pangu_stats["level_std"], surface=pangu_stats["surface_std"]
        )
        scaler = TensorDict(
            **torch.load(
                geoarches_stats_path / "deltapred24_aws_denorm.pt", weights_only=False
            )
        )
        scaler["level"][-1] *= 3  # we don't care too much about vertical velocity
        self.state_scaler = scaler / pangu_scaler  # inverse because we divide by state_scaler

    ##### methods #####

    def rollout(
        self, 
        N, 
        X_cond,
        X_guide_trajectory: list[torch.Tensor] | None = None, # shape: 
        mask: torch.Tensor | None = None,  # shape: 
    ):
        trajectory = []

        for n in range(N):
            X_guide = None if X_guide_trajectory is None else X_guide_trajectory[n]
            x_hat = self.rollout_step(
                X_cond=X_cond,
                X_guide=X_guide,
                mask=mask,
            )
            trajectory.append(x_hat)

            if n < N - 1:
                # TODO: update correctly!
                X_cond = [X_cond[1], x_hat]

        return trajectory

    def rollout_step(self,
        X_cond, X_guide, mask
    ):
        mu = self.det_model(X_cond)
        z = self.sample(X_cond, mu, X_guide, mask)
        x_hat = mu + self.state_scaler * z
        return x_hat
    
    def sample(
        self,
        batch,
        guidance_term: torch.Tensor,  # NOTE: when the guidance term is a full state, we can set the mask to torch.ones()
        mask: torch.Tensor,
        **kwargs,
    ):
        """
        kwargs args are fed to scheduler_step
        """
        # set up scheduler
        scheduler = self.inference_scheduler
        scheduler.set_timesteps(self.T)
        # TODO: probably need to check this part again now ...
        scheduler_kwargs = dict(s_churn=self.cfg.inference.s_churn)
        scheduler_kwargs.update(kwargs)

        # draw noise
        generator = torch.Generator(device=self.device)
        z = batch["state"].apply(
            lambda x: torch.empty_like(x).normal_(generator=generator)
        )
        # reason explained in paper
        z = z * self.scale_input_noise

        # det prediction
        with torch.no_grad():
            batch["pred_state"] = self.det_model(batch).detach()

        # remove next_state (save compute)
        loop_batch = {k: v for k, v in batch.items() if "next" not in k} 
        with torch.no_grad():
            # TODO: check wtf is happening with time going backwards
            for i, t in enumerate(tqdm(scheduler.timesteps)):
                # NOTE: 
                # t is in torch.linspace(1000, small_value, 25) 
                # loop_batch.shape == noisy_state and are Torch dicts with:
                    # level: Tensor(shape=torch.Size([1, 6, 13, 121, 240]), device=mps:0, dtype=torch.float32, is_shared=False),
                    # surface: Tensor(shape=torch.Size([1, 4, 1, 121, 240]), device=mps:0, dtype=torch.float32, is_shared=False)},
                # loop_batch is there for the conditioniing (X_{t-2}, X_{t-1})
                # noisy_state is being denoised progressively

                # predict noise model_output
                # NOTE: pred is the inverse velocity vector (x_t - r^theta(x^cond, x_t, t)) / s
                v = self.compute_velocity(
                    loop_batch,
                    z,
                    t=torch.tensor([t]).to(self.device),
                    guidance_term=guidance_term
                )  # also a tensor_dict

                # TODO: improve this bs
                # due to weird behavior of scheduler we need to use the following
                step_index = getattr(scheduler, "_step_index", None)
                def scheduler_step(*args, **kwargs):
                    # dt = sigma_next - sigma
                    # prev_sample = sample + dt * model_output (velocity)
                    out = scheduler.step(*args, **kwargs)
                    if hasattr(scheduler, "_step_index"):
                        scheduler._step_index = step_index
                    return out.prev_sample

                z = tensordict_apply(
                    scheduler_step, v, t, z, **scheduler_kwargs  
                )

                # at the end
                # TODO: need to remove this bs
                if step_index is not None:
                    scheduler._step_index = step_index + 1

        # z_T
        residual = z.detach()
        # r = sigma * z_T
        scaled_residual = tensordict_apply(
            torch.mul, residual, self.state_scaler.to(self.device)
        )
        # X_hat = X_det + r
        final_state = batch["pred_state"] + scaled_residual
        return final_state


    def compute_velocity(self, batch, noisy_next_state, t, guidance_term=None, mask=None):
        device = batch["state"].device
        eta = 0.1
        lambda_t = 0.1  # TODO: should input a time dependent lambda list to guided_sampling

        with torch.enable_grad():
            # TODO: choose the tensor we actually guide from outside
            #       --> pass variable_type and variable
            # for now: example on surface
            noisy_guided = noisy_next_state["surface"].detach().clone().requires_grad_(True)

            # rebuild noisy state with the differentiable tensor inserted
            guided_noisy_state = noisy_next_state.clone()
            guided_noisy_state["surface"] = noisy_guided

            # tensor_dict with single state (fields only)
            input_state = guided_noisy_state
    
            # concatenate inputs
            assert "pred_state" in batch
            pred_state = batch["pred_state"]
            prev_state = batch["prev_state"]
            input_state = tensordict_cat([prev_state, input_state], dim=1)
            input_state = tensordict_cat([pred_state, input_state], dim=1)
        
            # conditional by default
            times = pd.to_datetime(batch["timestamp"].cpu().numpy() * 10**9).tz_localize(None)
            month = torch.tensor(times.month).to(device)
            month_emb = self.month_embedder(month)
            hour = torch.tensor(times.hour).to(device)
            hour_emb = self.hour_embedder(hour)
            timestep_emb = self.timestep_embedder(t)

            cond_emb = month_emb + hour_emb + timestep_emb

            # NOTE: here we embedd prev_state (input_state[0]), current_state (batch["state"]), noisy_state (input_state[1])
            x = self.embedder.encode(batch["state"], input_state)
            x = self.backbone(x, cond_emb)
            out = self.embedder.decode(x)  # we get tdict

            ##### here is where we compute the guidance
            #     we use universal guidance for now

            # compute aggregate term for loss
            pred_guided_tensor = out["surface"]
            gen_agg_term = torch.mean(mask * pred_guided_tensor)  # NOTE: mask has should probably be 121x240

            # grad
            print("guidance_term", guidance_term)
            guidance_loss = torch.nn.functional.mse_loss(gen_agg_term, guidance_term)
            # TODO: check that grad is disabled again after this
            with torch.enable_grad():
                grad = torch.autograd.grad(
                    outputs=guidance_loss,
                    inputs=noisy_guided,
                    retain_graph=True,
                    create_graph=False,
                )[0]
            # make one update to the noisy state
            guided_noisy_state_hat = guided_noisy_state.clone()
            guided_noisy_state_hat["surface"] = noisy_guided - eta * grad

            # rebuild input_state
            input_state_hat = guided_noisy_state_hat
            input_state_hat = tensordict_cat([prev_state, input_state_hat], dim=1)
            input_state_hat = tensordict_cat([pred_state, input_state_hat], dim=1)

            # second forward pass
            x_hat = self.embedder.encode(batch["state"], input_state_hat)
            x_hat = self.backbone(x_hat, cond_emb)
            out_hat = self.embedder.decode(x_hat)

            # guided combo
            out_tilde = out.apply(lambda x: x.clone())
            for key in out.keys():
                out_tilde[key] = out[key] + lambda_t * (out_hat[key] - out[key])

            #####

        # compute velocity r-eps := (r-z)/s 
        # to pass the right thing to the euler sampler
        sigmas = t / self.num_train_timesteps  # t/T, e.g. 959.5 / 1000
        sigmas = sigmas[:, None, None, None, None]  # shape (batch_size,1,1,1,1), to divide elementwise
        out_tilde = (noisy_next_state - out_tilde).apply(lambda x: x / sigmas)
        return out_tilde