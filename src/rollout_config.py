from dataclasses import dataclass
from typing import Any
from datetime import datetime

import torch

##### guidance constants #####

GUIDANCE_REFERENCES = [
    "unguided_members",
    # "ground_truth",
]
MASK_MODES = [
    "bbox", 
    "normal"
]

GUIDANCE_MODES = [
    "DPS",
    "UG",
    "LBG",
    "FLOWGRAD",
    "FLOWGRAD_FREE",
]

##### config class #####
from src.constants import DATETIME_STR_FORMAT

def datetime_to_string(timestamp: datetime):
    return datetime.strftime(timestamp, DATETIME_STR_FORMAT)


def string_to_datetime(timestamp: str):
    return datetime.strptime(timestamp, DATETIME_STR_FORMAT)



@dataclass
class RolloutConfig:

    ### fixed params

    M: int | None = None
    N: int | None = None
    timestamp: str | None = None

    partition: str | None = None
    level: int | None = None
    var: str | None = None

    mask_corners: Any | None = None

    ### sweep params -> save them in config to use in rollout, but not extracted 
    delta_trajectory: list[torch.Tensor] | None = None
    mask_mode: str | None = None
    guidance_mode: str | None = None
    guidance_reference: str | None = None

    alpha: float | None = None
    w: float | None = None

    ### guidance hyperparameters (sweepable; None -> method default is used)

    # shared across guidance types
    regularized: bool | None = None
    beta: float | None = None
    normalize: bool | None = None
    eps: float | None = None

    # LBG only
    lbg_n_mc: int | None = None
    lbg_r_t: float | None = None

    # UG only
    ug_S: int | None = None
    ug_m: int | None = None
    ug_delta_lr: float | None = None

    # FLOWGRAD only
    fg_n_opt: int | None = None
    fg_lr: float | None = None
    fg_gamma: float | None = None
    fg_init_lambda: float | None = None
    fg_n_lambda: int | None = None  # FlowGrad coarse schedule: number of control windows K

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RolloutConfig":
        return cls(
            M=config.get("M"),
            N=config.get("N"),
            timestamp=string_to_datetime(config.get("timestamp")),

            partition=config.get("partition"),
            level=config.get("level"),
            var=config.get("var"),
    
            mask_corners=config.get("mask_corners"),


            mask_mode=config.get("mask_mode"),
            guidance_mode=config.get("guidance_mode"),
            guidance_reference=config.get("guidance_reference"),
            delta_trajectory=config.get("delta_trajectory"),

            alpha=config.get("alpha"),
            w=config.get("w"),

            regularized=config.get("regularized"),
            beta=config.get("beta"),
            normalize=config.get("normalize"),
            eps=config.get("eps"),

            lbg_n_mc=config.get("lbg_n_mc"),
            lbg_r_t=config.get("lbg_r_t"),

            ug_S=config.get("ug_S"),
            ug_m=config.get("ug_m"),
            ug_delta_lr=config.get("ug_delta_lr"),

            fg_n_opt=config.get("fg_n_opt"),
            fg_lr=config.get("fg_lr"),
            fg_gamma=config.get("fg_gamma"),
            fg_init_lambda=config.get("fg_init_lambda"),
            fg_n_lambda=config.get("fg_n_lambda"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "M": self.M,
            "N": self.N,
            "timestamp": datetime_to_string(self.timestamp) if self.timestamp is not None else None,

            "partition": self.partition,
            "level": self.level,
            "var": self.var,

            "mask_corners": self.mask_corners,

            "mask_mode": self.mask_mode,
            "guidance_mode": self.guidance_mode,
            "guidance_reference": self.guidance_reference,
            "delta_trajectory": self.delta_trajectory,

            "alpha": self.alpha,
            "w": self.w,

            "regularized": self.regularized,
            "beta": self.beta,
            "normalize": self.normalize,
            "eps": self.eps,

            "lbg_n_mc": self.lbg_n_mc,
            "lbg_r_t": self.lbg_r_t,

            "ug_S": self.ug_S,
            "ug_m": self.ug_m,
            "ug_delta_lr": self.ug_delta_lr,

            "fg_n_opt": self.fg_n_opt,
            "fg_lr": self.fg_lr,
            "fg_gamma": self.fg_gamma,
            "fg_init_lambda": self.fg_init_lambda,
            "fg_n_lambda": self.fg_n_lambda,
        }


    def to_dict_list(self) -> dict[str, Any]:
        return {
            "delta_trajectory": [self.delta_trajectory],
            "mask_mode": [self.mask_mode],
            "guidance_mode": [self.guidance_mode],
            "guidance_reference": [self.guidance_reference],

            "alpha": [self.alpha],
            "w": [self.w],

            "regularized": [self.regularized],
            "beta": [self.beta],
            "normalize": [self.normalize],
            "eps": [self.eps],

            "lbg_n_mc": [self.lbg_n_mc],
            "lbg_r_t": [self.lbg_r_t],

            "ug_S": [self.ug_S],
            "ug_m": [self.ug_m],
            "ug_delta_lr": [self.ug_delta_lr],

            "fg_n_opt": [self.fg_n_opt],
            "fg_lr": [self.fg_lr],
            "fg_gamma": [self.fg_gamma],
            "fg_init_lambda": [self.fg_init_lambda],
            "fg_n_lambda": [self.fg_n_lambda],
        }