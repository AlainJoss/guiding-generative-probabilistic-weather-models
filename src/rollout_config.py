from dataclasses import dataclass
from typing import Any

import torch

from src.utils.converters import (
    list_floats_to_tensors, list_tensors_to_floats, 
    datetime_to_string, string_to_datetime
)

##### guidance constants #####

ROLLOUT_TYPES = ["unguided", "guided"]

SWEEP_PARAMS = ["guidance_reference", "mask_mode", "alpha", "w"]

GUIDANCE_REFERENCES = [
    "unguided_members",
    # "ground_truth",
    # "lower_boundary",
    # "upper_boundary",
]
MASK_MODES = [
    "bbox", 
    "normal"
]
ALPHAS = [
    0, 2
]
WS = [
    100, 1000
]

##### config class #####

# TODO: should I actually implement Enums instead of lists?
@dataclass
class RolloutConfig:

    ### fixed params

    rollout_id: str | None = None
    guidance_flag: bool | None = None

    M: int | None = None
    N: int | None = None
    timestamp: str | None = None

    partition: str | None = None
    level: int | None = None
    var: str | None = None

    ### sweep params -> save them in config to use in rollout, but not extracted 

    mask_mode: str | None = None
    mask_corners: Any | None = None

    guidance_reference: str | None = None
    delta_trajectory: list[torch.Tensor] | None = None

    alpha: float | None = None
    w: float | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RolloutConfig":
        return cls(
            rollout_id=config.get("rollout_id"),
            guidance_flag=config.get("guidance_flag"),

            M=config.get("M"),
            N=config.get("N"),
            timestamp=string_to_datetime(config.get("timestamp")),

            partition=config.get("partition"),
            level=config.get("level"),
            var=config.get("var"),

            mask_mode=config.get("mask_mode"),
            mask_corners=config.get("mask_corners"),

            guidance_reference=config.get("guidance_reference"),
            delta_trajectory=config.get("delta_trajectory"),
            alpha=config.get("alpha"),
            w=config.get("w"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "guidance_flag": self.guidance_flag,

            "M": self.M,
            "N": self.N,
            "timestamp": datetime_to_string(self.timestamp),

            "partition": self.partition,
            "level": self.level,
            "var": self.var,

            "mask_mode": self.mask_mode,
            "mask_corners": self.mask_corners,

            "guidance_reference": self.guidance_reference,
            "delta_trajectory": self.delta_trajectory,

            "alpha": self.alpha,
            "w": self.w,
        }