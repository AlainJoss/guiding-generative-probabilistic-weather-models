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

    mask_mode: str | None = None
    guidance_reference: str | None = None
    delta_trajectory: list[torch.Tensor] | None = None

    alpha: float | None = None
    w: float | None = None

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
            guidance_reference=config.get("guidance_reference"),
            delta_trajectory=config.get("delta_trajectory"),

            alpha=config.get("alpha"),
            w=config.get("w"),
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
            "guidance_reference": self.guidance_reference,
            "delta_trajectory": self.delta_trajectory,

            "alpha": self.alpha,
            "w": self.w,
        }


    def to_dict_list(self) -> dict[str, Any]:
        return {
            "mask_mode": [self.mask_mode],
            "guidance_reference": [self.guidance_reference],
            "delta_trajectory": [self.delta_trajectory],

            "alpha": [self.alpha],
            "w": [self.w],
        }