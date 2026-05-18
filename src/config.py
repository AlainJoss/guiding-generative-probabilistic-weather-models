from dataclasses import dataclass
from typing import Any

import torch

from src.utils.converters import list_floats_to_tensors


##### guidance #####

ROLLOUT_TYPES = ["unguided", "guided"]

GUIDANCE_REFERENCES = [
    "unguided_members",
    "ground_truth",
    "lower_boundary",
    "upper_boundary",
]

GUIDANCE_PARAMS = ["guidance_mode", "alpha", "w"]

MASK_MODES = ["bbox", "normal", "normal_minmax"]

##### config class #####

# TODO: should I actually implement Enums instead of lists?
@dataclass
class RolloutConfig:
    rollout_id: str | None = None
    guidance_flag: bool | None = None

    M: int | None = None
    N: int | None = None
    timestamp: str | None = None

    partition: str | None = None
    level: int | None = None
    var: str | None = None

    mask_mode: str | None = None
    mask_params: Any | None = None

    guidance_reference: str | None = None
    delta_trajectory: list[torch.Tensor] | None = None

    alpha: float | None = None
    w: float | None = None
    lambda_schedule: list[torch.Tensor] | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RolloutConfig":
        return cls(
            rollout_id=config.get("rollout_id"),
            guidance_flag=config.get("guidance_flag"),

            M=config.get("M"),
            N=config.get("N"),
            timestamp=config.get("timestamp"),

            partition=config.get("partition"),
            level=config.get("level"),
            var=config.get("var"),

            mask_mode=config.get("mask_mode"),
            mask_params=config.get("mask_params"),

            guidance_reference=config.get("guidance_reference"),
            delta_trajectory=(
                None
                if config.get("delta_trajectory") is None
                else list_floats_to_tensors(config["delta_trajectory"])
            ),

            alpha=config.get("alpha"),
            w=config.get("w"),
            lambda_schedule=(
                None
                if config.get("lambda_schedule") is None
                else list_floats_to_tensors(config["lambda_schedule"])
            ),
        )