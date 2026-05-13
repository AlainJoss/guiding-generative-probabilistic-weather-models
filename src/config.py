from dataclasses import dataclass
from typing import Any

from src.utils import (
    list_floats_to_tensors
)

PARTITIONS = ["surface", "level"]

LEVELS_DICT = {
    "surface": [0],
    "level": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
}

VARIABLES_DICT = {
    "surface": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure"
    ],
    "level": [
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "specific_humidity",
        "vertical_velocity"
    ]
}

ROLLOUT_TYPES = ["unguided", "guided"]

GUIDANCE_MODES = [
    "planned_trajectory",
    "ground_truth",
    "lower_boundary",
    "upper_boundary",
]

GUIDANCE_PARAMS = ["guidance_mode", "alpha", "w"]


@dataclass
class UnguidedConfig:
    rollout_id: str
    guidance_flag: bool
    M: int
    N: int
    timestamp: str
    partition: str
    level_idx: int
    var_idx: int

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "UnguidedConfig":
        # TODO: make all assertions with value in ...
        return cls(
            rollout_id=config["rollout_id"],
            timestamp=config["timestamp"],
            partition=config["partition"],
            level_idx=config["level_idx"],
            var_idx=config["var_idx"],
            guidance_flag=config["guidance_flag"],
            N=config["N"],
            M=config["M"],
        )
    
@dataclass
class GuidedConfig:
    timestamp: str
    partition: str
    level_idx: int
    var_idx: int
    lambda_: list[float] | float
    guidance_flag: bool
    N: int
    M: int
    ...

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "UnguidedConfig":
        # TODO: make all assertions with value in ...
        return cls(
            timestamp=config["timestamp"],
            partition=config["partition"],
            level_idx=config["level_idx"],
            var_idx=config["var_idx"],
            lambda_=list_floats_to_tensors(config["lambda_"]),
            guidance_flag=config["guidance_flag"],
            N=config["N"],
            M=config["M"],
        )
    
# usage: 
# cfg = RolloutConfig.from_dict(config)