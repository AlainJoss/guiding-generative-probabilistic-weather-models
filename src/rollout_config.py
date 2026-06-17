from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch

##### guidance constants #####

GUIDANCE_MODES = ["DPS", "LBG", "UG", "FG", "FGF"]
GUI_REFS = ["UNG", "GT"]
MASK_MODES = ["BBOX", "GAUSSIAN"]
REG_TYPES = ["ID", "ACTIVITY", "POWER_SPECTRUM"]
LOSS_TYPES = ["STATE", "MASKED_AVERAGE"]  # derived from GUI_REF, never swept

# the hyperparameter tree: each guidance mode's own (swept) hypers. OPTIMIZE_K /
# OPTIMIZE_LR / SHIFT_INIT / CONTROL_GAMMA / N_WINDOWS are single shared fields --
# the GUIDANCE_MODE axis disambiguates which mode reads them.
MODE_HYPERS = {
    "DPS": [],
    "LBG": ["M_MC", "SIGMA_MC"],
    "UG": ["OPTIMIZE_K", "OPTIMIZE_LR", "RENOISE_S", "SHIFT_INIT"],
    "FG": ["OPTIMIZE_K", "OPTIMIZE_LR", "CONTROL_GAMMA", "LAMBDA_INIT", "N_WINDOWS"],
    "FGF": ["OPTIMIZE_K", "OPTIMIZE_LR", "SHIFT_INIT", "CONTROL_GAMMA", "N_WINDOWS"],
}

# swept axes shared by every guidance mode (GUIDANCE_MODE drives the tree)
COMMON_AXES = [
    "MASK_MODE",
    "GUIDANCE_DELTA",
    "GUI_REF",
    "REG_TYPE",
    "BETA_REG",
    "ALPHA",
    "W",
    "GUIDANCE_MODE",
]

# every swept axis written to sweep_params.json (flat union over all modes)
SWEEP_AXES = COMMON_AXES + sorted({h for hs in MODE_HYPERS.values() for h in hs})

##### config class #####
from src.constants import DATETIME_STR_FORMAT


def datetime_to_string(timestamp: datetime):
    return datetime.strftime(timestamp, DATETIME_STR_FORMAT)


def string_to_datetime(timestamp: str):
    return datetime.strptime(timestamp, DATETIME_STR_FORMAT)


@dataclass
class RolloutConfig:

    ### fixed params (config.json only, not swept)

    M: int | None = None
    N: int | None = None
    T: int | None = None  # number of flow/sampling steps (lower = faster, for testing)
    START_TS: datetime | None = None

    PARTITION: str | None = None
    LEVEL: int | None = None
    VAR: str | None = None
    MASK_CORNERS: Any | None = None

    ### common swept axes
    MASK_MODE: str | None = None
    GUIDANCE_DELTA: list[torch.Tensor] | None = None  # per-step delta target (non-scalar axis)
    GUI_REF: str | None = None  # UNG -> masked-average loss; GT -> state loss
    REG_TYPE: str | None = None  # ID | ACTIVITY | POWER_SPECTRUM
    BETA_REG: float | None = None
    ALPHA: float | None = None
    W: float | None = None
    GUIDANCE_MODE: str | None = None

    ### mode-specific swept hypers (None -> method default; pinned for irrelevant modes)

    # LBG
    M_MC: int | None = None
    SIGMA_MC: float | None = None

    # UG / FG / FGF (OPTIMIZE_K, OPTIMIZE_LR, SHIFT_INIT, CONTROL_GAMMA, N_WINDOWS shared)
    OPTIMIZE_K: int | None = None      # optimization iterations
    OPTIMIZE_LR: float | None = None
    RENOISE_S: int | None = None       # UG self-recurrence steps
    SHIFT_INIT: float | None = None    # UG dz init / FGF control init
    CONTROL_GAMMA: float | None = None  # FG/FGF L2 penalty weight
    LAMBDA_INIT: float | None = None   # FG learned-lambda init
    N_WINDOWS: int | None = None       # FG/FGF coarse schedule: number of windows K

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RolloutConfig":
        ts = config.get("START_TS")
        kwargs = {k: config.get(k) for k in cls.__dataclass_fields__ if k != "START_TS"}
        kwargs["START_TS"] = string_to_datetime(ts) if ts is not None else None
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}
        out["START_TS"] = datetime_to_string(self.START_TS) if self.START_TS is not None else None
        return out

    def to_dict_list(self) -> dict[str, Any]:
        # flat union of every swept axis, each wrapped as a single-element list
        return {axis: [getattr(self, axis)] for axis in SWEEP_AXES}
