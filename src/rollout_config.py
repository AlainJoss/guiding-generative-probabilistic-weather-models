from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch

##### guidance constants #####

GUIDANCE_MODES = ["LBG", "LBG-MC", "UG", "FG", "FGF"]
GUI_REFS = ["UNG", "GT"]
MASK_MODES = ["BBOX", "GAUSSIAN"]
REG_TYPES = ["ID", "ACTIVITY", "POWER_SPECTRUM"]
LOSS_TYPES = ["STATE", "MASKED_AVERAGE"]  # derived from GUI_REF, never swept

# the hyperparameter tree: each guidance mode's own (swept) hypers. Symbols match the
# presentation (W, R, S, K, ETA, N_MC). W (guidance strength w) is mode-specific: it
# applies only to LBG / LBG-MC / UG -- FG learns its own lambda schedule and FGF has no
# guidance direction, so neither uses W. FG/FGF are not finalized yet, so they keep the
# old OPTIMIZE_K / OPTIMIZE_LR / SHIFT_INIT / CONTROL_GAMMA / N_WINDOWS fields -- the
# GUIDANCE_MODE axis disambiguates which mode reads which field.
MODE_HYPERS = {
    "LBG": ["W"],
    "LBG-MC": ["W", "N_MC", "R"],
    "UG": ["W", "K", "ETA", "S"],
    "FG": ["OPTIMIZE_K", "OPTIMIZE_LR", "CONTROL_GAMMA", "LAMBDA_INIT", "N_WINDOWS"],
    "FGF": ["OPTIMIZE_K", "OPTIMIZE_LR", "SHIFT_INIT", "CONTROL_GAMMA", "N_WINDOWS"],
}

# swept axes shared by every guidance mode (GUIDANCE_MODE drives the tree).
# W is NOT here -- it is mode-specific (see MODE_HYPERS).
COMMON_AXES = [
    "MASK_MODE",
    "GUIDANCE_DELTA",
    "GUI_REF",
    "REG_TYPE",
    "BETA_REG",
    "GUIDANCE_MODE",
]

# every swept axis written to sweep_params.json (flat union over all modes)
SWEEP_AXES = COMMON_AXES + sorted({h for hs in MODE_HYPERS.values() for h in hs})


##### config class #####
DATETIME_STR_FORMAT = "%Y-%m-%dT%H:%M:%S"

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
    W: float | None = None  # guidance strength w: applied as w * a_t * unit(grad) per flow step
    GUIDANCE_MODE: str | None = None

    ### mode-specific swept hypers (None -> method default; pinned for irrelevant modes)

    # LBG-MC
    N_MC: int | None = None    # number of Monte-Carlo samples
    R: float | None = None     # posterior spread magnitude (cloud std sigma_t = R * a_t)

    # UG
    K: int | None = None       # inner optimization steps
    ETA: float | None = None   # inner-optimization learning rate
    S: int | None = None       # self-recurrence steps

    # FG / FGF (not finalized: keep legacy field names; OPTIMIZE_K/OPTIMIZE_LR/SHIFT_INIT/
    # CONTROL_GAMMA/N_WINDOWS shared between FG and FGF)
    OPTIMIZE_K: int | None = None      # optimization iterations
    OPTIMIZE_LR: float | None = None
    SHIFT_INIT: float | None = None    # FGF control init
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
