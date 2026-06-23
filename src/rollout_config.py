from dataclasses import dataclass
from datetime import datetime
from typing import Any


# hyperparameter tree
MODE_HYPERS = {
    "LBG": ["W"],
    "LBG-MC": ["W", "N_MC", "R"],
    "UG": ["W", "K", "ETA", "S"],
    "FG": ["OPTIMIZE_K", "OPTIMIZE_LR", "CONTROL_GAMMA", "LAMBDA_INIT", "N_WINDOWS"],
    "FGF": ["OPTIMIZE_K", "OPTIMIZE_LR", "SHIFT_INIT", "CONTROL_GAMMA", "N_WINDOWS"],
}

# common hypers 
GUIDANCE_MODES = ["LBG", "LBG-MC", "UG", "FG", "FGF"]
GUI_REFS = ["UNG", "GT"]
MASK_MODES = ["BBOX", "GAUSSIAN"]
REG_TYPES = ["ID", "ACTIVITY", "POWER_SPECTRUM"]
LOSS_TYPES = ["STATE", "MASKED_AVERAGE"]  # derived from GUI_REF, never swept

COMMON_AXES = [
    "MASK_MODE",
    "GUIDANCE_DELTA",
    "GUI_REF",
    "REG_TYPE",
    "BETA_REG",
    "GUIDANCE_MODE",
]

# union over all modes
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
    GUIDANCE_DELTA: list[float] | None = None  # per-step delta target (non-scalar axis)
    GUI_REF: str | None = None  # UNG -> TODO: remove if guiding towards gt is not of interest
    REG_TYPE: str | None = None  # ID | ACTIVITY | POWER_SPECTRUM
    BETA_REG: float | None = None
    GUIDANCE_MODE: str | None = None

    ### mode-specific swept hypers (None -> method default; pinned for irrelevant modes)
    
    # LBG, LBG-MC, UG
    W: float | None = None  # guidance strength w: applied as w * a_t * unit(grad) per flow step

    # LBG-MC
    N_MC: int | None = None    # number of Monte-Carlo samples
    R: float | None = None     # posterior spread magnitude (cloud std sigma_t = R * a_t)

    # UG
    K: int | None = None       # inner optimization steps
    ETA: float | None = None   # inner-optimization learning rate
    S: int | None = None       # self-recurrence steps

    # FG / FGF 
    OPTIMIZE_K: int | None = None      # optimization iterations
    OPTIMIZE_LR: float | None = None
    SHIFT_INIT: float | None = None    # FGF control init
    CONTROL_GAMMA: float | None = None  # FG/FGF L2 penalty weight
    LAMBDA_INIT: float | None = None   # FG learned-lambda init
    N_WINDOWS: int | None = None       # FG/FGF coarse schedule: number of windows K

    # NOTE: implementing from_dict and to_dict in this abstruse way, such that I can convert datetime objects

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