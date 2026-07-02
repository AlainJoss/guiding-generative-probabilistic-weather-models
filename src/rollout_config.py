from dataclasses import dataclass
from datetime import datetime
from typing import Any


# hyperparameter tree
# hyper names equal the guidance-fn kwarg names so build_guidance_kwargs is a straight
# pass-through (no map). w is flow-level (-> lambda_schedule) and is skipped there.
# UG/FG/FGF optimizer hypers are method-namespaced so each method's lr/steps can be swept
# independently even though the flow fns share a generic optimizer.
GUIDANCE_METHOD_HYPERS = {
    "LBG": ["w"],
    "LBG-MC": ["w", "n_mc", "r"],
    "UG": ["w", "ug_k", "ug_lr", "ug_s"],
    "FG": ["fg_k", "fg_lr", "fg_gamma", "fg_lambda_init", "fg_n_windows"],
    "FGF": ["fgf_k", "fgf_lr", "fgf_shift_init", "fgf_gamma", "fgf_n_windows"],
}

# common hypers 
GUIDANCE_METHODS = ["LBG", "LBG-MC", "UG", "FG", "FGF"]
GUI_REFS = ["UNG", "GT"]
MASK_MODES = ["BBOX", "GAUSSIAN"]

COMMON_AXES = [
    "MASK_MODE",
    "GUIDANCE_DELTA",
    "GUI_REF",
    "GUIDANCE_MODE",
]

# union over all modes
SWEEP_AXES = COMMON_AXES + sorted({h for hs in GUIDANCE_METHOD_HYPERS.values() for h in hs})

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
    GUIDANCE_MODE: str | None = None

    ### mode-specific swept hypers (None -> method default; pinned for irrelevant modes)
    # names match the guidance-fn kwargs (see GUIDANCE_METHOD_HYPERS)

    # LBG, LBG-MC, UG
    w: float | None = None  # guidance strength: applied as w * a_t * unit(grad) per flow step

    # LBG-MC
    n_mc: int | None = None    # number of Monte-Carlo samples
    r: float | None = None     # posterior spread magnitude (cloud std sigma_t = r * a_t)

    # UG
    ug_k: int | None = None    # inner optimization steps
    ug_lr: float | None = None  # inner-optimization learning rate
    ug_s: int | None = None    # self-recurrence steps

    # FG
    fg_k: int | None = None            # optimization iterations
    fg_lr: float | None = None
    fg_gamma: float | None = None      # L2 penalty weight
    fg_lambda_init: float | None = None  # learned-lambda init
    fg_n_windows: int | None = None    # coarse schedule: number of windows

    # FGF
    fgf_k: int | None = None           # optimization iterations
    fgf_lr: float | None = None
    fgf_shift_init: float | None = None  # free-control init
    fgf_gamma: float | None = None     # L2 penalty weight
    fgf_n_windows: int | None = None   # coarse schedule: number of windows

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