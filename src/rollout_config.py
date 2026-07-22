from dataclasses import dataclass
from datetime import datetime
from typing import Any


# hyperparameter tree
# hyper names equal the guidance-fn kwarg names so build_guidance_kwargs is a straight
# pass-through (no map).
GUIDANCE_METHOD_HYPERS = {
    # FGWNOLR: secant on the exact scalar dL/dw -- no learning rate, no iteration
    # count (optimizes until the hardcoded loss threshold is reached)
    "FGWNOLR": ["fgwnolr_w_init", "eta"],
    # FGWNOGAP: exact per-step gap closure (damped Newton on S = target); eta=1 -> full
    "FGWNOGAP": ["eta"],
    # FGWNORM: unit-gradient NOLR -- prescribed kick norm w*a_t (c_t = 1)
    "FGWNORM": ["fgwnorm_w_init", "eta"],
    # FGWFREE: Adam on the full lambda trajectory; phi = kick-energy regularizer strength
    "FGWFREE": ["phi"],
    # FGWRHO: secant on the guidance-to-flow ratio w (kick normalized to ||u_t||)
    "FGWRHO": ["fgwrho_w_init"],
}

# mask hypers, shared by every mask mode (extent / sigma_div rescaling)
MASK_HYPERS = ["sigma_div"]

# common hypers
GUIDANCE_METHODS = ["FGWNOLR", "FGWNOGAP", "FGWFREE", "FGWRHO", "FGWNORM"]
GUI_REFS = ["UNG", "GT"]
MASK_MODES = ["BBOX", "ELLIPTICAL"]

COMMON_AXES = [
    "MASK_MODE",
    "GUIDANCE_DELTA",
    "GUI_REF",
    "GUIDANCE_MODE",
]

# union over all modes
SWEEP_AXES = COMMON_AXES + sorted(
    {h for hs in GUIDANCE_METHOD_HYPERS.values() for h in hs} | set(MASK_HYPERS)
)

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
    # target percentage profile p_n vs the reference baseline (non-scalar axis);
    # the online loss delta is derived per step in rollout.py so the target is absolute
    GUIDANCE_DELTA: list[float] | None = None
    GUI_REF: str | None = None  # UNG -> TODO: remove if guiding towards gt is not of interest
    GUIDANCE_MODE: str | None = None

    ### mode-specific swept hypers (None -> method default; pinned for irrelevant modes)
    # names match the guidance-fn kwargs (see GUIDANCE_METHOD_HYPERS)

    # FGWNOLR (secant on the exact scalar dL/dw; no learning rate; runs until the
    # hardcoded loss threshold is met)
    fgwnolr_w_init: float | None = None  # starting w

    # eta: shared closure rate for FGWNOLR and FGWNOGAP. a_t = (1 - eta)^(t+1);
    # for FGWNOGAP it is the fraction of the remaining gap closed per step (1 = all).
    eta: float | None = None

    # phi: FGWFREE regularizer strength (penalty on the applied guidance kicks)
    phi: float | None = None

    # fgwrho_w_init: starting guidance-to-flow ratio (10% default: undershoot);
    # the from-below search grows it to the SMALLEST gap-closing ratio
    fgwrho_w_init: float | None = None

    # fgwnorm_w_init: starting kick scale for FGWNORM (kick norm = w*a_t, unit gradient)
    fgwnorm_w_init: float | None = None

    # sigma_div: mask extent divisor, shared by ALL mask modes (half-side or
    # sigma = box extent / sigma_div; 2.0 = base box, 4.0 = half, 1.0 = double).
    # None -> default 2.0 at the get_mask_2d call site.
    sigma_div: float | None = None

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