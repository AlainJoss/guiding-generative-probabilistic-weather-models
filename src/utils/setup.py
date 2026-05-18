import sys
import logging

from pathlib import Path 
from datetime import datetime

import torch

from src.paths import LOGS, ROLLOUTS



def setup_logging(log_prefix: str = "run") -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    log_file = LOGS / f"{log_prefix}_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def get_now_timestamp():
    date, time = str(datetime.now().replace(microsecond=0)).split(" ")
    return date + "_" + time


def ensure_rollout_dir(experiment_id: str = None) -> Path:
    if experiment_id is None:
        experiment_id = get_now_timestamp()
    rollout_dir = Path(ROLLOUTS, f"{experiment_id}")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    return rollout_dir


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"running on device: {device}")
    return device