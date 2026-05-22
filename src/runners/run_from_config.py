import argparse

from src.utils.read_write import (
    get_model,
    get_run_config,
)
from src.utils.setup import get_device
from src.rollout import rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_id", type=str, required=True)
    parser.add_argument("--rollout_type", choices=["guided_rollout", "unguided_rollout"], required=True)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()

def main():
    print("running experiment")

    args = parse_args()
    if args.test:
        flow_model = None 
    else:
        device = get_device()
        flow_model = get_model(device)

    config = get_run_config(args.rollout_type, args.rollout_id)
    rollout(
        config,
        flow_model,
        test=args.test,
    )


if __name__ == "__main__":
    main()