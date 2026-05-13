import argparse

from src.utils import (
    get_model,
    get_device,
    get_config
)
from src.rollout import rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", type=str, required=True)
    parser.add_argument("--config-type", choices=["guided", "unguided"], required=True)
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

    config = get_config(args.config_type, args.config_id)
    rollout(
        flow_model,
        config,
        test=args.test,
    )


if __name__ == "__main__":
    main()