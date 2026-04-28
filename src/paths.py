from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

CONFIGS = ROOT / "configs"

DATA = ROOT / "data"
# print(DATA)
MODELSTORE = DATA / "modelstore"
ROLLOUTS = DATA / "rollouts"
ERA5 = DATA / "era5"

# print(ERA5)

# if __name__ == "__main__":