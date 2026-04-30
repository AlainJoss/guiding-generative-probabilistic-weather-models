from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

DATA = ROOT / "data"
# print(DATA)
CONFIGS = ROOT / "configs"
MODELSTORE = DATA / "modelstore"
ROLLOUTS = DATA / "rollouts"
ERA5 = DATA / "era5"