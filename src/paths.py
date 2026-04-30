from pathlib import Path

import os


ROOT = Path(__file__).parent.parent.resolve()

CONFIGS = ROOT / "configs"

# on Renku: ln -s ../data data
DATA = ROOT / "data"

# use to connect: ln -s ~/switchdrive data 
if not os.path.exists(DATA): 
    DATA = Path("~", "switchdrive").expanduser()

print(f"DATA path: {DATA}")

MODELSTORE = DATA / "modelstore"
ROLLOUTS = DATA / "rollouts"
ERA5 = DATA / "era5"