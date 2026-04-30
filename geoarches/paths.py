import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

# on Renku: ln -s ../data data
DATA = ROOT / "data"

# use to connect: ln -s ~/switchdrive data 
if ROOT.parents[0] == Path("/Users/alain/Desktop/master-thesis"):
    DATA = Path("~", "switchdrive").expanduser()

print(f"DATA path: {DATA}")
STATS_PATH = DATA / "stats"
MODELSTORE = DATA / "modelstore"