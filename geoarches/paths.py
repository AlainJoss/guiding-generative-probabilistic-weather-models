import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

# on Renku: ln -s ../data data
DATA = ROOT / "data"

# use to connect: ln -s ~/switchdrive data 
if ROOT.parents[0] == Path("/Users/alain/Desktop/master-thesis"):
    DATA = Path("~", "switchdrive").expanduser()

STATS_PATH = DATA / "stats"
_fast = os.environ.get("FAST_MODELSTORE")
if _fast and Path(_fast, "archesweather-m-seed0").exists():
    MODELSTORE = Path(_fast)
else:
    MODELSTORE = DATA / "modelstore"