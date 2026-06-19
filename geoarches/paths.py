from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

# on Renku: ln -s ../data data
DATA = ROOT / "data"

# use to connect: ln -s ~/switchdrive data 
if ROOT.parents[0] == Path("/Users/alain/Desktop/master-thesis"):
    DATA = Path("~", "switchdrive").expanduser()

# prefer a fast local copy at ROOT/stats, else the data mount
if Path(ROOT, "stats").exists():
    STATS_PATH = Path(ROOT, "stats")
else:
    STATS_PATH = DATA / "stats"

# prefer a fast local copy at ROOT/modelstore (mirrors src/paths.py), else the data mount
if Path(ROOT, "modelstore").exists():
    MODELSTORE = Path(ROOT, "modelstore")
else:
    MODELSTORE = DATA / "modelstore"