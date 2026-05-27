from pathlib import Path


ROOT = Path(__file__).parent.parent.resolve()

# on Renku: ln -s ../data data
DATA = ROOT / "data"

# use to connect: ln -s ~/switchdrive data 
if ROOT.parents[0] == Path("/Users/alain/Desktop/master-thesis"):
    DATA = Path("~", "switchdrive").expanduser()
print(f"DATA path: {DATA}")

LOGS = DATA / "logs"
RUN_CONFIGS = DATA / "configs"
if Path(ROOT, "modelstore").exists():
    MODELSTORE = Path(ROOT, "modelstore")
else:
    MODELSTORE = DATA / "modelstore"
DET_MODEL_PATHS = [
    MODELSTORE / "archesweather-m-seed0",
    MODELSTORE / "archesweather-m-seed1",
    MODELSTORE / "archesweather-m-skip-seed0",
    MODELSTORE / "archesweather-m-skip-seed1",
]
GEN_MODEL_PATH = MODELSTORE / "archesweathergen"
ROLLOUTS = DATA / "rollouts"
ERA5 = DATA / "era5"