from pathlib import Path


ROOT = Path(__file__).parent.parent.resolve()
print(f"ROOT @ {ROOT}")

DATA = ROOT / "data"

if ROOT.parents[0] == Path("/Users/alain/Desktop/master-thesis"):
    DATA = Path("~", "switchdrive").expanduser()
    # DATA = ROOT / "local_data"

print(f"DATA @ {DATA}")

LOGS = DATA / "logs"
if Path(ROOT, "modelstore").exists():
    MODELSTORE = Path(ROOT, "modelstore")
else:
    MODELSTORE = DATA / "modelstore"
print(f"MODELSTORE @ {MODELSTORE}")
DET_MODEL_PATHS = [
    MODELSTORE / "archesweather-m-seed0",
    MODELSTORE / "archesweather-m-seed1",
    MODELSTORE / "archesweather-m-skip-seed0",
    MODELSTORE / "archesweather-m-skip-seed1",
]
GEN_MODEL_PATH = MODELSTORE / "archesweathergen"
ROLLOUTS = DATA / "rollouts"
ERA5 = DATA / "era5"