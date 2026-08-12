from pathlib import Path

# repo
ROOT = Path(__file__).parent.parent.resolve()
print(f"ROOT @ {ROOT}")

SWITCHDRIVE_DATA = ROOT / "data"
if ROOT.parents[0] == Path("/Users/alain/Desktop/master-thesis"):
    SWITCHDRIVE_DATA = Path("~", "switchdrive").expanduser()

ERA5 = SWITCHDRIVE_DATA / "era5"

# local root for heavy stuff: ground-truth ERA5 + climatology
LOCAL_DATA = ROOT / "local_data"

CLIM = LOCAL_DATA / "stats" / "era5_240_clim.nc"
GT = LOCAL_DATA / "era5"

# persisted PCA bases for the mask-region latent space (one .npz per var/level/bbox)
PCA_BASIS = LOCAL_DATA / "pca_basis"

# NOTE: switch between SWITCHDRIVE_DATA or LOCAL_DATA for defining source for notebooks 
ROLLOUTS = LOCAL_DATA / "rollouts"
print(f"DATA @ {SWITCHDRIVE_DATA}  |  GT_DATA @ {LOCAL_DATA}  |  CLIM @ {CLIM}")

if Path(ROOT, "modelstore").exists():
    MODELSTORE = Path(ROOT, "modelstore")
else:
    MODELSTORE = SWITCHDRIVE_DATA / "modelstore"
print(f"MODELSTORE @ {MODELSTORE}")

DET_MODEL_PATHS = [
    MODELSTORE / "archesweather-m-seed0",
    MODELSTORE / "archesweather-m-seed1",
    MODELSTORE / "archesweather-m-skip-seed0",
    MODELSTORE / "archesweather-m-skip-seed1",
]
GEN_MODEL_PATH = MODELSTORE / "archesweathergen"