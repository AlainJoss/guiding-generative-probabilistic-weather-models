from pathlib import Path

DATA = Path(__file__).parent.parent.resolve() / "data"
MODELSTORE = DATA / "modelstore"
ROLLOUTS = DATA / "rollouts"
ERA5 = DATA / "era5"

# print(ERA5)

# if __name__ == "__main__":