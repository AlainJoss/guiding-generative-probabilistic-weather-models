from pathlib import Path
import argparse

import pandas as pd
import torch
import xarray as xr

from src.paths import MODELSTORE
from src.utils import get_device

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.dataloaders import zarr
from geoarches.lightning_modules import load_module


DET_MODEL_PATHS = [
    MODELSTORE / "archesweather-m-seed0",
    MODELSTORE / "archesweather-m-seed1",
    MODELSTORE / "archesweather-m-skip-seed0",
    MODELSTORE / "archesweather-m-skip-seed1",
]

def load_det_avg_model(device):
    model, cfg = load_module(
        DET_MODEL_PATHS[0],
        avg_with_modules=DET_MODEL_PATHS[1:],
    )
    return [model.to(device).eval()], cfg

def load_det_ens_models(device):
    models = []
    cfg = None

    for path in DET_MODEL_PATHS:
        model, cfg_i = load_module(path)
        models.append(model.to(device).eval())
        if cfg is None:
            cfg = cfg_i

    return models, cfg

def load_gen_time_correct_model(device):
    model, cfg = load_module(
        MODELSTORE / "archesweathergen",
        module_target="geoarches.lightning_modules.diffusion_time_correct.DiffusionModuleTimeCorrect",
    )
    return model.to(device).eval(), cfg

def load_gen_model(device):
    model, cfg = load_module(
        MODELSTORE / "archesweathergen",
        module_target="geoarches.lightning_modules.diffusion.DiffusionModule",
    )
    return model.to(device).eval(), cfg

def rollout_to_xarray(ds, sample_multistep, init_timestamp, member):
    xr_rollout = ds.convert_trajectory_to_xarray(
        preds_future=sample_multistep,
        timestamp=init_timestamp.cpu(),
        denormalize=True,
    )

    return xr_rollout.expand_dims(member=[member])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["det_avg", "det_ens", "gen", "gen_time_correct"], required=True)
    parser.add_argument("--multistep", type=int, default=10)
    parser.add_argument("--num-members", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--force", action="store_true")  # if data/multistep/...zarr already exists, delete it first.
    args = parser.parse_args()

    device = get_device()
    print(f"running on device: {device}")

    from src.paths import ERA5
    ds = Era5Forecast(
        path=ERA5,
        domain="test_z0012",
        lead_time_hours=24,
        multistep=args.multistep,
        load_prev=True,
        norm_scheme="pangu",
    )

    if args.model == "det_avg":
        models, cfg = load_det_avg_model(device)
        num_members = 1

    elif args.model == "det_ens":
        models, cfg = load_det_ens_models(device)
        num_members = len(models)

    elif args.model == "gen":
        model, cfg = load_gen_model(device)
        models = [model]
        num_members = args.num_members
    else:
        model, cfg = load_gen_time_correct_model(device)
        models = [model]
        num_members = args.num_members

    save_name = args.output_name or args.model
    out_path = (
        Path("data/multistep")
        / f"{save_name}_multistep={args.multistep}_members={num_members}.zarr"
    )

    writer = zarr.ZarrIterativeWriter(out_path, force=args.force)
    written = 0
    with torch.no_grad():
        for idx in range(len(ds)):
            sample = ds[idx]

            batch = {
                k: v[None].to(device) if hasattr(v, "to") else v
                for k, v in sample.items()
            }

            member_datasets = []

            for member in range(num_members):
                print(f"idx={idx}, member={member}")
                                
                if args.model == "det_avg":
                    sample_multistep = models[0].forward_multistep(
                        batch,
                        iters=args.multistep,
                    )

                elif args.model == "det_ens":
                    sample_multistep = models[member].forward_multistep(
                        batch,
                        iters=args.multistep,
                        use_avg=False,
                    )

                else:
                    sample_multistep = models[0].sample_rollout(
                        batch,
                        batch_nb=idx,
                        member=member,
                        iterations=args.multistep,
                    )

                xr_member = rollout_to_xarray(
                    ds=ds,
                    sample_multistep=sample_multistep,
                    init_timestamp=batch["timestamp"],
                    member=member,
                )

                member_datasets.append(xr_member)

            xr_pred = xr.concat(member_datasets, dim="member")

            writer.write(xr_pred)

            written += 1
            timestamp = pd.to_datetime(int(sample["timestamp"]), unit="s")
            print(f"wrote sample {written}: idx={idx}, timestamp={timestamp}")

            if args.max_samples and written >= args.max_samples:
                break

    writer.to_netcdf(dump_id=0)
    print(f"saved rollouts to: {out_path}")
    print(f"also converted to NetCDF under: {out_path.parent}")


if __name__ == "__main__":
    main()