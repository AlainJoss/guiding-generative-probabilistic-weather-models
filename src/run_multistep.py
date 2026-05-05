from pathlib import Path
import argparse
import shutil

import torch
import xarray as xr

from src.paths import ERA5, DET_MODEL_PATHS, GEN_MODEL_PATH
from src.utils import get_device, tensor_timestamp_to_string, batchify_and_move, rollout_to_xarray

from geoarches.dataloaders.era5 import Era5Forecast
from geoarches.lightning_modules import load_module


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
        GEN_MODEL_PATH,
        module_target="geoarches.lightning_modules.diffusion_time_correct.DiffusionModuleTimeCorrect",
    )
    return model.to(device).eval(), cfg


def load_gen_model(device):
    model, cfg = load_module(
        GEN_MODEL_PATH,
        module_target="geoarches.lightning_modules.diffusion.DiffusionModule",
    )
    return model.to(device).eval(), cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["det_ens", "gen", "gen_time_correct"], required=True)
    parser.add_argument("--multistep", type=int, default=10)
    parser.add_argument("--num-members", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = get_device()
    print(f"running on device: {device}")

    ds = Era5Forecast(
        path=ERA5,
        domain="test_z0012",
        lead_time_hours=24,
        multistep=args.multistep,
        load_prev=True,
        norm_scheme="pangu",
    )

    if args.model == "det_ens":
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
    out_dir = (
        Path("data/multistep")
        / f"{save_name}_multistep={args.multistep}_members={num_members}"
    )
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with torch.no_grad():
        for idx in range(len(ds)):
            sample = ds[idx]
            batch = batchify_and_move(sample, device)

            member_datasets = []

            for member in range(num_members):
                print(f"idx={idx}, member={member}")

                if args.model == "det_ens":
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

            timestamp = tensor_timestamp_to_string(sample["timestamp"])
            out_file = out_dir / f"{save_name}_{timestamp}.nc"
            xr_pred.to_netcdf(out_file)

            written += 1
            print(f"wrote sample {written}: idx={idx}, timestamp={timestamp}, file={out_file}")

            if args.max_samples and written >= args.max_samples:
                break

    print(f"saved rollouts to directory: {out_dir}")


if __name__ == "__main__":
    main()