from pathlib import Path

import torch
import torch.nn as nn

from src.utils import (
    get_now_timestamp,
    get_device, get_dataset
)
from src.paths import MODELSTORE

from hydra.utils import instantiate
from geoarches.lightning_modules import load_module


rollout_iterations = 2
device = get_device()

metrics_kwargs = {
    "rollout_iterations": rollout_iterations,
    "lead_time_hours": 24,
    "save_memory": False
}
metrics = {
    "era5_ensemble": {
        "_target_": "geoarches.metrics.ensemble_metrics.Era5EnsembleMetrics",
    },
    "era5_brier": {
        "_target_": "geoarches.metrics.brier_skill_score.Era5BrierSkillScore",
    },
}
def make_metrics(device):
    return nn.ModuleDict(
        {
            metric_name: instantiate(metric_cfg, **metrics_kwargs)
            for metric_name, metric_cfg in metrics.items()
        }
    ).to(device)

test_metrics_det = make_metrics(device)
test_metrics_gen = make_metrics(device)

def test_step(batch, sample_multistep, test_metrics):
    # compute metrics
    for metric in test_metrics.values():
        metric.update(
            ds.denormalize(batch["future_states"]),  # TODO: or without denormalize? 
            [ds.denormalize(sample) for sample in sample_multistep],
        )


from geoarches.metrics.label_wrapper import convert_metric_dict_to_xarray

def save_metrics(test_metrics, output_dir, model_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, metric in test_metrics.items():
        labelled_metric_output = metric.compute()

        if isinstance(labelled_metric_output, dict):
            labelled_dict = {
                k: (v.cpu() if hasattr(v, "cpu") else v)
                for k, v in labelled_metric_output.items()
            }

            extra_dimensions = ["prediction_timedelta"]
            if "brier" in metric_name:
                extra_dimensions = ["quantile", "prediction_timedelta"]

            ds_metric = convert_metric_dict_to_xarray(
                labelled_dict,
                extra_dimensions,
            )

            torch.save(
                labelled_dict,
                output_dir / f"{model_name}-{metric_name}.pt",
            )
        else:
            ds_metric = labelled_metric_output

        ds_metric.to_netcdf(output_dir / f"{model_name}-{metric_name}.nc")

##### load #####

ds = get_dataset(multistep=rollout_iterations)
# has future_states key (with {rollout_iterations} states):
# level: Tensor(shape=torch.Size([2, 6, 13, 121, 240]) ...
# surface: Tensor(shape=torch.Size([2, 4, 1, 121, 240]) ...

##### det model
det_model, det_config = load_module(
    MODELSTORE / "archesweather-m-seed0",
    avg_with_modules=[
        MODELSTORE / "archesweather-m-seed1",
        MODELSTORE / "archesweather-m-skip-seed0",
        MODELSTORE / "archesweather-m-skip-seed1",
    ],
)
det_model = det_model.to(device).eval()

gen_model, gen_config = load_module(
    MODELSTORE / "archesweathergen",
    module_target="geoarches.lightning_modules.diffusion.DiffusionModule",
)

gen_model = gen_model.to(device).eval()

with torch.no_grad():
    for idx in range(100):
        # literally adds batch dim to fields (level, surface)
        batch = {k: v[None].to(device) for k, v in ds[idx].items()}

        print("starting det")
        sample_multistep = det_model.forward_multistep(batch, iters=rollout_iterations)
        print(f"finished det: {sample_multistep}")   

        print(type(sample_multistep), len(sample_multistep)) 

        test_step(batch, sample_multistep, test_metrics_det)

        print("starting gen")
        sample_multistep = gen_model.sample_rollout(
            batch,
            batch_nb=0,  # should be different for each input
            member=idx,
            iterations=rollout_iterations,
        )
        
        print(f"finished gen: {sample_multistep}")
        
        print(type(sample_multistep), len(sample_multistep))

        test_step(batch, sample_multistep, test_metrics_gen)

        path = ...
        save_metrics(test_metrics_det, "experiments/metrics/det", "det")
        save_metrics(test_metrics_gen, "experiments/metrics/gen", "gen")