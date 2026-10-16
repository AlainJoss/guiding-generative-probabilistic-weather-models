# Guiding Generative Probabilistic Weather Models

Analysis notebook:
```bash
uv run marimo edit src/ui/guidance.py --watch --no-token
uv run marimo edit src/ui/flow_time_tests.py --watch --no-token
uv run marimo edit src/ui/experiment_builder.py --watch --no-token
uv run marimo edit src/ui/latent_trajectories.py --watch --no-token
```

Experiment runner:
```bash
python -m src.run --rollout_id 2026-08-17_11:46:56 --rollout_type gui
python -m src.download --rollout_id  2026-08-17_11:46:54
2026-08-19_16:19:44
```

Terminal setup:
```bash
# on Renku only once after creating the session 
cd guiding-generative-probabilistic-weather-models
ln -s ../data data 
mkdir modelstore stats
cp -r data/modelstore/* modelstore/
cp -r data/stats/* stats/

# locally
ln -s ~/switchdrive data
git push
git pull
```