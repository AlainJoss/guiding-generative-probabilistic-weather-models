# Guiding Generative Probabilistic Weather Models to Simulate Realistic Extreme Weather Events

Resources:
- [Notes](/resources/latex-notes/main.pdf)

Run analysis:
```bash
python -m src.runners.run_sweep --rollout_id 2026-05-27_14:32:22
python -m src.runners.run_from_config --rollout_type guided_rollout --rollout_id 2026-05-27_14:32:22 --test
```

Notebooks:
```bash
# new one all in once
uv run marimo edit notebooks/guidance.py --watch --no-token

# old separate ones kept for consistency
uv run marimo edit notebooks/rollout.py --watch --no-token
uv run marimo edit notebooks/guide.py --watch --no-token
uv run marimo edit notebooks/analyze.py --watch --no-token
```

Setup:
```bash
# on Renku
cd guiding-generative-probabilistic-weather-models
ln -s ../data data
mkdir modelstore
cd ..  # work
cp -r data/modelstore/* guiding-generative-probabilistic-weather-models/modelstore/
# or rsync -av data/modelstore/ guiding-generative-probabilistic-weather-models/modelstore/

# locally
ln -s ~/switchdrive data
git push
git pull
```