# Guiding Generative Probabilistic Weather Models to Simulate Realistic Extreme Weather Events

Resources:
- [Notes](/resources/latex-notes/main.pdf)

Run analysis:
```bash
python -m src.runners.sweep --rollout_id 2026-05-27_10:01:30
python -m src.runners.run_from_config --rollout_id "2026-05-27_10:01:30" --rollout_type unguided_rollout --test
python -m src.runners.run_from_config --rollout_id "2026-05-22_17:53:46" --rollout_type guided_rollout --test
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
ln -s ../data data
mkdir modelstore
cp -r data/modelstore/. modelstore
# locally
ln -s ~/switchdrive data
# remember to git push after running config
git add . && git commit -m "Run configs." && git push origin main
# also when changing code locally
git pull
```