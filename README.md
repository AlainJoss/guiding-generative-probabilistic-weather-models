# Guiding Generative Probabilistic Weather Models to Simulate Realistic Extreme Weather Events

Resources:
- [Notes](/resources/latex-notes/main.pdf)

Run analysis:
```bash
python -m src.runners.run_all_configs --rollout_type unguided
python -m src.runners.run_from_config --rollout_id "2026-05-19_15:55:28" --rollout_type unguided --test
```

Notebooks:
```bash
uv run marimo edit notebooks/rollout.py --watch --no-token
uv run marimo edit notebooks/guide.py --watch --no-token
uv run marimo edit notebooks/analyze.py --watch --no-token
```

Setup:
```bash
# on Renku
ln -s ../data data
# locally
ln -s ~/switchdrive data
# remember to git push after running config
git add . && git commit -m "Run configs." && git push origin main
# also when changing code locally
git pull
```