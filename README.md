# Guiding Generative Probabilistic Weather Models to Simulate Realistic Extreme Weather Events

Resources:
- [Notes](/reporting/latex-notes/main.pdf)

Run analysis:
```bash
# compare original vs. new implementation
bash experiments/eval_pipeline.sh
uv run -m src.run
# run 
python -m src.run_from_config --config-id "2026-04-28_17:03:02" --config-type guided
python -m src.run_all_configs --config-type guided
```

Dashboards:
```bash
uv run marimo run notebooks/guide.py
uv run marimo run notebooks/rollout.py
uv run marimo run notebooks/analyze.py
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