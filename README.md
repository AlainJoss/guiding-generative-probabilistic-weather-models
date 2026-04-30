# Guiding Generative Probabilistic Weather Models to Simulate Realistic Extreme Weather Events

Resources:
- [Notes](/reporting/latex-notes/main.pdf)

Run analysis:
```
uv run marimo edit notebooks/rollout.py --watch --no-token
uv run marimo edit notebooks/guide.py --watch --no-token
uv run marimo edit notebooks/analyze.py --watch --no-token
uv run -m src.run
uv run -m src.from_cfg --config-id "2026-04-28_17:03:02" --test
python -m src.from_cfg --config-id "2026-04-28_17:03:02" --test
source .venv/bin/activate
python -m src.run_all_configs --config-type guided
uv run -m src.run_all_configs --test --config-type guided
```

Setup:
```
ln -s ../data data
# or
ln -s ~/switchdrive/data data

# remember to git pull after running config
```