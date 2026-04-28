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
```

