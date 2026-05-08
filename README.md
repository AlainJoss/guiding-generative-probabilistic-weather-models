# Guiding Generative Probabilistic Weather Models to Simulate Realistic Extreme Weather Events

# change plots font to match marimo theme:
A warm, paper-like theme with cream backgrounds, terracotta primary colors, and sage green accents. Uses Source Serif 4 for headings, Inter for body text, and IBM Plex Mono for code. Designed for comfortable reading during long sessions.



Resources:
- [Notes](/resources/latex-notes/main.pdf)

Run analysis:
```bash
# compare original vs. new implementation
bash experiments/eval_pipeline.sh
uv run -m src.run
# run 
python -m src.run_from_config --config-id "2026-05-05_17:37:52" --config-type unguided --test
python -m src.run_all_configs --config-type guided --test
uv run marimo edit notebooks/timestamp_discrepancy.py --watch --no-token
```

Dashboards:
```bash
uv run marimo run notebooks/guide.py
uv run marimo run notebooks/rollout.py
uv run marimo run notebooks/analyze.py
uv run marimo edit notebooks/rollout.py --watch --no-token
uv run marimo edit notebooks/guide.py --watch --no-token
uv run marimo edit notebooks/analyze_vibe.py --watch --no-token
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