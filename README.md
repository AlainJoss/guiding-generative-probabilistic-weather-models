# Guiding Generative Probabilistic Weather Models

Resources:
- [report](/reports/latex-notes/main.pdf)

Analysis notebook:
```bash
uv run marimo edit notebooks/guidance.py --watch --no-token
```

Experiment runner:
```bash
python -m src.run --rollout_id 2026-07-02_12:13:27 --rollout_type ung
python -m src.run --rollout_id 2026-07-02_12:18:10 --rollout_type gui

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