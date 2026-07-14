# Guiding Generative Probabilistic Weather Models

Resources:
- [report](/reports/latex-notes/main.pdf)

Analysis notebook:
```bash
uv run marimo edit src/ui/guidance.py --watch --no-token
uv run marimo edit src/ui/contour_demo.py --watch --no-token
```

Experiment runner:
```bash
python -m src.run --rollout_id ID --rollout_type ung
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