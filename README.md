# Guiding Generative Probabilistic Weather Models

Resources:
- [report](/reports/latex-notes/main.pdf)

Analysis notebook:
```bash
marimo edit notebooks/guidance.py --watch --no-token
```

Experiment runner:
```bash
python -m src.run --rollout_id 2026-06-09_11:34:54 --rollout_type ung
python -m src.run --rollout_id 2026-06-09_11:42:56--rollout_type ung
python -m src.run --rollout_id 2026-06-09_11:43:12 --rollout_type ung
python -m src.run --rollout_id 2026-06-09_11:44:55 --rollout_type ung

```

Terminal setup:
```bash
# on Renku
cd guiding-generative-probabilistic-weather-models
ln -s ../data data
mkdir modelstore
cd ..  
cp -r data/modelstore/* guiding-generative-probabilistic-weather-models/modelstore/
cd guiding-generative-probabilistic-weather-models
# or rsync -av data/modelstore/ guiding-generative-probabilistic-weather-models/modelstore/

# locally
ln -s ~/switchdrive data
git push
git pull
```