# Guiding Generative Probabilistic Weather Models

better describe the UI and make 
tune the schedule to 
isotrophic on the sphere
statements about scheduling
variables that mostly influence
write down the formula of 
sigmoud or poisson shape --> reach it in 10 days and have a function that imposes some prior assumption 
"force" states and how
define extremes
start much prior to heatwave
there are regions that have era5 anomalies

Done:
- Order variables in legend
- add option aggregate over the mask 
- compute cumulative percentage increase
- delta change flat line 1%
- compute difference to ground truth in addition to unguided_guided or unguided


Resources:
- [report](/reports/latex-notes/main.pdf)

Analysis notebook:
```bash
marimo edit notebooks/guidance.py --watch --no-token
```

Experiment runner:
```bash
python -m src.run --rollout_id 2026-06-10_10:21:49 --rollout_type ung
python -m src.run --rollout_id 2026-06-10_10:26:15 --rollout_type gui
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