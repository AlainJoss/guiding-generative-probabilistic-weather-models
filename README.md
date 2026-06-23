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
how do vf and gui_vec interplay and is there a rule?

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
python -m src.run --rollout_id 2026-06-23_13:56:02 --rollout_type ung
python -m src.run --rollout_id 2026-06-18_09:39:05 --rollout_type gui
python -m src.run --rollout_id 2026-06-19_11:36:43 --rollout_type ung gui
python -m src.run --rollout_id 2026-06-19_16:03:20 --rollout_type ung gui
2026-06-19_16:03:20

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