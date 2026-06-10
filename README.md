# Guiding Generative Probabilistic Weather Models

better describe the UI
Order variables in legend
add option aggregate over the mask 
tune the schedule to 
isotrophic on the sphere
statements about scheduling
masking size
variables that mostly influence
delta change flat line 1%
compute cumulative percentage increase
write down the formula of 
sigmoud or poisson shape --> reach it in 10 days and have a function that imposes some prior assumption 
"force" states and how
define extremes
compute difference to ground truth in addition to unguided_guided or unguided
start much prior to heatwave
there are regions that have era5 anomalies

Resources:
- [report](/reports/latex-notes/main.pdf)

Analysis notebook:
```bash
marimo edit notebooks/guidance.py --watch --no-token
```

Experiment runner:
```bash

python -m src.run --rollout_id 2026-06-10_10:21:49 --rollout_type ung
python -m src.run --rollout_id 2026-06-09_11:43:12 --rollout_type gui
python -m src.run --rollout_id 2026-06-09_11:44:55 --rollout_type gui
python -m src.run --rollout_id 2026-06-10_10:26:17 --rollout_type gui

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

```python
# doesnt fall into the loss then grad pattern
# must use another pattern to implement this
# probably best to use guidance instead of specific loss as protocol
# then use losses and grad inside
def UG_loss(self, x_hat_t, x_hat_ung, delta_t, mask):
    Delta = self.normalize(x_hat_ung) - x_hat_t_norm

def guidance_vector()

def UG_vector():
    Delta = self.normalize(x_hat_ung) - x_hat_t_norm
    eps = generator()
    g_vec = Delta + h * generator()

def LGB_vector()

u_t = u_t - delta_t * Delta + h * generator()

```