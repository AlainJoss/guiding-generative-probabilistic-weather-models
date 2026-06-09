# Guiding Generative Probabilistic Weather Models

Resources:
- [report](/reports/latex-notes/main.pdf)

Analysis notebook:
```bash
marimo edit notebooks/guidance.py --watch --no-token
```

Experiment runner:
```bash
python -m src.run --rollout_id 2026-06-09_18:27:33 --rollout_type ung
python -m src.run --rollout_id 2026-06-09_11:34:54 --rollout_type ung
python -m src.run --rollout_id 2026-06-09_11:42:56--rollout_type gui
python -m src.run --rollout_id 2026-06-09_11:43:12 --rollout_type ung
python -m src.run --rollout_id 2026-06-09_11:44:55 --rollout_type gui

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