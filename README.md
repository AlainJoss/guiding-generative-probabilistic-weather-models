# guiding generative probabilistic weather models

resources:
- [report](/resources/latex-notes/main.pdf)

analyze results as follows:
- launch a VSCode instance: 

    [![launch - renku](https://renkulab.io/renku-badge.svg)](https://renkulab.io/p/alain.joss.1998/generative-weather/sessions/01KPWVZG1FCADRDKX0A77VDYKM/start)

- run following commands in the terminal
    ```bash
    python -m src.run --rollout_id 2026-06-08_11:18:37 --rollout_type gui


    python -m src.runners.run_from_config --rollout_type unguided_rollout --rollout_id 2026-05-28_11:44:34 --test
    python -m src.runners.run_all_configs --rollout_type unguided_rollout
    ```

Run analysis:
```bash
python -m src.run --rollout_id 2026-06-08_11:18:37 --rollout_type gui


python -m src.runners.run_from_config --rollout_type unguided_rollout --rollout_id 2026-05-28_11:44:34 --test
python -m src.runners.run_all_configs --rollout_type unguided_rollout
```

Notebooks:
```bash
# new one all in once
marimo edit notebooks/guidance.py --watch --no-token

# old separate ones kept for consistency
uv run marimo edit notebooks/rollout.py --watch --no-token
uv run marimo edit notebooks/guide.py --watch --no-token
uv run marimo edit notebooks/analyze.py --watch --no-token
```

Setup:
```bash
# on Renku
cd guiding-generative-probabilistic-weather-models
ln -s ../data data
mkdir modelstore
cd ..  
cp -r data/modelstore/* guiding-generative-probabilistic-weather-models/modelstore/
# or rsync -av data/modelstore/ guiding-generative-probabilistic-weather-models/modelstore/

# locally
ln -s ~/switchdrive data
git push
git pull
```