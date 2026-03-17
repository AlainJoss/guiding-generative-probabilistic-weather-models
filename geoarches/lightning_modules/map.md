# map

- diffusion.py: train and sample flow matching on residuals
- forecast.py: 
    - ForecastModule: deterministic forecaster (direct state prediction) used to train the “m” models
    - ForecastModuleWithCond: forward passess some model/ensemble with data + time embeddings