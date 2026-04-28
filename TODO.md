# TODOs

Rollout:
- implement a logger for experiments and print error to log file instead of this

Analysis:
- produce trajectory that comes down again so middle can be extremified

Estaetics:
- explainers for all variables
- marimo tour for app usage
- guide and rollout should be a single interface that doesn't run anything by itself. I should just create folders, write config file, and launch experiments from somewhere else --> think about how this pipeline should work. Anyways, it makes sense that the presentation layer is only about guided experiments and their properties.


Guidance:
- define masks with physical priors
- define masks dynamically in N
- experiment with multiple variables (and masks correspondingly)
- define weighted average Gaussian Kernel or future difference around region in loss function. Refine latex-notes with new definition of mask.
- Try out regularization term ztKz or just ztIz=||z||2
- Guide using the ground truth and see whether the accuracy of other variables improves.
- Define an ensemble of  G guided models.
- as baseline compute some basic facts about ArchesWeatherGen. For instance, how well it does (compared to its deterministic brother)? How does performance degrade as N of rollout increases?
- swap rollout_dist_plot with newer version present in analyze.py

Results organization:
- save all separate tensors in single file