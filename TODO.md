# TODOs

Rollouts:
- run eval with correct timestamp and compare
- manage paths in single place
- merge all experiments in single xarrays
- merce ground truth into single .nc for xarray interface
- implement logger for experiments
- implement hydra experiment for automatic logging instead of bash script?

Analysis:
- reimplement puckchart option
- explainers for all variables

Guidance:
- define masks with physical priors
- define masks dynamically in N
- define weighted average Gaussian Kernel or future difference around region in loss function. Refine latex-notes with new definition of mask.
- Try out regularization term ztKz or just ztIz=||z||2
- Guide using the ground truth and see whether the accuracy of other variables improves.
- Define an ensemble of G guided models.
- as baseline run the same eval_pipeline on guided and unguided ensembles in ground truth version and different experimental settings
- swap rollout_dist_plot with newer version present in analyze.py

What we do not do:
- experiment with multiple variables (and masks correspondingly)

## THOUGHTS

- What about realism? What does this tell us about the model's learned weather dynamics in the first place? -> some distributional and magnitudal test?
- The gradient is not a reliable source of information. We can improve it by sampling multiple ones with different noises at each t. However, local dynamics will anyway not comply with the global real weather dynamics. It's like inpainting what we want. Defining global dynamics is feasible only by providing sampled states, but then what's the use of guidance?

14.05
- The problem I'm having with the targeted guidance is that we can achieve some tail event, but it's not easy at all to assess the realism of the generated weather states, and as a consequence the trajectory as a whole.