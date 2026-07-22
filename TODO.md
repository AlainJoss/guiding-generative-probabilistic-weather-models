# TODOs



Changed: storage requirements:
M * N 
4 * 12 = 48
M * N * T
4 * 12 * 25 = 1200
1 state: 6.9MB
3 * M * N + 4 * M * N * T
ung: 
gui: 
ung_gui: 
sampling trace: 

Random:
- better describe the UI and make 
- statements about scheduling
- variables that mostly influence
- write down the formulas used for plots
- start much prior to heatwave
- there are regions that have era5 anomalies
- Order variables in legend

Communicate:
- serve one experiment on some cloud machine for them to access and play around with
- explain pipeline and code base visually

Rollouts:
- compare in and outside mask and also all other possible comparisons
- enable collapsing markdown cells 
- altair plots instead?
- make analyzer for historical weather data and search for final experiments of interest -> use clim data
- Guide full slice of temperature and observe what happens
- find real extremes (maybe even unpredictable) as use case
- define delta trajectory in terms of extremity (whathever that is) or guide using some notion of std of the forecast bands 
- implement logger for experiments

Analysis:
- write dist bands dropdown or show multiple lines somehow or show something on right and something on left
- make menu to select the notebook type instead of the stupid dropdown
- add clim to the plots
- compute differences in terms of percentages? Computing differences in normalized space seem to be susceptible to the variance of the variables itself.
- Cross-check variables distribution 
- how to compare parameters on the same plots?
- visualize unguided from guided at n so we can always know what the guidance is producing compared to the unguided model
- explainers for all variables
- new analysis over T:
    - realized vs. planned guidance
    - gradients
    - vector fields
- make checkboxes for things to plot or not in analyze part
- try out 3d viz for maps difference
- removed unused vars from era5 files

Guidance:
- should probably rewrite data interface
- for final evaluation rehabilitate det model with all 4 submodels
- sweep var to test how model function reacts: see whether guided percentage corresponds in untargeted guided version. 
For instance t2m-mslp and mslp-t2m
- define masks with physical priors
- define masks dynamically in N
- define weighted average Gaussian Kernel or future difference around region in loss function. Refine latex-notes with new definition of mask.
- Localize gradients by masking them or applying a penalty outside of the mask
- Try out regularization term ztKz or just ztIz=||z||2
- Guide using the ground truth and see whether the accuracy of other variables improves.
- Define an ensemble of G guided models.
- as baseline run the same eval_pipeline on guided and unguided ensembles in ground truth version and different experimental settings
- swap rollout_dist_plot with newer version present in analyze.py
- implement sampling algorithms (AI+RES and FK+S)
- how to define a realistic mask for the event of interest? lets say, el nino?

What we do not do:
- experiment with multiple variables (and masks correspondingly)
- extremes that are not smooth, over localized areas, and not medium range -> explain type in report

## THOUGHTS

- What about realism? What does this tell us about the model's learned weather dynamics in the first place? -> some distributional and magnitudal test?
- The gradient is not a reliable source of information. We can improve it by sampling multiple ones with different noises at each t. However, local dynamics will anyway not comply with the global real weather dynamics. It's like inpainting what we want. Defining global dynamics is feasible only by providing sampled states, but then what's the use of guidance?

14.05
- The problem I'm having with the targeted guidance is that we can achieve some tail event, but it's not easy at all to assess the realism of the generated weather states, and as a consequence the trajectory as a whole.

08.06 
- We are bounded by the deterministic prediction in terms of what we can explore

Bug reports:
- domain all has another bug