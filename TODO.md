# TODOs

Storage requirements:
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
- Compute a priori the size of the experiment
- Copy data locally since I have 2TB disk!
- what happen to the rollout after we stop guiding?
- shouldn't show interactivity elements when they are fixed parameters -> calendar, N, M, ... present them in a compact way
- Use daterange calendar for presentation -> use in guided and analysis modes
- Remerge utils!

Experiments:
- tune w
- tune alpha with w
- fix alpha and w
- sweep mask size and mode
- try out guiding towards full state

Communicate:
- serve one experiment on some cloud machine for them to access and play around with
- explain pipeline and code base visuall
- domain all has another bug

Rollouts:
- handle delta trajectories from the UI
- Assess likelihood of gui vs ung_gui
- Guide full slice of temperature and observe what happens
- Guide both entire states (can define guidance at variable level and state level)
- Define mask for both state and gradients (or penalize in loss)
- New mask types: normal and cost-lines prior
- simply try to define a much broader area, or sweep over different sizes of the mask using the corners (nice, and make it a sweep param: like percentage corner increase)
- find real extremes (maybe even unpredictable) as use case
- define delta trajectory in terms of extremity (whathever that is) or guide using some notion of std of the forecast bands
- maybe define coarser gaussian mask?
- attempt new sampling route of AI+RES and FK-steering
- Implement gradient averaging and loss-based-guidance --> new sweep param
- run eval with correct timestamp and compare
- manage paths in single place
- merge all experiments in single xarrays
- merce ground truth into single .nc for xarray interface
- implement logger for experiments
- implement hydra experiment for automatic logging instead of bash script?

Analysis:
- Cross-check variables distribution 
- how to compare parameters on the same plots?
- visualize unguided from guided at n so we can always know what the guidance is producing compared to the unguided model
- explainers for all variables
- new analysis over T:
    - realized vs. planned guidance
    - gradients
    - vector fields
- make checkboxes for things to plot or not in analyze part

Guidance:
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