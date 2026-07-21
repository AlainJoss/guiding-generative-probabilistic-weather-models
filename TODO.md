# TODOs
All three methods apply ũ_t = u_t − λ_t g_t with g_t = ∇_{z_t}L(x̂_t). The key structural difference: NOGAP normalizes the gradient, NOLR and FREE apply it raw.

FGWNOLR — raw gradient, fixed profile, one optimized scalar:

$$\lambda_t = w^* , a_t, \qquad a_t = (1-\eta)^{t+1}, \qquad w^* = \arg\min_w L\bigl(z_T(w)\bigr) ;\text{(secant)}$$

No normalization: the realized per-step closure is $\propto |\partial S/\partial z_t|^2$, so the model's sensitivity shapes when guidance bites; $w^*$ only sets the global scale.

FGWNOGAP — Newton step, gradient used as direction only ($|g_t|^2$ cancels its magnitude):

$$\lambda_t = \frac{2, r_t ,(r_t - r_t^{\text{target}})}{h_t ,|g_t|^2}, \qquad r_t^{\text{target}} = (1-\eta)^{t+1} r_0$$

Closed-loop: $r_t$ is re-measured each step, so drift is corrected against the schedule. Normalizing NOLR's gradient would collapse it into (an open-loop) NOGAP — the raw gradient is what keeps them distinct.

FGWFREE — raw gradient like NOLR, but the whole trajectory is free (no $w/a_t$ split):

$$\lambda = \arg\min_{\lambda \ge 0} ; L\bigl(z_T(\lambda)\bigr) + \varphi \sum_{t=0}^{T-1} |h_t \lambda_t g_t|^2$$

solved by Adam with the exact frozen-g gradient

$$\frac{\partial J}{\partial \lambda_t} = -,h_t ,\langle a_{t+1},, g_t\rangle ;+; 2\varphi, h_t^2, \lambda_t ,|g_t|^2$$

where $a_{t+1}$ is the adjoint state. It sits between the two: sensitivity-shaped kicks like NOLR, but a free profile, with $\varphi$ charging the total injected guidance.

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

New guidance modes:
- LBG with 
- fix a_t and optimize w
- 

Random:
- Compute a priori the size of the experiment

better describe the UI and make 
tune the schedule to 
isotrophic on the sphere
statements about scheduling
variables that mostly influence
write down the formulas of evals
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



Experiments:
- tune w
- tune alpha with w
- fix alpha and w
- sweep mask size and mode
- try out guiding towards full state
- add flow grad to guidance modes

Communicate:
- visualize hypers tree
- marimo app tour and all the explanations with math formulas
- visualization of the pipeline
- serve one experiment on some cloud machine for them to access and play around with
- explain pipeline and code base visually
- report time and memory complexity of algos

Rollouts:
- problem: duplicated runs -> for instance lambda_reg = 0 produces same results for all reg types
- enable absolute values on the maps and add own scale dropdown to all widgets
- compare in and outside mask and also all other possible comparisons
- enable collapsing markdown cells
- implement the spherical gaussian 
- altair plots instead?
- make analyzer for historical weather data and search for final experiments of interest -> use clim data
- set better defaults for the sweep
- Keep rollout to less than 14 days ow it just explodes
- Assess likelihood of gui vs ung_gui
- Guide full slice of temperature and observe what happens
- Guide both entire states (can define guidance at variable level and state level)
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
- color pixels that are 0 in white if matplotlib is the library we use
- write dist bands dropdown or show multiple lines somehow or show something on right and something on left
- make menu to select the notebook type instead of the stupid dropdown
- remove flowgrad bullshit plots
- change of var should not change map in mask section
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