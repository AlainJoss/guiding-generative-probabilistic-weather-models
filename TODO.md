# TODOs
T
Changed: storage requirements:
M * N 
4 * 12 = 48
M * N * T
4 * 12 * 25 = 1200
1 state: 6.9MB
3 * M * N + 4 * M * N * T


New whiteboard:

- attempt wind
- compare mask region and not mask region rmSE and other metrics
- write marimo slidable latex formulas with values used in experiments
- try a new stop guide @ across different phi levels and assert whether they reconverge
- how to better initialize w? is there a relation with land portion, gradient norm, ...?
- test also wind variable on larger mask to recreate cyclones
- guide towards gt and measure RMSE inside and outside
- represent historical distribution in cross-var checks and implement sampler from historical dist for setup unguided diffusion
- the square bbox weights should also be weighted according to the latitude and longitude, and great sphere distance
- Correct the rankings or use a mode for them in the cross var section
- keep ranking of not diff in diff mode under cross var checks
- delta should be the notation for the applied perturbation (guidance vector)
- Use M=5 for final experiments
- Push: 
    - temperature land-mass instead of total temperature mass
    - (soft) maximum value
- compare total guidance across methods
- score guidance by push and gradient informativeness
- w depends on the mass that is being pushed and the schedule
- r^gui looks suspiscious
- if guide t1000 does 2tm follow equivalently?
- fully rebuild data API to retrieve gt needed for experiment and save in folder
- make views to visualize things side by side
- make gifs for levels (pressure level) / steps (n)
- can we run batches of guided experiments or is the GPU already maxed out?



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

Open questions:
- how to define realistic guidance target trajectory
- how to define a realistic mask for the event of interest? lets say, el nino?

Bug reports:
- domain all has another bug