

Done
- write data analyzer for playing around with N and mask before generating the rollout for the M distribution and start the guidance experiment.
- generate ensemble rollout as base trajectory
- define mask (or evolving masks) and define guidance (y) trajectory (extremified version of ensemble rollout trajectory over mask)
- Guide using dynamic mask instead of fixed one (future rollout but also regin difference)
- plot masks over N
- plot the sum of relative absolute change in variable from gen sample to gen guided sample.
- capture total relative change per variable in mask and overall.
- correct the loss function in latex doc and explain why y not possible in denormalized space.
- convert temperature to degrees celsius