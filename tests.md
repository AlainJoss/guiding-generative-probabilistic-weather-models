# tests

## setup
we have rho=2 * startdates=4 * masks=1 = 8 experiments (E)
M=5

## T1.1
for each e in E we have ung|gui and gui
    we then measure abs(Delta x^GE)
    this gives us a chart for each var with the levels on the x axis, and a line for each gamma, where m=0 should be shown and the shading should be done with min-max over the M members

    we then should make a table with gamma over cols and vars over rows, for each var sum over the levels and divide by the max over the different gammas for each m. the variance of the single entries is then given over the members m.
    from the variable scores we can build a single score in a new row that summarizes the variable scores just by summing them and then again normalize by the max (this should still retain the variability over M fro mean and std score)

    now for each experiment e we have the chart and the table (8 of them)

we now summarize can the tables into two views.
for each rho level
    view 1: shows the variable aggregated single score, has startdates in rows, and gammas in column. mean and std of the scores is across M.

    view 2: variables over rows of table, gammas in column, mean and std is taken over experiments and M.

now for the two views we build the same tables but compute mean and std also across rho level.

T1.2

for each e in E we have ung|gui and gui
    we then measure for each v,l : diff = normspatially(abs(Delta x^GE))_(v,l) - mask
    and then do sum(|diff_(i,j)|) 
    this is our per v,l total variation.

    notice that normspatially(abs(Delta x^GE))_(v,l) is now the object we use in the visual example subsection (the chart we already have is perfect).

    then sum(|diff_(i,j)|) is again an object which is per var and per level,
    which we can visualize in the same plot type as in T1.1.
    same goes for the summary tables.
    so in a nutshell, the only thing that is changing is the considered object, but the section should have the same structure, as is for the eval procedure.

T1.3
After T1.2 we now understand that we will have the same structure again.
Now the object to consider is 
    
