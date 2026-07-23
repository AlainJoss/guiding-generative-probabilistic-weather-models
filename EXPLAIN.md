# Optimal guidance

Optimal guidance is the one that brings the gap down to zero, while satisfying some trajectory constraint.
In other words, we want to get from a to b, and do it by optimizing some criteria.

In this particular case, there are multiple possible b's.
In addition, there are many possible constraints that can encode our path optimality, preferences.

You probably want to constrain your trajectory to have some kind of minimal deviation behavior. You for sure don't want to make any step to big (max deviation), but also want to keep the aggregate deviation low (total deviation summed over trajectory).