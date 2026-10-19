# Baselines on the test set

Mean ± std AOCC over seeds (runs that found a feasible point / runs). Higher is better; AOCC > 0.5 requires being feasible for most of the run.

| problem | random_search | lshade_penalty | lshade_deb | cmaes_penalty | cmaes_deb |
|---|---|---|---|---|---|
| lens/original | 0.576 ± 0.027 (5/5) | 0.596 ± 0.015 (5/5) | 0.608 ± 0.006 (5/5) | 0.567 ± 0.008 (5/5) | 0.567 ± 0.013 (5/5) |
| **mean over problems** | **0.576** | **0.596** | **0.608** | **0.567** | **0.567** |

Runs: 25; failed (error/crash/timeout): 0.
