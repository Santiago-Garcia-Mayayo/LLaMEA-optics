# Baselines on the test set

Mean ± std AOCC over seeds (runs that found a feasible point / runs). Higher is better; AOCC > 0.5 requires being feasible for most of the run.

| problem | llamea_penalty |
|---|---|
| lens/original | 0.588 ± 0.024 (15/15) |
| lens/realistic_draft | 0.669 ± 0.038 (15/15) |
| cec/RC12 | 0.526 ± 0.024 (15/15) |
| cec/RC13 | 0.977 ± 0.008 (15/15) |
| cec/RC15 | 0.972 ± 0.003 (15/15) |
| cec/RC16 | 0.566 ± 0.043 (15/15) |
| cec/RC18 | 0.609 ± 0.036 (15/15) |
| cec/RC21 | 0.943 ± 0.041 (15/15) |
| cec/RC25 | 0.515 ± 0.014 (15/15) |
| cec/RC26 | 0.520 ± 0.033 (15/15) |
| **mean over problems** | **0.688** |

Runs: 150; failed (error/crash/timeout): 0.
