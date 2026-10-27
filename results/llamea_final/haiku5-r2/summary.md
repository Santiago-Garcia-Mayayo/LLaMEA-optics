# Baselines on the test set

Mean ± std AOCC over seeds (runs that found a feasible point / runs). Higher is better; AOCC > 0.5 requires being feasible for most of the run.

| problem | llamea_free | llamea_informed_v2 | llamea_penalty |
|---|---|---|---|
| lens/original | 0.705 ± 0.030 (15/15) | 0.647 ± 0.027 (15/15) | 0.594 ± 0.031 (15/15) |
| lens/realistic_draft | 0.693 ± 0.022 (15/15) | 0.642 ± 0.024 (15/15) | 0.666 ± 0.022 (15/15) |
| cec/RC12 | 0.687 ± 0.105 (15/15) | 0.706 ± 0.126 (15/15) | 0.521 ± 0.019 (15/15) |
| cec/RC13 | 0.956 ± 0.017 (15/15) | 0.942 ± 0.018 (15/15) | 0.951 ± 0.049 (15/15) |
| cec/RC15 | 0.967 ± 0.006 (15/15) | 0.955 ± 0.009 (15/15) | 0.976 ± 0.005 (15/15) |
| cec/RC16 | 0.530 ± 0.030 (15/15) | 0.499 ± 0.014 (15/15) | 0.613 ± 0.109 (15/15) |
| cec/RC18 | 0.602 ± 0.072 (15/15) | 0.640 ± 0.113 (15/15) | 0.655 ± 0.207 (15/15) |
| cec/RC21 | 0.928 ± 0.024 (15/15) | 0.918 ± 0.084 (15/15) | 0.921 ± 0.090 (15/15) |
| cec/RC25 | 0.520 ± 0.015 (15/15) | 0.515 ± 0.013 (15/15) | 0.515 ± 0.008 (15/15) |
| cec/RC26 | 0.536 ± 0.082 (14/15) | 0.508 ± 0.030 (13/15) | 0.530 ± 0.023 (15/15) |
| **mean over problems** | **0.712** | **0.697** | **0.694** |

Runs: 450; failed (error/crash/timeout): 1.
* llamea_penalty cec/RC18 seed 1013: timeout — time limit exceeded (no func calls)
