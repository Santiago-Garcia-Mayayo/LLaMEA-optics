# Ablation: the same algorithm with and without the constraint information

Mean test AOCC over seeds. blind = f is the penalised value, g = 0 (what the penalty arm sees).
penalty arm = best algorithm of the penalty arm of the same repetition, for reference.

| problem | repetition | arm | algorithm | normal | blind | blind - normal | p (Wilcoxon) | penalty arm |
|---|---|---|---|---|---|---|---|---|
| cec/RC12 | haiku5 | free | AdaptiveMomentumDE | 0.786 | 0.513 | -0.273 | 6.1e-05 | 0.523 |
| cec/RC12 | haiku5 | informed | MixedFeasibilityJDEDense | 0.601 | 0.505 | -0.095 | 0.000122 | 0.523 |
| cec/RC12 | haiku5 | informed_v2 | DecayedEliteCrossoverDE | 0.813 | 0.517 | -0.296 | 6.1e-05 | 0.523 |
| cec/RC12 | haiku5-r2 | free | RefinedMixedFeasibleDE | 0.740 | 0.516 | -0.223 | 6.1e-05 | 0.521 |
| cec/RC12 | haiku5-r2 | informed_v2 | RefinedFeasBanditDE | 0.732 | 0.509 | -0.224 | 6.1e-05 | 0.521 |
| cec/RC12 | haiku5-r3 | free | EliteFeasPoolDE | 0.766 | 0.514 | -0.253 | 6.1e-05 | 0.511 |
| cec/RC12 | haiku5-r3 | informed_v2 | FRMIDEPlusV13 | 0.682 | 0.513 | -0.169 | 6.1e-05 | 0.511 |
| cec/RC12 | haiku5-r4 | free | FeasibilityIPOPSHADEMixed | 0.652 | 0.510 | -0.142 | 0.000183 | 0.511 |
| cec/RC12 | haiku5-r4 | informed_v2 | AdaptiveWeightedJADEPolish | 0.736 | 0.511 | -0.225 | 6.1e-05 | 0.511 |
| cec/RC12 | haiku5-r5 | free | FeasibilityRuleMixedDE | 0.720 | 0.505 | -0.215 | 6.1e-05 | 0.519 |
| cec/RC12 | haiku5-r5 | informed_v2 | ConstraintWeightedElitistMixedDE | 0.688 | 0.505 | -0.184 | 6.1e-05 | 0.519 |
| lens/original | haiku5 | free | EliteEpsilonJADEPlus | 0.685 | 0.595 | -0.090 | 6.1e-05 | 0.604 |
| lens/original | haiku5 | informed | ScreenedLogitRefineDE | 0.600 | 0.585 | -0.015 | 0.151 | 0.604 |
| lens/original | haiku5 | informed_v2 | ConstraintFrontierMIDE | 0.619 | 0.576 | -0.043 | 0.0125 | 0.604 |
| lens/original | haiku5-r2 | free | EliteExtrapMomentumDE | 0.705 | 0.617 | -0.088 | 0.000122 | 0.594 |
| lens/original | haiku5-r2 | informed_v2 | DominantRepairBisectionES2 | 0.660 | 0.583 | -0.077 | 6.1e-05 | 0.594 |
| lens/original | haiku5-r3 | free | RefinedArchivePatternES | 0.659 | 0.585 | -0.074 | 0.000183 | 0.633 |
| lens/original | haiku5-r3 | informed_v2 | ConstraintGradientCreditES | 0.620 | 0.585 | -0.035 | 0.0103 | 0.633 |
| lens/original | haiku5-r4 | free | EpsDiscountBanditLocalSHADE | 0.694 | 0.611 | -0.083 | 6.1e-05 | 0.601 |
| lens/original | haiku5-r4 | informed_v2 | ConstraintArchiveJADEPlus | 0.706 | 0.632 | -0.074 | 0.000122 | 0.601 |
| lens/original | haiku5-r5 | free | RefinedFeasibilityMIDEv7 | 0.706 | 0.616 | -0.090 | 6.1e-05 | 0.610 |
| lens/original | haiku5-r5 | informed_v2 | ConstraintSurrogateBanditEA | 0.631 | 0.581 | -0.050 | 0.00427 | 0.610 |

Per problem, over all algorithms: mean normal / blind / penalty arm
* cec/RC12: 0.720 / 0.511 / 0.517 (11 algorithms)
* lens/original: 0.662 / 0.597 / 0.608 (11 algorithms)
