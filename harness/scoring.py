"""Scores and anytime performance (AOCC).

Per evaluation, the harness keeps the best point so far under the common criterion
(any feasible point beats any infeasible one; among feasible, lower f; among infeasible,
lower total violation). That best point gets a score:

  feasible:    s = 1 at gap <= gap_min, 0 at gap >= gap_max, log-linear in between,
               gap = (f - f_ref) / scale                                  -> s in [0, 1]
  infeasible:  s = -T / (1 + T),  T = sum_i v_i (1 if f undefined and no violation) -> s in (-1, 0]

The unit score u = (1 + s) / 2 maps this to [0, 1] with every feasible point in [0.5, 1]
and every infeasible one in [0, 0.5). AOCC is the mean of u over evaluations 1..budget
(area over the convergence curve, as in BLADE/IOH), counting u = 0 before the first
evaluation and keeping the last value if the algorithm stops early.
"""
from __future__ import annotations

import numpy as np


def total_violation(f: float, v: np.ndarray) -> float:
    """Sum of normalised violations. An undefined objective with no recorded violation
    (should not happen with the suite's problems) counts as one unit."""
    t = float(np.sum(v))
    return t if (t > 0 or np.isfinite(f)) else 1.0


def score(f: float, feasible: bool, T: float, spec) -> float:
    if not feasible:
        return -1.0 if not np.isfinite(T) else -T / (1.0 + T)
    gap = (f - spec.f_ref) / spec.scale
    if gap <= spec.gap_min:
        return 1.0
    if gap >= spec.gap_max:
        return 0.0
    lo, hi = np.log10(spec.gap_min), np.log10(spec.gap_max)
    return float((hi - np.log10(gap)) / (hi - lo))


def unit(s: float) -> float:
    return 0.5 * (1.0 + s)


def aocc(unit_curve: np.ndarray, budget: int) -> float:
    """unit_curve[t] = unit score of best-so-far after evaluation t+1 (len <= budget)."""
    if budget <= 0:
        return 0.0
    c = np.asarray(unit_curve, float)[:budget]
    if c.size == 0:
        return 0.0
    pad = np.full(budget - c.size, c[-1])
    return float(np.mean(np.concatenate([c, pad])))
