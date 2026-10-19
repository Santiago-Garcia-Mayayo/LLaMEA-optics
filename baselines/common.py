"""Comparisons shared by the baselines.

Each baseline is written once and paired with one of two comparisons, so that the pair
differs ONLY in how the constraint information is used:

  PenaltyComparison  (penalty arm)   rank by the penalised scalar y = func(x)
  DebComparison      (free arm)      Deb's feasibility rules on (f, g) = func(x):
                                     feasible < infeasible; feasible by f; infeasible by sum(g)

Keys are (n, 2) arrays [class, value], class 0 = feasible (or "no information" in the
penalty arm), compared lexicographically.
"""
from __future__ import annotations

import numpy as np

BIG = 1e300


class PenaltyComparison:
    arm = "penalty"

    def evaluate(self, func, X):
        y = np.atleast_1d(np.asarray(func(X), float))
        y = np.where(np.isfinite(y), y, BIG)
        return np.column_stack([np.zeros_like(y), y])


class DebComparison:
    arm = "free"

    def evaluate(self, func, X):
        f, g = func(X)
        f = np.atleast_1d(np.asarray(f, float))
        g = np.atleast_2d(np.asarray(g, float))
        t = g.sum(axis=1)
        feasible = (t == 0) & np.isfinite(f)
        # undefined objective without any recorded violation counts as one unit of violation
        t = np.where(~feasible & (t == 0), 1.0, t)
        val = np.where(feasible, f, np.where(np.isfinite(t), t, BIG))
        return np.column_stack([(~feasible).astype(float), val])


def leq(a, b):
    """a <= b elementwise for key arrays (n, 2)."""
    return (a[:, 0] < b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] <= b[:, 1]))


def lt(a, b):
    return (a[:, 0] < b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] < b[:, 1]))


def order(K):
    """Indices sorting keys from best to worst."""
    return np.lexsort((K[:, 1], K[:, 0]))


def improvement(old, new):
    """Positive improvement magnitude old -> new (for success-history weights)."""
    same = old[:, 0] == new[:, 0]
    d = np.where(same, np.abs(old[:, 1] - new[:, 1]), np.minimum(old[:, 1], BIG))
    d = np.where(np.isfinite(d), d, 1.0)
    return np.maximum(d, 1e-300)


def clip_to_bounds(X, lb, ub):
    return np.minimum(np.maximum(X, lb), ub)
