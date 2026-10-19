"""CMA-ES (pycma) with IPOP restarts and a pluggable comparison.

Both variants give CMA-ES the *ranks* of the population under their comparison (CMA-ES is
invariant to monotone transformations of the fitness, so for the penalty variant this equals
using the penalty values), so the pair differs only in how solutions are ordered.
Bounds by pycma's BoundTransform; integer variables via pycma's `integer_variables`
(minimum step size per integer coordinate). Initial step 0.3 x range per coordinate.
"""
from __future__ import annotations

import numpy as np

from .common import DebComparison, PenaltyComparison, order


class _CMAES:
    comparison = None
    SIGMA0 = 0.3
    INC_POPSIZE = 2

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        import cma
        rng = np.random.default_rng(np.random.randint(2**31))
        cmp = self.comparison()
        lb, ub = np.asarray(func.bounds.lb, float), np.asarray(func.bounds.ub, float)
        popsize = int(4 + 3 * np.log(self.dim))
        while True:   # IPOP restarts until the harness ends the run (BudgetExhausted)
            opts = {
                "bounds": [lb.tolist(), ub.tolist()],
                "CMA_stds": (ub - lb).tolist(),
                "popsize": popsize,
                "integer_variables": [int(i) for i in func.integer_indices],
                "seed": int(rng.integers(1, 2**31 - 1)),
                "verbose": -9,
                "maxfevals": np.inf,
            }
            es = cma.CMAEvolutionStrategy(rng.uniform(lb, ub).tolist(), self.SIGMA0, opts)
            while not es.stop():
                X = np.asarray(es.ask())
                K = cmp.evaluate(func, X)
                ranks = np.empty(len(X))
                ranks[order(K)] = np.arange(len(X))
                es.tell(list(X), ranks.tolist())
            popsize *= self.INC_POPSIZE


class CMAESPenalty(_CMAES):
    comparison = PenaltyComparison


class CMAESDeb(_CMAES):
    comparison = DebComparison


class RandomSearch:
    """Uniform random sampling of the encoded box (batches of 100). Arm-agnostic: it never
    uses the returned values (the harness keeps the record), so one implementation serves
    both arms and gives identical results in both."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        rng = np.random.default_rng(np.random.randint(2**31))
        lb, ub = np.asarray(func.bounds.lb, float), np.asarray(func.bounds.ub, float)
        while True:
            func(rng.uniform(lb, ub, size=(100, self.dim)))
