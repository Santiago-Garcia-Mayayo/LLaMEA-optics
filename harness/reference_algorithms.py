"""Simple, trusted algorithms with the LLaMEA interface, used to test the harness end to end
and as sanity baselines. Not the study's baselines (those go in baselines/).

Run via the isolated runner with e.g. algorithm="harness.reference_algorithms:RandomSearchFree".
"""
from __future__ import annotations

import numpy as np


def _random_point(func, rng):
    return rng.uniform(func.bounds.lb, func.bounds.ub)


class RandomSearchPenalty:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(np.random.randint(2**31))
        best = np.inf
        for _ in range(self.budget):
            best = min(best, func(_random_point(func, rng)))
        return best


class RandomSearchFree:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(np.random.randint(2**31))
        best = np.inf
        for _ in range(self.budget):
            f, g = func(_random_point(func, rng))
            if np.all(g == 0) and f < best:
                best = f
        return best


class _DE:
    """DE/rand/1/bin with a pluggable comparison; integers handled by the harness rounding."""
    F, CR = 0.6, 0.9

    def __init__(self, budget, dim, pop=None):
        self.budget, self.dim = budget, dim
        self.pop = pop or min(40, max(10, 5 * dim))

    def keys(self, out):  # list of sortable keys, lower is better
        raise NotImplementedError

    def __call__(self, func):
        rng = np.random.default_rng(np.random.randint(2**31))
        lb, ub = func.bounds.lb, func.bounds.ub
        P = rng.uniform(lb, ub, size=(self.pop, self.dim))
        out = func(P)                       # batch evaluation
        keys = self.keys(out)
        while True:
            idx = np.arange(self.pop)
            a, b, c = (rng.permutation(self.pop) for _ in range(3))
            M = np.clip(P[a] + self.F * (P[b] - P[c]), lb, ub)
            mask = rng.random((self.pop, self.dim)) < self.CR
            mask[idx, rng.integers(self.dim, size=self.pop)] = True
            U = np.where(mask, M, P)
            out_u = func(U)
            ku = self.keys(out_u)
            better = np.array([ku[i] <= keys[i] for i in idx])
            P[better] = U[better]
            keys = [ku[i] if better[i] else keys[i] for i in idx]


class DEPenalty(_DE):
    def keys(self, y):
        return [(float(v),) for v in y]


class DEFeasibilityRules(_DE):
    """Deb's rules: feasible < infeasible; feasible by f; infeasible by total violation."""
    def keys(self, out):
        f, g = out
        tv = g.sum(axis=1)
        return [(0, float(fi)) if t == 0 and np.isfinite(fi) else (1, float(t) if t > 0 else 1.0)
                for fi, t in zip(f, tv)]


class CatchesEverything:
    """Adversarial: swallows exceptions and keeps calling func. Must not exceed the budget."""
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        for _ in range(10 * self.budget):
            try:
                func(np.zeros(self.dim))
            except Exception:
                pass


class Crashes:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        func(np.zeros(self.dim))
        raise ZeroDivisionError("boom")


class SpinsForever:
    """Never calls func again after the first evaluation (tests SIGALRM / hard kill)."""
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        func(np.zeros(self.dim))
        while True:
            pass


class SwallowsEverything:
    """Worst case: a bare except also swallows BudgetExhausted/HarnessTimeout and loops forever.
    Only the parent's hard kill can stop it."""
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        while True:
            try:
                func(np.zeros(self.dim))
            except:  # noqa: E722
                pass
