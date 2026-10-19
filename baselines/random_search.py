"""Random search: the floor every baseline must beat."""
from __future__ import annotations

import numpy as np


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