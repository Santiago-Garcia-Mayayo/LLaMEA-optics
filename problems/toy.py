"""Tiny synthetic problem with known optimum, for testing the harness (not part of the study).

minimise  f = x0^2 + x1^2 + (k - 3)^2 + [c != 2]     x0, x1 in [-5, 5], k in {1..5} ordinal,
s.t.      g1 = 1 - x0 <= 0        (x0 >= 1)                c in {0, 1, 2} categorical
          g2 = x1 - 2 <= 0
optimum:  x0 = 1, x1 = 0, k = 3, c = 2  ->  f* = 1
"""
from __future__ import annotations

import numpy as np

from .suite import Evaluation, ScoreSpec, Variable, _Base


class ToyProblem(_Base):
    def __init__(self):
        self.pid = "toy/basic"
        self.title = "toy"
        self.variables = [Variable("continuous", -5.0, 5.0), Variable("continuous", -5.0, 5.0),
                          Variable("ordinal", 1.0, 5.0), Variable("categorical", 0.0, 2.0)]
        self.constraint_names = ["g1", "g2"]
        self.f_star = 1.0
        self.score_spec = ScoreSpec(1.0, 1.0, 1e-6, 1.0)
        self.rho = 100.0
        self.n_calls = 0

    def evaluate(self, x) -> Evaluation:
        self.n_calls += 1
        x = self.seal(x)
        f = x[0] ** 2 + x[1] ** 2 + (x[2] - 3) ** 2 + float(x[3] != 2)
        g = np.array([1 - x[0], x[1] - 2])
        v = np.maximum(0.0, g)
        return Evaluation(float(f), v, float(f + self.rho * v.sum()))
