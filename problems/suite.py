"""Problem suite for the constraint-representation study.

Every problem exposes the same interface, so the harness, the baselines and the
three LLaMEA arms all see the same thing:

    p.name, p.dim, p.lb, p.ub, p.integer_indices, p.n_constraints, p.f_star
    p.seal(x)          -> clipped + rounded x (applied by the harness before every eval)
    p.constrained(x)   -> (f, v): objective and per-constraint violations v_i >= 0
                          (0 = satisfied); f = inf where the objective is undefined
    p.penalty(x)       -> single scalar (objective + penalties), the "penalty" arm
    p.is_feasible(x)   -> all(v <= feas_tol) and f finite

Search space shown to the algorithms is the problem's native box; the harness may map
it to [-1, 1]^d (recommended: hides problem identity from the LLM).
"""
from __future__ import annotations

import json
import os

import numpy as np

from cec2020 import CEC2020Problem

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROL_SET = (12, 13, 15, 16, 18, 21, 25, 26)
LENS_SET = ("original", "realistic_draft")


class CECProblem:
    """CEC 2020 RW problem with inequality constraints only.

    Violations are normalised per constraint by a fixed scale s_i = median |g_i(x)| over
    2000 uniform samples (seeded), so that one unit of violation is comparable across
    constraints. The penalty arm uses a static penalty  f + rho * sum_i v_i  with
    rho = 10 * max(1, |f*|)  (DECISION TO CONFIRM: any fixed rho is a modelling choice,
    exactly like the (10, 1, 1, 1, 1) weights of the lens).
    """

    feas_tol = 0.0  # competition convention: g_i(x) > 0 is a violation

    def __init__(self, num: int):
        self._p = CEC2020Problem(num)
        assert self._p.nh == 0, "control set restricted to inequality-only problems"
        self.num = num
        self.name = f"cec2020/RC{num:02d}"
        self.title = self._p.name
        self.dim = self._p.dim
        self.lb, self.ub = self._p.lb.copy(), self._p.ub.copy()
        self.integer_indices = self._p.integer_indices
        self.n_constraints = self._p.ng
        self.constraint_names = [f"g{i + 1}" for i in range(self.n_constraints)]
        cons = json.load(open(os.path.join(HERE, "cec_fstar_consensus.json")))
        self.f_star = cons[str(num)]["consensus_best"]
        self.scale = self._scales()
        self.rho = 10.0 * max(1.0, abs(self.f_star))

    def _scales(self, n=2000, seed=12345):
        rng = np.random.default_rng(seed + self.num)
        G = np.array([self._p.raw(self.lb + rng.random(self.dim) * (self.ub - self.lb)).g for _ in range(n)])
        s = np.median(np.abs(G), axis=0)
        return np.where(s > 0, s, 1.0)

    def seal(self, x):
        return self._p.seal(x)

    def constrained(self, x):
        r = self._p.raw(x)
        f = r.f if np.isfinite(r.f) else np.inf
        return f, np.maximum(0.0, r.g) / self.scale

    def raw_violations(self, x):
        return np.maximum(0.0, self._p.raw(x).g)

    def penalty(self, x):
        f, v = self.constrained(x)
        return float(f + self.rho * np.sum(v))

    def is_feasible(self, x):
        f, _ = self.constrained(x)
        return bool(np.isfinite(f) and np.all(self.raw_violations(x) <= self.feas_tol))


class LensSuiteProblem:
    feas_tol = 1e-9  # thickness constraints sit on box bounds; float noise ~1e-15

    def __init__(self, instance: str):
        from lens import LensProblem
        self._p = LensProblem(instance)
        self.name = self._p.name
        self.title = f"Double-Gauss lens ({instance})"
        self.dim = self._p.dim
        self.lb, self.ub = self._p.lb.copy(), self._p.ub.copy()
        self.integer_indices = self._p.integer_indices
        self.constraint_names = list(self._p.CONSTRAINT_NAMES)
        self.n_constraints = len(self.constraint_names)
        self.f_star = None  # unknown; LDG-EA best designs to be added as reference
        self.spec = self._p.spec

    def seal(self, x):
        return self._p.seal(x)

    def constrained(self, x):
        return self._p.constrained(x)

    def penalty(self, x):
        return self._p.penalty(x)

    def is_feasible(self, x):
        f, v = self.constrained(x)
        return bool(np.isfinite(f) and np.all(v <= self.feas_tol))

    def template(self):
        return self._p.template()


def load_suite(lens: bool = True, control: bool = True):
    probs = []
    if lens:
        probs += [LensSuiteProblem(i) for i in LENS_SET]
    if control:
        probs += [CECProblem(n) for n in CONTROL_SET]
    return probs
