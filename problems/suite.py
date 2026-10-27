"""Problem suite for the constraint-representation study.

Every problem has the same interface (native, i.e. physical, variable space):

    p.pid, p.title, p.dim, p.n_constraints, p.constraint_names
    p.variables          -> list[Variable]: kind, bounds / integer values
    p.lb, p.ub           -> native box (integers: [first value, last value])
    p.seal(x)            -> clip to box + round integer variables
    p.evaluate(x)        -> Evaluation(f, v, penalty) from ONE simulation
    p.is_feasible(ev)    -> f finite and all v_i <= FEAS_TOL
    p.score_spec         -> how f is mapped to a [0, 1] score (harness/scoring.py)

The harness (harness/) maps the algorithm's encoded space to this native space, counts
evaluations and decides which view (penalty scalar or (f, v)) each arm sees.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .cec2020 import INTEGER_KINDS, CEC2020Problem

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# One feasibility tolerance for every problem, on the normalised violations v_i.
# Needed because constraints that sit on a bound evaluate to ~1e-15 by float round-off.
FEAS_TOL = 1e-9

CONTROL_SET = (12, 13, 15, 16, 18, 21, 25, 26)
LENS_SET = ("original", "realistic_draft")
PROBLEM_IDS = tuple(f"lens/{i}" for i in LENS_SET) + tuple(f"cec/RC{n:02d}" for n in CONTROL_SET)


def violation_scale(R: np.ndarray) -> np.ndarray:
    """Per-constraint scale from a sample of raw violations R (n_samples x m, >= 0):
    the median of the positive values (a 'typical violation'); 1.0 if never violated.
    One unit of normalised violation then means 'a typical violation of that constraint',
    so that no constraint dominates a sum of violations because of its units."""
    R = np.asarray(R, float)
    out = np.ones(R.shape[1])
    for j in range(R.shape[1]):
        pos = R[:, j][np.isfinite(R[:, j]) & (R[:, j] > 0)]
        if pos.size:
            out[j] = max(float(np.median(pos)), 1e-12)
    return out


@dataclass(frozen=True)
class Variable:
    """One decision variable in native units.

    kind: "continuous" | "ordinal" | "binary" | "categorical"
    continuous: lb <= x <= ub.  Integer kinds: x in {values[0], ..., values[-1]} (consecutive ints).
    """
    kind: str
    lb: float
    ub: float

    @property
    def is_integer(self) -> bool:
        return self.kind != "continuous"

    @property
    def n_values(self) -> int:
        return int(round(self.ub - self.lb)) + 1 if self.is_integer else 0


@dataclass
class Evaluation:
    f: float                 # objective (inf where undefined)
    v: np.ndarray            # normalised violations, >= 0, 0 = satisfied
    penalty: float           # scalar seen by the penalty arm
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreSpec:
    """score = 1 at gap <= gap_min, 0 at gap >= gap_max, log-linear in between, where
    gap = (f - f_ref) / scale. See harness/scoring.py."""
    f_ref: float
    scale: float
    gap_min: float
    gap_max: float
    note: str = ""


class _Base:
    pid: str
    title: str
    variables: list
    constraint_names: list

    @property
    def dim(self):
        return len(self.variables)

    @property
    def n_constraints(self):
        return len(self.constraint_names)

    @property
    def lb(self):
        return np.array([v.lb for v in self.variables], float)

    @property
    def ub(self):
        return np.array([v.ub for v in self.variables], float)

    @property
    def integer_indices(self):
        return [i for i, v in enumerate(self.variables) if v.is_integer]

    def seal(self, x):
        x = np.clip(np.asarray(x, float).ravel(), self.lb, self.ub)
        ii = self.integer_indices
        if ii:
            x[ii] = np.rint(x[ii])
        return x

    def is_feasible(self, ev: Evaluation) -> bool:
        return bool(np.isfinite(ev.f) and np.all(ev.v <= FEAS_TOL))


class CECProblem(_Base):
    """CEC 2020 RW problem with inequality constraints only.

    v_i = max(0, g_i(x)) / s_i, with s_i = median of the positive g_i over 2000 seeded uniform
    samples (violation_scale), so that one unit of violation is comparable across constraints.
    Penalty arm: f + rho * sum_i v_i with rho = 10 * max(1, |f*|)  (DESIGN.md, decision P1).
    """

    def __init__(self, num: int, rho_scale: float = 1.0):
        self._p = CEC2020Problem(num)
        assert self._p.nh == 0, "control set restricted to inequality-only problems"
        self.num = num
        self.rho_scale = float(rho_scale)
        # 'cec/RC12@rho10' = the same problem with 10x the penalty weight (sensitivity study only)
        self.pid = f"cec/RC{num:02d}" + ("" if self.rho_scale == 1.0 else f"@rho{self.rho_scale:g}")
        self.title = self._p.name
        kinds = {i: INTEGER_KINDS.get(num, "ordinal") for i in self._p.integer_indices}
        self.variables = []
        for i in range(self._p.dim):
            lo, hi = float(self._p.lb[i]), float(self._p.ub[i])
            if i in kinds:  # official encoding [k-0.51, m+0.49] -> values k..m
                self.variables.append(Variable(kinds[i], float(np.ceil(lo)), float(np.floor(hi))))
            else:
                self.variables.append(Variable("continuous", lo, hi))
        self.constraint_names = [f"g{i + 1}" for i in range(self._p.ng)]
        cons = json.load(open(os.path.join(DATA, "cec_fstar_consensus.json")))
        self.f_star = float(cons[str(num)]["consensus_best"])
        self.scale = self._scales()
        self.rho = self.rho_scale * 10.0 * max(1.0, abs(self.f_star))
        self.score_spec = ScoreSpec(self.f_star, abs(self.f_star), 1e-6, 1.0,
                                    "relative gap to best-known value of the CEC 2020 competition")

    def _scales(self, n=2000, seed=12345):
        rng = np.random.default_rng(seed + self.num)
        lb, ub = self.lb, self.ub
        R = np.array([np.maximum(0.0, self._p.raw(self.seal(lb + rng.random(self.dim) * (ub - lb))).g)
                      for _ in range(n)])
        return violation_scale(R)

    def evaluate(self, x) -> Evaluation:
        r = self._p.raw(self.seal(x))
        f = float(r.f) if np.isfinite(r.f) else np.inf
        v = np.maximum(0.0, r.g) / self.scale
        pen = f + self.rho * float(np.sum(v)) if np.isfinite(f) else np.inf
        return Evaluation(f, v, pen, {"g_raw": r.g})


class LensSuiteProblem(_Base):
    """Double-Gauss lens. f = RMS^2 spot [mm^2] (inf if any ray is lost)."""

    def __init__(self, instance: str):
        from .lens import LensProblem
        self._p = LensProblem(instance)
        self.instance = instance
        self.pid = f"lens/{instance}"
        self.title = f"Double-Gauss lens ({instance})"
        nx = self._p.nx
        lb, ub = self._p.lb, self._p.ub
        self.variables = []
        for i in range(self._p.dim):
            if i < nx:
                self.variables.append(Variable("continuous", float(lb[i]), float(ub[i])))
            else:
                # glass index into the Schott catalogue (sorted by refractive index n_d;
                # neighbours have similar n_d but very different Abbe numbers)
                self.variables.append(Variable("categorical", float(lb[i]), float(ub[i])))
        self.constraint_names = list(self._p.CONSTRAINT_NAMES)
        self.spec = self._p.spec
        self.f_star = None
        self.score_spec = ScoreSpec(0.0, 1.0, 1e-5, 1.0,
                                    "absolute RMS^2 spot in mm^2: 1e-5 (~3 um RMS radius) -> score 1, "
                                    "1 mm^2 -> score 0")

    def evaluate(self, x) -> Evaluation:
        f, v, pen = self._p.evaluate_all(self.seal(x))
        return Evaluation(float(f), np.asarray(v, float), float(pen))

    @property
    def batch_ok(self) -> bool:
        """Vectorised (jax.vmap) evaluation pays off on GPU only: on CPU each lens already traces
        8,658 rays in one vectorised call and vmap is 1.5-3x *slower* per lens (measured).
        Force with LLAMEA_LENS_BATCH=1 / disable with =0."""
        flag = os.environ.get("LLAMEA_LENS_BATCH")
        if flag is not None:
            return flag == "1"
        import jax
        return jax.default_backend() != "cpu"

    def evaluate_batch(self, X) -> list:
        """Same values as [evaluate(x) for x in X] up to float round-off (~1e-15 relative)."""
        return [Evaluation(float(f), np.asarray(v, float), float(p))
                for f, v, p in self._p.evaluate_all_batch(np.array([self.seal(x) for x in X]))]

    def template(self):
        return self._p.template()


@lru_cache(maxsize=None)
def get_problem(pid: str):
    """'lens/original', 'lens/realistic_draft', 'cec/RC12', ... (cached per process).
    'cec/RC12@rho10': RC12 with the penalty weight multiplied by 10 (only the penalty view
    changes; objective, constraints and scoring are those of RC12)."""
    family, name = pid.split("/", 1)
    if family == "lens":
        return LensSuiteProblem(name)
    if family == "cec":
        base, _, scale = name.partition("@rho")
        return CECProblem(int(base.replace("RC", "")), float(scale) if scale else 1.0)
    if family == "toy":  # synthetic problem for tests only
        from .toy import ToyProblem
        return ToyProblem()
    raise KeyError(pid)


def load_suite(lens: bool = True, control: bool = True):
    return [get_problem(p) for p in PROBLEM_IDS
            if (lens and p.startswith("lens/")) or (control and p.startswith("cec/"))]