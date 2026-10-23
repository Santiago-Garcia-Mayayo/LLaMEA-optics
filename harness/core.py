"""The harness: the only thing an algorithm can touch.

    h = Harness(problem, arm="free", budget=2000, seed=0)
    algorithm(h.func)          # may raise BudgetExhausted / HarnessTimeout (BaseException)
    res = h.result()

Arms (identical evaluation underneath, one simulation per point):
    "penalty"          func(x) -> y                 (objective + penalties, one float)
    "free", "informed", "informed_v2" func(x) -> (f, g)            (objective, violations g_i >= 0, 0 = satisfied)
("informed" and "informed_v2" differ from "free" only in the feedback the LLM receives, see evaluate.py.)
For a batch X of shape (n, d): penalty -> y (n,), free -> (f (n,), g (n, m)).

The harness, not the algorithm, is the source of truth: it seals points, counts the
budget, and tracks the best point under the common criterion for every arm.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

import numpy as np

from problems import FEAS_TOL

from . import scoring
from .space import EncodedSpace

ARMS = ("penalty", "free", "informed", "informed_v2")


class BudgetExhausted(BaseException):
    """Raised when the evaluation budget is used up. Derives from BaseException so that
    `except Exception:` in generated code does not swallow it."""


class HarnessTimeout(BaseException):
    """Raised when a run exceeds its wall-clock limit."""


@dataclass
class RunResult:
    problem: str
    arm: str
    seed: int
    budget: int
    n_evals: int = 0
    status: str = "ok"                 # ok | budget_exhausted | timeout | error | crashed
    error: str = ""
    found_feasible: bool = False
    first_feasible_eval: int | None = None
    best_f: float | None = None        # best feasible objective (None if none feasible)
    best_total_violation: float | None = None  # of the best point (0 if feasible)
    best_x: list | None = None         # native coordinates of the best point
    final_score: float = -1.0          # s of best point, in [-1, 1]
    final_unit: float = 0.0            # u = (1+s)/2, in [0, 1]
    aocc: float = 0.0
    feasible_fraction: float = 0.0     # fraction of evaluations that were feasible
    wall_time: float = 0.0
    curve: list = field(default_factory=list)   # [(eval, unit score)] at improvements
    # behaviour statistics (used in the feedback of the informed arm and in the analysis)
    best_gap: float | None = None      # (f - f_ref) / scale of the best feasible point
    viol_freq: list = field(default_factory=list)  # per constraint: fraction of evals violating it
    best_violated: list = field(default_factory=list)  # constraints violated at the best point
    last_improvement_eval: int | None = None
    eval_90: int | None = None         # first eval reaching 90% of the run's total improvement

    def to_dict(self):
        return asdict(self)


def _sanitise(ev):
    """NaN anywhere becomes +inf (worst), so comparisons and scores stay well defined."""
    if not np.isfinite(ev.f):
        ev.f = np.inf
    ev.v = np.nan_to_num(np.asarray(ev.v, float), nan=np.inf, posinf=np.inf, neginf=0.0)
    if not np.isfinite(ev.penalty):
        ev.penalty = np.inf
    return ev


class _Func:
    """Callable handed to the algorithm, with LLaMEA-style attributes."""
    __slots__ = ("_call", "bounds", "dim", "variable_types", "n_values", "integer_indices",
                 "continuous_indices", "n_constraints", "budget")

    def __call__(self, x):
        return self._call(x)

    def __repr__(self):
        return f"<func dim={self.dim}>"


class Harness:
    def __init__(self, problem, arm: str, budget: int, seed: int = 0,
                 time_limit: float | None = None, keep_trace: bool = True):
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if budget < 1:
            raise ValueError("budget must be >= 1")
        self.problem = problem
        self.arm = arm
        self.budget = int(budget)
        self.seed = int(seed)
        self.time_limit = time_limit
        self.keep_trace = keep_trace
        self.space = EncodedSpace(problem.variables)
        self.spec = problem.score_spec
        self.n_evals = 0
        self._t0 = time.perf_counter()
        self._exhausted = False
        self._timed_out = False
        # best point so far (common criterion)
        self._best_key = (2, np.inf)   # (0, f) feasible | (1, T) infeasible
        self._best_eval = None
        self._first_feasible = None
        self._n_feasible = 0
        self._curve = []               # (eval, unit) at each improvement
        self._unit_now = 0.0
        self._viol_count = np.zeros(problem.n_constraints, np.int64)
        # per-evaluation trace (for analysis)
        self.trace = {"f": [], "T": [], "n_violated": [], "feasible": [], "penalty": [], "x": []}
        self.func = self._make_func()

    # ------------------------------------------------------------------ public
    @property
    def remaining(self) -> int:
        return self.budget - self.n_evals

    def restart_clock(self):
        self._t0 = time.perf_counter()

    def result(self, status: str | None = None, error: str = "") -> RunResult:
        b = self._best_eval
        if status is None:
            status = "timeout" if self._timed_out else ("budget_exhausted" if self._exhausted else "ok")
        unit_curve = self._dense_unit_curve()
        r = RunResult(
            problem=self.problem.pid, arm=self.arm, seed=self.seed, budget=self.budget,
            n_evals=self.n_evals, status=status, error=error,
            found_feasible=self._first_feasible is not None,
            first_feasible_eval=self._first_feasible,
            best_f=(float(b["f"]) if b is not None and b["feasible"] else None),
            best_total_violation=(float(b["T"]) if b is not None else None),
            best_x=(b["x"].tolist() if b is not None else None),
            final_score=(float(b["s"]) if b is not None else -1.0),
            final_unit=float(self._unit_now),
            aocc=scoring.aocc(unit_curve, self.budget),
            feasible_fraction=(self._n_feasible / self.n_evals if self.n_evals else 0.0),
            wall_time=time.perf_counter() - self._t0,
            curve=[(int(e), float(u)) for e, u in self._curve],
            best_gap=(float((b["f"] - self.spec.f_ref) / self.spec.scale)
                      if b is not None and b["feasible"] else None),
            viol_freq=((self._viol_count / self.n_evals).round(4).tolist() if self.n_evals else []),
            best_violated=([int(j) for j in np.flatnonzero(b["v"] > FEAS_TOL)] if b is not None else []),
            last_improvement_eval=(int(self._curve[-1][0]) if self._curve else None),
            eval_90=self._eval_90(),
        )
        return r

    def _eval_90(self):
        if not self._curve:
            return None
        u0, uf = self._curve[0][1], self._curve[-1][1]
        thr = u0 + 0.9 * (uf - u0)
        return int(next(e for e, u in self._curve if u >= thr - 1e-12))

    # ------------------------------------------------------------------ internals
    def _dense_unit_curve(self) -> np.ndarray:
        c = np.zeros(self.n_evals)
        for (e, u), nxt in zip(self._curve, self._curve[1:] + [(self.n_evals + 1, None)]):
            c[e - 1:nxt[0] - 1] = u
        return c

    def _make_func(self):
        f = _Func()
        f._call = self._call
        f.bounds = SimpleNamespace(lb=self.space.lb.copy(), ub=self.space.ub.copy())
        f.dim = self.space.dim
        f.variable_types = list(self.space.kinds)
        f.n_values = self.space.n_values.copy()
        f.integer_indices = self.space.integer_indices.copy()
        f.continuous_indices = self.space.continuous_indices.copy()
        f.n_constraints = self.problem.n_constraints if self.arm != "penalty" else None
        f.budget = self.budget
        return f

    def _check_time(self):
        if self.time_limit is not None and time.perf_counter() - self._t0 > self.time_limit:
            self._timed_out = True
            raise HarnessTimeout(f"time limit of {self.time_limit:.0f} s exceeded")

    def _call(self, x):
        if self._exhausted or self.n_evals >= self.budget:
            self._exhausted = True
            raise BudgetExhausted(f"budget of {self.budget} evaluations exhausted")
        self._check_time()
        single = np.ndim(x) == 1
        X = self.space.decode(x)          # validates shape / finiteness
        n_ok = min(len(X), self.remaining)
        F = np.empty(n_ok)
        G = np.empty((n_ok, self.problem.n_constraints))
        P = np.empty(n_ok)
        if n_ok > 1 and getattr(self.problem, "batch_ok", False):
            evs = self.problem.evaluate_batch(X[:n_ok])     # vectorised (GPU), same values
        else:
            evs = None
        for j in range(n_ok):
            ev = _sanitise(evs[j] if evs is not None else self.problem.evaluate(X[j]))
            self.n_evals += 1
            self._record(X[j], ev)
            # violations below the feasibility tolerance are returned as exactly 0, so that
            # "feasible <=> all(g == 0)" (as stated in the prompt) matches the harness criterion
            F[j], G[j], P[j] = ev.f, np.where(ev.v <= FEAS_TOL, 0.0, ev.v), ev.penalty
        if n_ok < len(X):                 # batch larger than the remaining budget
            self._exhausted = True
            raise BudgetExhausted(f"budget of {self.budget} evaluations exhausted "
                                  f"(batch of {len(X)}, {n_ok} evaluated)")
        if self.arm == "penalty":
            return float(P[0]) if single else P
        return (float(F[0]), G[0].copy()) if single else (F, G)

    def _record(self, x, ev):
        feasible = self.problem.is_feasible(ev)
        T = 0.0 if feasible else scoring.total_violation(ev.f, ev.v)
        self._viol_count += np.asarray(ev.v) > FEAS_TOL
        if feasible:
            self._n_feasible += 1
            if self._first_feasible is None:
                self._first_feasible = self.n_evals
        key = (0, ev.f) if feasible else (1, T)
        if key < self._best_key:
            self._best_key = key
            s = scoring.score(ev.f, feasible, T, self.spec)
            self._best_eval = {"f": ev.f, "T": T, "feasible": feasible, "x": np.array(x), "s": s,
                               "v": np.array(ev.v, float)}
            u = scoring.unit(s)
            if u != self._unit_now or not self._curve:
                self._unit_now = u
                self._curve.append((self.n_evals, u))
        if self.keep_trace:
            t = self.trace
            t["f"].append(ev.f)
            t["T"].append(T)
            t["n_violated"].append(int(np.sum(ev.v > FEAS_TOL)))
            t["feasible"].append(feasible)
            t["penalty"].append(ev.penalty)
            t["x"].append(np.asarray(x, np.float32))

    def trace_arrays(self) -> dict:
        t = self.trace
        return {
            "f": np.asarray(t["f"], float), "T": np.asarray(t["T"], float),
            "n_violated": np.asarray(t["n_violated"], np.int16),
            "feasible": np.asarray(t["feasible"], bool), "penalty": np.asarray(t["penalty"], float),
            "x": (np.stack(t["x"]) if t["x"] else np.zeros((0, self.space.dim), np.float32)),
        }