"""Harness core: interfaces, budget, best-feasible tracking, scores."""
import numpy as np
import pytest

from harness import BudgetExhausted, Harness, HarnessTimeout
from harness import scoring
from problems.toy import ToyProblem

OPT = np.array([2 * (1 - (-5)) / 10 - 1, 0.0, 2, 2])   # encoded optimum: x0=1, x1=0, k=3 (index 2), c=2


def test_optimum_encoding():
    h = Harness(ToyProblem(), "free", 10)
    f, g = h.func(OPT)
    assert f == pytest.approx(1.0) and np.all(g == 0)


def test_arms_see_the_same_evaluation():
    rng = np.random.default_rng(3)
    Z = rng.uniform([-1, -1, -0.5, -0.5], [1, 1, 4.5, 2.5], size=(50, 4))
    hp, hf = Harness(ToyProblem(), "penalty", 100), Harness(ToyProblem(), "free", 100)
    y = hp.func(Z)
    f, g = hf.func(Z)
    assert np.allclose(y, f + 100 * g.sum(axis=1))
    assert hp.result().best_f == hf.result().best_f       # same criterion in both arms
    assert hp.result().aocc == hf.result().aocc


def test_return_types_single_and_batch():
    hp, hf = Harness(ToyProblem(), "penalty", 10), Harness(ToyProblem(), "free", 10)
    assert isinstance(hp.func(OPT), float)
    f, g = hf.func(OPT)
    assert isinstance(f, float) and g.shape == (2,)
    y = hp.func(np.stack([OPT, OPT]))
    assert y.shape == (2,)
    f, g = hf.func(np.stack([OPT, OPT, OPT]))
    assert f.shape == (3,) and g.shape == (3, 2)


def test_penalty_arm_hides_constraint_count():
    assert Harness(ToyProblem(), "penalty", 5).func.n_constraints is None
    assert Harness(ToyProblem(), "free", 5).func.n_constraints == 2


def test_budget_is_hard():
    p = ToyProblem()
    h = Harness(p, "free", 5)
    for _ in range(5):
        h.func(OPT)
    with pytest.raises(BudgetExhausted):
        h.func(OPT)
    with pytest.raises(BudgetExhausted):
        h.func(OPT)
    assert p.n_calls == 5 and h.n_evals == 5


def test_batch_crossing_budget_evaluates_only_the_rest():
    p = ToyProblem()
    h = Harness(p, "penalty", 7)
    h.func(np.stack([OPT] * 4))
    with pytest.raises(BudgetExhausted):
        h.func(np.stack([OPT] * 4))
    assert p.n_calls == 7 and h.n_evals == 7


def test_budget_not_swallowed_by_except_exception():
    p = ToyProblem()
    h = Harness(p, "free", 3)
    with pytest.raises(BudgetExhausted):
        for _ in range(100):
            try:
                h.func(OPT)
            except Exception:
                pass
    assert p.n_calls == 3


def test_best_feasible_tracking_matches_brute_force():
    rng = np.random.default_rng(7)
    Z = rng.uniform([-1, -1, -0.5, -0.5], [1, 1, 4.5, 2.5], size=(300, 4))
    h = Harness(ToyProblem(), "free", 300)
    F, G = h.func(Z)
    feas = np.all(G == 0, axis=1)
    r = h.result()
    assert r.found_feasible == feas.any()
    assert r.best_f == pytest.approx(F[feas].min())
    assert r.first_feasible_eval == int(np.argmax(feas)) + 1
    assert r.feasible_fraction == pytest.approx(feas.mean())


def test_infeasible_only_run_tracks_least_violation():
    h = Harness(ToyProblem(), "free", 3)
    # x0 = -5 (g1 = 6), then x0 = -1 (g1 = 2), then x0 = -3 (g1 = 4): all infeasible
    for x0 in (-5.0, -1.0, -3.0):
        h.func(np.array([2 * (x0 + 5) / 10 - 1, 0, 2, 2]))
    r = h.result()
    assert not r.found_feasible and r.best_f is None
    assert r.best_total_violation == pytest.approx(2.0)
    assert r.final_score == pytest.approx(-2 / 3)
    assert 0 <= r.final_unit < 0.5


def test_aocc_monotone_curve_and_bounds():
    h = Harness(ToyProblem(), "free", 100)
    rng = np.random.default_rng(0)
    for _ in range(60):  # algorithm stops early: the last value is kept until the budget
        h.func(rng.uniform([-1, -1, -0.5, -0.5], [1, 1, 4.5, 2.5]))
    r = h.result()
    units = [u for _, u in r.curve]
    assert all(b >= a for a, b in zip(units, units[1:]))
    assert 0 <= r.aocc <= 1
    dense = h._dense_unit_curve()
    assert len(dense) == 60
    assert r.aocc == pytest.approx((dense.sum() + 40 * dense[-1]) / 100)


def test_aocc_of_immediate_optimum_is_one():
    h = Harness(ToyProblem(), "free", 10)
    h.func(OPT)
    assert h.result().aocc == pytest.approx(1.0)


def test_score_scale():
    spec = ToyProblem().score_spec  # f_ref = 1, scale 1, gaps 1e-6..1
    assert scoring.score(1.0, True, 0, spec) == 1.0
    assert scoring.score(2.0, True, 0, spec) == 0.0
    assert scoring.score(1.0 + 1e-3, True, 0, spec) == pytest.approx(0.5)
    assert scoring.score(1.0 + 1e-2, True, 0, spec) == pytest.approx(1 / 3)
    assert scoring.score(50.0, False, np.inf, spec) == -1.0
    # any feasible point beats any infeasible one in unit score
    assert scoring.unit(scoring.score(1e9, True, 0, spec)) >= 0.5 > scoring.unit(scoring.score(1, False, 1e-12, spec))


def test_sub_tolerance_violation_returned_as_zero():
    class Tiny(ToyProblem):
        def evaluate(self, x):
            ev = super().evaluate(x)
            ev.v = ev.v + 1e-15
            return ev
    h = Harness(Tiny(), "free", 2)
    f, g = h.func(OPT)
    assert np.all(g == 0) and h.result().found_feasible


def test_nan_objective_is_worst_not_best():
    class NaNy(ToyProblem):
        def evaluate(self, x):
            ev = super().evaluate(x)
            if self.n_calls == 1:
                ev.f, ev.v = np.nan, np.array([np.nan, 0.0])
            return ev
    h = Harness(NaNy(), "free", 5)
    h.func(OPT)                     # NaN evaluation
    h.func(OPT)                     # proper optimum
    r = h.result()
    assert r.found_feasible and r.best_f == pytest.approx(1.0)


def test_time_limit_raises():
    h = Harness(ToyProblem(), "free", 10**6, time_limit=0.0)
    with pytest.raises(HarnessTimeout):
        h.func(OPT)


def test_bad_arm_and_budget():
    with pytest.raises(ValueError):
        Harness(ToyProblem(), "both", 10)
    with pytest.raises(ValueError):
        Harness(ToyProblem(), "free", 0)
