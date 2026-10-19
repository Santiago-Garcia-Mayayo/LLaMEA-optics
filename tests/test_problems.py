"""Problem suite checks."""
import json
import os


import numpy as np

from problems.suite import CONTROL_SET, DATA, CECProblem, LensSuiteProblem  # noqa: E402

REF = {r["num"]: r for r in json.load(open(os.path.join(DATA, "reference_summary.json")))}


def test_cec_reference_points_feasible_and_near_fstar():
    for n in CONTROL_SET:
        p = CECProblem(n)
        x = np.array(REF[n]["x_best"])
        ev = p.evaluate(x)
        f = ev.f
        assert p.is_feasible(ev), n
        tol = 0.2 if n in (16, 26) else 1e-6  # RC16/RC26: reference DE did not reach f*
        assert (f - p.f_star) / abs(p.f_star) <= tol, (n, f, p.f_star)


def test_cec_seal_rounds_integers():
    p = CECProblem(12)
    x = p.seal(p.lb + 0.37 * (p.ub - p.lb))
    assert np.all(x[p.integer_indices] == np.rint(x[p.integer_indices]))


def test_lens_original_matches_lensgopt():
    p = LensSuiteProblem("original")
    t = p.template()
    assert abs(p.evaluate(t).penalty - p._p.obj.objective_theta(t)) < 1e-12


def test_lens_template_feasible_both_instances():
    for inst in ("original", "realistic_draft"):
        p = LensSuiteProblem(inst)
        assert p.is_feasible(p.evaluate(p.template())), inst


def test_lens_ray_loss_is_a_graded_constraint():
    """f is undefined exactly when rays are lost, and the free arm sees how many (graded),
    like the penalty arm does (10 x rays lost)."""
    p = LensSuiteProblem("original")
    assert p.constraint_names[0] == "rays_lost"
    rng = np.random.default_rng(0)
    for _ in range(40):
        ev = p.evaluate(p.lb + rng.random(p.dim) * (p.ub - p.lb))
        assert (ev.v[0] > 0) == (not np.isfinite(ev.f))
        assert 0 <= ev.v[0] <= 1 / p._p.scales[0] + 1e-12   # raw fraction in [0, 1], then scaled


def test_feedback_lens_descriptor_matches_problem():
    from harness.evaluate import _LENS_DESCR
    for inst in ("original", "realistic_draft"):
        p = LensSuiteProblem(inst)
        assert _LENS_DESCR == {"dim": p.dim, "n_int": len(p.integer_indices), "m": p.n_constraints}
