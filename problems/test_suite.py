"""Quick checks: python -m pytest test_suite.py -q"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import CONTROL_SET, CECProblem, LensSuiteProblem  # noqa: E402

REF = {r["num"]: r for r in json.load(open(os.path.join(os.path.dirname(__file__), "reference_summary.json")))}


def test_cec_reference_points_feasible_and_near_fstar():
    for n in CONTROL_SET:
        p = CECProblem(n)
        x = np.array(REF[n]["x_best"])
        f, v = p.constrained(x)
        assert p.is_feasible(x), n
        tol = 0.2 if n in (16, 26) else 1e-6  # RC16/RC26: reference DE did not reach f*
        assert (f - p.f_star) / abs(p.f_star) <= tol, (n, f, p.f_star)


def test_cec_seal_rounds_integers():
    p = CECProblem(12)
    x = p.seal(p.lb + 0.37 * (p.ub - p.lb))
    assert np.all(x[p.integer_indices] == np.rint(x[p.integer_indices]))


def test_lens_original_matches_lensgopt():
    p = LensSuiteProblem("original")
    t = p.template()
    assert abs(p.penalty(t) - p._p.obj.objective_theta(t)) < 1e-12


def test_lens_template_feasible_both_instances():
    for inst in ("original", "realistic_draft"):
        p = LensSuiteProblem(inst)
        assert p.is_feasible(p.template()), inst
