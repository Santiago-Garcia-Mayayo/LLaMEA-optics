"""Candidate evaluation, feedback, prompts, the LLaMEA integration and run driver (fake LLM)."""
import json
import re
import textwrap

import pytest

from harness import config, prompts
from harness.evaluate import evaluate_candidate

RA = "harness.reference_algorithms:"
SMALL = ("toy/basic", "cec/RC15")


@pytest.fixture
def small_budgets(monkeypatch):
    monkeypatch.setattr(config, "BUDGET_PER_DIM", {"train": 10, "test": 10})
    monkeypatch.setattr(config, "TIME_MIN", 30.0)


def test_fitness_and_feedback(small_budgets):
    res = evaluate_candidate(RA + "DEFeasibilityRules", "free", SMALL, (0, 1))
    assert 0 < res.fitness <= 1
    assert set(res.per_problem) == set(SMALL)
    assert len(res.runs) == 4 and res.error == ""
    # problems are anonymised in the feedback
    assert "P1" in res.feedback and "P2" in res.feedback
    for leak in ("toy", "RC15", "cec", "lens", "speed reducer"):
        assert leak not in res.feedback.lower()


def test_feedback_identical_for_penalty_and_free(small_budgets):
    fp = evaluate_candidate(RA + "DEPenalty", "penalty", SMALL, (0,)).feedback
    ff = evaluate_candidate(RA + "DEFeasibilityRules", "free", SMALL, (0,)).feedback
    strip = lambda s: re.sub(r"[0-9.]+", "#", s)  # noqa: E731
    assert strip(fp) == strip(ff)


def test_informed_feedback_is_free_feedback_plus_behaviour(small_budgets):
    ff = evaluate_candidate(RA + "DEFeasibilityRules", "free", SMALL, (0,)).feedback
    fi = evaluate_candidate(RA + "DEFeasibilityRules", "informed", SMALL, (0,)).feedback
    extra = [l for l in fi.splitlines() if l not in ff.splitlines()]
    assert [l for l in fi.splitlines() if l in ff.splitlines()] == ff.splitlines()   # superset
    assert all(l.startswith("    ") for l in extra)                  # only the indented behaviour block
    assert "feasible evaluations" in fi and "first feasible point" in fi and "last improvement" in fi
    for leak in ("toy", "rc15", "cec", "lens"):
        assert leak not in fi.lower()


def test_informed_v2_feedback_is_free_feedback_plus_diagnosis(small_budgets):
    ff = evaluate_candidate(RA + "DEFeasibilityRules", "free", SMALL, (0,)).feedback
    fd = evaluate_candidate(RA + "DEFeasibilityRules", "informed_v2", SMALL, (0,)).feedback
    assert [l for l in fd.splitlines() if l in ff.splitlines()] == ff.splitlines()   # superset
    extra = [l for l in fd.splitlines() if l not in ff.splitlines()]
    assert all(l.startswith("    ") for l in extra) and "Diagnosis" in fd
    assert "Behaviour of the algorithm" not in fd                     # not the v1 statistics block
    for leak in ("toy", "rc15", "cec", "lens"):
        assert leak not in fd.lower()


def _summary(**kw):
    s = {"n_runs": 5, "feasible_runs": 5, "feasible_fraction": 0.8, "first_feasible_pct_median": 1.0,
         "most_violated": [(2, 0.1)], "last_improvement_pct_median": 90.0, "aocc_mean": 0.9,
         "aocc_median": 0.9, "aocc_min": 0.88, "n_never_feasible": 0, "worst_run_first_feasible_pct": 1.0}
    s.update(kw)
    return s


def test_diagnosis_rules():
    from harness.evaluate import diagnosis_lines
    text = lambda **kw: " ".join(diagnosis_lines(_summary(**kw)))   # noqa: E731
    assert "No constraint-related weakness" in text()
    assert "slow to reach feasibility" in text(first_feasible_pct_median=40.0)
    assert "infeasible points" in text(feasible_fraction=0.1)
    assert "Constraint 2 is violated in 60%" in text(most_violated=[(2, 0.6)])
    assert "converges prematurely" in text(last_improvement_pct_median=5.0)
    t = text(aocc_min=0.55, aocc_median=0.95, worst_run_first_feasible_pct=45.0)
    assert "unreliable" in t and "45% of the budget" in t
    assert "never finds a feasible point in 2 of 5" in text(n_never_feasible=2, aocc_min=0.2)
    assert "No constraint-related weakness" not in text(feasible_fraction=0.1)
    parent = _summary(aocc_mean=0.7, feasible_fraction=0.4, first_feasible_pct_median=15.0)
    cmp_ = diagnosis_lines(_summary(), parent)[-1]
    assert "Compared with its parent" in cmp_ and "(parent 0.7000)" in cmp_ and "(parent 15% of the budget)" in cmp_


def test_broken_candidate_gets_zero_and_error(small_budgets):
    res = evaluate_candidate("class A:\n  def __init__(self, budget, dim): pass\n  def __call__(self, f): 1/0",
                             "free", SMALL, (0, 1), class_name="A")
    assert res.fitness == 0 and "ZeroDivisionError" in res.error and "failed" in res.feedback


def test_prompts_differ_only_where_intended():
    p, f, i = (prompts.task_prompt(a) for a in ("penalty", "free", "informed"))
    assert "f, g = func(x)" in f and "f, g = func(x)" not in p
    assert i == f and prompts.example_prompt("informed") == prompts.example_prompt("free")
    assert prompts.task_prompt("informed_v2") == f and prompts.example_prompt("informed_v2") == prompts.example_prompt("free")
    # penalty and free differ only in the interface paragraph
    assert p.replace(textwrap.dedent(prompts._INTERFACE["penalty"]), textwrap.dedent(prompts._INTERFACE["free"])) == f
    head, tail = prompts._COMMON_HEAD, prompts._COMMON_TAIL
    for t in (p, f, i):
        assert head.strip() in t and tail.strip() in t
        assert "Hints" not in t
        for leak in ("lens", "RC1", "CEC", "Double-Gauss", "gearbox"):   # no problem identity
            assert leak not in t


class _FakeLLM:
    """Stands in for an LLM: returns valid code for the arm, slightly different each call."""

    def __init__(self, arm):
        from llamea.llm import LLM
        self._llm = type("F", (LLM,), {"query": lambda s, m: self._reply()})("", "fake", None)
        self.arm = arm
        self.n = 0

    def _reply(self):
        self.n += 1
        call = "f, g = func(x)\n            ok = np.all(g == 0)" if self.arm != "penalty" else "f = func(x)\n            ok = True"
        return f"""# Description: random search variant {self.n}
# Code:
```python
import numpy as np

class RS{self.n}:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        best = np.inf
        for _ in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            {call}
            if ok and f < best:
                best = f
        return best
```
"""


@pytest.mark.parametrize("arm", ["penalty", "informed", "informed_v2"])
def test_llamea_loop_with_fake_llm(arm, small_budgets, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # LLaMEA writes its logs to the working directory
    from harness.llamea_adapter import make_llamea
    fake = _FakeLLM(arm)
    es = make_llamea(arm, fake._llm, "toy/basic", budget=4, n_parents=2, n_offspring=2, max_workers=1,
                     seeds=(0,), log=False, log_dir=str(tmp_path / "cand"))
    best = es.run()
    assert best.fitness > 0, (best.error, best.feedback)
    assert "P1" in best.feedback and "P2" not in best.feedback        # one problem per run
    assert ("Behaviour of the algorithm" in best.feedback) == (arm == "informed")
    assert len(list((tmp_path / "cand").glob("*.json"))) == 4
    assert es.feature_guided_mutation == (arm == "informed")          # SAGE only in informed
    if arm == "informed":
        feats = best.get_metadata("ast_features")
        assert "time_to_first_feasible_point" in feats and "violation_rate_of_constraint_0" in feats
        assert len(feats) > 10                                         # AST features kept too
    if arm == "informed_v2":
        fbs = [json.load(open(f))["feedback"] for f in (tmp_path / "cand").glob("*.json")]
        assert all("Diagnosis" in fb for fb in fbs)
        assert any("Compared with its parent" in fb for fb in fbs)    # offspring see their parent


def test_sage_guidance_uses_behaviour_features(small_budgets):
    """SAGE's own archive model accepts the merged features and can return a behaviour hint."""
    import numpy as np
    from llamea.feature_guidance import compute_feature_guidance
    from llamea.solution import Solution
    from harness.llamea_adapter import sage_features
    code = "class A:\n    def __init__(self, budget, dim):\n        pass\n"
    sols = []
    for k, t in enumerate(np.linspace(0.05, 0.9, 8)):   # earlier feasibility <-> higher fitness
        s = Solution(code=code, name="A")
        per = {"p": {"feasible_fraction": 0.5, "first_feasible_pct_median": 100 * t,
                     "eval90_pct_median": 50.0, "last_improvement_pct_median": 80.0,
                     "viol_freq_mean": [0.3, 0.1]}}
        s.set_scores(1.0 - t)
        s.add_metadata("ast_features", sage_features(code, per))
        sols.append(s)
    g = compute_feature_guidance(sols, minimization=False)
    assert g is not None and g.feature_name == "time_to_first_feasible_point" and g.action == "decrease"
    assert "time to first feasible point" in g.message


def test_defaults_are_the_protocol():
    import inspect
    from harness.llamea_adapter import make_llamea
    sig = inspect.signature(make_llamea).parameters
    assert (sig["n_parents"].default, sig["n_offspring"].default, sig["budget"].default,
            sig["elitism"].default) == (5, 5, 100, True)


# ---------------------------------------------------------------- run driver (no network)
class _Resp:
    def __init__(self, text):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason = "end_turn"


def test_run_llamea_single_end_to_end(small_budgets, tmp_path, monkeypatch):
    """experiments.run_llamea.single with LLaMEA's Anthropic_LLM on a fake client: writes best.json and candidates."""
    import llamea.llm
    from experiments import run_llamea
    fake = _FakeLLM("free")

    class FakeClient:
        def __init__(self, **kw):
            pass

        class messages:  # noqa: N801
            @staticmethod
            def create(**kw):
                return _Resp(fake._reply())

    monkeypatch.setattr(llamea.llm.anthropic, "Anthropic", FakeClient)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.chdir(tmp_path)
    run_llamea.single("toy/basic", "free", str(tmp_path / "out"), "t", "claude-sonnet-5-5",
                      budget=4, n_parents=2, n_offspring=2, elitism=True, temperature=0.8,
                      max_workers=1)   # in-process, so the fake client is used
    d = tmp_path / "out" / "t" / "free" / "toy_basic"
    best = json.loads((d / "best.json").read_text())
    assert best["fitness"] > 0 and "class RS" in best["code"]
    assert len(list((d / "candidates").glob("*.json"))) == 4