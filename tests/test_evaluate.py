"""Candidate evaluation, feedback and the LLaMEA integration (with a fake LLM)."""
import re

import pytest

from harness import config, prompts
from harness.evaluate import evaluate_candidate

RA = "harness.reference_algorithms:"
SMALL = ("toy/basic", "cec/RC15")


@pytest.fixture
def small_budgets(monkeypatch):
    monkeypatch.setattr(config, "CEC_BUDGET_PER_DIM", 40)
    monkeypatch.setattr(config, "LENS_BUDGET", 40)
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


def test_feedback_template_identical_across_arms(small_budgets):
    fp = evaluate_candidate(RA + "DEPenalty", "penalty", SMALL, (0,)).feedback
    ff = evaluate_candidate(RA + "DEFeasibilityRules", "free", SMALL, (0,)).feedback
    strip = lambda s: re.sub(r"[0-9.]+", "#", s)  # noqa: E731
    assert strip(fp) == strip(ff)


def test_broken_candidate_gets_zero_and_error(small_budgets):
    res = evaluate_candidate("class A:\n  def __init__(self, budget, dim): pass\n  def __call__(self, f): 1/0",
                             "free", SMALL, (0, 1), class_name="A")
    assert res.fitness == 0 and "ZeroDivisionError" in res.error and "failed" in res.feedback


def test_prompts_differ_only_where_intended():
    p, f, g = (prompts.task_prompt(a) for a in ("penalty", "free", "guided"))
    assert "f, g = func(x)" in f and "f, g = func(x)" not in p
    assert "Hints:" in g and "Hints:" not in f and "Hints:" not in p
    import textwrap
    assert g.replace(textwrap.dedent(prompts._HINTS) + "\n", "", 1) == f   # guided = free + hints, nothing else
    # penalty and free differ only in the interface paragraph
    assert p.replace(textwrap.dedent(prompts._INTERFACE["penalty"]), textwrap.dedent(prompts._INTERFACE["free"])) == f
    head, tail = prompts._COMMON_HEAD, prompts._COMMON_TAIL
    for t in (p, f, g):
        assert head.strip() in t and tail.strip() in t
    for t in (p, f, g):  # no problem identity in any prompt
        for leak in ("lens", "RC1", "CEC", "Double-Gauss", "gearbox"):
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


@pytest.mark.parametrize("arm", ["penalty", "free"])
def test_llamea_loop_with_fake_llm(arm, small_budgets, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # LLaMEA writes its logs to the working directory
    from harness.llamea_adapter import make_llamea
    fake = _FakeLLM(arm)
    es = make_llamea(arm, fake._llm, budget=3, n_parents=1, n_offspring=1, max_workers=1,
                     problem_ids=SMALL, seeds=(0,), log=False)
    best = es.run()
    assert best.fitness > 0, (best.error, best.feedback)
    assert "P1" in best.feedback
