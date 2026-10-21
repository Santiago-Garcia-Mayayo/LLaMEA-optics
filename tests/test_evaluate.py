"""Candidate evaluation, feedback, prompts, the LLaMEA integration (fake LLM) and Claude_LLM."""
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


def test_broken_candidate_gets_zero_and_error(small_budgets):
    res = evaluate_candidate("class A:\n  def __init__(self, budget, dim): pass\n  def __call__(self, f): 1/0",
                             "free", SMALL, (0, 1), class_name="A")
    assert res.fitness == 0 and "ZeroDivisionError" in res.error and "failed" in res.feedback


def test_prompts_differ_only_where_intended():
    p, f, i = (prompts.task_prompt(a) for a in ("penalty", "free", "informed"))
    assert "f, g = func(x)" in f and "f, g = func(x)" not in p
    assert i == f and prompts.example_prompt("informed") == prompts.example_prompt("free")
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


@pytest.mark.parametrize("arm", ["penalty", "informed"])
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


def test_defaults_are_the_protocol():
    import inspect
    from harness.llamea_adapter import make_llamea
    sig = inspect.signature(make_llamea).parameters
    assert (sig["n_parents"].default, sig["n_offspring"].default, sig["budget"].default,
            sig["elitism"].default) == (5, 5, 100, True)


# ---------------------------------------------------------------- Claude_LLM (no network)
class _Resp:
    def __init__(self, text):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason = "end_turn"
        self.usage = type("U", (), {"input_tokens": 1000, "output_tokens": 500,
                                    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})()


def test_claude_llm_messages_and_usage(tmp_path, monkeypatch):
    from harness import claude_llm
    sent = {}

    class FakeClient:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kw):
                sent.update(kw)
                return _Resp("hello")

    monkeypatch.setattr(claude_llm.Claude_LLM, "_make_client", lambda self: FakeClient())
    log = tmp_path / "u.jsonl"
    llm = claude_llm.Claude_LLM(api_key="sk-test", model="claude-sonnet-5-5", usage_log=str(log))
    out = llm.query([{"role": "system", "content": "S"}, {"role": "user", "content": "a"},
                     {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"},
                     {"role": "user", "content": "d"}])
    assert out == "hello" and sent["system"] == "S" and "temperature" not in sent
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user"]
    assert sent["messages"][0]["content"] == "a\n\nb"
    rec = json.loads(log.read_text())
    assert rec["input_tokens"] == 1000 and rec["output_tokens"] == 500
    c = claude_llm.cost_of(str(log))
    assert c["calls"] == 1 and c["usd"] == pytest.approx((1000 * 2 + 500 * 10) / 1e6)
    import pickle
    llm2 = pickle.loads(pickle.dumps(llm))      # LLaMEA ships the LLM to worker processes
    assert llm2.query([{"role": "user", "content": "x"}]) == "hello"


def test_run_llamea_single_end_to_end(small_budgets, tmp_path, monkeypatch):
    """experiments.run_llamea.single with a fake Claude client: writes best.json, candidates, usage."""
    from experiments import run_llamea
    from harness import claude_llm
    fake = _FakeLLM("free")

    class FakeClient:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kw):
                return _Resp(fake._reply())

    monkeypatch.setattr(claude_llm.Claude_LLM, "_make_client", lambda self: FakeClient())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.chdir(tmp_path)
    run_llamea.single("toy/basic", "free", str(tmp_path / "out"), "t", "claude-sonnet-5-5",
                      budget=4, n_parents=2, n_offspring=2, elitism=True, temperature=None,
                      max_workers=1)   # in-process, so the fake client is used
    d = tmp_path / "out" / "t" / "free" / "toy_basic"
    best = json.loads((d / "best.json").read_text())
    assert best["fitness"] > 0 and "class RS" in best["code"]
    assert best["llm_cost"]["calls"] >= 4
    assert len(list((d / "candidates").glob("*.json"))) == 4
