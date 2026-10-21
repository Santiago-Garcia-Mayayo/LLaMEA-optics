"""Isolated execution of (generated) code: errors, runaway loops, adversarial code."""
import textwrap
import time



from harness.runner import run_job

RA = "harness.reference_algorithms:"


def test_reference_algorithms_run_in_all_arms():
    for arm, algo in [("penalty", "RandomSearchPenalty"), ("free", "RandomSearchFree"),
                      ("informed", "RandomSearchFree"), ("penalty", "DEPenalty"), ("free", "DEFeasibilityRules")]:
        rs = run_job(RA + algo, "toy/basic", arm, [0, 1], budget=300, time_limit=30)
        assert all(r.status in ("ok", "budget_exhausted") for r in rs), (arm, algo, [r.error for r in rs])
        assert all(r.n_evals == 300 and r.found_feasible for r in rs)


def test_seeding_is_reproducible():
    a = run_job(RA + "DEFeasibilityRules", "toy/basic", "free", [5], budget=200, time_limit=30)[0]
    b = run_job(RA + "DEFeasibilityRules", "toy/basic", "free", [5], budget=200, time_limit=30)[0]
    c = run_job(RA + "DEFeasibilityRules", "toy/basic", "free", [6], budget=200, time_limit=30)[0]
    assert a.best_f == b.best_f and a.aocc == b.aocc
    assert (a.best_f, a.aocc) != (c.best_f, c.aocc)


def test_source_code_candidate():
    code = textwrap.dedent('''
        import numpy as np
        class MySearch:
            def __init__(self, budget, dim):
                self.budget, self.dim = budget, dim
            def __call__(self, func):
                print("noise on stdout is suppressed")
                for _ in range(self.budget):
                    f, g = func(np.random.uniform(func.bounds.lb, func.bounds.ub))
    ''')
    r = run_job(code, "toy/basic", "free", [0], class_name="MySearch", budget=50, time_limit=30)[0]
    assert r.status in ("ok", "budget_exhausted") and r.n_evals == 50


def test_wrong_interface_use_is_an_error_not_a_crash():
    code = "class A:\n    def __init__(self, budget, dim): pass\n    def __call__(self, func):\n        y = func([0.0])\n"
    r = run_job(code, "toy/basic", "free", [0], class_name="A", budget=10, time_limit=30)[0]
    assert r.status == "error" and "ValueError" in r.error


def test_syntax_error_and_missing_class():
    rs = run_job("class A(:\n  pass", "toy/basic", "free", [0, 1], class_name="A", budget=10, time_limit=30)
    assert [r.status for r in rs] == ["error", "error"] and "SyntaxError" in rs[0].error
    r = run_job("x = 1", "toy/basic", "free", [0], class_name="A", budget=10, time_limit=30)[0]
    assert r.status == "error" and "NameError" in r.error


def test_exception_in_algorithm_keeps_partial_record():
    r = run_job(RA + "Crashes", "toy/basic", "free", [0], budget=10, time_limit=30)[0]
    assert r.status == "error" and "ZeroDivisionError" in r.error and r.n_evals == 1


def test_catching_everything_cannot_exceed_budget():
    r = run_job(RA + "CatchesEverything", "toy/basic", "free", [0], budget=25, time_limit=30)[0]
    assert r.status == "budget_exhausted" and r.n_evals == 25


def test_infinite_loop_without_func_calls_is_stopped():
    t0 = time.time()
    rs = run_job(RA + "SpinsForever", "toy/basic", "free", [0, 1], budget=10, time_limit=2)
    assert [r.status for r in rs] == ["timeout", "timeout"]
    assert rs[0].n_evals == 1                     # partial record survives
    assert time.time() - t0 < 60


def test_watchdog_without_sigalrm_stops_spinning_code(monkeypatch):
    """Windows path (no SIGALRM): a watchdog thread interrupts the main thread."""
    from harness import runner
    from harness.reference_algorithms import SpinsForever
    from problems import get_problem
    monkeypatch.setattr(runner, "_USE_SIGALRM", False)
    t0 = time.time()
    res, _ = runner.run_one(SpinsForever, get_problem("toy/basic"), "free", 0, 10, 1.0)
    assert res.status == "timeout" and res.n_evals == 1
    assert time.time() - t0 < 15


def test_lens_and_cec_in_isolated_processes():
    r = run_job(RA + "RandomSearchFree", "lens/original", "free", [0], budget=30, time_limit=120)[0]
    assert r.status == "ok" and r.n_evals == 30, r.error
    r = run_job(RA + "DEPenalty", "cec/RC15", "penalty", [0], budget=200, time_limit=60)[0]
    assert r.status == "budget_exhausted" and r.n_evals == 200, r.error


def test_bare_except_loop_is_killed_by_parent(monkeypatch):
    from harness import config
    monkeypatch.setattr(config, "STARTUP_GRACE_S", 5.0)
    t0 = time.time()
    rs = run_job(RA + "SwallowsEverything", "toy/basic", "free", [0, 1], budget=10, time_limit=1)
    assert [r.status for r in rs] == ["timeout", "timeout"]
    # either the alarm lands outside the bare try (partial record kept) or the parent kills
    # the process (record lost); in both cases the budget is never exceeded
    assert all(r.n_evals <= 10 for r in rs)
    assert time.time() - t0 < 60


def test_works_from_script_without_main_guard(tmp_path):
    """Regression: with multiprocessing 'spawn' every worker re-imported the caller's script."""
    import subprocess
    import sys
    script = tmp_path / "launch.py"
    script.write_text(
        "from harness.runner import run_job\n"
        "rs = run_job('harness.reference_algorithms:RandomSearchFree', 'toy/basic', 'free', [0, 1],"
        " budget=20, time_limit=30)\n"
        "print('STATUS', [r.status for r in rs], [r.n_evals for r in rs])\n")
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert "STATUS ['ok', 'ok'] [20, 20]" in out.stdout, out.stdout + out.stderr


def test_crash_message_includes_stderr():
    code = "import os\nclass A:\n  def __init__(self, budget, dim): pass\n  def __call__(self, f):\n    os._exit(3)"
    r = run_job(code, "toy/basic", "free", [0], class_name="A", budget=10, time_limit=30)[0]
    assert r.status == "crashed" and "exit code 3" in r.error


def test_few_cores_restores_affinity(monkeypatch):
    import os
    from problems import _threads
    if not hasattr(os, "sched_getaffinity"):
        return
    monkeypatch.setenv("LLAMEA_CORES_PER_PROCESS", "1")
    before = os.sched_getaffinity(0)
    with _threads.few_cores():
        assert len(os.sched_getaffinity(0)) == 1
    assert os.sched_getaffinity(0) == before
