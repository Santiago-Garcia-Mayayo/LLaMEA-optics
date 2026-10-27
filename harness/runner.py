"""Run an algorithm (LLM-generated code or a library class) on one problem, isolated.

One job = one algorithm x one problem x several seeds, executed in a fresh Python
interpreter (`python -m harness._worker`): no shared state with the caller or with other
candidates, no dependence on the caller's __main__ (works from scripts without a
`__main__` guard and from inside LLaMEA's worker processes). Results are written after
every seed, so a crash or hard kill only loses the seed in progress.

Time is limited at three levels:
  1. inside func: HarnessTimeout once the run exceeds its limit (cross-platform);
  2. a few seconds later, for code that loops without calling func: SIGALRM on POSIX,
     a watchdog thread that interrupts the main thread on Windows;
  3. the parent kills the process (tree) if the whole job overruns (always).

The algorithm interface is LLaMEA's: a class with __init__(self, budget, dim) and
__call__(self, func). Its return value is ignored: the harness keeps the record.
"""
from __future__ import annotations

import importlib
import json
import os
import random
import subprocess
import sys
import tempfile
import traceback

from . import config
from .core import BudgetExhausted, Harness, HarnessTimeout, RunResult

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_USE_SIGALRM = hasattr(__import__("signal"), "setitimer")   # False on Windows


def _kill_tree(proc):
    """Kill the worker and its children. On Windows a venv's python.exe is a launcher that
    starts the real interpreter as a child, so killing only proc.pid would leave it running."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.kill()


def _load_class(algorithm: str, class_name: str | None):
    """`algorithm` is either Python source code, or 'package.module:ClassName'."""
    if class_name is None and ":" in algorithm and "\n" not in algorithm:
        mod, cls = algorithm.split(":")
        return getattr(importlib.import_module(mod), cls)
    _install_seeded_default_rng()   # also covers `from numpy.random import default_rng`
    ns = {"__name__": "candidate"}
    exec(compile(algorithm, "<candidate>", "exec"), ns)
    if class_name is None:
        raise ValueError("class_name is required when passing source code")
    if class_name not in ns:
        raise NameError(f"class '{class_name}' is not defined in the generated code")
    return ns[class_name]


# Generated code often creates its own generator with np.random.default_rng() and no seed,
# which draws from OS entropy: the same algorithm on the same seed gave different results.
# The worker replaces np.random.default_rng by a version that, when called without a seed,
# seeds the new generator from a SeedSequence of the run's seed (one child per call, in call
# order). Explicit seeds are passed through unchanged, and the global np.random / random
# streams are not touched, so code that never calls default_rng() without a seed behaves
# exactly as before.
_RNG = {"ss": None}
_ORIG_DEFAULT_RNG = None


def _install_seeded_default_rng():
    global _ORIG_DEFAULT_RNG
    import numpy as np
    if _ORIG_DEFAULT_RNG is not None:
        return
    orig = _ORIG_DEFAULT_RNG = np.random.default_rng
    if _RNG["ss"] is None:   # calls before the first seed (e.g. at class definition)
        _RNG["ss"] = np.random.SeedSequence(0)

    def default_rng(seed=None):
        if seed is None:
            seed = _RNG["ss"].spawn(1)[0]
        return orig(seed)

    default_rng.__doc__ = orig.__doc__
    np.random.default_rng = default_rng


def _seed_everything(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % (2**32))
    _install_seeded_default_rng()
    _RNG["ss"] = np.random.SeedSequence(seed % (2**32))


def run_one(cls, problem, arm: str, seed: int, budget: int, time_limit: float | None,
            init_kwargs: dict | None = None, keep_trace: bool = False):
    """Run in the *current* process (used inside workers, and for trusted baselines).
    Returns (RunResult, Harness)."""
    _seed_everything(seed)
    h = Harness(problem, arm, budget, seed=seed, time_limit=time_limit, keep_trace=keep_trace)
    status, err = None, ""
    use_alarm = time_limit is not None and _USE_SIGALRM
    watchdog = None
    if time_limit is not None and not use_alarm:   # Windows: same role as SIGALRM
        import _thread
        import threading
        watchdog = threading.Timer(time_limit + 5.0, _thread.interrupt_main)
        watchdog.daemon = True
        watchdog.start()
    if use_alarm:
        import signal

        def _alarm(signum, frame):
            raise HarnessTimeout("time limit exceeded (no func calls)")
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, time_limit + 5.0)
    try:
        h.restart_clock()
        algo = cls(budget=budget, dim=problem.dim, **(init_kwargs or {}))
        algo(h.func)
    except BudgetExhausted:
        status = "budget_exhausted"
    except HarnessTimeout as e:
        status, err = "timeout", str(e)
    except KeyboardInterrupt:
        if watchdog is None:
            raise
        status, err = "timeout", "time limit exceeded (no func calls)"
    except Exception as e:  # error in the algorithm
        status = "error"
        err = f"{type(e).__name__}: {e}\n" + "".join(traceback.format_exc(limit=8))
    finally:
        if watchdog is not None:
            watchdog.cancel()
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    return h.result(status=status, error=err), h


def _save_trace(trace_dir, res, h):
    import numpy as np
    os.makedirs(trace_dir, exist_ok=True)
    name = f"{res.problem.replace('/', '_')}__{res.arm}__s{res.seed}.npz"
    np.savez_compressed(os.path.join(trace_dir, name), **h.trace_arrays())


# Failures of the machine, not of the algorithm (thread/process limit reached, e.g. when other
# programs use many threads): such runs are retried instead of scoring 0.
RESOURCE_ERRORS = ("pthread_create", "Resource temporarily unavailable", "BlockingIOError",
                   "Thread tf_", "Cannot allocate memory", "[Errno 11]", "[Errno 12]")
RESOURCE_RETRIES = 4
RESOURCE_WAIT_S = (5, 15, 30, 60)   # seconds, x (1..2) jitter; fits within the candidate timeout


def _is_resource_failure(r: RunResult) -> bool:
    return r.status in ("error", "crashed") and any(p in (r.error or "") for p in RESOURCE_ERRORS)


def run_job(algorithm: str, problem_id: str, arm: str, seeds, **kw) -> list[RunResult]:
    """Run `algorithm` on one problem for every seed, in a fresh, isolated Python process
    (see _run_job_once). Seeds that fail because the machine ran out of threads/processes
    are run again, up to RESOURCE_RETRIES times with growing waits; if they still fail they
    are returned as failed, with the resource error as message."""
    import random
    import time
    seeds = [int(s) for s in seeds]
    try:
        results = _run_job_once(algorithm, problem_id, arm, seeds, **kw)
    except OSError as e:   # the worker process itself could not be started (fork failed)
        results = [RunResult(problem=problem_id, arm=arm, seed=s, budget=0, status="crashed",
                             error=f"BlockingIOError: could not start worker: {e}") for s in seeds]
    for attempt in range(RESOURCE_RETRIES):
        redo = [r.seed for r in results if _is_resource_failure(r)]
        if not redo:
            break
        time.sleep(RESOURCE_WAIT_S[attempt] * (1 + random.random()))
        try:
            again = {r.seed: r for r in _run_job_once(algorithm, problem_id, arm, redo, **kw)}
        except OSError:
            continue
        results = [again.get(r.seed, r) for r in results]
    return results


def _run_job_once(algorithm: str, problem_id: str, arm: str, seeds, *, class_name: str | None = None,
                  budget: int | None = None, time_limit: float | None = None, init_kwargs: dict | None = None,
                  trace_dir: str | None = None, phase: str = "train") -> list[RunResult]:
    """Run `algorithm` on one problem for every seed, in a fresh, isolated Python process.
    Unless given, the budget is config.budget_for(problem, phase) (phase "train" or "test")."""
    from problems import get_problem
    seeds = [int(s) for s in seeds]
    if budget is None or time_limit is None:
        p = get_problem(problem_id) if not problem_id.startswith("lens/") else _LensStub(problem_id)
        budget = budget or config.budget_for(p, phase)
        time_limit = time_limit or config.time_limit_for(p, budget)
    job = dict(algorithm=algorithm, class_name=class_name, problem=problem_id, arm=arm, seeds=seeds,
               budget=int(budget), time_limit=float(time_limit), init_kwargs=init_kwargs,
               trace_dir=os.path.abspath(trace_dir) if trace_dir else None)
    deadline_s = config.STARTUP_GRACE_S + len(seeds) * (time_limit + 15.0)
    results, fatal, hard_timeout, stderr_tail, exitcode = [], None, False, "", None
    with tempfile.TemporaryDirectory(prefix="harness_job_", ignore_cleanup_errors=True) as td:
        job_path, out_path, err_path = (os.path.join(td, n) for n in ("job.json", "out.jsonl", "err.txt"))
        with open(job_path, "w") as fh:
            json.dump(job, fh)
        env = os.environ.copy()
        env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        with open(err_path, "wb") as ferr:
            proc = subprocess.Popen([sys.executable, "-m", "harness._worker", job_path, out_path],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=ferr,
                                    env=env, cwd=td)
            try:
                proc.wait(timeout=deadline_s)
            except subprocess.TimeoutExpired:
                hard_timeout = True
                _kill_tree(proc)
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    pass
            exitcode = proc.returncode
        if os.path.exists(out_path):
            with open(out_path) as fh:
                for line in fh:
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:   # partial line written when killed
                        continue
                    if msg["kind"] == "result":
                        results.append(RunResult(**msg["payload"]))
                    elif msg["kind"] == "fatal":
                        fatal = msg["payload"]
        with open(err_path, "rb") as fh:
            stderr_tail = fh.read()[-3000:].decode(errors="replace")
    done = {r.seed for r in results}
    for s in seeds:
        if s not in done:
            status = "error" if fatal else ("timeout" if hard_timeout else "crashed")
            if fatal:
                msg = fatal
            elif hard_timeout:
                msg = "job exceeded its wall-clock limit and was killed"
            else:
                msg = f"worker process died (exit code {exitcode}). stderr tail:\n{stderr_tail}"
            results.append(RunResult(problem=problem_id, arm=arm, seed=s, budget=int(budget),
                                     status=status, error=msg))
    return sorted(results, key=lambda r: seeds.index(r.seed))


class _LensStub:
    """Avoids building the (JAX) lens problem in the parent just to read its budget."""
    def __init__(self, pid):
        self.pid = pid
        self.dim = 24