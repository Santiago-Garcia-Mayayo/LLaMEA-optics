"""Evaluate one candidate algorithm on a set of problems x seeds.

fitness = mean over problems of (mean over seeds of AOCC), in [0, 1], higher is better.
Runs that end in error / crash / hard kill count as AOCC 0 (LLaMEA convention: broken code
gets the worst fitness and the error message as feedback).

Feedback to the LLM is the SAME template in every arm; problems are anonymised (P1, P2, ...)
and described only by dimension, number of integer variables and number of constraints.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import config
from .runner import run_job

FAILED = ("error", "crashed")


@dataclass
class CandidateResult:
    fitness: float
    per_problem: dict                   # pid -> summary dict
    runs: list = field(default_factory=list)   # RunResult dicts
    error: str = ""                     # first error message (for LLaMEA)
    feedback: str = ""


def _run_aocc(r, penalise_failures=True):
    if penalise_failures and r.status in FAILED + ("timeout",):
        return 0.0
    return r.aocc


def evaluate_candidate(algorithm: str, arm: str, problem_ids=config.TRAIN_PROBLEMS,
                       seeds=config.TRAIN_SEEDS, *, class_name: str | None = None,
                       n_workers: int | None = None, trace_dir: str | None = None,
                       algorithm_name: str | None = None, phase: str = "train") -> CandidateResult:
    problem_ids = list(problem_ids)
    n_workers = n_workers or min(len(problem_ids), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {pid: ex.submit(run_job, algorithm, pid, arm, seeds, class_name=class_name,
                               trace_dir=trace_dir, phase=phase) for pid in problem_ids}
        results = {pid: f.result() for pid, f in futs.items()}
    per, errors = {}, []
    for pid, rs in results.items():
        a = [_run_aocc(r) for r in rs]
        per[pid] = {
            "aocc_mean": float(np.mean(a)), "aocc_std": float(np.std(a)),
            "feasible_runs": int(sum(r.found_feasible for r in rs)), "n_runs": len(rs),
            "final_unit_median": float(np.median([0.0 if r.status in FAILED + ("timeout",) else r.final_unit
                                                  for r in rs])),
            "best_f": min((r.best_f for r in rs if r.best_f is not None), default=None),
            "statuses": [r.status for r in rs],
        }
        errors += [r.error for r in rs if r.status in FAILED + ("timeout",) and r.error]
    fitness = float(np.mean([v["aocc_mean"] for v in per.values()]))
    descr = {pid: _describe(pid) for pid in problem_ids}
    fb = feedback_text(algorithm_name or class_name or "The algorithm", fitness, per, descr, errors)
    return CandidateResult(fitness, per, [r.to_dict() for rs in results.values() for r in rs],
                           errors[0] if errors else "", fb)


_LENS_DESCR = {"dim": 24, "n_int": 6, "m": 24}


def _describe(pid):
    if pid.startswith("lens/"):
        return dict(_LENS_DESCR)
    from problems import get_problem
    p = get_problem(pid)
    return {"dim": p.dim, "n_int": len(p.integer_indices), "m": p.n_constraints}


def feedback_text(name, fitness, per, descr, errors) -> str:
    lines = [f"{name} obtained a mean anytime score (AOCC) of {fitness:.4f} "
             f"(range 0 to 1, higher is better; any feasible solution scores above 0.5 at that moment)."]
    lines.append("Per problem (problems are anonymised):")
    for k, (pid, s) in enumerate(per.items(), 1):
        d = descr[pid]
        lines.append(
            f"  P{k} (dim {d['dim']}, {d['n_int']} integer variables, {d['m']} constraints): "
            f"AOCC {s['aocc_mean']:.4f}, feasible solution found in {s['feasible_runs']}/{s['n_runs']} runs, "
            f"median final score {s['final_unit_median']:.3f}.")
    if errors:
        first = errors[0].strip().splitlines()
        lines.append("Some runs failed. First error: " + " | ".join(first[:1] + first[-3:]))
    return "\n".join(lines)
