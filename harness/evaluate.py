"""Evaluate one candidate algorithm on a set of problems x seeds.

fitness = mean over problems of (mean over seeds of AOCC), in [0, 1], higher is better.
Runs that end in error / crash / hard kill count as AOCC 0 (LLaMEA convention: broken code
gets the worst fitness and the error message as feedback).

Feedback to the LLM: the same basic template in every arm. On top of it, per problem:
  informed     a block of behaviour statistics (DESIGN.md §9), plus LLaMEA-SAGE (llamea_adapter);
  informed_v2  1-3 diagnoses derived from those statistics by fixed rules (diagnosis_lines),
               and the same statistics compared with the candidate's parent. Problems are anonymised
(P1, P2, ...) and described only by dimension, number of integer variables and constraints;
constraints by their index.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import config
from .runner import run_job

FAILED = ("error", "crashed")
TOP_CONSTRAINTS = 3


@dataclass
class CandidateResult:
    fitness: float
    per_problem: dict                   # pid -> summary dict
    runs: list = field(default_factory=list)   # RunResult dicts
    error: str = ""                     # first error message (for LLaMEA)
    feedback: str = ""


def _bad(r):
    return r.status in FAILED + ("timeout",)


def evaluate_candidate(algorithm: str, arm: str, problem_ids=config.PROBLEMS,
                       seeds=config.TRAIN_SEEDS, *, class_name: str | None = None,
                       n_workers: int | None = None, trace_dir: str | None = None,
                       algorithm_name: str | None = None, phase: str = "train",
                       parent_per_problem: dict | None = None) -> CandidateResult:
    """One isolated job per (problem, seed), run in parallel. parent_per_problem: the parent's
    per-problem summaries (informed_v2 only, compared in the feedback)."""
    problem_ids, seeds = list(problem_ids), [int(s) for s in seeds]
    jobs = [(pid, s) for pid in problem_ids for s in seeds]
    n_workers = n_workers or min(len(jobs), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {(pid, s): ex.submit(run_job, algorithm, pid, arm, [s], class_name=class_name,
                                    trace_dir=trace_dir, phase=phase) for pid, s in jobs}
        done = {k: f.result()[0] for k, f in futs.items()}
    results = {pid: [done[(pid, s)] for s in seeds] for pid in problem_ids}
    per, errors = {}, []
    for pid, rs in results.items():
        per[pid] = summarise(rs)
        errors += [r.error for r in rs if _bad(r) and r.error]
    fitness = float(np.mean([v["aocc_mean"] for v in per.values()]))
    descr = {pid: _describe(pid) for pid in problem_ids}
    fb = feedback_text(algorithm_name or class_name or "The algorithm", fitness, per, descr, errors,
                       informed=(arm == "informed"), diagnosed=(arm == "informed_v2"),
                       parent=parent_per_problem)
    return CandidateResult(fitness, per, [r.to_dict() for rs in results.values() for r in rs],
                           errors[0] if errors else "", fb)


def summarise(rs) -> dict:
    """Per-problem summary over seeds (failed runs count as AOCC 0, as never feasible)."""
    ok = [r for r in rs if not _bad(r)]
    a = [0.0 if _bad(r) else r.aocc for r in rs]
    s = {
        "aocc_mean": float(np.mean(a)), "aocc_std": float(np.std(a)),
        "feasible_runs": int(sum(r.found_feasible for r in ok)), "n_runs": len(rs),
        "final_unit_median": float(np.median([0.0 if _bad(r) else r.final_unit for r in rs])),
        "best_f": min((r.best_f for r in ok if r.best_f is not None), default=None),
        "statuses": [r.status for r in rs],
    }
    # behaviour statistics (shown to the informed arm only)
    s["feasible_fraction"] = float(np.mean([r.feasible_fraction for r in ok])) if ok else 0.0
    ff = [r.first_feasible_eval / r.budget for r in ok if r.first_feasible_eval is not None]
    s["first_feasible_pct_median"] = 100 * float(np.median(ff)) if ff else None
    gaps = [r.best_gap for r in ok if r.best_gap is not None]
    s["log10_gap_median"] = float(np.median(np.log10(np.maximum(gaps, 1e-300)))) if gaps else None
    s["eval90_pct_median"] = (100 * float(np.median([r.eval_90 / r.budget for r in ok if r.eval_90]))
                              if any(r.eval_90 for r in ok) else None)
    s["last_improvement_pct_median"] = (
        100 * float(np.median([r.last_improvement_eval / r.budget for r in ok if r.last_improvement_eval]))
        if any(r.last_improvement_eval for r in ok) else None)
    vf = [r.viol_freq for r in ok if r.viol_freq]
    if vf:
        m = np.mean(np.asarray(vf, float), axis=0)
        top = [int(j) for j in np.argsort(-m)[:TOP_CONSTRAINTS] if m[j] > 0]
        s["most_violated"] = [(j, float(m[j])) for j in top]
        s["viol_freq_mean"] = [float(v) for v in m]
    else:
        s["most_violated"] = []
        s["viol_freq_mean"] = []
    bv = sorted({j for r in ok if not r.found_feasible for j in r.best_violated})
    s["violated_at_best_infeasible"] = bv
    # reliability over seeds (used by the diagnoses of informed_v2)
    s["aocc_min"], s["aocc_median"] = float(np.min(a)), float(np.median(a))
    s["n_never_feasible"] = int(sum(not r.found_feasible for r in rs))
    worst = rs[int(np.argmin(a))]
    s["worst_run_first_feasible_pct"] = (100 * worst.first_feasible_eval / worst.budget
                                         if not _bad(worst) and worst.first_feasible_eval is not None else None)
    return s


_LENS_DESCR = {"dim": 24, "n_int": 6, "m": 24}


def _describe(pid):
    if pid.startswith("lens/"):
        return dict(_LENS_DESCR)
    from problems import get_problem
    p = get_problem(pid)
    return {"dim": p.dim, "n_int": len(p.integer_indices), "m": p.n_constraints}


def _pct(v):
    return "never" if v is None else f"{v:.0f}% of the budget"


def informed_lines(s) -> list:
    out = [f"      feasible evaluations: {100 * s['feasible_fraction']:.1f}% of all evaluations;",
           f"      first feasible point (median over runs): {_pct(s['first_feasible_pct_median'])};"]
    if s["log10_gap_median"] is not None:
        out.append(f"      final gap to the reference value (median, log10; lower is better): {s['log10_gap_median']:.2f};")
    out.append(f"      90% of the final improvement reached at: {_pct(s['eval90_pct_median'])}; "
               f"last improvement at: {_pct(s['last_improvement_pct_median'])};")
    if s["most_violated"]:
        out.append("      most often violated constraints: " + ", ".join(
            f"constraint {j} ({100 * f:.0f}% of evaluations)" for j, f in s["most_violated"]) + ";")
    if s["violated_at_best_infeasible"]:
        out.append("      in runs without a feasible point, the best point still violates constraints "
                   + ", ".join(str(j) for j in s["violated_at_best_infeasible"]) + ";")
    return out


# Thresholds of the diagnoses (informed_v2). Fixed before any informed_v2 run.
DIAG = {"slow_feasibility_pct": 10.0, "infeasible_waste": 0.30, "dominant_constraint": 0.30,
        "premature_pct": 20.0, "unreliable_gap": 0.10}


def diagnosis_lines(s, parent=None) -> list:
    """Behaviour statistics turned into short, actionable diagnoses, plus a comparison with the
    parent. Every rule is a fixed threshold on the statistics of summarise()."""
    out, n = [], s["n_runs"]
    if s.get("n_never_feasible"):
        out.append(f"It never finds a feasible point in {s['n_never_feasible']} of {n} runs: reaching the "
                   f"feasible region is the first thing to fix (only feasible points are scored).")
    ff = s.get("first_feasible_pct_median")
    if ff is not None and ff > DIAG["slow_feasibility_pct"]:
        out.append(f"It is slow to reach feasibility: the first feasible point comes at {ff:.0f}% of the "
                   f"budget (median over runs), and the score counts nothing before it.")
    if s["feasible_runs"] and s["feasible_fraction"] < DIAG["infeasible_waste"]:
        out.append(f"It spends {100 * (1 - s['feasible_fraction']):.0f}% of its evaluations on infeasible points.")
    if s.get("most_violated"):
        j, f = s["most_violated"][0]
        if f >= DIAG["dominant_constraint"]:
            out.append(f"Constraint {j} is violated in {100 * f:.0f}% of the evaluations, more than any other: "
                       f"handling it explicitly may help.")
    li = s.get("last_improvement_pct_median")
    if li is not None and li < DIAG["premature_pct"]:
        out.append(f"It stops improving after {li:.0f}% of the budget (median over runs): it converges "
                   f"prematurely and the rest of the budget is not used.")
    if s.get("aocc_median", 0) - s.get("aocc_min", 0) > DIAG["unreliable_gap"]:
        w = s.get("worst_run_first_feasible_pct")
        extra = (f"; in that run the first feasible point came at {w:.0f}% of the budget" if w is not None
                 else "; that run found no feasible point" if s.get("n_never_feasible") else "")
        out.append(f"It is unreliable across runs: the worst run scores {s['aocc_min']:.3f} against a median "
                   f"of {s['aocc_median']:.3f}{extra}.")
    if not out:
        out.append("No constraint-related weakness detected: feasible points are found early and the search "
                   "keeps improving; further gains must come from the optimisation itself.")
    lines = [f"      - {t}" for t in out]
    if parent:
        pf = parent.get("first_feasible_pct_median")
        lines.append(
            f"      Compared with its parent: AOCC {s['aocc_mean']:.4f} (parent {parent['aocc_mean']:.4f}); "
            f"feasible evaluations {100 * s['feasible_fraction']:.0f}% (parent {100 * parent.get('feasible_fraction', 0):.0f}%); "
            f"first feasible point at {_pct(ff)} (parent {_pct(pf)}).")
    return lines


def feedback_text(name, fitness, per, descr, errors, informed: bool = False, diagnosed: bool = False,
                  parent: dict | None = None) -> str:
    lines = [f"{name} obtained a mean anytime score (AOCC) of {fitness:.4f} "
             f"(range 0 to 1, higher is better; any feasible solution scores above 0.5 at that moment)."]
    lines.append("Per problem (problems are anonymised):")
    for k, (pid, s) in enumerate(per.items(), 1):
        d = descr[pid]
        lines.append(
            f"  P{k} (dim {d['dim']}, {d['n_int']} integer variables, {d['m']} constraints): "
            f"AOCC {s['aocc_mean']:.4f}, feasible solution found in {s['feasible_runs']}/{s['n_runs']} runs, "
            f"median final score {s['final_unit_median']:.3f}.")
        if informed:
            lines.append("    Behaviour of the algorithm on this problem:")
            lines += informed_lines(s)
        if diagnosed:
            lines.append("    Diagnosis of the algorithm's behaviour on this problem:")
            lines += diagnosis_lines(s, (parent or {}).get(pid))
    if errors:
        first = errors[0].strip().splitlines()
        lines.append("Some runs failed. First error: " + " | ".join(first[:1] + first[-3:]))
    return "\n".join(lines)