"""Ablation: does an algorithm found with access to the constraints still work without them?

    python -m experiments.ablation_blind                          # RC12, all haiku5* repetitions
    python -m experiments.ablation_blind --problems cec/RC12 lens/original --workers 4

For every best algorithm of the free / informed / informed_v2 arms (results/llamea/<tag>/...),
two conditions on the test seeds and test budget, through the usual harness:

  normal   f, g = func(x) as during evolution (true objective, per-constraint violations)
  blind    f, g = func(x) where f is the PENALISED value the penalty arm sees and g is all
           zeros: the algorithm believes every point is feasible, so feasibility rules,
           repair or feasibility search have nothing to work with. Everything else (code,
           budget, seeds, bounds, variable types) is identical.

The harness scores both with the true objective and constraints (same AOCC as everywhere).
If blind drops to the level of the penalty arm, the advantage comes from seeing the
constraints; if it stays, it comes from other components of the algorithm.
Writes results/ablation_blind/runs.jsonl (resumable) and summary.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from harness import config
from harness.runner import run_job

ARMS_WITH_G = ("free", "informed", "informed_v2")
OUT = "results/ablation_blind"

_BLIND = '''

import numpy as _np_blind


class _BlindFunc:
    """Penalty-arm func dressed as a free-arm func: f = penalised value, g = zeros."""

    def __init__(self, func, m):
        self._func, self.n_constraints = func, m
        for k in ("bounds", "dim", "variable_types", "n_values", "integer_indices",
                  "continuous_indices", "budget"):
            if hasattr(func, k):
                setattr(self, k, getattr(func, k))

    def __call__(self, x):
        y = self._func(x)
        if _np_blind.ndim(x) == 1:
            return float(y), _np_blind.zeros(self.n_constraints)
        y = _np_blind.asarray(y, float)
        return y, _np_blind.zeros((len(y), self.n_constraints))


class BlindWrapper__:
    def __init__(self, budget, dim):
        self._alg = {cls}(budget, dim)

    def __call__(self, func):
        return self._alg(_BlindFunc(func, {m}))
'''


def _n_constraints(pid):
    if pid.startswith("lens/"):
        return 24
    from problems import get_problem
    return get_problem(pid).n_constraints


def blind_source(code: str, class_name: str, m: int) -> str:
    """The candidate's code, unchanged, followed by the wrapper (appended, so that
    `from __future__` lines and the candidate's own imports keep their place)."""
    return code + _BLIND.format(cls=class_name, m=m)


def _jobs(tags, problems, seeds, seen):
    jobs = []
    for tag in tags:
        for arm in ARMS_WITH_G:
            for pid in problems:
                f = os.path.join("results", "llamea", tag, arm, pid.replace("/", "_"), "best.json")
                if not os.path.exists(f):
                    continue
                b = json.load(open(f))
                m = _n_constraints(pid)
                for cond in ("normal", "blind"):
                    todo = [s for s in seeds if (tag, arm, pid, cond, s) not in seen]
                    for s in todo:
                        jobs.append((tag, arm, pid, cond, s, b, m))
    return jobs


def _run(job):
    tag, arm, pid, cond, s, b, m = job
    if cond == "normal":
        r = run_job(b["code"], pid, "free", [s], class_name=b["name"], phase="test")[0]
    else:
        r = run_job(blind_source(b["code"], b["name"], m), pid, "penalty", [s],
                    class_name="BlindWrapper__", phase="test")[0]
    d = r.to_dict()
    d.update(tag=tag, llamea_arm=arm, condition=cond, algorithm=b["name"])
    return d


def _aocc(r):
    return 0.0 if r["status"] in ("error", "crashed", "timeout") else float(r["aocc"])


def summarise(runs, penalty_ref: dict) -> str:
    by = {}
    for r in runs:
        by.setdefault((r["problem"], r["tag"], r["llamea_arm"]), {}).setdefault(r["condition"], []).append(r)
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    L = ["# Ablation: the same algorithm with and without the constraint information", "",
         "Mean test AOCC over seeds. blind = f is the penalised value, g = 0 (what the penalty arm sees).",
         "penalty arm = best algorithm of the penalty arm of the same repetition, for reference.", "",
         "| problem | repetition | arm | algorithm | normal | blind | blind - normal | p (Wilcoxon) | penalty arm |",
         "|---|---|---|---|---|---|---|---|---|"]
    drops = {}
    for (pid, tag, arm), c in sorted(by.items()):
        if "normal" not in c or "blind" not in c:
            continue
        n = {r["seed"]: _aocc(r) for r in c["normal"]}
        bl = {r["seed"]: _aocc(r) for r in c["blind"]}
        common = sorted(set(n) & set(bl))
        a, b = np.array([n[s] for s in common]), np.array([bl[s] for s in common])
        p = ""
        if wilcoxon is not None and len(common) >= 5 and np.any(a != b):
            p = f"{wilcoxon(a, b).pvalue:.3g}"
        ref = penalty_ref.get((tag, pid))
        L.append(f"| {pid} | {tag} | {arm} | {c['normal'][0]['algorithm']} | {a.mean():.3f} | {b.mean():.3f} | "
                 f"{b.mean() - a.mean():+.3f} | {p} | {'' if ref is None else f'{ref:.3f}'} |")
        drops.setdefault(pid, []).append((a.mean(), b.mean(), ref))
    L += ["", "Per problem, over all algorithms: mean normal / blind / penalty arm"]
    for pid, v in sorted(drops.items()):
        v = np.array([(x, y, np.nan if z is None else z) for x, y, z in v], float)
        L.append(f"* {pid}: {np.nanmean(v[:, 0]):.3f} / {np.nanmean(v[:, 1]):.3f} / {np.nanmean(v[:, 2]):.3f} "
                 f"({len(v)} algorithms)")
    return "\n".join(L) + "\n"


def _penalty_reference(tags):
    ref = {}
    for tag in tags:
        p = os.path.join("results", "llamea_final", tag, "runs.jsonl")
        if not os.path.exists(p):
            continue
        acc = {}
        for line in open(p):
            r = json.loads(line)
            if r.get("baseline") == "llamea_penalty":
                acc.setdefault(r["problem"], []).append(_aocc(r))
        for pid, v in acc.items():
            ref[(tag, pid)] = float(np.mean(v))
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", nargs="*", default=["cec/RC12"])
    ap.add_argument("--tags", nargs="*", default=None, help="default: every results/llamea/haiku5*")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(config.TEST_SEEDS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    tags = a.tags or sorted(os.path.basename(d) for d in glob.glob("results/llamea/haiku5*")
                            if re.fullmatch(r"haiku5(-r\d+)?", os.path.basename(d)))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "runs.jsonl")
    runs = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    seen = {(r["tag"], r["llamea_arm"], r["problem"], r["condition"], r["seed"]) for r in runs}
    jobs = _jobs(tags, a.problems, a.seeds, seen)
    print(f"{len(jobs)} runs ({', '.join(tags)}) -> {path}", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(path, "a") as fh:
        futs = [ex.submit(_run, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            d = f.result()
            runs.append(d)
            fh.write(json.dumps(d, default=float) + "\n")
            fh.flush()
            if k % 20 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}]", flush=True)
    md = summarise(runs, _penalty_reference(tags))
    open(os.path.join(a.out, "summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()