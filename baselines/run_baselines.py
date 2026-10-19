"""Run the baselines on the test problems and seeds, through the isolated harness runner.

    python -m baselines.run_baselines                      # all baselines, 10 problems, 15 test seeds
    python -m baselines.run_baselines --baselines lshade_deb cmaes_deb --problems cec/RC15 --seeds 0 1
    python -m baselines.run_baselines --workers 8 --out results/baselines --trace

Writes <out>/runs.jsonl (one line per run, appended; existing (baseline, problem, seed) are skipped,
so an interrupted run can be resumed) and <out>/summary.md.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from harness import config
from harness.runner import run_job

from . import BASELINES


def _done(path):
    seen = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    seen.add((r["baseline"], r["problem"], r["seed"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return seen


def run(baselines, problems, seeds, out, workers, trace=False):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "runs.jsonl")
    seen = _done(path)
    jobs = []
    for b in baselines:
        algo, arm = BASELINES[b]
        for pid in problems:
            todo = [s for s in seeds if (b, pid, s) not in seen]
            if todo:
                jobs.append((b, algo, arm, pid, todo))
    print(f"{len(jobs)} jobs ({sum(len(j[4]) for j in jobs)} runs), {workers} workers -> {path}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex, open(path, "a") as fh:
        futs = {ex.submit(run_job, algo, pid, arm, todo,
                          trace_dir=os.path.join(out, "traces", b) if trace else None): (b, pid)
                for b, algo, arm, pid, todo in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            b, pid = futs[fut]
            for r in fut.result():
                d = r.to_dict()
                d["baseline"] = b
                fh.write(json.dumps(d, default=float) + "\n")
            fh.flush()
            print(f"[{k}/{len(jobs)}] {b} {pid} done ({time.time() - t0:.0f} s)", flush=True)
    write_summary(path, os.path.join(out, "summary.md"))


def load_runs(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_summary(runs_path, md_path):
    runs = load_runs(runs_path)
    names = [b for b in BASELINES if any(r["baseline"] == b for r in runs)]
    probs = [p for p in config.TEST_PROBLEMS if any(r["problem"] == p for r in runs)]
    probs += sorted({r["problem"] for r in runs} - set(probs))

    def cell(b, p):
        rs = [r for r in runs if r["baseline"] == b and r["problem"] == p]
        if not rs:
            return "—"
        a = np.array([0.0 if r["status"] in ("error", "crashed", "timeout") else r["aocc"] for r in rs])
        feas = sum(r["found_feasible"] for r in rs)
        return f"{a.mean():.3f} ± {a.std():.3f} ({feas}/{len(rs)})"

    lines = ["# Baselines on the test set", "",
             "Mean ± std AOCC over seeds (runs that found a feasible point / runs). Higher is better; "
             "AOCC > 0.5 requires being feasible for most of the run.",
             "", "| problem | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for p in probs:
        lines.append(f"| {p} | " + " | ".join(cell(b, p) for b in names) + " |")
    means = []
    for b in names:
        per = [np.mean([0.0 if r["status"] in ("error", "crashed", "timeout") else r["aocc"]
                        for r in runs if r["baseline"] == b and r["problem"] == p]) for p in probs
               if any(r["baseline"] == b and r["problem"] == p for r in runs)]
        means.append(f"**{np.mean(per):.3f}**")
    lines.append("| **mean over problems** | " + " | ".join(means) + " |")
    bad = [r for r in runs if r["status"] in ("error", "crashed", "timeout")]
    lines += ["", f"Runs: {len(runs)}; failed (error/crash/timeout): {len(bad)}."]
    for r in bad[:10]:
        lines.append(f"* {r['baseline']} {r['problem']} seed {r['seed']}: {r['status']} — {r['error'][:200]}")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", nargs="*", default=list(BASELINES))
    ap.add_argument("--problems", nargs="*", default=list(config.TEST_PROBLEMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(config.TEST_SEEDS))
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--trace", action="store_true", help="also save per-evaluation traces (.npz)")
    a = ap.parse_args()
    unknown = set(a.baselines) - set(BASELINES)
    if unknown:
        ap.error(f"unknown baselines {sorted(unknown)}; choose from {list(BASELINES)}")
    run(a.baselines, a.problems, a.seeds, a.out, a.workers, a.trace)


if __name__ == "__main__":
    main()
