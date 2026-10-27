"""Sensitivity of the penalty baselines to the penalty weight rho (CEC problems).

    python -m experiments.rho_sensitivity run --workers 16     # L-SHADE / CMA-ES + penalty, rho x 0.1, 10, 100
    python -m experiments.rho_sensitivity summary              # table, with rho x 1 from results/baselines_1000d

Every CEC problem keeps its objective, constraints, budget and scoring; only the weight of the
penalty arm's scalar f + rho * sum(v) changes (rho = 10 * max(1, |f*|) in the study). The lens
has no rho (its penalty is lensgopt's loss) and is not included. Test seeds, test budget.
Writes results/baselines_rho/runs.jsonl (resumable) and results/baselines_rho/rho_summary.md.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from harness import config

CEC = [p for p in config.PROBLEMS if p.startswith("cec/")]
SCALES = (0.1, 10, 100)
BASELINES = ("lshade_penalty", "cmaes_penalty")
REF = {"lshade_penalty": "lshade_deb", "cmaes_penalty": "cmaes_deb"}
OUT = "results/baselines_rho"
FAILED = ("error", "crashed", "timeout")


def _means(path):
    acc = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            acc.setdefault((r["baseline"], r["problem"]), []).append(0.0 if r["status"] in FAILED else r["aocc"])
    return {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}


def summary(out, main_path="results/baselines_1000d/runs.jsonl"):
    m = _means(os.path.join(out, "runs.jsonl"))
    m.update({k: v for k, v in _means(main_path).items()})
    cols = (0.1, 1, 10, 100)
    L = ["# Penalty baselines vs penalty weight rho", "",
         "Mean test AOCC (15 seeds). rho x 1 = the study's value, 10 * max(1, |f*|). "
         "Last column: the same algorithm with Deb's rules (does not use rho).", ""]
    for b in BASELINES:
        L += [f"## {b}", "", "| problem | " + " | ".join(f"rho x {c:g}" for c in cols) + f" | best rho | {REF[b]} |",
              "|---|" + "---|" * (len(cols) + 2)]
        for pid in CEC:
            vals = [m.get((b, pid if c == 1 else f"{pid}@rho{c:g}"), (np.nan, 0))[0] for c in cols]
            best = cols[int(np.nanargmax(vals))] if not np.all(np.isnan(vals)) else None
            deb = m.get((REF[b], pid), (np.nan, 0))[0]
            cells = [("**%.3f**" % v if c == best else "%.3f" % v) if not np.isnan(v) else "—" for c, v in zip(cols, vals)]
            L.append(f"| {pid} | " + " | ".join(cells) + f" | x {best:g} | {deb:.3f} |" if best is not None
                     else f"| {pid} | " + " | ".join(cells) + " | — | — |")
        L.append("")
    md = "\n".join(L) + "\n"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "rho_summary.md"), "w").write(md)
    print(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("run", "summary"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.action == "run":
        from baselines.run_baselines import run
        problems = [f"{p}@rho{s:g}" for p in CEC for s in SCALES]
        run(list(BASELINES), problems, list(config.TEST_SEEDS), a.out, a.workers)
    summary(a.out)


if __name__ == "__main__":
    main()