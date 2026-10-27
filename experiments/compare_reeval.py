"""Compare two final evaluations of the same LLaMEA algorithms (before / after a harness change).

    python -m experiments.compare_reeval                       # results/llamea_final_unseeded vs results/llamea_final

Per (repetition, arm, problem): mean test AOCC before and after, the largest per-seed change,
and whether the algorithm calls np.random.default_rng() without a seed. Algorithms that do not
should give exactly the same numbers (their randomness was already seeded); the others change
because before the fix each evaluation drew fresh OS entropy.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

FAILED = ("error", "crashed", "timeout")
UNSEEDED = re.compile(r"default_rng\(\s*(seed\s*=\s*)?(None)?\s*\)")


def _load(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("baseline", "").startswith("llamea_"):
                out[(r["baseline"][len("llamea_"):], r["problem"], r["seed"])] = \
                    0.0 if r["status"] in FAILED else r["aocc"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="results/llamea_final_unseeded")
    ap.add_argument("--new", default="results/llamea_final")
    ap.add_argument("--runs", default="results/llamea")
    a = ap.parse_args()
    rows, same, total, missing = [], 0, 0, 0
    for d in sorted(glob.glob(os.path.join(a.old, "haiku5*"))):
        tag = os.path.basename(d)
        old, new = _load(os.path.join(d, "runs.jsonl")), _load(os.path.join(a.new, tag, "runs.jsonl"))
        for arm, pid in sorted({(k[0], k[1]) for k in old}):
            f = os.path.join(a.runs, tag, arm, pid.replace("/", "_"), "best.json")
            uns = bool(UNSEEDED.search(json.load(open(f))["code"])) if os.path.exists(f) else None
            seeds = sorted(s for (x, p, s) in old if (x, p) == (arm, pid))
            if not all((arm, pid, s) in new for s in seeds):
                missing += 1
                continue
            o = np.array([old[(arm, pid, s)] for s in seeds]); n = np.array([new[(arm, pid, s)] for s in seeds])
            dmax = float(np.max(np.abs(n - o)))
            rows.append((tag, arm, pid, uns, o.mean(), n.mean(), dmax))
            if not uns:
                total += 1
                same += dmax < 1e-9
    print(f"{'repetition':11s} {'arm':12s} {'problem':22s} unseeded  before  after   max|Δ| per seed")
    for tag, arm, pid, uns, o, n, dmax in rows:
        print(f"{tag:11s} {arm:12s} {pid:22s} {'yes' if uns else 'no ':8s} {o:.3f}   {n:.3f}   {dmax:.4f}")
    print(f"\nAlgorithms without unseeded default_rng(): {same}/{total} reproduce the old numbers exactly.")
    u = [r for r in rows if r[3]]
    if u:
        print(f"Algorithms with unseeded default_rng(): {len(u)}, mean change of their mean AOCC "
              f"{np.mean([r[5] - r[4] for r in u]):+.4f} (mean |change| {np.mean([abs(r[5] - r[4]) for r in u]):.4f}).")
    if missing:
        print(f"{missing} (repetition, arm, problem) not yet re-evaluated.")


if __name__ == "__main__":
    main()