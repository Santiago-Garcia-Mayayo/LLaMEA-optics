"""Final evaluation: the best algorithm of every LLaMEA run on its problem, with the test
seeds and the test budget, through the same harness as the baselines.

    python -m experiments.final_eval                      # all finished runs of results/llamea/main
    python -m experiments.final_eval --tag pilot --workers 16

Writes <out>/runs.jsonl in the baselines' format, with "baseline" = "llamea_<arm>", so the
summary and statistics treat LLaMEA's algorithms and the baselines alike; resumable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from baselines.run_baselines import _done, write_summary
from harness import config
from harness.runner import run_job


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/llamea")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=None, help="default: results/llamea_final/<tag>")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(config.TEST_SEEDS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = a.out or os.path.join("results", "llamea_final", a.tag)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "runs.jsonl")
    seen = _done(path)
    jobs = []
    for f in sorted(glob.glob(os.path.join(a.runs, a.tag, "*", "*", "best.json"))):
        b = json.load(open(f))
        name = f"llamea_{b['arm']}"
        for s in a.seeds:
            if (name, b["problem"], s) not in seen:
                jobs.append((name, b, s))
    print(f"{len(jobs)} runs -> {path}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(path, "a") as fh:
        futs = {ex.submit(run_job, b["code"], b["problem"], b["arm"], [s], class_name=b["name"],
                          phase="test"): (name, b) for name, b, s in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            name, b = futs[fut]
            for r in fut.result():
                d = r.to_dict()
                d["baseline"] = name
                fh.write(json.dumps(d, default=float) + "\n")
            fh.flush()
            if k % 10 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] ({time.time() - t0:.0f} s)", flush=True)
    write_summary(path, os.path.join(out, "summary.md"))


if __name__ == "__main__":
    main()
