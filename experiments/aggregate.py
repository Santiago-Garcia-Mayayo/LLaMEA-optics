"""Combine the final evaluations of all LLaMEA repetitions.

    python -m experiments.aggregate                     # every results/llamea_final/haiku5*
    python -m experiments.aggregate --tags haiku5 haiku5-r2

Unit of analysis: one LLaMEA run = the mean test AOCC of its best algorithm over the 15 test
seeds (failed seeds count 0). Per problem and arm: mean +- std over runs. Arms compared with
two-sided Mann-Whitney U and Cliff's delta over runs, Holm-corrected over the problems of
each comparison. Overall: per run, the mean over problems. Optional: Spearman correlation
between the screening effect (results/screening/selection.json, d_mean) and the LLaMEA effect.
Writes results/aggregate/summary.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

COMPARISONS = (("free", "penalty"), ("informed_v2", "penalty"), ("informed_v2", "free"))
FAILED = ("error", "crashed", "timeout")


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None, :]
    return float(np.mean(a > b) - np.mean(a < b))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def load(tags):
    """{(arm, problem): {tag: mean AOCC over seeds}}, and the resource failures per tag."""
    per, resource = {}, {}
    for tag in tags:
        acc = {}
        for line in open(os.path.join("results", "llamea_final", tag, "runs.jsonl")):
            r = json.loads(line)
            if not r.get("baseline", "").startswith("llamea_"):
                continue
            if r["status"] in FAILED and "pthread_create" in (r.get("error") or ""):
                resource.setdefault(tag, []).append((r["baseline"], r["problem"], r["seed"]))
            arm = r["baseline"][len("llamea_"):]
            acc.setdefault((arm, r["problem"]), []).append(0.0 if r["status"] in FAILED else r["aocc"])
        for k, v in acc.items():
            per.setdefault(k, {})[tag] = float(np.mean(v))
    return per, resource


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--screening", default="results/screening/selection.json")
    ap.add_argument("--out", default="results/aggregate")
    a = ap.parse_args()
    tags = a.tags or sorted(os.path.basename(d) for d in glob.glob("results/llamea_final/haiku5*")
                            if re.fullmatch(r"haiku5(-r\d+)?", os.path.basename(d))
                            and os.path.exists(os.path.join(d, "runs.jsonl")))
    per, resource = load(tags)
    arms = [x for x in ("penalty", "free", "informed", "informed_v2") if any(k[0] == x for k in per)]
    problems = sorted({k[1] for k in per}, key=lambda p: (not p.startswith("lens"), p))
    from scipy.stats import mannwhitneyu, spearmanr

    L = [f"# LLaMEA over repetitions ({', '.join(tags)})", "",
         "Mean +- std over LLaMEA runs of the mean test AOCC (15 seeds) of each run's best algorithm; "
         "n = number of runs.", "",
         "| problem | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for pid in problems:
        cells = []
        for arm in arms:
            v = list(per.get((arm, pid), {}).values())
            cells.append("—" if not v else f"{np.mean(v):.3f} ± {np.std(v):.3f} (n={len(v)})")
        L.append(f"| {pid} | " + " | ".join(cells) + " |")
    # overall: per run (tag), mean over problems
    overall = {}
    for arm in arms:
        for tag in tags:
            v = [per[(arm, p)][tag] for p in problems if tag in per.get((arm, p), {})]
            if len(v) == len(problems):
                overall.setdefault(arm, []).append(float(np.mean(v)))
    L.append("| **mean over problems** | " + " | ".join(
        f"**{np.mean(overall[x]):.3f} ± {np.std(overall[x]):.3f}** (n={len(overall[x])})" if x in overall else "—"
        for x in arms) + " |")

    for x, y in COMPARISONS:
        if x not in arms or y not in arms:
            continue
        rows, ps = [], []
        for pid in problems:
            A, B = list(per.get((x, pid), {}).values()), list(per.get((y, pid), {}).values())
            if len(A) < 2 or len(B) < 2:
                continue
            p = mannwhitneyu(A, B, alternative="two-sided").pvalue
            rows.append((pid, np.mean(A) - np.mean(B), cliffs_delta(A, B), p, len(A), len(B)))
            ps.append(p)
        if not rows:
            continue
        adj = holm(ps)
        L += ["", f"## {x} vs {y}", "",
              "| problem | mean diff | Cliff's δ | p | p (Holm) | n |", "|---|---|---|---|---|---|"]
        for (pid, d, cd, p, na, nb), pa in zip(rows, adj):
            mark = " *" if pa < 0.05 else ""
            L.append(f"| {pid} | {d:+.3f} | {cd:+.2f} | {p:.3g} | {pa:.3g}{mark} | {na} vs {nb} |")
        if x in overall and y in overall and len(overall[x]) >= 2 and len(overall[y]) >= 2:
            p = mannwhitneyu(overall[x], overall[y], alternative="two-sided").pvalue
            L.append(f"| **mean over problems** | {np.mean(overall[x]) - np.mean(overall[y]):+.3f} | "
                     f"{cliffs_delta(overall[x], overall[y]):+.2f} | {p:.3g} | | "
                     f"{len(overall[x])} vs {len(overall[y])} |")

    if os.path.exists(a.screening):
        rows = {r["problem"]: r for r in json.load(open(a.screening))["rows"]}
        L += ["", "## Screening effect (baselines) vs LLaMEA effect", ""]
        for x in ("free", "informed_v2"):
            pts = [(rows[p]["d_mean"], np.mean(list(per[(x, p)].values())) - np.mean(list(per[("penalty", p)].values())))
                   for p in problems if p in rows and (x, p) in per and ("penalty", p) in per]
            if len(pts) >= 4:
                r = spearmanr([u for u, _ in pts], [v for _, v in pts])
                L.append(f"* {x} - penalty: Spearman ρ = {r.statistic:+.2f}, p = {r.pvalue:.3g} ({len(pts)} problems)")

    if resource:
        L += ["", "**Test runs lost to the machine's thread limit (count as 0 above; re-evaluate them):**"]
        for tag, v in resource.items():
            L.append(f"* {tag}: {len(v)} — " + ", ".join(f"{b} {p} seed {s}" for b, p, s in v[:6]))
    os.makedirs(a.out, exist_ok=True)
    md = "\n".join(L) + "\n"
    open(os.path.join(a.out, "summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()