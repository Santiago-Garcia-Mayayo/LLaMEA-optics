import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from problems.cec2020 import CEC2020Problem  # noqa: E402

runs = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "reference_runs.json")))
by = {}
for r in runs:
    by.setdefault(r["num"], []).append(r)

print(f"{'prob':5} {'D':>3} {'g':>3} {'int':>3} | {'f*':>12} {'best f':>12} {'rel gap':>9} {'feas':>5} | "
      f"{'active@x*':>9} {'reached@5%':>10} {'gap@2k':>9}")
rows = []
for num in sorted(by):
    p = CEC2020Problem(num)
    rs = by[num]
    feas = [r for r in rs if r["v"] == 0]
    best = min(feas, key=lambda r: r["f"]) if feas else min(rs, key=lambda r: r["v"])
    fs = p.f_star
    rel = (best["f"] - fs) / max(abs(fs), 1e-12) if fs is not None else np.nan
    raw = p.raw(best["x"])
    scale = np.maximum(1.0, np.abs(raw.g))
    # active: g_i within 1e-4 relative of the boundary
    active = int(np.sum(np.abs(raw.g) <= 1e-4 * scale)) if p.ng else 0
    # difficulty proxy: best-so-far at 2000 evals (median over seeds), relative gap to f*
    def at(r, n):
        pts = [c for c in r["curve"] if c[0] <= n]
        if not pts:
            return np.nan
        u, f, v = pts[-1]
        return np.inf if v > 0 else (f - fs) / max(abs(fs), 1e-12)
    g2k = np.median([at(r, 2000) for r in rs])
    # fraction of total budget until within 1e-4 rel of f* (median); measure of hardness
    def reach(r):
        for u, f, v in r["curve"]:
            if v == 0 and (f - fs) / max(abs(fs), 1e-12) <= 1e-4:
                return u
        return np.inf
    reach_u = np.median([reach(r) for r in rs])
    print(f"RC{num:02d} {p.dim:3d} {p.ng:3d} {len(p.integer_indices):3d} | {fs:12.5e} {best['f']:12.5e} {rel:9.1e} "
          f"{len(feas)}/{len(rs)} | {active:4d}/{p.ng:<4d} {reach_u:10.0f} {g2k:9.1e}  {p.name}")
    rows.append(dict(num=num, name=p.name, D=p.dim, ng=p.ng, nint=len(p.integer_indices), f_star=fs,
                     best_f=best["f"], rel_gap=rel, n_feasible_runs=len(feas), n_runs=len(rs),
                     active_at_xstar=active, evals_to_1e4=float(reach_u), gap_at_2k=float(g2k),
                     x_best=best["x"]))
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "reference_summary.json"), "w"), indent=1)
