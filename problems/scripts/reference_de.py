"""Reference solver for verifying problem implementations and finding x*.

L-SHADE-style current-to-pbest/1/bin DE with Deb's feasibility rules
(feasible beats infeasible; among feasible lower f; among infeasible lower total violation).
Not a baseline of the study: only used to (1) check we reach the published f*,
(2) obtain a near-optimal x* to inspect which constraints are active.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from problems.cec2020 import CEC2020Problem  # noqa: E402


def better(f1, v1, f2, v2):
    if v1 == 0 and v2 == 0:
        return f1 <= f2
    if v1 == 0 or v2 == 0:
        return v1 == 0
    return v1 <= v2


def run(num, seed, budget, pop=None):
    p = CEC2020Problem(num)
    rng = np.random.default_rng(seed)
    n = p.dim
    NP = pop or max(30, 10 * n)
    NP = min(NP, 200)
    lb, ub = p.lb, p.ub

    def ev(x):
        x = p.seal(x)
        r = p.raw(x)
        v = float(np.sum(np.maximum(0, r.g)) + np.sum(np.maximum(0, np.abs(r.h) - 1e-4)))
        return x, r.f, v

    X = lb + rng.random((NP, n)) * (ub - lb)
    F, V = np.empty(NP), np.empty(NP)
    for i in range(NP):
        X[i], F[i], V[i] = ev(X[i])
    used = NP
    MF, MCR = 0.5, 0.5
    curve = []
    while used < budget:
        order = sorted(range(NP), key=lambda i: (V[i] > 0, V[i] if V[i] > 0 else F[i]))
        pbest = order[: max(2, NP // 10)]
        SF, SCR = [], []
        for i in range(NP):
            if used >= budget:
                break
            Fm = np.clip(MF + 0.1 * rng.standard_cauchy(), 0.05, 1.0)
            CR = np.clip(rng.normal(MCR, 0.1), 0, 1)
            b = rng.choice(pbest)
            r1, r2 = rng.choice([j for j in range(NP) if j != i], 2, replace=False)
            m = X[i] + Fm * (X[b] - X[i]) + Fm * (X[r1] - X[r2])
            m = np.where(m < lb, (lb + X[i]) / 2, m)
            m = np.where(m > ub, (ub + X[i]) / 2, m)
            mask = rng.random(n) < CR
            mask[rng.integers(n)] = True
            u = np.where(mask, m, X[i])
            u, fu, vu = ev(u)
            used += 1
            if better(fu, vu, F[i], V[i]):
                X[i], F[i], V[i] = u, fu, vu
                SF.append(Fm)
                SCR.append(CR)
        if SF:
            SF = np.array(SF)
            MF = 0.9 * MF + 0.1 * (np.sum(SF**2) / np.sum(SF))
            MCR = 0.9 * MCR + 0.1 * np.mean(SCR)
        k = min(range(NP), key=lambda i: (V[i] > 0, V[i] if V[i] > 0 else F[i]))
        curve.append((used, float(F[k]), float(V[k])))
    k = min(range(NP), key=lambda i: (V[i] > 0, V[i] if V[i] > 0 else F[i]))
    return {"num": num, "seed": seed, "f": float(F[k]), "v": float(V[k]), "x": X[k].tolist(), "curve": curve}


def budget_for(num):
    # CEC 2020 RW guideline: maxFEs = 1e5 (D<=10), 2e5 (10<D<=30), 4e5 (30<D<=50)
    d = CEC2020Problem(num).dim
    return 100_000 if d <= 10 else 200_000 if d <= 30 else 400_000


if __name__ == "__main__":
    nums = [int(a) for a in sys.argv[1].split(",")]
    seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    jobs = [(n, s, budget_for(n)) for n in nums for s in seeds]
    t0 = time.time()
    with Pool() as pool:
        out = pool.starmap(run, jobs)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "reference_runs.json"), "w"))
    print("done", time.time() - t0)
