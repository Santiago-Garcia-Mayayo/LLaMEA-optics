import sys, numpy as np, cma, json
sys.path.insert(0, ".")
from suite import LensSuiteProblem
out = {}
for inst in ["realistic_draft", "original"]:
    p = LensSuiteProblem(inst); t0 = p.template(); nx = p._p.nx
    rec = []
    for run in range(4):
        rng = np.random.default_rng(run)
        start = t0[:nx] * (1 + 0.15 * rng.normal(size=nx)) if run else t0[:nx]
        scale = np.abs(t0[:nx]) * 0.2 + 0.5
        es = cma.CMAEvolutionStrategy(np.zeros(nx), 1.0, {"popsize": 16, "verbose": -9, "seed": run + 1, "maxfevals": 2500})
        while not es.stop():
            Z = np.array(es.ask()); fs = []
            for z in Z:
                th = np.r_[start + z * scale, t0[nx:]]
                pen = p.penalty(th); f, v = p.constrained(th)
                fs.append(pen if np.isfinite(pen) else 1e6); rec.append((pen, f, v.tolist()))
            es.tell(list(Z), fs)
        print(inst, run, es.result.fbest, flush=True)
    pen = np.array([r[0] for r in rec]); V = np.array([r[2] for r in rec]); F = np.array([r[1] for r in rec])
    q = pen <= np.quantile(pen, 0.05)
    act = (V[q] > 0).mean(0)
    near = {}
    out[inst] = {"frac_viol_top5": dict(zip(p.constraint_names, act.round(3).tolist())),
                 "frac_any_viol_top5": float((V[q] > 1e-9).any(1).mean()),
                 "share_penalty_top5": float(np.median((pen[q] - F[q]) / pen[q]))}
    print(inst, json.dumps(out[inst]), flush=True)
json.dump(out, open("diag_realistic.json", "w"), indent=1)
