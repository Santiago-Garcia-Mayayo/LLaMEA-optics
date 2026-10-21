import numpy as np


class AdaptiveElitistMixedES:
    """
    (mu+lambda) mixed-integer evolution strategy.
    - continuous genes: Gaussian mutation scaled by per-individual self-adapted sigma
    - binary genes: flip; ordinal genes: +/-1 step (occasional jump); categorical: resample
    - crossover: uniform, or BLX-style blend on continuous genes + uniform on discrete genes
    - global 1/5-type success-rate control of step sizes
    - duplicate-free elitist survival, and restarts anchored on the incumbent best solution
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        dim = self.dim
        budget = self.budget

        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        types = [str(t) for t in func.variable_types]

        cont = np.array([t == "continuous" for t in types])
        binm = np.array([t == "binary" for t in types])
        cat = np.array([t == "categorical" for t in types])
        ordm = np.array([t == "ordinal" for t in types])
        disc = ~cont

        nv = np.ones(dim)
        for i in range(dim):
            if disc[i]:
                nv[i] = int(round(float(func.n_values[i])))

        lo = np.where(cont, lb, 0.0)
        hi = np.where(cont, ub, nv - 1.0)
        span = np.where(cont, ub - lb, 1.0)
        ci = np.where(cont)[0]
        di = np.where(disc)[0]

        state = {"used": 0, "best_y": np.inf, "best_x": None}
        seen = set()

        def repair(X):
            X = np.clip(X, lo, hi)
            if len(di) > 0:
                X[:, di] = np.rint(X[:, di])
            return X

        def sample(n):
            X = np.empty((n, dim))
            if len(ci) > 0:
                X[:, ci] = rng.uniform(lb[ci], ub[ci], size=(n, len(ci)))
            for j in di:
                X[:, j] = rng.integers(0, int(nv[j]), size=n)
            return X

        def evaluate(X):
            n = len(X)
            y = np.asarray(func(X), dtype=float).reshape(-1)
            if y.size != n:
                raise RuntimeError("func must return one value per row in batch mode")
            state["used"] += n
            for i in range(n):
                seen.add(X[i].tobytes())
                if y[i] < state["best_y"]:
                    state["best_y"] = float(y[i])
                    state["best_x"] = X[i].copy()
            return y

        def mutate(parent, s):
            c = parent.copy()
            k = 1 + rng.binomial(dim - 1, min(0.5, 2.0 / dim))
            genes = rng.choice(dim, size=min(k, dim), replace=False)
            for g in genes:
                if cont[g]:
                    if rng.random() < 0.1:
                        step = np.clip(rng.standard_cauchy(), -5, 5)
                    else:
                        step = rng.standard_normal()
                    c[g] += s * span[g] * step
                elif binm[g]:
                    c[g] = 1.0 - c[g]
                elif cat[g]:
                    k_v = int(nv[g])
                    if k_v > 1:
                        r = int(rng.integers(0, k_v - 1))
                        if r >= int(c[g]):
                            r += 1
                        c[g] = r
                elif ordm[g]:
                    if rng.random() < 0.1:
                        c[g] = float(rng.integers(0, int(nv[g])))
                    else:
                        c[g] += rng.choice([-1.0, 1.0])
            return repair(c[None, :])[0]

        def dedup_select(Xall, Fall, Sall, n_keep):
            idx = np.argsort(Fall, kind="stable")
            keep = []
            keys = set()
            for i in idx:
                key = Xall[i].tobytes()
                if key in keys:
                    continue
                keys.add(key)
                keep.append(i)
                if len(keep) == n_keep:
                    break
            keep = np.array(keep, dtype=int)
            return keep

        # ----- parameters -----
        mu = 15
        lam = 40
        stall_limit = 20
        gscale = 1.0

        # ----- initialisation -----
        n0 = min(budget, max(4 * mu, 8 * dim))
        X = sample(n0)
        F = evaluate(X)
        S = np.full(n0, 0.2)
        keep = dedup_select(X, F, S, mu)
        X, F, S = X[keep], F[keep], S[keep]

        prev_best = state["best_y"]
        stall = 0

        # ----- main loop -----
        while state["used"] < budget:
            remaining = budget - state["used"]
            nb = min(lam, remaining)
            if nb <= 0:
                break

            imm_frac = 0.05 if stall < stall_limit // 2 else 0.25
            n_imm = int(round(imm_frac * nb))

            children = np.empty((nb, dim))
            csig = np.empty(nb)
            for k in range(nb):
                if k < n_imm:
                    children[k] = sample(1)[0]
                    csig[k] = 0.3
                    continue

                # binary tournament parents
                i1, i2 = rng.integers(0, len(X), size=2)
                p1 = i1 if F[i1] <= F[i2] else i2
                i3, i4 = rng.integers(0, len(X), size=2)
                p2 = i3 if F[i3] <= F[i4] else i4
                a, b = X[p1], X[p2]

                if rng.random() < 0.5:
                    mask = rng.random(dim) < 0.5
                    base = np.where(mask, a, b)
                else:
                    base = np.where(rng.random(dim) < 0.5, a, b)
                    if len(ci) > 0:
                        r = rng.uniform(-0.25, 1.25)
                        base[ci] = a[ci] + r * (b[ci] - a[ci])
                    base = repair(base[None, :])[0]

                s = np.sqrt(max(S[p1] * S[p2], 1e-12))
                s = s * np.exp(0.3 * rng.standard_normal()) * gscale
                s = float(np.clip(s, 1e-3, 0.5))

                c = base
                for _ in range(5):
                    c = mutate(base, s)
                    if c.tobytes() not in seen:
                        break
                children[k] = c
                csig[k] = s

            Y = evaluate(children)

            # elitist (mu+lambda) survival with duplicate removal
            Xall = np.vstack([X, children])
            Fall = np.concatenate([F, Y])
            Sall = np.concatenate([S, csig])
            keep = dedup_select(Xall, Fall, Sall, mu)
            n_par = len(X)
            survived = int(np.sum(keep >= n_par))
            X, F, S = Xall[keep], Fall[keep], Sall[keep]

            # 1/5-type global step-size control
            rate = survived / max(nb, 1)
            if rate > 0.2:
                gscale = min(gscale * 1.15, 10.0)
            elif rate < 0.2:
                gscale = max(gscale * 0.88, 1e-3)

            # stagnation tracking
            if state["best_y"] < prev_best - 1e-12:
                prev_best = state["best_y"]
                stall = 0
            else:
                stall += 1

            # restart anchored on the incumbent
            if stall >= stall_limit and state["used"] < budget and state["best_x"] is not None:
                nr = min(mu - 1, budget - state["used"])
                if nr > 0:
                    best_x = state["best_x"].copy()
                    n_pert = nr // 2
                    Xr = np.empty((nr, dim))
                    Sr = np.full(nr, 0.2)
                    for j in range(n_pert):
                        Xr[j] = mutate(best_x, 0.1)
                    Xr[n_pert:] = sample(nr - n_pert)
                    Yr = evaluate(Xr)
                    Xc = np.vstack([state["best_x"][None, :], Xr])
                    Fc = np.concatenate([[state["best_y"]], Yr])
                    Sc = np.concatenate([[0.1], Sr])
                    X, F, S = Xc, Fc, Sc
                    gscale = 1.0
                stall = 0
                prev_best = state["best_y"]

            # mild decay of step sizes in the population to favour exploitation
            S = np.clip(S * 0.98, 1e-3, 0.5)

        best_x = state["best_x"]
        if best_x is None:
            best_x = X[0]
        return state["best_y"], np.asarray(best_x)