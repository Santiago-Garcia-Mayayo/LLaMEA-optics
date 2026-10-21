import numpy as np


class MixedAdaptiveES:
    """
    Elitist (mu+lambda) evolution strategy for mixed-integer black-box problems.
    - continuous genes: Gaussian mutation with per-individual self-adapted step size
    - binary genes: bit flip
    - ordinal genes: +/-1 step (clipped to the valid range)
    - categorical genes: uniform resampling among the other labels
    - uniform crossover, binary tournament selection, duplicate avoidance,
      and random immigrants when the search stalls.
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
        span = hi - lo
        span[cont] = ub[cont] - lb[cont]
        nd = int(disc.sum())

        state = {"used": 0, "best_y": np.inf, "best_x": None}
        seen = set()

        def repair(X):
            X = np.clip(X, lo, hi)
            if nd > 0:
                X[..., disc] = np.rint(X[..., disc])
            return X

        def sample(n):
            X = np.empty((n, dim))
            if cont.any():
                X[:, cont] = rng.uniform(lb[cont], ub[cont], size=(n, int(cont.sum())))
            if nd > 0:
                X[:, disc] = rng.integers(0, nv[disc].astype(int), size=(n, nd))
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

        def mutate(c, s):
            pm = np.clip(2.0 / dim, 0.1, 0.5)
            m = rng.random(dim) < pm
            if not m.any():
                m[rng.integers(dim)] = True
            mc = m & cont
            if mc.any():
                c[mc] += s * span[mc] * rng.standard_normal(int(mc.sum()))
            mb = m & binm
            if mb.any():
                c[mb] = 1.0 - c[mb]
            mk = m & cat
            for j in np.where(mk)[0]:
                k = int(nv[j])
                if k > 1:
                    r = rng.integers(0, k - 1)
                    if r >= int(c[j]):
                        r += 1
                    c[j] = r
            mo = m & ordm
            if mo.any():
                steps = rng.choice([-1.0, 1.0], size=int(mo.sum()))
                jump = rng.random(int(mo.sum())) < 0.1
                rnd = rng.integers(0, nv[mo].astype(int))
                c[mo] = np.where(jump, rnd, c[mo] + steps)
            return repair(c)

        # ----- initialisation -----
        mu = 20
        lam = 40
        n0 = min(max(5 * mu, 10 * dim), max(mu, budget // 3))
        n0 = min(n0, budget)
        X = sample(n0)
        F = evaluate(X)
        S = np.full(n0, 0.2)
        order = np.argsort(F)[:mu]
        X, F, S = X[order], F[order], S[order]

        prev_best = state["best_y"]
        stall = 0

        # ----- main loop -----
        while state["used"] < budget:
            nb = min(lam, budget - state["used"])
            if nb <= 0:
                break

            # binary tournament parents
            t1 = rng.integers(0, len(X), size=(nb, 2))
            t2 = rng.integers(0, len(X), size=(nb, 2))
            p1 = np.where(F[t1[:, 0]] < F[t1[:, 1]], t1[:, 0], t1[:, 1])
            p2 = np.where(F[t2[:, 0]] < F[t2[:, 1]], t2[:, 0], t2[:, 1])

            # fraction of random immigrants (more when stalled)
            rand_frac = 0.25 if stall < 10 else 0.75
            n_rand = int(round(rand_frac * nb))

            children = np.empty((nb, dim))
            csig = np.empty(nb)
            for k in range(nb):
                if k < n_rand:
                    children[k] = sample(1)[0]
                    csig[k] = 0.3
                    continue
                a, b = X[p1[k]], X[p2[k]]
                if rng.random() < 0.7:
                    mask = rng.random(dim) < 0.5
                    base = np.where(mask, a, b)
                    s = 0.5 * (S[p1[k]] + S[p2[k]])
                else:
                    base = a.copy()
                    s = S[p1[k]]
                s = float(np.clip(s * np.exp(0.3 * rng.standard_normal()), 1e-3, 0.5))
                c = base.copy()
                for _ in range(5):
                    c = mutate(base.copy(), s)
                    if c.tobytes() not in seen:
                        break
                children[k] = c
                csig[k] = s

            Y = evaluate(children)

            # (mu + lambda) elitist selection
            Xall = np.vstack([X, children])
            Fall = np.concatenate([F, Y])
            Sall = np.concatenate([S, csig])
            order = np.argsort(Fall)[:mu]
            X, F, S = Xall[order], Fall[order], Sall[order]

            # stagnation tracking
            if state["best_y"] < prev_best - 1e-12:
                prev_best = state["best_y"]
                stall = 0
            else:
                stall += 1

        best_x = state["best_x"]
        if best_x is None:
            best_x = X[0]
        return state["best_y"], np.asarray(best_x)