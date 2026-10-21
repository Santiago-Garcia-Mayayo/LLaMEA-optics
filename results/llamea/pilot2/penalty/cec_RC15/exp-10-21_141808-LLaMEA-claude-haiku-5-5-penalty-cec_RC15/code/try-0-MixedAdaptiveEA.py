import numpy as np


class MixedAdaptiveEA:
    """
    (mu + lambda) evolution strategy for mixed-integer constrained black-box problems.
    - Continuous variables: Gaussian mutation with per-individual self-adaptive step size.
    - Ordinal/binary variables: +-1 neighbourhood steps or random resampling.
    - Categorical variables: random resampling to a different label.
    - Uniform crossover, tournament selection, elitist survival (penalised y is used directly).
    - Duplicate avoidance via an evaluation cache.
    - Partial restarts (keeping the incumbent) when the search stagnates.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        d = self.dim
        budget = self.budget

        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        vt = [str(t) for t in func.variable_types]
        nv_raw = getattr(func, "n_values", None)
        if nv_raw is None:
            nv_raw = np.ones(d)
        nv = np.asarray(nv_raw, dtype=float)
        nv_int = np.maximum(np.round(nv).astype(int), 1)

        is_cont = np.array([t == "continuous" for t in vt], dtype=bool)
        is_ord = np.array([t in ("ordinal", "binary") for t in vt], dtype=bool)
        is_cat = np.array([t == "categorical" for t in vt], dtype=bool)
        is_disc = ~is_cont
        width = ub - lb
        width = np.where(width <= 0, 1.0, width)

        evals = 0
        best_y = np.inf
        best_x = None
        cache = set()

        def key(x):
            return tuple(np.round(x, 8))

        def repair(X):
            X = np.asarray(X, dtype=float)
            X[:, is_cont] = np.clip(X[:, is_cont], lb[is_cont], ub[is_cont])
            Xd = np.clip(np.round(X[:, is_disc]), 0, nv_int[is_disc] - 1)
            X[:, is_disc] = Xd
            return X

        def random_points(n):
            X = np.empty((n, d))
            if is_cont.any():
                X[:, is_cont] = rng.uniform(lb[is_cont], ub[is_cont], size=(n, is_cont.sum()))
            if is_disc.any():
                X[:, is_disc] = rng.integers(0, nv_int[is_disc], size=(n, is_disc.sum()))
            return X.astype(float)

        def evaluate(X):
            nonlocal evals, best_y, best_x
            n = len(X)
            if n == 0:
                return np.empty(0)
            Y = np.asarray(func(X), dtype=float).reshape(-1)
            evals += n
            for x, y in zip(X, Y):
                cache.add(key(x))
                if y < best_y:
                    best_y = float(y)
                    best_x = x.copy()
            return Y

        def mutate(x, s):
            y = x.copy()
            pm = max(1.0 / d, 0.1)
            mask = rng.random(d) < pm
            if not mask.any():
                mask[rng.integers(d)] = True
            # continuous
            cm = mask & is_cont
            if cm.any():
                y[cm] += rng.normal(0.0, 1.0, cm.sum()) * s * width[cm]
            # ordinal / binary
            om = mask & is_ord
            for i in np.where(om)[0]:
                if nv_int[i] <= 1:
                    continue
                if rng.random() < 0.6:
                    step = 1 if rng.random() < 0.5 else -1
                    ni = y[i] + step
                    if ni < 0 or ni > nv_int[i] - 1:
                        ni = y[i] - step
                    y[i] = ni
                else:
                    y[i] = rng.integers(0, nv_int[i])
            # categorical
            cam = mask & is_cat
            for i in np.where(cam)[0]:
                if nv_int[i] > 1:
                    v = rng.integers(0, nv_int[i] - 1)
                    y[i] = v if v < y[i] else v + 1
            return repair(y[None, :])[0]

        def crossover(a, b):
            m = rng.random(d) < 0.5
            return np.where(m, a, b)

        def tournament(Y, k=2):
            idx = rng.integers(0, len(Y), size=k)
            return idx[np.argmin(Y[idx])]

        # ---------- initial population ----------
        mu = int(min(30, max(8, d)))
        lam = 2 * mu
        n0 = min(budget, max(4 * d, 2 * mu))
        X0 = random_points(n0)
        Y0 = evaluate(X0)
        order = np.argsort(Y0)
        Xp = X0[order[:mu]]
        Yp = Y0[order[:mu]]
        Sp = np.full(len(Xp), 0.2)

        last_improve_gen = 0
        gen = 0
        ref_best = best_y

        # ---------- main loop ----------
        while evals < budget:
            gen += 1
            n_off = min(lam, budget - evals)
            children = np.empty((n_off, d))
            child_s = np.empty(n_off)
            for k in range(n_off):
                p1 = tournament(Yp)
                p2 = tournament(Yp)
                c = crossover(Xp[p1], Xp[p2])
                s = np.exp(0.5 * (np.log(Sp[p1]) + np.log(Sp[p2]))) if False else 0.5 * (Sp[p1] + Sp[p2])
                s = float(np.clip(s * np.exp(0.3 * rng.normal()), 0.005, 0.5))
                child = mutate(c, s)
                tries = 0
                while key(child) in cache and tries < 5:
                    child = mutate(c, s)
                    tries += 1
                children[k] = child
                child_s[k] = s

            Yc = evaluate(children)

            # survival: (mu + lambda)
            allX = np.vstack([Xp, children])
            allY = np.concatenate([Yp, Yc])
            allS = np.concatenate([Sp, child_s])
            order = np.argsort(allY, kind="stable")[:mu]
            Xp, Yp, Sp = allX[order], allY[order], allS[order]

            # stagnation tracking
            if best_y < ref_best - 1e-10 * max(1.0, abs(ref_best)):
                ref_best = best_y
                last_improve_gen = gen

            # partial restart on stagnation
            if gen - last_improve_gen > 40 and evals < budget:
                n_new = min(mu - 1, budget - evals)
                if n_new > 0:
                    Xn = random_points(n_new)
                    Yn = evaluate(Xn)
                    Xp = np.vstack([best_x[None, :], Xn])
                    Yp = np.concatenate([[best_y], Yn])
                    Sp = np.full(len(Xp), 0.2)
                    order = np.argsort(Yp, kind="stable")
                    Xp, Yp, Sp = Xp[order], Yp[order], Sp[order]
                last_improve_gen = gen
                ref_best = best_y

        return best_y, best_x