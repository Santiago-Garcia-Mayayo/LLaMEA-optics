import numpy as np


class AdaptiveMixedEvolution:
    """
    Self-adaptive (mu+lambda) evolution strategy for mixed-integer black-box problems.
    - Continuous genes: BLX-style blend crossover + Gaussian mutation with per-individual,
      log-normally self-adapted step sizes (occasional heavy-tailed Cauchy jumps).
    - Discrete genes (ordinal/binary/categorical): uniform crossover; mutation uses
      +-1 steps for ordinal genes and uniform re-sampling for categorical/binary genes.
    - Mutation rate for discrete genes is adapted from the offspring success rate.
    - A small fraction of offspring are fresh random samples (exploration).
    - Partial restart (keeping the best individual) when the best value stagnates.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lam = max(12, 4 * self.dim)
        self.mu = max(4, self.lam // 3)

    def __call__(self, func):
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        d = self.dim
        budget = self.budget

        lb = np.asarray(func.bounds.lb, dtype=float).copy()
        ub = np.asarray(func.bounds.ub, dtype=float).copy()
        vt = [str(v).lower() for v in func.variable_types]
        disc = np.array([v != "continuous" for v in vt], dtype=bool)
        cont = ~disc
        ordinal = np.array([v == "ordinal" for v in vt], dtype=bool)

        nvals = np.ones(d)
        if disc.any():
            nraw = np.asarray(func.n_values, dtype=float)
            nvals[disc] = nraw[disc]

        span = ub - lb
        d_idx = np.where(disc)[0]
        c_idx = np.where(cont)[0]
        nd, nc = len(d_idx), len(c_idx)
        tau = 1.0 / np.sqrt(max(nc, 1))
        pm_base = 1.0 / max(d, 1)
        pm_min, pm_max = 0.5 / max(d, 1), 0.5
        init_sigma = 0.2

        best_y = np.inf
        best_x = None
        used = 0

        def evaluate(X):
            nonlocal used, best_y, best_x
            rem = budget - used
            if rem <= 0 or len(X) == 0:
                return np.empty((0, d)), np.empty(0)
            X = X[:rem]
            y = np.asarray(func(X), dtype=float).ravel()
            y = np.where(np.isnan(y), np.inf, y)
            used += len(X)
            i = int(np.argmin(y))
            if y[i] < best_y:
                best_y = float(y[i])
                best_x = X[i].copy()
            return X, y

        def repair(X):
            if nc:
                X[:, c_idx] = np.clip(X[:, c_idx], lb[c_idx], ub[c_idx])
            if nd:
                X[:, d_idx] = np.clip(np.rint(X[:, d_idx]), 0, nvals[d_idx] - 1)
            return X

        def sample(n):
            X = np.empty((n, d))
            if nc:
                X[:, c_idx] = rng.uniform(lb[c_idx], ub[c_idx], size=(n, nc))
            if nd:
                X[:, d_idx] = np.floor(rng.uniform(size=(n, nd)) * nvals[d_idx])
            return X

        def mutate_discrete(C, pm):
            for j in d_idx:
                m = rng.random(len(C)) < pm
                k = int(m.sum())
                if k == 0:
                    continue
                if ordinal[j] and rng.random() < 0.7:
                    step = rng.choice([-1.0, 1.0], size=k)
                    C[m, j] = np.clip(C[m, j] + step, 0, nvals[j] - 1)
                else:
                    C[m, j] = np.floor(rng.uniform(size=k) * nvals[j])
            return C

        # ---- initial population ----
        n0 = min(max(2 * self.lam, 20), max(budget // 4, self.lam))
        n0 = max(1, min(n0, budget))
        X, y = evaluate(sample(n0))
        if len(X) == 0:
            return best_y, best_x

        order = np.argsort(y)[: self.mu]
        P, Py = X[order], y[order]
        S = np.full(len(P), init_sigma)
        pm = pm_base
        prev_best = best_y
        last_improve = 0
        stall_limit = 12
        gen = 0

        # ---- main loop ----
        while used < budget:
            gen += 1
            lam = self.lam
            n_pop = len(P)

            # binary tournament (P is sorted ascending, so min index = better)
            pa = rng.integers(0, n_pop, (lam, 2)).min(axis=1)
            pb = rng.integers(0, n_pop, (lam, 2)).min(axis=1)
            A, B = P[pa], P[pb]

            child = np.empty((lam, d))
            if nc:
                r = rng.uniform(-0.25, 1.25, size=(lam, 1))
                child[:, c_idx] = A[:, c_idx] + r * (B[:, c_idx] - A[:, c_idx])
            if nd:
                mask = rng.random((lam, nd)) < 0.5
                child[:, d_idx] = np.where(mask, A[:, d_idx], B[:, d_idx])

            # self-adaptive step sizes (fractions of the variable span)
            sig = np.sqrt(S[pa] * S[pb]) * np.exp(tau * rng.standard_normal(lam))
            sig = np.clip(sig, 1e-3, 0.5)

            if nc:
                z = rng.standard_normal((lam, nc))
                heavy = rng.random(lam) < 0.1
                if heavy.any():
                    z[heavy] = np.clip(rng.standard_cauchy((int(heavy.sum()), nc)), -10, 10)
                child[:, c_idx] += sig[:, None] * span[c_idx] * z

            if nd:
                child = mutate_discrete(child, pm)

            # exploration: a few fresh random samples
            ex = rng.random(lam) < 0.03
            if ex.any():
                child[ex] = sample(int(ex.sum()))

            child = repair(child)
            C, Cy = evaluate(child)
            m = len(C)
            if m == 0:
                break

            # adapt discrete mutation rate from success rate
            succ = Cy < Py[pa[:m]]
            rate = succ.mean()
            pm = float(np.clip(pm * (1.25 if rate > 0.2 else 0.85), pm_min, pm_max))

            # (mu+lambda) survivor selection
            allX = np.vstack([P, C])
            allY = np.concatenate([Py, Cy])
            allS = np.concatenate([S, sig[:m]])
            order = np.argsort(allY, kind="stable")[: self.mu]
            P, Py, S = allX[order], allY[order], allS[order]

            # stagnation tracking and partial restart
            if best_y < prev_best - 1e-12:
                prev_best = best_y
                last_improve = gen
            elif gen - last_improve > stall_limit and used < budget:
                n_new = self.mu - 1
                Xn, yn = evaluate(sample(n_new))
                if len(Xn) > 0:
                    P = np.vstack([P[:1], Xn])
                    Py = np.concatenate([Py[:1], yn])
                    S = np.concatenate([S[:1], np.full(len(Xn), init_sigma)])
                    order = np.argsort(Py, kind="stable")
                    P, Py, S = P[order], Py[order], S[order]
                pm = pm_base
                last_improve = gen
                prev_best = best_y

        return best_y, best_x