import numpy as np


class AdaptiveMixedEvolution:
    """
    Refined self-adaptive mixed-integer (mu+lambda) ES.
    - Continuous genes: BLX-alpha crossover, Gaussian mutation with per-individual
      log-normal step sizes scaled by a global 1/5-success-rule factor, plus rare Cauchy jumps.
    - Discrete genes: uniform crossover; ordinal +-1 steps, binary flips, categorical resampling.
      The discrete mutation rate is adapted from offspring success.
    - Stagnation-triggered restarts (IPOP-style growth of lambda); restarts re-seed half of the
      population in a shrinking neighbourhood of the incumbent and half uniformly at random.
    - Elitist survivor selection, so the best feasible point found is never lost.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lam0 = max(12, 4 * self.dim)
        self.lam_max = max(self.lam0, 10 * self.dim)

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
        binary = np.array([v == "binary" for v in vt], dtype=bool)

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

        def sample_near(n, center, rad):
            X = sample(n)
            if nc:
                z = rng.standard_normal((n, nc))
                X[:, c_idx] = center[c_idx] + rad * span[c_idx] * z
            if nd:
                keep = rng.random((n, nd)) < 0.7
                X[:, d_idx] = np.where(keep, center[d_idx], X[:, d_idx])
            return repair(X)

        def mutate_discrete(C, pm):
            for j in d_idx:
                m = rng.random(len(C)) < pm
                k = int(m.sum())
                if k == 0:
                    continue
                if binary[j]:
                    C[m, j] = 1.0 - C[m, j]
                elif ordinal[j] and rng.random() < 0.7:
                    step = rng.choice([-1.0, 1.0], size=k)
                    C[m, j] = np.clip(C[m, j] + step, 0, nvals[j] - 1)
                else:
                    C[m, j] = np.floor(rng.uniform(size=k) * nvals[j])
            return C

        # ---- initial population ----
        lam = self.lam0
        mu = max(4, lam // 3)
        n0 = min(max(2 * lam, 20), max(budget // 4, lam))
        n0 = max(1, min(n0, budget))
        X, y = evaluate(sample(n0))
        if len(X) == 0:
            return best_y, best_x

        order = np.argsort(y)[:mu]
        P, Py = X[order], y[order]
        S = np.full(len(P), init_sigma)
        pm = pm_base
        sc = 1.0                  # global 1/5-rule step scale
        prev_best = best_y
        last_improve = 0
        stall_limit = 10
        gen = 0
        restarts = 0

        # ---- main loop ----
        while used < budget:
            gen += 1
            n_pop = len(P)

            # binary tournament selection (P sorted ascending)
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

            # self-adaptive step sizes scaled by global factor
            sig = np.sqrt(S[pa] * S[pb]) * np.exp(tau * rng.standard_normal(lam))
            sig = np.clip(sig * sc, 1e-4, 0.5)

            if nc:
                z = rng.standard_normal((lam, nc))
                heavy = rng.random(lam) < 0.1
                if heavy.any():
                    z[heavy] = np.clip(rng.standard_cauchy((int(heavy.sum()), nc)), -10, 10)
                child[:, c_idx] += sig[:, None] * span[c_idx] * z

            if nd:
                child = mutate_discrete(child, pm)

            # exploration: fresh random samples
            ex = rng.random(lam) < 0.03
            if ex.any():
                child[ex] = sample(int(ex.sum()))

            child = repair(child)
            C, Cy = evaluate(child)
            m = len(C)
            if m == 0:
                break

            # success-based adaptation of discrete rate and global step scale
            succ = Cy < Py[pa[:m]]
            rate = succ.mean()
            pm = float(np.clip(pm * (1.25 if rate > 0.2 else 0.85), pm_min, pm_max))
            sc = float(np.clip(sc * (1.2 if rate > 0.2 else 0.9), 0.01, 2.0))

            # (mu+lambda) survivor selection
            allX = np.vstack([P, C])
            allY = np.concatenate([Py, Cy])
            allS = np.concatenate([S, sig[:m] / max(sc, 1e-12)])
            order = np.argsort(allY, kind="stable")[:mu]
            P, Py, S = allX[order], allY[order], allS[order]

            # stagnation tracking and restart
            if best_y < prev_best - 1e-12:
                prev_best = best_y
                last_improve = gen
            elif gen - last_improve > stall_limit and used < budget:
                restarts += 1
                # IPOP-style population growth
                lam = min(int(lam * 1.5), self.lam_max)
                mu = max(4, lam // 3)
                n_new = max(mu - 1, 1)
                n_local = n_new // 2
                parts = []
                if n_local > 0 and best_x is not None:
                    rad = 0.1 / (1 + restarts)
                    parts.append(sample_near(n_local, best_x, rad))
                parts.append(sample(n_new - n_local))
                Xn, yn = evaluate(np.vstack(parts))
                if len(Xn) > 0:
                    P = np.vstack([P[:1], Xn])
                    Py = np.concatenate([Py[:1], yn])
                    S = np.concatenate([S[:1], np.full(len(Xn), init_sigma)])
                    order = np.argsort(Py, kind="stable")[:mu]
                    P, Py, S = P[order], Py[order], S[order]
                pm = pm_base
                sc = 1.0
                last_improve = gen
                prev_best = best_y

        return best_y, best_x