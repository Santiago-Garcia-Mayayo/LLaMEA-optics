import numpy as np


class EliteGuidedMixedES:
    """
    Elite-guided mixed-integer evolution strategy.
    - Continuous variables: arithmetic crossover + log-normal self-adaptive Gaussian mutation,
      global step size tuned with a 1/5-success-style rule.
    - Discrete variables (binary/ordinal/categorical): per-variable probability tables learned
      from the elite set (smoothed toward uniform), used to resample mutated entries. This
      treats categorical labels as unordered and avoids assuming neighbouring indices are similar.
    - Penalised objective is used directly: infeasible points carry penalties, so ranking by y
      favours feasible, good points.
    - Restarts from fresh uniform samples when the best value stagnates.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        d = self.dim
        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        vtypes = list(func.variable_types)
        is_cont = np.array([vt == "continuous" for vt in vtypes], dtype=bool)
        cont_idx = np.where(is_cont)[0]
        disc_idx = np.where(~is_cont)[0]
        nv = np.ones(d, dtype=int)
        for i in disc_idx:
            nv[i] = int(func.n_values[i])
        clb, cub = lb[cont_idx], ub[cont_idx]
        crange = cub - clb

        evals = 0
        best_y, best_x = np.inf, None

        def evaluate(X):
            nonlocal evals, best_y, best_x
            rem = self.budget - evals
            if rem <= 0 or len(X) == 0:
                return None
            X = X[:rem]
            Y = np.asarray(func(X), dtype=float).reshape(-1)
            evals += len(X)
            k = int(np.argmin(Y))
            if Y[k] < best_y:
                best_y = float(Y[k])
                best_x = X[k].copy()
            return Y

        def repair(X):
            if cont_idx.size:
                X[:, cont_idx] = np.clip(X[:, cont_idx], clb, cub)
            if disc_idx.size:
                X[:, disc_idx] = np.clip(np.rint(X[:, disc_idx]), 0, nv[disc_idx] - 1)
            return X

        def sample(n):
            X = np.zeros((n, d))
            if cont_idx.size:
                X[:, cont_idx] = rng.uniform(clb, cub, size=(n, cont_idx.size))
            for i in disc_idx:
                X[:, i] = rng.integers(0, nv[i], size=n)
            return X

        def reset_tables():
            return {i: np.full(nv[i], 1.0 / nv[i]) for i in disc_idx}

        def update_tables(P, E, lr):
            for i in disc_idx:
                counts = np.bincount(E[:, i].astype(int), minlength=nv[i]).astype(float)
                freq = counts / max(counts.sum(), 1.0)
                target = 0.7 * freq + 0.3 / nv[i]          # smoothed toward uniform
                newp = (1 - lr) * P[i] + lr * target
                P[i] = newp / newp.sum()

        pop_size = int(max(16, min(60, 4 * d)))
        mu = max(2, pop_size // 3)
        stall_limit = 12 + d // 2
        pm = float(np.clip(1.5 / max(1, disc_idx.size), 0.1, 0.5))
        w = np.arange(mu, 0, -1, dtype=float)
        w /= w.sum()

        while evals < self.budget:
            # ---- (re)start ----
            X = sample(pop_size)
            Y = evaluate(X)
            if Y is None:
                break
            X = X[:len(Y)]
            P = reset_tables()
            sigma = 0.25
            gen_best = Y.min()
            stall = 0

            while evals < self.budget:
                order = np.argsort(Y)
                E = X[order[:mu]]
                EY = Y[order[:mu]]
                update_tables(P, E, lr=0.3)

                lam = pop_size
                pa = rng.choice(mu, size=lam, p=w)
                pb = rng.integers(0, mu, size=lam)
                A, B = E[pa], E[pb]

                # uniform crossover for discrete, arithmetic blend for continuous
                C = np.where(rng.random((lam, d)) < 0.5, A, B)
                if cont_idx.size:
                    r = rng.random((lam, 1))
                    base = A[:, cont_idx] + r * (B[:, cont_idx] - A[:, cont_idx])
                    s = sigma * np.exp(0.2 * rng.standard_normal((lam, 1)))  # self-adaptive
                    C[:, cont_idx] = base + s * crange * rng.standard_normal((lam, cont_idx.size))

                # discrete mutation from elite-frequency tables
                for i in disc_idx:
                    m = rng.random(lam) < pm
                    k = int(m.sum())
                    if k:
                        C[m, i] = rng.choice(nv[i], size=k, p=P[i])

                C = repair(C)
                Yc = evaluate(C)
                if Yc is None:
                    break
                C = C[:len(Yc)]
                parent_y = EY[pa][:len(Yc)]

                # 1/5-success-style step-size control
                success = float(np.mean(Yc < parent_y))
                sigma = float(np.clip(sigma * np.exp(0.5 * (success - 0.2)), 1e-3, 0.5))

                # (mu + lambda) survivor selection
                allX = np.vstack([X, C])
                allY = np.concatenate([Y, Yc])
                keep = np.argsort(allY)[:pop_size]
                X, Y = allX[keep], allY[keep]

                if Yc.min() < gen_best - 1e-12:
                    gen_best = float(Yc.min())
                    stall = 0
                else:
                    stall += 1
                if stall >= stall_limit:
                    break

        return best_y, best_x