import numpy as np


class MixedIntegerSelfAdaptiveES:
    """
    (mu+lambda) self-adaptive evolution strategy for mixed-integer black-box problems.
    - Continuous genes: log-normal self-adapted Gaussian mutation per individual.
    - Ordinal genes: mostly +-1 neighbourhood steps, occasionally uniform resampling.
    - Binary/categorical genes: resampling to a different label (no adjacency assumed).
    - Uniform crossover between parents, random immigrants for exploration,
      and restarts when the search stagnates.
    Offspring are evaluated in batches to use the budget efficiently.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        d = self.dim
        budget = self.budget
        if budget <= 0:
            return np.inf, None

        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        vtypes = [str(t) for t in func.variable_types]
        n_values = getattr(func, "n_values", None)
        if n_values is None:
            n_values = np.ones(d)
        n_values = np.asarray(n_values, dtype=float)

        cont = np.array([t == "continuous" for t in vtypes], dtype=bool)
        cont_idx = np.where(cont)[0]
        disc_idx = np.where(~cont)[0]
        ordinal = np.array([t == "ordinal" for t in vtypes], dtype=bool)
        width = ub - lb
        nv = np.ones(d)
        for i in disc_idx:
            nv[i] = max(2, int(round(n_values[i])))
        n_cont = max(len(cont_idx), 1)
        tau = 1.0 / np.sqrt(2.0 * np.sqrt(n_cont))

        state = {"evals": 0, "best_y": np.inf, "best_x": None}

        def evaluate(X):
            X = np.array(X, dtype=float, copy=True)
            X = np.clip(X, lb, ub)
            if len(disc_idx):
                X[:, disc_idx] = np.clip(np.round(X[:, disc_idx]), 0, nv[disc_idx] - 1)
            y = np.asarray(func(X), dtype=float).reshape(-1)
            state["evals"] += len(X)
            k = int(np.argmin(y))
            if y[k] < state["best_y"]:
                state["best_y"] = float(y[k])
                state["best_x"] = X[k].copy()
            return X, y

        def random_pop(n):
            X = np.empty((n, d))
            if len(cont_idx):
                X[:, cont_idx] = rng.uniform(lb[cont_idx], ub[cont_idx],
                                             size=(n, len(cont_idx)))
            for i in disc_idx:
                X[:, i] = rng.integers(0, int(nv[i]), size=n)
            return X

        def mutate(parents, sig):
            n = len(parents)
            C = parents.copy()
            s = sig * np.exp(tau * rng.standard_normal(n))
            s = np.clip(s, 1e-4, 0.5)
            if len(cont_idx):
                noise = rng.standard_normal((n, len(cont_idx)))
                C[:, cont_idx] += s[:, None] * noise * width[cont_idx]
                C[:, cont_idx] = np.clip(C[:, cont_idx], lb[cont_idx], ub[cont_idx])
            if len(disc_idx):
                pm = min(0.5, 1.5 / len(disc_idx))
                mask = rng.random((n, len(disc_idx))) < pm
                # guarantee at least one discrete mutation per child
                for r in np.where(~mask.any(axis=1))[0]:
                    mask[r, rng.integers(len(disc_idx))] = True
                for j, i in enumerate(disc_idx):
                    m = mask[:, j]
                    k = int(m.sum())
                    if k == 0:
                        continue
                    cur = C[m, i]
                    if ordinal[i] and rng.random() < 0.7:
                        new = cur + rng.choice([-1.0, 1.0], size=k)
                        new = np.clip(new, 0, nv[i] - 1)
                    else:
                        off = rng.integers(1, int(nv[i]), size=k)
                        new = (cur + off) % nv[i]
                    C[m, i] = new
            return C, s

        # ---- initialisation ----
        n0 = min(budget, max(10, 5 * d))
        X0 = random_pop(n0)
        X0, Y0 = evaluate(X0)
        S0 = np.full(n0, 0.2)
        mu = max(4, min(20, n0 // 2))
        lam = 2 * mu
        order = np.argsort(Y0)[:mu]
        P, PY, PS = X0[order], Y0[order], S0[order]

        stall = 0
        prev_best = state["best_y"]
        stall_limit = 12 * lam

        # ---- main loop ----
        while state["evals"] < budget:
            rem = budget - state["evals"]
            n = int(min(lam, rem))
            if n <= 0:
                break

            # rank-weighted parent selection
            w = 1.0 / (np.arange(len(P)) + 1.0)
            w /= w.sum()
            ia = rng.choice(len(P), size=n, p=w)
            ib = rng.choice(len(P), size=n, p=w)

            cross = rng.random(n) < 0.5
            mask = rng.random((n, d)) < 0.5
            px = np.where(cross[:, None] & mask, P[ib], P[ia])
            ps = np.where(cross, 0.5 * (PS[ia] + PS[ib]), PS[ia])

            C, cs = mutate(px, ps)

            # random immigrants for exploration
            imm = rng.random(n) < 0.1
            if imm.any():
                C[imm] = random_pop(int(imm.sum()))
                cs[imm] = 0.2

            CX, CY = evaluate(C)

            # (mu+lambda) survivor selection
            AX = np.vstack([P, CX])
            AY = np.concatenate([PY, CY])
            AS = np.concatenate([PS, cs])
            keep = np.argsort(AY)[:mu]
            P, PY, PS = AX[keep], AY[keep], AS[keep]

            # stagnation detection and restart
            if state["best_y"] < prev_best - 1e-12 * max(1.0, abs(prev_best)):
                prev_best = state["best_y"]
                stall = 0
            else:
                stall += n
            if stall >= stall_limit and state["evals"] < budget:
                rem = budget - state["evals"]
                nr = int(min(mu, rem))
                if nr > 0:
                    R = random_pop(nr)
                    R, RY = evaluate(R)
                    # keep the global incumbent as an elite seed
                    if state["best_x"] is not None:
                        P = np.vstack([state["best_x"][None, :], R])
                        PY = np.concatenate([[state["best_y"]], RY])
                        PS = np.full(len(P), 0.2)
                    else:
                        P, PY, PS = R, RY, np.full(len(R), 0.2)
                    o = np.argsort(PY)[:mu]
                    P, PY, PS = P[o], PY[o], PS[o]
                stall = 0
                prev_best = state["best_y"]

        return state["best_y"], state["best_x"]