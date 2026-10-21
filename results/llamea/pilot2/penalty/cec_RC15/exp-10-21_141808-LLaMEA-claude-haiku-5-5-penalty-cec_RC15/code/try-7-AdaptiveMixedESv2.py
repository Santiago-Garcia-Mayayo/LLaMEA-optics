import numpy as np


class AdaptiveMixedESv2:
    """
    Mixed-integer (mu+lambda) evolution strategy, refined from MixedAdaptiveES.
    - vectorised, type-aware mutation (Gaussian for continuous, flip for binary,
      +/-1 or random jump for ordinal, resampling for categorical)
    - per-individual log-normal step sizes multiplied by a global 1/5-success-rule factor
    - per-child mutation rate drawn from {1,2,4}/dim
    - uniform crossover with probability 0.6, binary tournament selection
    - duplicate avoidance via a seen-set with repeated re-mutation
    - stagnation-triggered restarts that keep the elite and re-seed around the incumbent
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
        nd = int(disc.sum())

        nv = np.ones(dim)
        for i in range(dim):
            if disc[i]:
                nv[i] = int(round(float(func.n_values[i])))

        lo = np.where(cont, lb, 0.0)
        hi = np.where(cont, ub, nv - 1.0)
        span = hi - lo
        span[cont] = ub[cont] - lb[cont]

        state = {"used": 0, "best_y": np.inf, "best_x": None}
        seen = set()

        def repair(X):
            X = np.clip(X, lo, hi)
            if nd > 0:
                X[:, disc] = np.rint(X[:, disc])
            return X

        def sample(n):
            X = np.empty((n, dim))
            if cont.any():
                X[:, cont] = rng.uniform(lb[cont], ub[cont], size=(n, int(cont.sum())))
            if nd > 0:
                X[:, disc] = np.floor(rng.random((n, nd)) * nv[disc][None, :])
            return repair(X)

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

        def mutate_batch(P, sig, pm):
            n = P.shape[0]
            C = P.copy()
            M = rng.random((n, dim)) < pm[:, None]
            empty = ~M.any(axis=1)
            if empty.any():
                idx = np.where(empty)[0]
                M[idx, rng.integers(0, dim, size=idx.size)] = True
            # continuous: Gaussian step scaled by span and sigma
            Cm = M & cont[None, :]
            G = rng.standard_normal((n, dim))
            C = np.where(Cm, C + sig[:, None] * span[None, :] * G, C)
            # binary: flip
            Bm = M & binm[None, :]
            C = np.where(Bm, 1.0 - C, C)
            # ordinal: +/-1 step with occasional random jump
            Om = M & ordm[None, :]
            steps = rng.choice([-1.0, 1.0], size=(n, dim))
            jump = rng.random((n, dim)) < 0.15
            rnd = np.floor(rng.random((n, dim)) * nv[None, :])
            C = np.where(Om, np.where(jump, rnd, C + steps), C)
            # categorical: uniform resample among other labels
            Km = M & cat[None, :]
            rc = np.floor(rng.random((n, dim)) * np.maximum(nv - 1.0, 1.0)[None, :])
            rc = rc + (rc >= C)
            C = np.where(Km, rc, C)
            return repair(C)

        # ----- initialisation -----
        mu = 15
        lam = 60
        n0 = min(budget, max(4 * mu, 10 * dim))
        X = sample(n0)
        F = evaluate(X)
        S = np.full(len(X), 0.2)
        order = np.argsort(F)[:mu]
        X, F, S = X[order], F[order], S[order]

        g = 1.0            # global step multiplier (1/5 success rule)
        tau = 0.25         # log-normal self-adaptation rate
        prev_best = state["best_y"]
        stall = 0
        restart_patience = 25

        # ----- main loop -----
        while state["used"] < budget:
            nb = min(lam, budget - state["used"])
            if nb <= 0:
                break

            # binary tournament parent selection
            t1 = rng.integers(0, len(X), size=(nb, 2))
            t2 = rng.integers(0, len(X), size=(nb, 2))
            p1 = np.where(F[t1[:, 0]] < F[t1[:, 1]], t1[:, 0], t1[:, 1])
            p2 = np.where(F[t2[:, 0]] < F[t2[:, 1]], t2[:, 0], t2[:, 1])

            # uniform crossover (prob 0.6)
            A, B = X[p1], X[p2]
            cross = rng.random(nb) < 0.6
            umask = rng.random((nb, dim)) < 0.5
            base = np.where(cross[:, None], np.where(umask, A, B), A)
            s_par = np.where(cross, 0.5 * (S[p1] + S[p2]), S[p1])

            # self-adapted step sizes with global multiplier
            sig = s_par * np.exp(tau * rng.standard_normal(nb)) * g
            sig = np.clip(sig, 1e-4, 0.5)

            # per-child mutation rate
            pm = rng.choice([1.0, 2.0, 4.0], size=nb) / dim
            pm = np.clip(pm, 1.0 / dim, 0.6)

            children = mutate_batch(base, sig, pm)

            # duplicate avoidance: re-mutate duplicates a few times
            for _ in range(3):
                dup = np.array([i for i in range(nb) if children[i].tobytes() in seen], dtype=int)
                if dup.size == 0:
                    break
                children[dup] = mutate_batch(base[dup], sig[dup], pm[dup])

            Y = evaluate(children)

            # (mu+lambda) elitist survival
            Xall = np.vstack([X, children])
            Fall = np.concatenate([F, Y])
            Sall = np.concatenate([S, sig])
            order = np.argsort(Fall)[:mu]
            survived = int(np.sum(order >= len(X)))
            X, F, S = Xall[order], Fall[order], Sall[order]

            # 1/5 success rule on global step multiplier
            ratio = survived / float(nb)
            g = float(np.clip(g * np.exp(0.5 * (ratio - 0.2)), 0.02, 3.0))

            # stagnation tracking
            if state["best_y"] < prev_best - 1e-12:
                prev_best = state["best_y"]
                stall = 0
            else:
                stall += 1

            # restart around incumbent while keeping elites
            if stall >= restart_patience and state["used"] < budget:
                n_new = min(mu - 2, budget - state["used"])
                if n_new > 0 and state["best_x"] is not None:
                    elite = X[:2].copy()
                    elite_F = F[:2].copy()
                    elite_S = S[:2].copy()
                    n_local = n_new // 2
                    local = np.repeat(state["best_x"][None, :], n_local, axis=0)
                    pm_r = np.full(n_local, 2.0 / dim)
                    sig_r = np.full(n_local, 0.3)
                    local = mutate_batch(local, sig_r, pm_r)
                    rand_part = sample(n_new - n_local)
                    NX = np.vstack([local, rand_part])
                    NF = evaluate(NX)
                    NS = np.full(len(NX), 0.25)
                    X = np.vstack([elite, NX])
                    F = np.concatenate([elite_F, NF])
                    S = np.concatenate([elite_S, NS])
                    order = np.argsort(F)[:mu]
                    X, F, S = X[order], F[order], S[order]
                g = 1.0
                stall = 0

        best_x = state["best_x"]
        if best_x is None:
            best_x = X[0]
        return state["best_y"], np.asarray(best_x)