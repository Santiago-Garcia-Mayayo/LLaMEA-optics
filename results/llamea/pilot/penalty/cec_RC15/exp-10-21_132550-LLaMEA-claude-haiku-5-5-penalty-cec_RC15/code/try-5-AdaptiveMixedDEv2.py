import numpy as np


class AdaptiveMixedDEv2:
    """JADE-style mixed-integer differential evolution.

    - Continuous genes: current-to-pbest/1 with external archive, F and CR adapted
      from successful trials (Lehmer mean / arithmetic mean of successes).
    - Integer genes: donor from pbest or random parent, then type-aware mutation
      (ordinal: +/-1 steps; categorical/binary: random relabel).
    - Duplicate cache: repeated points are never re-evaluated.
    - Restarts: half random immigrants, half multi-scale perturbations of the incumbent.
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        d = self.dim
        lb = np.asarray(func.bounds.lb, dtype=float).copy()
        ub = np.asarray(func.bounds.ub, dtype=float).copy()
        vt = [str(t) for t in func.variable_types]
        is_int = np.array([t != "continuous" for t in vt])
        is_ord = np.array([t == "ordinal" for t in vt])
        nv = np.ones(d, dtype=int)
        if is_int.any():
            nv_all = np.asarray(func.n_values, dtype=int)
            nv[is_int] = nv_all[is_int]
        ci = np.where(~is_int)[0]
        ii = np.where(is_int)[0]
        is_int_row = is_int[None, :]
        is_ord_row = is_ord[None, :]
        span = ub - lb

        def repair(X):
            X = np.array(X, dtype=float, copy=True)
            if len(ci):
                X[:, ci] = np.clip(X[:, ci], lb[ci], ub[ci])
            if len(ii):
                X[:, ii] = np.clip(np.rint(X[:, ii]), 0, nv[ii] - 1)
            return X

        def rand_pop(n):
            X = np.empty((n, d))
            if len(ci):
                X[:, ci] = rng.uniform(lb[ci], ub[ci], size=(n, len(ci)))
            if len(ii):
                X[:, ii] = np.floor(rng.random((n, len(ii))) * nv[ii])
            return X

        cache = {}
        state = {"used": 0, "best_y": np.inf, "best_x": None}

        def key(x):
            return tuple(np.round(x, 8))

        def evaluate(X):
            X = repair(X)
            ys = np.full(len(X), np.inf)
            groups = {}
            for j in range(len(X)):
                k = key(X[j])
                if k in cache:
                    ys[j] = cache[k]
                else:
                    groups.setdefault(k, []).append(j)
            items = list(groups.items())
            remaining = self.budget - state["used"]
            items = items[:max(0, remaining)]
            if items:
                Xe = np.array([X[idx[0]] for _, idx in items])
                Ye = np.asarray(func(Xe), dtype=float).reshape(-1)
                state["used"] += len(items)
                for (k, idx), y in zip(items, Ye):
                    cache[k] = y
                    for j in idx:
                        ys[j] = y
                    if y < state["best_y"]:
                        state["best_y"] = y
                        state["best_x"] = X[idx[0]].copy()
            return X, ys

        # ---- initial design ----
        P = max(20, 5 * d)
        n_init = min(self.budget, max(10 * d, 50))
        X0, Y0 = evaluate(rand_pop(n_init))
        order = np.argsort(Y0)[:P]
        pop, pY = X0[order].copy(), Y0[order].copy()
        archive = np.empty((0, d))
        mu_F, mu_CR = 0.5, 0.5
        stall = 0
        p_top = 0.11
        arch_cap = P

        while state["used"] < self.budget:
            prev_best = state["best_y"]
            used_before = state["used"]

            order = np.argsort(pY)
            n_top = max(2, int(p_top * P))

            # JADE parameter sampling
            F = np.clip(mu_F + 0.1 * rng.standard_cauchy(P), 0.1, 1.0)[:, None]
            CR = np.clip(rng.normal(mu_CR, 0.1, P), 0.0, 1.0)[:, None]
            pm = rng.uniform(0.5 / d, 3.0 / d, P)[:, None]

            pb = pop[order[rng.integers(0, n_top, P)]]
            r1 = rng.integers(0, P, P)
            U = np.vstack([pop, archive]) if len(archive) else pop
            r2 = rng.integers(0, len(U), P)
            R1 = pop[r1]
            R2 = U[r2]

            # current-to-pbest/1 donor for continuous genes
            D = pop + F * (pb - pop) + F * (R1 - R2)
            # integer donor: mixture of pbest and random parent
            if len(ii):
                pick_best = rng.random((P, d)) < 0.5
                int_donor = np.where(pick_best, pb, R1)
                D[:, ii] = int_donor[:, ii]

            # binomial crossover with guaranteed donor gene
            mask = rng.random((P, d)) < CR
            mask[np.arange(P), rng.integers(0, d, P)] = True
            T = np.where(mask, D, pop)

            # type-aware integer mutation
            if len(ii):
                mut = (rng.random((P, d)) < pm) & is_int_row
                rnd_val = np.floor(rng.random((P, d)) * nv[None, :])
                step = np.clip(T + rng.choice([-1.0, 1.0], size=(P, d)), 0, nv[None, :] - 1)
                use_step = is_ord_row & (rng.random((P, d)) < 0.7)
                newv = np.where(use_step, step, rnd_val)
                T = np.where(mut, newv, T)

            Tr, Ty = evaluate(T)
            success = Ty < pY
            equal = (Ty == pY)

            # archive replaced parents (only strict improvements)
            if success.any():
                new_arch = pop[success].copy()
                archive = np.vstack([archive, new_arch]) if len(archive) else new_arch
                if len(archive) > arch_cap:
                    keep = rng.choice(len(archive), arch_cap, replace=False)
                    archive = archive[keep]

            # JADE adaptation of success memories
            if success.any():
                Fs = F[success, 0]
                CRs = CR[success, 0]
                lehmer = np.sum(Fs ** 2) / max(np.sum(Fs), 1e-12)
                mu_F = 0.9 * mu_F + 0.1 * lehmer
                mu_CR = 0.9 * mu_CR + 0.1 * np.mean(CRs)

            improve = success | equal
            pop[improve] = Tr[improve]
            pY[improve] = Ty[improve]

            # stagnation bookkeeping
            if state["best_y"] < prev_best - 1e-12:
                stall = 0
            else:
                stall += 1
            if state["used"] == used_before:
                stall += 5

            # restart: immigrants + multi-scale perturbations of incumbent
            if stall > 30 and state["used"] < self.budget and state["best_x"] is not None:
                n_new = P
                half = n_new // 2
                inc = state["best_x"].copy()
                Xn = np.empty((n_new, d))
                Xn[:half] = rand_pop(half)
                m = n_new - half
                loc = np.tile(inc, (m, 1))
                if len(ci):
                    scales = rng.choice([0.01, 0.05, 0.2], size=(m, 1))
                    loc[:, ci] += rng.normal(0, 1.0, (m, len(ci))) * scales * span[ci]
                    loc[:, ci] = np.clip(loc[:, ci], lb[ci], ub[ci])
                if len(ii):
                    flip_p = rng.choice([0.5 / len(ii), 1.0 / len(ii), 3.0 / len(ii)], size=(m, 1))
                    flip = rng.random((m, len(ii))) < flip_p
                    loc[:, ii] = np.where(flip, np.floor(rng.random((m, len(ii))) * nv[ii]), loc[:, ii])
                Xn[half:] = loc
                Xr, Yr = evaluate(Xn)
                pop, pY = Xr.copy(), Yr.copy()
                archive = np.empty((0, d))
                mu_F, mu_CR = 0.5, 0.5
                stall = 0

        return state["best_y"], state["best_x"]