import numpy as np


class AdaptiveMixedDE:
    """Type-aware differential evolution for mixed-integer black-box problems.

    - Continuous genes: current-to-best/1 DE mutation with binomial crossover.
    - Integer genes: donor from best or random parent, then type-aware mutation
      (ordinal: +/-1 steps; categorical/binary: random relabel).
    - Duplicate cache: repeated points are never re-evaluated (saves budget).
    - Stagnation restarts: half random immigrants, half perturbations of the incumbent.
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
        stall = 0

        while state["used"] < self.budget:
            prev_best = state["best_y"]
            used_before = state["used"]

            order = np.argsort(pY)
            best = pop[order[0]]

            # per-individual adaptive parameters
            F = np.clip(rng.standard_cauchy(P) * 0.1 + 0.6, 0.3, 1.0)[:, None]
            CR = np.clip(rng.normal(0.8, 0.1, P), 0.1, 1.0)[:, None]
            pm = rng.uniform(0.5 / d, 3.0 / d, P)[:, None]

            R1 = pop[rng.integers(0, P, P)]
            R2 = pop[rng.integers(0, P, P)]

            # continuous donor: current-to-best/1
            D = pop + F * (best - pop) + F * (R1 - R2)
            # integer donor: mixture of best and random parent
            if len(ii):
                pick_best = rng.random((P, d)) < 0.5
                int_donor = np.where(pick_best, best, R1)
                D[:, ii] = int_donor[:, ii]

            # binomial crossover with guaranteed one donor gene
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
            improve = Ty <= pY
            pop[improve] = Tr[improve]
            pY[improve] = Ty[improve]

            # stagnation bookkeeping
            if state["best_y"] < prev_best - 1e-12:
                stall = 0
            else:
                stall += 1
            if state["used"] == used_before:
                # nothing new was evaluated (all duplicates): inject diversity
                stall += 5

            # restart with immigrants + local exploration around incumbent
            if stall > 25 and state["used"] < self.budget:
                n_new = P
                half = n_new // 2
                inc = state["best_x"].copy()
                Xn = np.empty((n_new, d))
                Xn[:half] = rand_pop(half)
                loc = np.tile(inc, (n_new - half, 1))
                if len(ci):
                    span = ub[ci] - lb[ci]
                    loc[:, ci] += rng.normal(0, 0.1, (n_new - half, len(ci))) * span
                if len(ii):
                    flip = rng.random((n_new - half, len(ii))) < 1.0 / max(1, len(ii))
                    loc[:, ii] = np.where(flip, np.floor(rng.random((n_new - half, len(ii))) * nv[ii]), loc[:, ii])
                Xn[half:] = loc
                Xr, Yr = evaluate(Xn)
                pop, pY = Xr.copy(), Yr.copy()
                stall = 0

        return state["best_y"], state["best_x"]