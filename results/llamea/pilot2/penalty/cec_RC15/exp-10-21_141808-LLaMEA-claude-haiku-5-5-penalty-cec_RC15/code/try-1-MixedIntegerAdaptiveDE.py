import numpy as np


class MixedIntegerAdaptiveDE:
    """
    Self-adaptive DE for mixed-integer black-box problems.
    - Continuous variables: jDE-adapted DE/current-to-best/1 with binomial crossover.
    - Discrete variables: donor recombination (best or random parent) followed by
      type-specific mutation (ordinal: +-1 steps, categorical: resample label,
      binary: flip).
    - Batched evaluation to reduce overhead; strict budget accounting.
    - Stagnation-triggered restarts: half the new population is seeded around the
      global incumbent (exploitation), half is random (exploration).
    """

    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        rng = np.random.default_rng()
        d = self.dim
        budget = self.budget

        vt = [str(t) for t in func.variable_types]
        cont = np.array([t == "continuous" for t in vt])
        ints = ~cont
        binm = np.array([t == "binary" for t in vt])
        catm = np.array([t == "categorical" for t in vt])
        ordm = ints & ~binm & ~catm

        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        nv = np.ones(d)
        if getattr(func, "n_values", None) is not None:
            raw = np.asarray(func.n_values, dtype=float)
            nv[ints] = raw[ints]
        lo = np.where(cont, lb, 0.0)
        hi = np.where(cont, ub, nv - 1.0)
        n_int = int(ints.sum())
        nv_int = nv[ints]

        npop = int(min(max(12, 6 * d), 80))
        patience = int(max(10, 3 * d))

        used = 0
        best_y = np.inf
        best_x = None

        def repair(X):
            X = np.clip(X, lo, hi)
            X[:, ints] = np.clip(np.round(X[:, ints]), 0, nv_int - 1)
            return X

        def random_pop(n):
            X = rng.uniform(lo, hi, size=(n, d))
            if n_int > 0:
                X[:, ints] = np.minimum(
                    np.floor(rng.random((n, n_int)) * nv_int), nv_int - 1
                )
            return X

        def evaluate(X):
            nonlocal used, best_y, best_x
            rem = budget - used
            if rem <= 0 or len(X) == 0:
                return None, None
            X = X[:rem]
            y = np.asarray(func(X), dtype=float).reshape(-1)
            used += len(X)
            i = int(np.argmin(y))
            if y[i] < best_y:
                best_y = float(y[i])
                best_x = X[i].copy()
            return X, y

        def pick_others(n, exclude):
            r = rng.integers(0, n - 1, size=n)
            r = r + (r >= exclude)
            return r

        while used < budget:
            n = min(npop, budget - used)
            if n <= 0:
                break

            # Initial population: half random, half seeded near global best on restarts
            if best_x is None:
                X = random_pop(n)
            else:
                n_seed = n // 2
                Xs = np.repeat(best_x[None, :], n_seed, axis=0)
                # continuous perturbation
                Xs[:, cont] += rng.normal(0, 0.15, size=(n_seed, int(cont.sum())))
                # discrete perturbation
                flip = rng.random((n_seed, d)) < 0.2
                flip &= ints[None, :]
                Xs[flip] = np.nan
                rnd = random_pop(n_seed)
                Xs[np.isnan(Xs)] = rnd[np.isnan(Xs)]
                X = np.vstack([Xs, random_pop(n - n_seed)])
            X = repair(X)
            X, Y = evaluate(X)
            if X is None:
                break
            n = len(X)

            F = np.full(n, 0.5)
            CR = np.full(n, 0.9)
            stall = 0
            gen_best = Y.min()

            while used < budget:
                n = len(X)
                gi = int(np.argmin(Y))
                xb = X[gi]
                idx = np.arange(n)

                # jDE parameter regeneration
                Ft = np.where(rng.random(n) < 0.1, 0.1 + 0.9 * rng.random(n), F)
                CRt = np.where(rng.random(n) < 0.1, rng.random(n), CR)

                r1 = pick_others(n, idx)
                r2 = pick_others(n, idx)
                r3 = pick_others(n, idx)

                # Continuous: current-to-best/1
                mutant = X + Ft[:, None] * (xb[None, :] - X) + \
                    Ft[:, None] * (X[r1] - X[r2])

                # Crossover mask with guaranteed at least one component
                cmask = rng.random((n, d)) < CRt[:, None]
                jrand = rng.integers(0, d, size=n)
                cmask[idx, jrand] = True

                trial = X.copy()
                trial[cmask & cont] = mutant[cmask & cont]

                # Discrete: recombine with best or random donor
                if n_int > 0:
                    use_best = rng.random((n, d)) < 0.5
                    donor = np.where(use_best, xb[None, :], X[r3])
                    dmask = cmask & ints
                    trial[dmask] = donor[dmask]

                    # Type-specific mutation
                    pm = 1.0 / d + 0.05
                    mut = (rng.random((n, d)) < pm) & ints
                    # binary: flip
                    mb = mut & binm
                    trial[mb] = 1.0 - trial[mb]
                    # ordinal: local +-1 step (with occasional jump)
                    mo = mut & ordm
                    step = np.where(rng.random((n, d)) < 0.5, 1.0, -1.0)
                    jump = rng.random((n, d)) < 0.2
                    rand_lbl = np.floor(rng.random((n, d)) * nv[None, :])
                    newo = np.where(jump, rand_lbl, trial + step)
                    trial[mo] = newo[mo]
                    # categorical: resample label
                    mc = mut & catm
                    trial[mc] = rand_lbl[mc]

                trial = repair(trial)
                Xt, Yt = evaluate(trial)
                if Xt is None:
                    break
                m = len(Xt)
                sel = Yt <= Y[:m]
                X[:m][sel] = Xt[sel]
                Y[:m][sel] = Yt[sel]
                F[:m][sel] = Ft[:m][sel]
                CR[:m][sel] = CRt[:m][sel]

                cur = Y.min()
                if cur < gen_best - 1e-12 * max(1.0, abs(gen_best)):
                    gen_best = cur
                    stall = 0
                else:
                    stall += 1
                if stall >= patience:
                    break  # restart

                if used >= budget:
                    break

        return best_y, best_x