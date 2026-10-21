import numpy as np


class MixedIntegerAdaptiveDEv2:
    """
    Refined mixed-integer self-adaptive DE.
    - Continuous: jDE-adapted current-to-pbest/1 with an external archive of replaced parents.
    - Discrete: donor recombination (from pbest or a random parent) followed by
      type-specific mutations (binary flip, ordinal +-1 with jumps, categorical resampling).
    - Batched evaluation, strict budget accounting, elitist selection.
    - Stagnation-triggered restarts with growing population (IPOP-like). Restart population
      is seeded around the global incumbent with multiple perturbation scales plus random points.
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
        n_cont = int(cont.sum())
        nv_int = nv[ints]

        npop_base = int(min(max(12, 6 * d), 80))
        npop = npop_base
        patience = int(max(12, 4 * d))

        used = 0
        best_y = np.inf
        best_x = None

        def repair(X):
            X = np.clip(X, lo, hi)
            if n_int > 0:
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

        restart_count = 0
        while used < budget:
            n = min(npop, budget - used)
            if n <= 0:
                break

            # Initial population
            if best_x is None:
                X = random_pop(n)
            else:
                n_seed = n // 2
                # diverse perturbation scales for seeds
                scales = np.exp(rng.uniform(np.log(0.01), np.log(0.3), size=n_seed))
                Xs = np.repeat(best_x[None, :], n_seed, axis=0)
                if n_cont > 0:
                    Xs[:, cont] += rng.normal(0, 1, size=(n_seed, n_cont)) * scales[:, None]
                if n_int > 0:
                    flip = (rng.random((n_seed, d)) < (0.05 + 0.25 * rng.random((n_seed, 1)))) & ints[None, :]
                    rnd = random_pop(n_seed)
                    Xs = np.where(flip, rnd, Xs)
                X = np.vstack([Xs, random_pop(n - n_seed)])
            X = repair(X)
            X, Y = evaluate(X)
            if X is None:
                break
            n = len(X)

            F = np.full(n, 0.5)
            CR = np.full(n, 0.9)
            arch = np.empty((0, d))
            stall = 0
            gen_best = Y.min()

            while used < budget:
                n = len(X)
                idx = np.arange(n)
                order = np.argsort(Y)
                n_top = max(1, int(0.15 * n))
                pb = order[rng.integers(0, n_top, size=n)]

                # jDE parameter regeneration
                Ft = np.where(rng.random(n) < 0.1, 0.1 + 0.9 * rng.random(n), F)
                CRt = np.where(rng.random(n) < 0.1, rng.random(n), CR)

                pool = np.vstack([X, arch]) if len(arch) > 0 else X
                r1 = pick_others(n, idx)
                r2 = rng.integers(0, len(pool), size=n)
                r3 = pick_others(n, idx)

                # Continuous: current-to-pbest/1 with archive
                mutant = X + Ft[:, None] * (X[pb] - X) + Ft[:, None] * (X[r1] - pool[r2])

                cmask = rng.random((n, d)) < CRt[:, None]
                jrand = rng.integers(0, d, size=n)
                cmask[idx, jrand] = True

                trial = X.copy()
                trial[cmask & cont] = mutant[cmask & cont]

                # Discrete: recombination + type-specific mutation
                if n_int > 0:
                    use_best = rng.random((n, d)) < 0.5
                    donor = np.where(use_best, X[pb], X[r3])
                    dmask = cmask & ints
                    trial[dmask] = donor[dmask]

                    pm = 1.0 / d + 0.05
                    mut = (rng.random((n, d)) < pm) & ints
                    mb = mut & binm
                    trial[mb] = 1.0 - trial[mb]
                    mo = mut & ordm
                    step = np.where(rng.random((n, d)) < 0.5, 1.0, -1.0)
                    jump = rng.random((n, d)) < 0.2
                    rand_lbl = np.floor(rng.random((n, d)) * nv[None, :])
                    newo = np.where(jump, rand_lbl, trial + step)
                    trial[mo] = newo[mo]
                    mc = mut & catm
                    trial[mc] = rand_lbl[mc]

                trial = repair(trial)
                Xt, Yt = evaluate(trial)
                if Xt is None:
                    break
                m = len(Xt)
                sel = Yt <= Y[:m]

                # archive replaced parents
                if sel.any():
                    arch = np.vstack([arch, X[:m][sel].copy()])
                    if len(arch) > npop:
                        keep = rng.choice(len(arch), npop, replace=False)
                        arch = arch[keep]

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
                    break
                if used >= budget:
                    break

            # grow population on restarts (IPOP-like), capped
            restart_count += 1
            npop = int(min(npop_base * (1.2 ** min(restart_count, 10)), 120))

        return best_y, best_x