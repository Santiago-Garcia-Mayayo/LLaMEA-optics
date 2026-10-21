import numpy as np


class RefinedMixedEA:
    """
    Refined (mu + lambda) evolution strategy for mixed-integer constrained black-box problems.
    - Cache-aware evaluation: duplicate candidates reuse stored objective values (no budget spent).
    - Continuous: arithmetic crossover, self-adaptive log-normal step sizes multiplied by a
      global 1/5-success-rule scale, occasional heavy-tailed (Cauchy) steps.
    - Ordinal/binary: +-1 neighbourhood steps or random resampling.
    - Categorical: resampling to a different label.
    - Survival: (mu + lambda) elitism on penalised objective values.
    - IPOP-style restarts keeping the incumbent: half the new population is random, half is
      a local cloud around the best, and the offspring count grows on each restart.
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
        cache = {}

        def key(x):
            return tuple(np.round(x, 8))

        def repair(X):
            X = np.array(X, dtype=float, copy=True)
            if is_cont.any():
                X[:, is_cont] = np.clip(X[:, is_cont], lb[is_cont], ub[is_cont])
            if is_disc.any():
                X[:, is_disc] = np.clip(np.round(X[:, is_disc]), 0, nv_int[is_disc] - 1)
            return X

        def random_points(n):
            X = np.empty((n, d))
            if is_cont.any():
                X[:, is_cont] = rng.uniform(lb[is_cont], ub[is_cont], size=(n, int(is_cont.sum())))
            if is_disc.any():
                X[:, is_disc] = rng.integers(0, nv_int[is_disc], size=(n, int(is_disc.sum())))
            return X.astype(float)

        def evaluate(X):
            """Evaluate rows of X; cached rows cost nothing, unique unseen rows are batched."""
            nonlocal evals, best_y, best_x
            X = repair(X)
            n = len(X)
            if n == 0:
                return np.empty(0)
            Y = np.empty(n)
            keys = [key(x) for x in X]
            uniq = {}
            for i, k in enumerate(keys):
                if k in cache:
                    Y[i] = cache[k]
                else:
                    uniq.setdefault(k, []).append(i)
            if uniq:
                ukeys = list(uniq.keys())
                first = [uniq[k][0] for k in ukeys]
                Yu = np.asarray(func(X[first]), dtype=float).reshape(-1)
                evals += len(first)
                for k, y in zip(ukeys, Yu):
                    y = float(y)
                    cache[k] = y
                    idxs = uniq[k]
                    if y < best_y:
                        best_y = y
                        best_x = X[idxs[0]].copy()
                    for i in idxs:
                        Y[i] = y
            return Y

        def mutate(x, s, pm):
            y = x.copy()
            mask = rng.random(d) < pm
            if not mask.any():
                mask[rng.integers(d)] = True
            cm = mask & is_cont
            if cm.any():
                z = rng.normal(0.0, 1.0, int(cm.sum()))
                if rng.random() < 0.1:
                    z = np.clip(rng.standard_cauchy(int(cm.sum())), -5.0, 5.0)
                y[cm] += z * s * width[cm]
            for i in np.where(mask & is_ord)[0]:
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
            for i in np.where(mask & is_cat)[0]:
                if nv_int[i] > 1:
                    v = rng.integers(0, nv_int[i] - 1)
                    y[i] = v if v < y[i] else v + 1
            return y

        def crossover(a, b):
            c = np.empty(d)
            if is_cont.any():
                alpha = rng.random()
                c[is_cont] = alpha * a[is_cont] + (1.0 - alpha) * b[is_cont]
            if is_disc.any():
                m = rng.random(int(is_disc.sum())) < 0.5
                c[is_disc] = np.where(m, a[is_disc], b[is_disc])
            return c

        def tournament(Y, k=2):
            idx = rng.integers(0, len(Y), size=k)
            return idx[np.argmin(Y[idx])]

        # ---------- parameters ----------
        mu = int(min(30, max(8, d)))
        lam_base = 2 * mu
        lam = lam_base
        lam_max = 6 * mu
        pm = float(min(0.5, max(1.0 / d, 0.1)))
        tau = 0.3
        g = 1.0  # global 1/5-rule step multiplier
        stall_limit = 30

        # ---------- initial population ----------
        n0 = min(budget, max(4 * d, 2 * mu))
        X0 = random_points(n0)
        Y0 = evaluate(X0)
        order = np.argsort(Y0, kind="stable")[:mu]
        Xp = repair(X0[order])
        Yp = Y0[order].copy()
        Sp = np.full(len(Xp), 0.2)

        last_improve_gen = 0
        gen = 0
        ref_best = best_y

        # ---------- main loop ----------
        while evals < budget:
            gen += 1
            n_off = int(min(lam, budget - evals))
            if n_off <= 0:
                break
            children = np.empty((n_off, d))
            child_s = np.empty(n_off)
            parent_idx = np.empty(n_off, dtype=int)
            for k in range(n_off):
                p1 = tournament(Yp)
                p2 = tournament(Yp)
                c = crossover(Xp[p1], Xp[p2])
                sbase = 0.5 * (Sp[p1] + Sp[p2]) * np.exp(tau * rng.normal())
                sbase = float(np.clip(sbase, 0.002, 0.5))
                s = float(np.clip(sbase * g, 0.001, 0.5))
                child = mutate(c, s, pm)
                children[k] = child
                child_s[k] = sbase
                parent_idx[k] = p1

            children = repair(children)
            Yc = evaluate(children)

            # (mu + lambda) survival
            allX = np.vstack([Xp, children])
            allY = np.concatenate([Yp, Yc])
            allS = np.concatenate([Sp, child_s])
            sel = np.argsort(allY, kind="stable")[:mu]
            n_success = int(np.sum(sel >= len(Xp)))
            Xp, Yp, Sp = allX[sel], allY[sel], allS[sel]

            # 1/5 success rule on global scale
            rate = n_success / max(1, n_off)
            if rate > 0.2:
                g *= 1.22
            else:
                g *= 0.82
            g = float(np.clip(g, 0.02, 3.0))

            # stagnation tracking
            if best_y < ref_best - 1e-10 * max(1.0, abs(ref_best)):
                ref_best = best_y
                last_improve_gen = gen

            # IPOP-style restart on stagnation
            if gen - last_improve_gen > stall_limit and evals < budget and best_x is not None:
                lam = min(lam_max, int(lam * 1.5))
                n_new = int(min(mu - 1, budget - evals))
                if n_new > 0:
                    n_rand = n_new // 2
                    n_loc = n_new - n_rand
                    parts = []
                    if n_rand > 0:
                        parts.append(random_points(n_rand))
                    if n_loc > 0:
                        Xl = np.empty((n_loc, d))
                        for j in range(n_loc):
                            Xl[j] = mutate(best_x, 0.15, max(pm, 0.3))
                        parts.append(repair(Xl))
                    Xn = np.vstack(parts)
                    Yn = evaluate(Xn)
                    Xp = np.vstack([best_x[None, :], Xn])
                    Yp = np.concatenate([[best_y], Yn])
                    Sp = np.full(len(Xp), 0.2)
                    order = np.argsort(Yp, kind="stable")
                    Xp, Yp, Sp = Xp[order], Yp[order], Sp[order]
                g = 1.0
                last_improve_gen = gen
                ref_best = best_y

        return best_y, best_x