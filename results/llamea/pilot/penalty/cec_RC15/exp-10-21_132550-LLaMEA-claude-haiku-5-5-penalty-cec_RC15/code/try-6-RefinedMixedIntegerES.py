import numpy as np


class RefinedMixedIntegerES:
    """
    Refined (mu+lambda) self-adaptive mixed-integer ES.
    - Latin-hypercube initialisation for continuous genes.
    - Tournament parent selection, uniform crossover, per-individual log-normal step sizes
      multiplied by a global 1/5-success-rule scale.
    - Ordinal genes: mostly +-1 steps; binary/categorical genes: resample to a different label.
    - A fraction of offspring exploit the incumbent with small local perturbations.
    - On stagnation, restart with a doubled population seeded partly by the incumbent (IPOP-style).
    Offspring are evaluated in batches; the budget is never exceeded.
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

        st = {"evals": 0, "best_y": np.inf, "best_x": None}

        def fix(X):
            X = np.clip(np.array(X, dtype=float, copy=True), lb, ub)
            if len(disc_idx):
                X[:, disc_idx] = np.clip(np.round(X[:, disc_idx]), 0, nv[disc_idx] - 1)
            return X

        def evaluate(X):
            X = fix(X)
            y = np.asarray(func(X), dtype=float).reshape(-1)
            y = np.where(np.isfinite(y), y, 1e30)
            st["evals"] += len(X)
            k = int(np.argmin(y))
            if y[k] < st["best_y"]:
                st["best_y"] = float(y[k])
                st["best_x"] = X[k].copy()
            return X, y

        def random_pop(n):
            X = np.empty((n, d))
            if len(cont_idx):
                # Latin hypercube sampling for continuous variables
                for i in cont_idx:
                    perm = rng.permutation(n)
                    u = (perm + rng.random(n)) / n
                    X[:, i] = lb[i] + u * width[i]
            for i in disc_idx:
                X[:, i] = rng.integers(0, int(nv[i]), size=n)
            return X

        def mutate(parents, sig, gscale, pm_scale=1.0):
            n = len(parents)
            C = parents.copy()
            s = sig * np.exp(tau * rng.standard_normal(n)) * gscale
            s = np.clip(s, 1e-5, 0.5)
            if len(cont_idx):
                noise = rng.standard_normal((n, len(cont_idx)))
                C[:, cont_idx] += s[:, None] * noise * width[cont_idx]
                C[:, cont_idx] = np.clip(C[:, cont_idx], lb[cont_idx], ub[cont_idx])
            if len(disc_idx):
                pm = min(0.5, pm_scale * 1.5 / len(disc_idx))
                mask = rng.random((n, len(disc_idx))) < pm
                for r in np.where(~mask.any(axis=1))[0]:
                    mask[r, rng.integers(len(disc_idx))] = True
                for j, i in enumerate(disc_idx):
                    m = mask[:, j]
                    k = int(m.sum())
                    if k == 0:
                        continue
                    cur = C[m, i]
                    if ordinal[i] and rng.random() < 0.7:
                        new = np.clip(cur + rng.choice([-1.0, 1.0], size=k), 0, nv[i] - 1)
                    else:
                        off = rng.integers(1, int(nv[i]), size=k)
                        new = (cur + off) % nv[i]
                    C[m, i] = new
            return C, s

        def tournament(P_len, n, k=2):
            cand = rng.integers(0, P_len, size=(n, k))
            return cand.min(axis=1)  # population is sorted by fitness, lower index = better

        def local_children(xbest, n, sig_loc):
            C = np.repeat(xbest[None, :], n, axis=0)
            if len(cont_idx):
                sc = sig_loc * np.exp(0.3 * rng.standard_normal(n))
                sc = np.clip(sc, 1e-6, 0.1)
                C[:, cont_idx] += sc[:, None] * rng.standard_normal((n, len(cont_idx))) * width[cont_idx]
                C[:, cont_idx] = np.clip(C[:, cont_idx], lb[cont_idx], ub[cont_idx])
            if len(disc_idx):
                for j, i in enumerate(disc_idx):
                    m = rng.random(n) < 0.08
                    k = int(m.sum())
                    if k:
                        off = rng.integers(1, int(nv[i]), size=k)
                        C[m, i] = (C[m, i] + off) % nv[i]
            return C

        base_mu = max(4, min(12, self.dim))
        pop_mult = 1
        restarts = 0
        max_mult = 4

        # ---- outer loop: (restarting) runs ----
        while st["evals"] < budget:
            mu = int(min(40, base_mu * pop_mult))
            lam = 3 * mu
            rem = budget - st["evals"]
            if rem <= 0:
                break

            n0 = int(min(rem, max(mu, 2 * mu)))
            if restarts > 0 and st["best_x"] is not None:
                n_rand = max(1, n0 - 1)
                R = random_pop(n_rand)
                R = np.vstack([st["best_x"][None, :], R])[:n0]
            else:
                R = random_pop(n0)
            R, RY = evaluate(R)
            sigs = np.full(len(R), 0.2)
            o = np.argsort(RY)[:mu]
            P, PY, PS = R[o], RY[o], sigs[o]
            gscale = 1.0
            prev_best = st["best_y"]
            stall = 0
            stall_limit = 15 * lam

            # ---- inner evolution loop ----
            while st["evals"] < budget and stall < stall_limit:
                rem = budget - st["evals"]
                n = int(min(lam, rem))
                if n <= 0:
                    break

                n_loc = int(round(0.15 * n)) if st["best_x"] is not None else 0
                n_loc = min(n_loc, n)
                n_ev = n - n_loc

                children = []
                child_sig = []
                if n_ev > 0:
                    ia = tournament(len(P), n_ev)
                    ib = tournament(len(P), n_ev)
                    cross = rng.random(n_ev) < 0.5
                    cmask = rng.random((n_ev, d)) < 0.5
                    px = np.where(cross[:, None] & cmask, P[ib], P[ia])
                    ps = np.where(cross, 0.5 * (PS[ia] + PS[ib]), PS[ia])
                    C, cs = mutate(px, ps, gscale)
                    imm = rng.random(n_ev) < 0.05
                    if imm.any():
                        C[imm] = random_pop(int(imm.sum()))
                        cs[imm] = 0.2
                    children.append(C)
                    child_sig.append(cs)
                if n_loc > 0:
                    L = local_children(st["best_x"], n_loc, sig_loc=0.02 * gscale)
                    children.append(L)
                    child_sig.append(np.full(n_loc, 0.05))

                C = np.vstack(children)
                cs = np.concatenate(child_sig)
                CX, CY = evaluate(C)

                AX = np.vstack([P, CX])
                AY = np.concatenate([PY, CY])
                AS = np.concatenate([PS, cs])
                keep = np.argsort(AY)[:mu]
                succ = np.mean(keep >= len(P))
                P, PY, PS = AX[keep], AY[keep], AS[keep]

                # 1/5-type success rule on global step scale
                if succ > 0.2:
                    gscale = min(gscale * 1.3, 3.0)
                elif succ < 0.2:
                    gscale = max(gscale * 0.85, 0.05)

                if st["best_y"] < prev_best - 1e-10 * max(1.0, abs(prev_best)):
                    prev_best = st["best_y"]
                    stall = 0
                else:
                    stall += n

            restarts += 1
            pop_mult = min(pop_mult * 2, max_mult)

        return st["best_y"], st["best_x"]