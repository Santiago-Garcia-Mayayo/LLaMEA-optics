"""COLSHADE (Gurrola-Ramos, Hernandez-Aguirre & Dalmau-Cedeno, CEC 2020): 2nd of the CEC 2020
real-world constrained competition, and the best-ranked entry that is a black-box method
(1st SASS and 3rd sCMAgES repair infeasible points with problem-specific analytic Jacobians).

Port of the authors' MATLAB code (colshade.m and helpers, competition submission E-24586):
L-SHADE with two mutations chosen per individual with an adaptive probability
(current-to-pbest/1 with archive, and a Levy-flight move towards a p-best), separate
success-history memories for each, Deb's feasibility tournament that always preserves the
best individual, linear population size reduction. The dynamic tolerance of the original only
concerns equality constraints, which our problems do not have, so the sum of violations is
used directly (svc = sum(g)). Parameters as in the submission (18*d initial population, archive
rate 2.6, p-best 0.11, H = 6, initial Levy probability 0.5, momentum 0.25); the only change is
the small-budget cap of the initial population we also apply to L-SHADE (budget // 25), because
the competition budgets (1e5-1e6) are 15-100x ours, and the last generation is evaluated only
up to the budget (the original stops when a full generation no longer fits), so that every
baseline uses exactly the same number of evaluations.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import levy


class COLSHADE:
    R_INIT, R_ARC, MIN_PS, P_BEST, H = 18, 2.6, 4, 0.11, 6
    MOMENTUM, PROB_MIN, PROB_LEVY0 = 0.25, 1e-3, 0.5

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def _eval(self, func, X):
        f, g = func(X)
        f = np.atleast_1d(np.asarray(f, float))
        svc = np.atleast_2d(np.asarray(g, float)).sum(axis=1)
        return f, svc

    @staticmethod
    def _sort(X, f, svc):
        """By total violation; feasible ones by objective (sort_pop.m)."""
        idx = np.lexsort((np.where(svc == 0, f, 0.0), svc))
        return X[idx], f[idx], svc[idx]

    def __call__(self, func):
        rng = np.random.default_rng(np.random.randint(2**31))
        lb, ub = np.asarray(func.bounds.lb, float), np.asarray(func.bounds.ub, float)
        D, max_fes = self.dim, self.budget
        max_ps = int(np.ceil(self.R_INIT * D))
        max_ps = max(self.MIN_PS, min(max_ps, max_fes // 25))
        ps = max_ps
        H = self.H
        M_F, M_CR, M_Fl, M_CRl = (np.full(H, 0.5) for _ in range(4))
        k = k_l = 0
        prob_levy = self.PROB_LEVY0
        X = lb + rng.random((ps, D)) * (ub - lb)
        f, svc = self._eval(func, X)
        fes = ps
        X, f, svc = self._sort(X, f, svc)
        arch = np.empty((0, D))
        while fes < max_fes:
            # ---- indices (get_index.m)
            as_ = len(arch)
            pbest = rng.integers(int(np.ceil(self.P_BEST * ps)), size=ps)
            r1 = np.empty(ps, int); r2 = np.empty(ps, int)
            for i in range(ps):
                while True:
                    r1[i] = rng.integers(ps)
                    if r1[i] != i:
                        break
                while True:
                    r2[i] = rng.integers(ps + as_)
                    if r2[i] != i and r2[i] != r1[i]:
                        break
            # ---- parameters (get_cx_params.m)
            use_pbest = rng.random(ps) > prob_levy
            r = rng.integers(H, size=ps)
            m_cr = np.where(use_pbest, M_CR[r], M_CRl[r])
            m_f = np.where(use_pbest, M_F[r], M_Fl[r])
            CR = np.where(m_cr == -1, 0.0, m_cr + rng.normal(0, 0.1, ps))
            F = np.zeros(ps)
            for i in range(ps):
                crit = np.sqrt((1 - CR[i] / 2) / ps) if CR[i] < 2 else 0.0
                while F[i] <= crit:
                    F[i] = m_f[i] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))   # Cauchy(0, 0.1)
            CR = np.clip(CR, 0, 1)
            F = np.minimum(F, 1.0)
            # ---- mutation and crossover (crossover.m)
            pool = np.vstack([X, arch]) if as_ else X
            Xp = X[pbest]
            V = np.where(use_pbest[:, None],
                         X + F[:, None] * (Xp - X) + F[:, None] * (X[r1] - pool[r2]),
                         X + F[:, None] * levy.rvs(loc=0.01, scale=0.01, size=(ps, D), random_state=rng) * (Xp - X))
            base = 0.1 * rng.random((ps, D))
            V = np.where(V > ub, (1 - base) * ub + base * X, V)
            V = np.where(V < lb, (1 - base) * lb + base * X, V)
            cross = rng.random((ps, D))
            cross[np.arange(ps), rng.integers(D, size=ps)] = 0.0
            U = np.where(cross <= CR[:, None], V, X)
            n = min(ps, max_fes - fes)          # last generation: only what the budget allows
            f_u, svc_u = np.full(ps, np.inf), np.full(ps, np.inf)
            f_u[:n], svc_u[:n] = self._eval(func, U[:n])
            fes += n
            # ---- Deb tournament (deb_tournament.m); index 0 = best, kept or improved
            d_f = np.where((svc_u == 0) & (svc == 0) & (f_u < f), f - f_u, 0.0)
            d_svc = np.where(svc_u < svc, svc - svc_u, 0.0)
            d_f = np.where(d_svc > 0, 0.0, d_f)
            if d_f.max() > 0:
                d_f = d_f / d_f.max()
            if d_svc.max() > 0:
                d_svc = d_svc / d_svc.max()
            delta = d_f + d_svc
            win = delta > 0
            if win.any():
                arch = np.vstack([arch, X[win]]) if as_ else X[win].copy()
                X = np.where(win[:, None], U, X); f = np.where(win, f_u, f); svc = np.where(win, svc_u, svc)
            # ---- memories (update_memory.m)
            for flag, MFm, MCRm, kk in ((True, M_F, M_CR, k), (False, M_Fl, M_CRl, k_l)):
                sel = win & (use_pbest == flag)
                if sel.any():
                    w = delta[sel] / delta[sel].sum()
                    scr, sf = CR[sel], F[sel]
                    MCRm[kk] = -1.0 if (MCRm[kk] == -1 or scr.max() == 0) else np.sum(w * scr ** 2) / np.sum(w * scr)
                    MFm[kk] = np.sum(w * sf ** 2) / np.sum(w * sf)
            dp, dl = delta * use_pbest, delta * ~use_pbest
            if dp.any():
                k = (k + 1) % H
            if dl.any():
                k_l = (k_l + 1) % H
            if (dp > 0).any() or (dl > 0).any():
                prob_levy = self.MOMENTUM * prob_levy + (1 - self.MOMENTUM) * dl.sum() / (dl.sum() + dp.sum())
                prob_levy = min(max(prob_levy, self.PROB_MIN), 1 - self.PROB_MIN)
            X, f, svc = self._sort(X, f, svc)
            # ---- population and archive size
            ps = int(round(self.MIN_PS + (1 - fes / max_fes) * (max_ps - self.MIN_PS)))
            X, f, svc = X[:ps], f[:ps], svc[:ps]
            max_as = int(round(ps * self.R_ARC))
            if len(arch) > max_as:
                arch = arch[rng.permutation(len(arch))[:max_as]]
        i = 0
        return f[i], X[i]