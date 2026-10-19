"""Encoded search space shown to the algorithms, and its decoding to native variables.

Continuous variables:  encoded in [-1, 1], decoded  x = lb + (z + 1) / 2 * (ub - lb).
Integer variables (ordinal / binary / categorical) with n values:
    encoded bounds [-0.5, n - 0.5], decoded  k = clip(rint(z), 0, n - 1),  x = lb + k.
    * an exact integer k in {0, ..., n-1} decodes to itself (integer-aware algorithms);
    * uniform sampling over the bounds gives every value the same probability (each value
      owns an interval of width 1). Using bounds [0, n-1] instead would give the two end
      values half the probability of the others; mapping [-1, 1] with round() reaches only
      a few values (the LDG-EA/TFG v5 bug).
"""
from __future__ import annotations

import numpy as np

INTEGER_KINDS = ("ordinal", "binary", "categorical")


class EncodedSpace:
    def __init__(self, variables):
        self.variables = list(variables)
        self.dim = len(self.variables)
        self.kinds = [v.kind for v in self.variables]
        self.continuous_indices = np.array([i for i, v in enumerate(self.variables) if not v.is_integer], int)
        self.integer_indices = np.array([i for i, v in enumerate(self.variables) if v.is_integer], int)
        self.n_values = np.array([v.n_values for v in self.variables], int)  # 0 for continuous
        self._nat_lb = np.array([v.lb for v in self.variables], float)
        self._nat_ub = np.array([v.ub for v in self.variables], float)
        lb = np.full(self.dim, -1.0)
        ub = np.full(self.dim, 1.0)
        lb[self.integer_indices] = -0.5
        ub[self.integer_indices] = self.n_values[self.integer_indices] - 0.5
        self.lb, self.ub = lb, ub
        for v in self.variables:
            if v.is_integer:
                assert v.kind in INTEGER_KINDS, v.kind
                assert v.n_values >= 2, v
                assert float(v.lb).is_integer() and float(v.ub).is_integer(), v
            else:
                assert v.ub > v.lb, v

    def check(self, Z) -> np.ndarray:
        """Validate shape/finiteness of an algorithm-provided point or batch; returns 2-D array."""
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[None, :]
        if Z.ndim != 2 or Z.shape[1] != self.dim:
            raise ValueError(f"func expects a vector of length {self.dim} (or an (n, {self.dim}) batch), "
                             f"got shape {np.shape(Z)}")
        if not np.all(np.isfinite(Z)):
            raise ValueError("func received a point with NaN or inf coordinates")
        return Z

    def decode(self, Z) -> np.ndarray:
        """Encoded (n, d) -> native (n, d). Out-of-bounds coordinates are clipped."""
        Z = self.check(Z)
        X = np.empty_like(Z)
        c, i = self.continuous_indices, self.integer_indices
        if c.size:
            zc = np.clip(Z[:, c], -1.0, 1.0)
            X[:, c] = self._nat_lb[c] + (zc + 1.0) / 2.0 * (self._nat_ub[c] - self._nat_lb[c])
        if i.size:
            k = np.clip(np.rint(Z[:, i]), 0, self.n_values[i] - 1)
            X[:, i] = self._nat_lb[i] + k
        return X

    def encode(self, X) -> np.ndarray:
        """Native (n, d) -> encoded (n, d); integers map to their index k."""
        X = np.atleast_2d(np.asarray(X, float))
        Z = np.empty_like(X)
        c, i = self.continuous_indices, self.integer_indices
        if c.size:
            Z[:, c] = 2.0 * (X[:, c] - self._nat_lb[c]) / (self._nat_ub[c] - self._nat_lb[c]) - 1.0
        if i.size:
            Z[:, i] = np.rint(X[:, i] - self._nat_lb[i])
        return Z

    def describe(self) -> dict:
        """What the algorithm is told about the space (identical in every arm)."""
        return {
            "dim": self.dim,
            "lb": self.lb.tolist(),
            "ub": self.ub.tolist(),
            "variable_types": list(self.kinds),
            "n_values": self.n_values.tolist(),
        }
