"""CEC 2020 real-world constrained suite (Kumar et al., 2020) via the C++ port of
V. Stanovov (https://github.com/VladimirStanovov/CEC-2020-RWC-C), linked from the
official repository P-N-Suganthan/2020-RW-Constrained-Optimisation.

Convention used by the official code: integer/binary variables are encoded as
continuous variables with bounds [k-0.51, m+0.49] and rounded INSIDE the function.
We expose them explicitly as integer indices and round them in the harness (sealing),
so both arms see exactly the same thing.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import numpy as np

_LIB_PATH = os.environ.get("CEC2020RWC_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cec2020_cpp", "cec2020rwc.so"))
_lib = None


def _get_lib():
    """Load the compiled C++ library on first use (so the lens works without it)."""
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(
                f"CEC 2020 library not found at {_LIB_PATH}. Compile problems/cec2020_cpp/cec2020rwc.cpp "
                "(see README) or set CEC2020RWC_LIB to the compiled .so/.dll.")
        _lib = ctypes.CDLL(_LIB_PATH)
        _lib.cec20_func.restype = None
        _lib.get_bounds.restype = None
    return _lib

D = [9,11,7,6,9,38,48,2,3,3,7,7,5,10,7,14,3,4,4,2,5,9,5,7,4,22,10,10,4,3,4,5,30,118,153,158,126,126,126,76,74,86,86,30,25,25,25,30,30,30,59,59,59,59,64,64,64]
GN = [0,0,14,1,2,0,0,2,1,3,4,9,3,10,11,15,4,4,5,3,8,10,8,7,7,86,3,9,1,8,0,6,30,0,0,0,0,0,0,0,0,0,0,91,24,24,24,29,29,29,14,14,14,14,0,0,0]
HN = [8,9,0,4,4,32,38,0,1,0,4,0,0,0,0,0,0,0,0,0,0,1,3,0,0,0,0,0,0,0,0,0,0,108,148,148,116,116,116,76,74,76,76,0,1,1,1,1,1,1,1,1,1,1,6,6,6]

# best-known feasible objective values, Table 3 of Kumar et al. (2020)
F_STAR = {
    3: -4.5291197395e03, 10: 1.0765430833e00, 12: 2.9248305537e00, 13: 2.6887000000e04,
    14: 5.3638942722e04, 15: 2.9944244658e03, 16: 3.2213000814e-02, 21: 2.3524245790e-01,
    24: 2.5287918415e00, 25: 1.6254428092e03, 26: 3.5359231973e01, 28: 1.4614135715e04,
    30: 2.6138840583e00, 17: 1.2665232788e-02, 18: 5.8853327736e03, 19: 1.6702177263e00,
}

NAMES = {
    3: "Optimal operation of alkylation unit", 10: "Process flowsheeting problem",
    12: "Process synthesis problem", 13: "Process design problem", 14: "Multi-product batch plant",
    15: "Weight minimisation of a speed reducer", 16: "Industrial refrigeration system",
    17: "Tension/compression spring (case 1)", 18: "Pressure vessel design", 19: "Welded beam design",
    21: "Multiple disk clutch brake", 24: "Robot gripper", 25: "Hydrostatic thrust bearing",
    26: "Four-stage gearbox", 28: "Rolling element bearing", 30: "Tension/compression spring (case 2)",
}


# 0-based indices of variables that the official code rounds (read from cec2020rwc.cpp),
# with their kind. All selected integer variables are ordinal or binary:
#   RC12: 4 binary design decisions; RC13: x4 in {78..102}, x5 in {33..45};
#   RC14: N_i in {1..3}; RC18: plate thicknesses as integer multiples of 0.0625 in (code
#   rounds x and multiplies by 0.0625), x in {1..99}; RC26: tooth counts {7..76}, module
#   index {1..4} into an increasing table, position index {1..9} into an increasing table.
INTEGER_VARS = {
    8: [1], 9: [2], 10: [2], 11: [4, 5], 12: [3, 4, 5, 6], 13: [3, 4], 14: [0, 1, 2],
    18: [0, 1], 22: list(range(9)), 26: list(range(22)), 28: [2], 30: [0, 2],
}
INTEGER_KINDS = {12: "binary"}  # default for the others: "ordinal"


@dataclass
class RawEval:
    f: float
    g: np.ndarray  # g_i(x) <= 0 is feasible
    h: np.ndarray  # h_j(x) == 0 is feasible


class CEC2020Problem:
    def __init__(self, num: int):
        self.num = num
        self.name = NAMES.get(num, f"RC{num:02d}")
        self.dim, self.ng, self.nh = D[num - 1], GN[num - 1], HN[num - 1]
        if num == 31:  # known quirk of the port: box-constrained only
            self.ng = self.nh = 0
        lb = (ctypes.c_double * self.dim)()
        ub = (ctypes.c_double * self.dim)()
        _get_lib().get_bounds(ctypes.c_int(num), lb, ub)
        self.lb, self.ub = np.array(lb[:]), np.array(ub[:])
        # variables rounded inside the official code (read from cec2020rwc.cpp)
        self.integer_indices = list(INTEGER_VARS.get(num, []))
        self.f_star = F_STAR.get(num)

    def seal(self, x):
        x = np.clip(np.asarray(x, float).ravel(), self.lb, self.ub)
        if self.integer_indices:
            ii = self.integer_indices
            x[ii] = np.clip(np.rint(x[ii]), np.ceil(self.lb[ii]), np.floor(self.ub[ii]))
        return x

    def raw(self, x) -> RawEval:
        x = self.seal(x)
        xc = (ctypes.c_double * self.dim)(*x)
        f = ctypes.c_double(0.0)
        g = (ctypes.c_double * max(self.ng, 1))()
        h = (ctypes.c_double * max(self.nh, 1))()
        _get_lib().cec20_func(xc, ctypes.c_int(self.num), ctypes.byref(f), g, h)
        return RawEval(float(f.value), np.array(g[: self.ng]), np.array(h[: self.nh]))

    def violations(self, x, eps_h: float = 1e-4) -> np.ndarray:
        """Per-constraint violation >= 0 (competition convention: |h| - 1e-4)."""
        r = self.raw(x)
        return np.concatenate([np.maximum(0.0, r.g), np.maximum(0.0, np.abs(r.h) - eps_h)])
