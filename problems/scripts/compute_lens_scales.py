"""Violation scales for the lens instances: s_j = median of the positive raw violations of
constraint j over N seeded uniform samples of the native box (same rule as for CEC).
Writes problems/data/lens_violation_scales.json. Usage: python problems/scripts/compute_lens_scales.py"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from problems.lens import INSTANCES, LensProblem  # noqa: E402
from problems.suite import violation_scale  # noqa: E402

N, SEED = 2000, 2026
out = {}
for name in INSTANCES:
    P = LensProblem(name)
    rng = np.random.default_rng(SEED)
    R = np.array([P.raw_violations(P.evaluate(P.seal(P.lb + rng.random(P.dim) * (P.ub - P.lb))))
                  for _ in range(N)])
    scales = violation_scale(R)
    out[name] = {"scales": scales.tolist(), "names": list(P.CONSTRAINT_NAMES), "n_samples": N, "seed": SEED,
                 "frac_violated": (R > 0).mean(0).round(4).tolist()}
    print(name, dict(zip(P.CONSTRAINT_NAMES, np.round(scales, 4))))
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lens_violation_scales.json")
json.dump(out, open(path, "w"), indent=1)
print("written", path)
