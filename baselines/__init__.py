"""Baselines of the study (DESIGN.md §8). All use the LLaMEA interface and run through the
harness with the same budgets as the generated algorithms.

BASELINES maps a name to (import path, arm). Pairs differ only in the comparison:
penalty scalar (penalty arm) vs Deb's feasibility rules on (f, g) (free arm).
"""
BASELINES = {
    "random_search":  ("baselines.cmaes:RandomSearch", "free"),
    "lshade_penalty": ("baselines.lshade:LSHADEPenalty", "penalty"),
    "lshade_deb":     ("baselines.lshade:LSHADEDeb", "free"),
    "cmaes_penalty":  ("baselines.cmaes:CMAESPenalty", "penalty"),
    "cmaes_deb":      ("baselines.cmaes:CMAESDeb", "free"),
}
