"""Baselines of the study.

BASELINES maps a name to (import path, arm). Pairs differ only in the comparison:
penalty scalar (penalty arm) vs Deb's feasibility rules on (f, g) (free arm).
"""
BASELINES = {
    "random_search":  ("baselines.random_search:RandomSearch", "free"),
    "lshade_penalty": ("baselines.lshade:LSHADEPenalty", "penalty"),
    "lshade_deb":     ("baselines.lshade:LSHADEDeb", "free"),
    "cmaes_penalty":  ("baselines.cmaes:CMAESPenalty", "penalty"),
    "cmaes_deb":      ("baselines.cmaes:CMAESDeb", "free"),
}
