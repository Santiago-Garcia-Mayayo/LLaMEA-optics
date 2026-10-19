"""Experimental configuration (see DESIGN.md). Change values here, nowhere else."""
from __future__ import annotations

# Problems LLaMEA evolves on, and problems the final algorithms are evaluated on.
# DECISION D6: lens instance for training is pending (original vs realistic, after Kirill).
TRAIN_PROBLEMS = ("lens/original", "cec/RC12", "cec/RC15", "cec/RC21", "cec/RC25")
TEST_PROBLEMS = ("lens/original", "lens/realistic_draft",
                 "cec/RC12", "cec/RC13", "cec/RC15", "cec/RC16",
                 "cec/RC18", "cec/RC21", "cec/RC25", "cec/RC26")

TRAIN_SEEDS = (0, 1, 2)                       # during evolution
TEST_SEEDS = tuple(range(1000, 1015))         # final evaluation (15 seeds, never seen)

# Evaluation budget per run (DECISION D3 / D11). The lens gets a larger budget at test time,
# like the TFG (50,000 evaluations as standard); during evolution 10,000 keeps one candidate
# evaluation affordable (~70 s per lens run at ~7 ms per evaluation on one CPU core).
LENS_BUDGET = {"train": 10_000, "test": 50_000}
CEC_BUDGET_PER_DIM = {"train": 500, "test": 500}


def budget_for(problem, phase: str = "train") -> int:
    if phase not in ("train", "test"):
        raise ValueError("phase must be 'train' or 'test'")
    if problem.pid.startswith("lens/"):
        return LENS_BUDGET[phase]
    return CEC_BUDGET_PER_DIM[phase] * problem.dim


# Wall-clock limit per run: TIME_FACTOR x (budget x typical cost of one evaluation),
# never below TIME_MIN seconds. Generous on purpose: it only stops runaway code.
EVAL_COST_S = {"lens": 0.010, "cec": 0.0003, "toy": 0.0001}   # measured: lens ~7 ms on one core
TIME_FACTOR = 10.0
TIME_MIN = 120.0
STARTUP_GRACE_S = 120.0    # process start + imports + JAX compilation, per job


def time_limit_for(problem, budget: int) -> float:
    family = problem.pid.split("/")[0]
    return max(TIME_MIN, TIME_FACTOR * budget * EVAL_COST_S[family])
