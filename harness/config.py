"""Experimental configuration (see DESIGN.md). Change values here, nowhere else."""
from __future__ import annotations

# The 10 problems. LLaMEA is run once per (problem, arm), evolving on that problem only;
# the final algorithm is then evaluated on the same problem with new seeds.
PROBLEMS = ("lens/original", "lens/realistic_draft",
            "cec/RC12", "cec/RC13", "cec/RC15", "cec/RC16",
            "cec/RC18", "cec/RC21", "cec/RC25", "cec/RC26")
TEST_PROBLEMS = PROBLEMS                      # baselines and final evaluation

TRAIN_SEEDS = (0, 1, 2)                       # during evolution (3 runs per candidate)
TEST_SEEDS = tuple(range(1000, 1015))         # final evaluation (15 seeds, never seen)

# Evaluation budget per run (DECISION D3/D11, revised 2026-10-20): 1000 x d for every problem,
# the same during evolution and in the final evaluation (lens: 24,000).
BUDGET_PER_DIM = {"train": 1000, "test": 1000}


def budget_for(problem, phase: str = "train") -> int:
    if phase not in ("train", "test"):
        raise ValueError("phase must be 'train' or 'test'")
    return BUDGET_PER_DIM[phase] * problem.dim


# LLaMEA (DECISION D8 and protocol, revised 2026-10-20): (5 + 5) elitist, as in the original
# LLaMEA; 100 candidates per run (LLaMEA's budget counts evaluated candidates).
LLAMEA_N_PARENTS = 5
LLAMEA_N_OFFSPRING = 5
LLAMEA_ELITISM = True
LLAMEA_BUDGET = 100
LLM_MODEL = "claude-sonnet-5-5"   # PENDING confirmation (D8)
LLM_TEMPERATURE = None            # None = API default

# Wall-clock limit per run: TIME_FACTOR x (budget x typical cost of one evaluation),
# never below TIME_MIN seconds. Generous on purpose: it only stops runaway code.
EVAL_COST_S = {"lens": 0.010, "cec": 0.0003, "toy": 0.0001}   # measured: lens ~7-9 ms on one core
TIME_FACTOR = 10.0
TIME_MIN = 120.0
STARTUP_GRACE_S = 120.0    # process start + imports + JAX compilation, per job


def time_limit_for(problem, budget: int) -> float:
    family = problem.pid.split("/")[0]
    return max(TIME_MIN, TIME_FACTOR * budget * EVAL_COST_S[family])
