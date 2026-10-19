"""Glue between LLaMEA (pip install llamea) and the harness.

    from harness.llamea_adapter import make_evaluator, make_llamea
    es = make_llamea(arm="free", llm=my_llm, budget=100, experiment_name="free-run0")
    best = es.run()

LLaMEA calls `f(solution, logger)`; we run solution.code through the harness on the training
problems and seeds, and set fitness (mean AOCC, maximised), feedback and error.
Everything arm-specific lives in harness.prompts; everything else is identical across arms.
"""
from __future__ import annotations

import json
import os

from . import config, prompts
from .evaluate import evaluate_candidate


def make_evaluator(arm: str, problem_ids=config.TRAIN_PROBLEMS, seeds=config.TRAIN_SEEDS,
                   n_workers: int | None = None, log_dir: str | None = None):
    def f(solution, logger=None):
        try:
            res = evaluate_candidate(solution.code, arm, problem_ids, seeds, class_name=solution.name,
                                     n_workers=n_workers, algorithm_name=solution.name)
        except Exception as e:  # harness-side failure: never let it look like a good algorithm
            solution.set_scores(0.0, f"Evaluation failed: {e}", e)
            return solution
        err = RuntimeError(res.error.splitlines()[0]) if res.error else None
        solution.set_scores(res.fitness, res.feedback, err)
        solution.add_metadata("per_problem", res.per_problem)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"{solution.id}.json"), "w") as fh:
                json.dump({"name": solution.name, "arm": arm, "fitness": res.fitness,
                           "per_problem": res.per_problem, "runs": res.runs}, fh, default=float)
        return solution
    return f


def candidate_timeout(problem_ids=config.TRAIN_PROBLEMS, seeds=config.TRAIN_SEEDS) -> int:
    """Upper bound on the wall time of one candidate evaluation (problems run in parallel,
    seeds sequentially), used as LLaMEA's eval_timeout."""
    from problems import get_problem
    from .runner import _LensStub
    limits = []
    for pid in problem_ids:
        p = _LensStub(pid) if pid.startswith("lens/") else get_problem(pid)
        limits.append(config.time_limit_for(p, config.budget_for(p)))
    return int(config.STARTUP_GRACE_S + len(seeds) * (max(limits) + 15.0) + 60)


def make_llamea(arm: str, llm, budget: int = 100, n_parents: int = 4, n_offspring: int = 12,
                experiment_name: str = "", max_workers: int = 2, n_workers_per_candidate: int | None = None,
                log_dir: str | None = None, problem_ids=config.TRAIN_PROBLEMS, seeds=config.TRAIN_SEEDS,
                **kwargs):
    """LLaMEA instance with the arm's prompts. Fitness is maximised (minimization=False).

    Parallelism: LLaMEA evaluates up to `max_workers` candidates at once and each candidate
    runs its problems in parallel (`n_workers_per_candidate`, default = number of problems),
    so the machine needs about max_workers x n_problems cores."""
    from llamea import LLaMEA
    evaluator = make_evaluator(arm, problem_ids, seeds, n_workers=n_workers_per_candidate, log_dir=log_dir)
    return LLaMEA(evaluator, llm=llm,
                  n_parents=n_parents, n_offspring=n_offspring,
                  role_prompt=prompts.ROLE, task_prompt=prompts.task_prompt(arm),
                  example_prompt=prompts.example_prompt(arm), experiment_name=experiment_name or arm,
                  budget=budget, eval_timeout=candidate_timeout(problem_ids, seeds), max_workers=max_workers,
                  minimization=False, **kwargs)
