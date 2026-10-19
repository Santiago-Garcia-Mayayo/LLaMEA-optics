from ._threads import limit_threads as _limit_threads

_limit_threads()  # before numpy/jax load, see _threads.py

from .suite import (FEAS_TOL, PROBLEM_IDS, Evaluation, ScoreSpec, Variable, get_problem,  # noqa: F401
                    load_suite)
