"""Harness for the constraint-representation study. See DESIGN.md."""
from problems._threads import limit_threads as _limit_threads

_limit_threads()  # before numpy/jax load, see problems/_threads.py

from .core import ARMS, BudgetExhausted, Harness, HarnessTimeout, RunResult  # noqa: F401
from .space import EncodedSpace  # noqa: F401
