"""Experiment drivers: LLaMEA runs (run_llamea.py) and final evaluation (final_eval.py)."""
from problems._threads import limit_threads as _limit_threads

_limit_threads()  # before numpy/jax load in any driver (BLAS would start one thread per core)