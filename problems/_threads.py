"""Keep the number of threads per process small.

Shared servers cap the tasks (processes + threads) per user: 2048 on the DSlab machines.
By default numpy's BLAS and JAX's CPU client each start about one thread per visible core
(256 on drogium), so a few processes are enough to hit the cap, and pthread_create fails
inside JAX ("Thread tf_XLAPjRtCpuClient creation via pthread_create() failed", abort).
Our evaluations are small and single-threaded anyway; parallelism comes from running many
jobs side by side. Override with LLAMEA_CORES_PER_PROCESS (0 = do not touch affinity).
"""
from __future__ import annotations

import contextlib
import os
import random

_BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def limit_threads(n: int = 1):
    """Thread counts for BLAS/OpenMP and XLA's Eigen. Effective only before numpy/jax load."""
    for v in _BLAS_VARS:
        os.environ.setdefault(v, str(n))
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_cpu_multi_thread_eigen" not in flags:
        os.environ["XLA_FLAGS"] = (flags + " --xla_cpu_multi_thread_eigen=false").strip()


def cores_per_process() -> int:
    return int(os.environ.get("LLAMEA_CORES_PER_PROCESS", "4"))


def _sample(k):
    cur = sorted(os.sched_getaffinity(0))
    return cur if len(cur) <= k else random.SystemRandom().sample(cur, k)


def pin_to_random_cores():
    """Restrict this process to k random cores (used by the isolated workers). With several
    choices per process the kernel still balances load well when many workers run at once."""
    k = cores_per_process()
    if k > 0 and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, _sample(k))


@contextlib.contextmanager
def few_cores():
    """Within the block the process sees at most k cores (thread pools created here, such as
    JAX's CPU client, are sized to k); the previous affinity is restored afterwards."""
    k = cores_per_process()
    if k <= 0 or not hasattr(os, "sched_setaffinity"):
        yield
        return
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, _sample(k))
    try:
        yield
    finally:
        os.sched_setaffinity(0, old)
