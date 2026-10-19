"""Worker entry point: `python -m harness._worker job.json results.jsonl`.

Runs in a fresh interpreter (independent of the caller's __main__, so callers need no
`if __name__ == "__main__"` guard and it works inside LLaMEA's own worker processes).
Writes one JSON line per finished seed, flushed immediately.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback


def main(job_path: str, out_path: str):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from problems._threads import limit_threads, pin_to_random_cores
    limit_threads()          # before numpy/jax load: few threads per worker
    pin_to_random_cores()    # thread pools sized to a few cores; see problems/_threads.py
    # persistent XLA cache: the lens simulator compiles once per machine instead of once per job
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                          os.path.join(os.path.expanduser("~"), ".cache", "llamea_constraints_jax"))
    with open(job_path) as fh:
        job = json.load(fh)
    out = open(out_path, "a", buffering=1)

    def emit(kind, payload):
        out.write(json.dumps({"kind": kind, "payload": payload}, default=float) + "\n")
        out.flush()
        os.fsync(out.fileno())

    try:
        from problems import get_problem
        from harness.runner import _load_class, _save_trace, run_one
        problem = get_problem(job["problem"])
        cls = _load_class(job["algorithm"], job.get("class_name"))
    except Exception as e:
        emit("fatal", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=8)}")
        return 0
    sink = io.StringIO()
    for seed in job["seeds"]:
        with contextlib.redirect_stdout(sink):
            res, h = run_one(cls, problem, job["arm"], seed, job["budget"], job["time_limit"],
                             job.get("init_kwargs"), keep_trace=bool(job.get("trace_dir")))
        if job.get("trace_dir"):
            _save_trace(job["trace_dir"], res, h)
        sink.truncate(0)
        sink.seek(0)
        emit("result", res.to_dict())
    emit("done", None)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
