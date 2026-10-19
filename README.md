# llamea-constraints

Does the way constraints are exposed change what LLaMEA discovers? Problems, harness and
protocol for the study. **Start with [DESIGN.md](DESIGN.md)** (the pre-registered design).

```
problems/            the 10 problems behind one interface (lens + CEC 2020)  -> problems/README.md
harness/             what algorithms see and how they are judged
  space.py           encoded search space ([-1,1] continuous, typed integer indices)
  core.py            Harness: func per arm, hard budget, best-feasible record, AOCC
  scoring.py         score / unit score / AOCC
  runner.py          isolated execution of generated code (one process per job)
  evaluate.py        candidate -> fitness + feedback (identical template across arms)
  prompts.py         task prompts of the three arms
  llamea_adapter.py  LLaMEA integration
  config.py          problems, splits, seeds, budgets, time limits
  reference_algorithms.py   trusted test algorithms (random search, DE)
tests/               54 tests (run them after any change)
```

## Setup

```bash
pip install -e ".[lens,llamea,dev]"
# CEC 2020 (C++, needs Eigen 3: apt install libeigen3-dev / brew install eigen)
cd problems/cec2020_cpp && g++ -std=c++14 -O2 -fPIC -shared -I/usr/include/eigen3 \
      -o cec2020rwc.so cec2020rwc.cpp -lm && cd ../..
# lensgopt: pip install -e the CameraLensSimulation repo, or set LENSGOPT_ROOT, or keep it
# next to this repo as ../CameraLensSimulation
python -m pytest -q
```

## Using the harness

```python
from problems import get_problem
from harness import Harness

h = Harness(get_problem("cec/RC15"), arm="free", budget=3500, seed=0)
f, g = h.func(x)          # the algorithm only ever sees h.func
print(h.result())         # best feasible f, AOCC, feasibility, curve ...
```

Isolated run of generated code (what LLaMEA uses):

```python
from harness.runner import run_job
results = run_job(code, "lens/original", "free", seeds=[0, 1, 2], class_name="MyAlgorithm")
```

LLaMEA run of one arm:

```python
from llamea import Gemini_LLM
from harness.llamea_adapter import make_llamea
es = make_llamea("free", Gemini_LLM(api_key=..., model=...), budget=100, experiment_name="free-r0")
best = es.run()
```

Each candidate runs its training problems in parallel (5 processes) and LLaMEA evaluates
`max_workers` candidates at once: plan for about `max_workers x 5` cores.
