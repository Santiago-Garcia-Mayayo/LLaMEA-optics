# Problem suite — constraint representation in LLaMEA

10 problems, one interface (`suite.py`): 2 lens instances + 8 CEC 2020 real-world problems.

```python
from problems import get_problem, PROBLEM_IDS
p = get_problem("cec/RC15")       # or "lens/original", ...
ev = p.evaluate(x)                # ONE simulation -> ev.f, ev.v (violations >= 0), ev.penalty
p.is_feasible(ev)                 # f finite and all v <= FEAS_TOL (1e-9)
p.variables                       # typed variables: continuous / ordinal / binary / categorical
```
Algorithms never call problems directly: they go through the harness (`harness/`).

## Setup

```bash
# 1) CEC 2020 C++ port (needs Eigen 3: apt install libeigen3-dev)
cd cec2020_cpp && g++ -std=c++14 -O2 -fPIC -shared -I/usr/include/eigen3 -o cec2020rwc.so cec2020rwc.cpp -lm && cd ..
# 2) lensgopt: point LENSGOPT_ROOT at the CameraLensSimulation repo (or pip install -e it)
export LENSGOPT_ROOT=/path/to/CameraLensSimulation-main
python -m pytest -q   # from the repo root
```
`cec2020_cpp/` is V. Stanovov's code (GPL-3.0), unchanged.

## Lens (lensgopt, Double-Gauss, 24 vars: 10 curvatures, 8 thicknesses, 6 glasses)

| | original | realistic_draft |
|---|---|---|
| glass centre thickness | ≥ 0.1 mm (box bound) | ≥ 2.0 mm (constraint) |
| air centre gap | ≥ 0.1 mm (box bound) | ≥ 0.5 mm (constraint) |
| glass edge thickness | ≥ 0.25 mm | ≥ 1.0 mm |
| air edge gap | ≥ 0.25 mm | ≥ 0.1 mm |
| free working distance | ≥ 3 mm | ≥ 3 mm |
| EFFL | penalty (EFFL−99.5)²; feasibility \|ΔEFFL\| ≤ 0.5 mm | \|ΔEFFL\| ≤ 0.5 mm |
| ray loss / TIR | hidden constraint: f = inf | same |

* `original.penalty(x)` equals lensgopt's loss exactly (checked on the template; per-gap
  edge/axis terms reproduce lensgopt's penalties to 7e-12 on 150 perturbed designs).
* 24 constraints: rays lost (fraction, graded), 10 centre thicknesses, 11 edge thicknesses (incl. last surface→sensor), FWD, EFFL. Violations are normalised by data-driven scales (DESIGN.md §6, D9).
* RMS is computed by lensgopt over surviving rays only (all rays lost ⇒ RMS = 0), hence f = inf
  whenever any ray is lost.
* **realistic_draft values are a draft**: sources in `lens.py` (`LensSpec.source`); to be fixed with
  Kirill Antonov / Teus Tukker *before* any LLaMEA run. The group template is feasible on both.
* Gap indexing: gap i = surface i → i+1. Glass: 0,2,3,6,7,9. Air: 1,4,5,8 (4,5 = around the
  stop, fixed). Gap 10 = last surface → sensor.

## Control: CEC 2020 real-world constrained (Kumar et al. 2020)

Code: C++ port by V. Stanovov (linked from the official repo), compiled as `cec2020rwc.so`.
Scripts used for selection and verification: `scripts/` (results in `data/`).
**Verified against the official MATLAB code in Octave**: identical f, g on 21 points per problem
(max rel. diff ≤ 3e-11) for all 16 candidates except RC24 (f differs at infeasible points → excluded).

Selection rules (fixed before any LLaMEA run): inequality constraints only; port = MATLAB;
best-known value reproducible; ≥ 1 constraint active at x*; not solved by a 2k-eval DE.

| prob | name | D | int | g | f* (competition consensus) | active at ref. x |
|---|---|---|---|---|---|---|
| RC12 | process synthesis | 7 | 4 | 9 | 2.92483055 | 2 |
| RC13 | process design | 5 | 2 | 3 | 26887.422 | 1 |
| RC15 | speed reducer | 7 | 0 | 11 | 2994.42446576 | 4 |
| RC16 | industrial refrigeration | 14 | 0 | 15 | 0.032213 | 2* |
| RC18 | pressure vessel (plates in 1/16 in) | 4 | 2 | 4 | 6059.7143 | 2 |
| RC21 | multiple disk clutch brake | 5 | 0 | 8 | 0.23524246 | 1 |
| RC25 | hydrostatic thrust bearing | 4 | 0 | 7 | 1616.11976505 | 4 |
| RC26 | four-stage gearbox | 22 | 22 | 86 | 35.359232 | 24* |

\* reference DE (L-SHADE-like + Deb rules, 1e5–2e5 evals, 5 seeds) did not reach f* on RC16
(gap 1.5 %) and RC26 (14 %): hardest problems of the set. All others reached f* within 1e-7.

f*: Table 3 of the paper is outdated for several problems (e.g. RC18 = 6059.71 after the
0.0625-inch fix, RC25 = 1616.12 < 1625.44 in the table). We use the best feasible final value
over all competition submissions (`data/cec_fstar_consensus.json`).

Excluded and why: RC03, RC14, RC28, RC30 (competition submissions disagree → code-version
ambiguity); RC10 (solved in < 2k evals); RC17, RC19 (trivial classics); RC24 (C++≠MATLAB);
all problems with equality constraints (different regime).

## Decisions

All design decisions (penalty weights, tolerance, encoding, splits) are in ../DESIGN.md.
