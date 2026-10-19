# Design: constraint representation in LLM-driven algorithm discovery

**Status:** design frozen for implementation, 2026-10-18 (pre-registration).
Items marked **PENDING** must be fixed, and committed here, before any LLaMEA run whose results
are reported. Any later change goes in the change log at the end, with its reason.

Author: Santiago Garcia Mayayo (LIACS, NACO). Supervisor: Niki van Stein.

---

## 1. Question

When LLaMEA designs optimisers for constrained problems, does it matter how the constraints
are *exposed* to the generated algorithm?

* **RQ1 (performance).** Does LLaMEA discover better algorithms when the algorithm sees the
  objective and the constraint violations separately than when it sees one penalised scalar?
* **RQ2 (knowledge).** Does telling the LLM about constraint-handling techniques change this?
* **RQ3 (strategies).** Without instructions, which constraint-handling strategies does the
  LLM invent (penalties, feasibility rules, repair, ε-constraints, violation-as-objective,
  constrained solvers, hybrids), how do they evolve over generations, and which win?
* **RQ4 (generalisation).** Do the results hold on problems not used during evolution, and
  outside lens design?

Existing work (llmEA/AwesomeDE, LMCOEA, LLM4CMO, AutoCO) builds constraint-handling
knowledge into the prompt or fixes the component the LLM designs. To our knowledge no work
isolates the effect of the constraint *interface* with everything else held fixed.

## 2. Arms

Three arms, identical except for what `func` returns and one paragraph of the prompt
(`harness/prompts.py`; the equality of all other text is enforced by a test).

| arm | `func(x)` returns | prompt |
|---|---|---|
| `penalty` | `y`: objective + penalties (one float) | interface + evaluation criterion |
| `free` | `(f, g)`: objective, vector of normalised violations `g_j >= 0` (0 = satisfied) | same, no method mentioned |
| `guided` | `(f, g)` as in `free` | same + one paragraph listing standard techniques |

`penalty` vs `free` isolates **information** (scalar vs vector); `free` vs `guided` isolates
**prescribed knowledge**. `guided` is the closest analogue of prior work.

## 3. Problems

10 problems, all through one interface (`problems/suite.py`). Integer/categorical variables
are exposed as such.

| id | problem | d | int (kind) | m | role |
|---|---|---|---|---|---|
| `lens/original` | Double-Gauss, group specification | 24 | 6 (categorical) | 24 | train + test |
| `lens/realistic_draft` | Double-Gauss, manufacturability specification | 24 | 6 (categorical) | 24 | test |
| `cec/RC12` | process synthesis | 7 | 4 (binary) | 9 | train + test |
| `cec/RC13` | process design | 5 | 2 (ordinal) | 3 | test |
| `cec/RC15` | speed reducer | 7 | 0 | 11 | train + test |
| `cec/RC16` | industrial refrigeration | 14 | 0 | 15 | test |
| `cec/RC18` | pressure vessel (plates in 1/16 in) | 4 | 2 (ordinal) | 4 | test |
| `cec/RC21` | multiple disk clutch brake | 5 | 0 | 8 | train + test |
| `cec/RC25` | hydrostatic thrust bearing | 4 | 0 | 7 | train + test |
| `cec/RC26` | four-stage gearbox | 22 | 22 (ordinal) | 86 | test |

**Lens.** lensgopt (K. Antonov), 10 curvatures + 8 thicknesses + 6 glasses (Schott catalogue,
120 glasses). Objective: RMS² spot size [mm²]. 24 constraints: rays lost, centre thickness
(10 gaps), edge thickness (11 gaps incl. last surface → sensor), free working distance, EFFL.
Ray loss / TIR makes the objective undefined (lensgopt computes RMS over surviving rays only, so it
would reward losing rays), hence `f = inf`; it is *also* exposed as a graded constraint
(`rays_lost` = fraction of the 6,390 traced rays lost), because the penalty arm sees it graded
(10 × rays lost) and the free arms must not get less information. When all rays are lost the edge
thicknesses cannot be computed and report no violation (simulator limitation).
* `original`: the group's values (min thickness 0.1 mm for every gap via box bounds, edge ≥ 0.25 mm,
  FWD ≥ 3 mm, EFFL 99.5 mm). Its penalty arm is *exactly* lensgopt's loss (weights 10, 1, 1, 1, 1),
  verified to machine precision. The values coincide with a CODE V course example for a 6.43 mm
  lens (to confirm with K. Antonov). Constraint that is active at good designs in pilots: the
  edge thickness of the first air gap.
* `realistic_draft`: glass centre ≥ 2 mm, air ≥ 0.5 mm, glass edge ≥ 1 mm, air edge ≥ 0.1 mm,
  |ΔEFFL| ≤ 0.5 mm; sources in `problems/lens.py`. **PENDING (D1):** values to be fixed with
  K. Antonov / T. Tukker on physical grounds only — never tuned to make constraints active.
* The glass index is categorical, but the catalogue is sorted by refractive index n_d:
  neighbouring indices have similar n_d and very different Abbe numbers.

**Control: CEC 2020 real-world constrained suite** (Kumar et al. 2020), C++ port by V. Stanovov,
verified identical to the official MATLAB code (Octave, 21 points per problem, rel. diff ≤ 3e-11).
Selection rules, fixed before any LLaMEA run: inequality constraints only; port = MATLAB;
best-known value reproducible; ≥ 1 constraint active at the reference solution; not solved by a
2,000-evaluation DE. Excluded: RC03/RC14/RC28/RC30 (competition submissions disagree → code-version
ambiguity), RC10/RC17/RC19 (trivial), RC24 (C++ ≠ MATLAB), all problems with equalities.
f*: best feasible value over all competition submissions (Table 3 of the paper is outdated).

## 4. What the algorithm sees

* **Encoded space** (`harness/space.py`):
  * continuous variables in **[-1, 1]**, decoded affinely to the native box;
  * integer variables (ordinal / binary / categorical) with n values as **indices 0..n-1** with
    bounds **[-0.5, n-0.5]**, decoded `k = clip(rint(z), 0, n-1)`. An exact integer maps to
    itself, and uniform sampling over the bounds gives every value equal probability (tested).
    (Bounds [0, n-1] would halve the probability of the end values; mapping [-1, 1] with
    `round` reached 2 of 6 glasses in the TFG's v5.)
* **Attributes of `func`**: `bounds.lb/ub`, `dim`, `variable_types`, `n_values`,
  `integer_indices`, `continuous_indices`, `budget`; `n_constraints` only in `free`/`guided`.
* **No problem identity**: no names in prompts or feedback (RC15, RC18, RC21 are textbook problems).
* `func` accepts one point or an (n, d) batch (n evaluations). Invalid input (wrong shape, NaN)
  raises `ValueError`.

## 5. Budget and time

* **1 call = 1 evaluation**, objective and constraints together, same cost in every arm.
  Repeated points are counted. No gradients in the main study (D2, see §11).
* **Budget per run (D3, D11):** lens 10,000 evaluations during evolution and 50,000 at test time
  (the TFG's standard budget); CEC 500·d (2,000–11,000; < 2 s) in both phases. A lens evaluation
  costs ~7 ms on one CPU core (~70 s per training run, ~6 min per test run). `harness/config.py`.
* **Vectorisation:** `jax.vmap` batch evaluation of the lens is implemented and tested equal to
  single evaluation (≤ 1e-15 relative), but on CPU it is 1.5–3× *slower* per lens (each lens already
  traces 8,658 rays in one vectorised call), so it is used only on GPU (`LLAMEA_LENS_BATCH` to force).
  On CPU the speed lever is running seeds/problems in parallel processes (scales linearly).
* Beyond the budget `func` raises `BudgetExhausted`, which derives from `BaseException` so
  `except Exception` in generated code cannot swallow it; a batch crossing the budget is
  evaluated up to the budget, then raises.
* **Wall clock:** per run max(120 s, 10 × budget × typical evaluation cost), enforced inside `func`,
  by SIGALRM (POSIX), and by a hard kill of the worker process.

## 6. Evaluation criterion (common to all arms)

The **harness**, never the algorithm, records every evaluation and the best point.

* **Normalised violations (D9):** for every problem, v_j = (raw violation of constraint j) / s_j with
  s_j = median of the *positive* raw violations of j over 2,000 seeded uniform samples of the native
  box (1 if never violated): one unit = a typical violation of that constraint, so that no
  constraint dominates a sum because of its units. Lens raw units: fraction of rays, mm; CEC: g_j.
  Scales: computed at load time for CEC, stored for the lens (`problems/data/lens_violation_scales.json`,
  `problems/scripts/compute_lens_scales.py`). Pilot evidence for the need: with violations scaled by
  their tolerances, a DE with Deb's rules was drawn to designs losing *all* rays (T = 1 there,
  versus T ≫ 1 for a few mm of focal-length error) and never found a feasible lens; with D9 it does.
* **Feasible:** objective finite and every normalised violation ≤ `FEAS_TOL = 1e-9` (float noise
  of constraints sitting on bounds is ~1e-15). Violations below the tolerance are returned to the
  algorithm as exactly 0, so "feasible ⇔ all(g == 0)" as stated in the prompt.
* **Best point:** any feasible beats any infeasible; feasible by lower f; infeasible by lower
  total violation T = Σ v_j (T = 1 if f is undefined and no violation is recorded).
* **Score (D4):** feasible: `s = clip((log10 gap_max − log10 gap) / (log10 gap_max − log10 gap_min), 0, 1)`
  with `gap = (f − f_ref)/scale`. CEC: f_ref = f*, scale = |f*|, gap ∈ [1e-6, 1] (1 = 100 % worse
  than f*). Lens: absolute RMS², f_ref = 0, scale = 1 mm², range [1e-5, 1] mm² (1e-5 ≈ 3 µm RMS
  radius). Infeasible: `s = −T/(1+T)` ∈ (−1, 0].
* **Unit score** u = (1+s)/2 ∈ [0, 1]: every feasible point ≥ 0.5, every infeasible < 0.5.
* **AOCC (D5):** mean of u of the best-so-far over evaluations 1..budget (u = 0 before the first
  evaluation; last value kept if the algorithm stops early). Errors, crashes and timeouts count as 0.
* **LLaMEA fitness** of a candidate: mean over training problems of the mean AOCC over seeds
  (maximised).

## 7. Penalties

* Lens, penalty arm: `original` = lensgopt loss; `realistic_draft` = RMS² + 10·(rays lost) +
  squared violations of each constraint (same structure).
* CEC, penalty arm (D7): `f + ρ Σ_j v_j`, ρ = 10·max(1, |f*|), v_j as in §6 (D9). Any fixed ρ is a modelling choice, like the lens
  weights; this is the "typical practitioner" penalty the study questions.
* The free arms see the same normalised v_j.

## 8. Protocol

* **LLaMEA:** (4 + 12) elitist ES, 100 candidates per run, standard mutation prompts, standard
  feedback (§9). **PENDING (D8):** LLM (model, version, temperature), fixed for all arms.
* **Training problems** (evolution): `lens/original`, RC12, RC15, RC21, RC25; seeds 0, 1, 2.
  **PENDING (D6):** lens instance used for training (`original` vs final realistic).
* **Test** (final algorithms): all 10 problems, 15 new seeds (1000–1014). Held-out problems
  (lens realistic, RC13, RC16, RC18, RC26) are reported separately from training problems.
* **Replicates:** 10 independent LLaMEA runs per arm (30 runs).
* **Baselines** (`baselines/`) on the same harness, budgets and test seeds, in matched pairs that
  differ only in the comparison (penalty scalar vs Deb's feasibility rules on (f, g)):
  L-SHADE (success-history DE, current-to-pbest/1 + archive, linear population reduction;
  initial population capped at budget/25 for our budgets), CMA-ES (pycma, rank-based, IPOP
  restarts, pycma integer handling), plus random search. **PENDING:** LDG-EA on the lens (code or
  results from K. Antonov). Multistart SLSQP only in the gradient experiment (§11): without
  gradients each finite-difference gradient would cost d + 1 evaluations of the budget.

## 9. Feedback to the LLM

Same template in every arm (enforced by a test): mean AOCC; per anonymised problem P1..Pk
(dimension, number of integer variables, number of constraints): AOCC, feasible runs k/n, median
final score; first error message if any. Telling the penalty arm how often it is feasible is
deliberate: the selection criterion depends on it and a practitioner would know it.

## 10. Analysis

* **Primary endpoint:** for each LLaMEA run, its best algorithm's mean test AOCC over the 10
  problems (15 seeds). Arms compared with two-sided Mann-Whitney U and Cliff's δ (n = 10 per arm);
  pairs: free vs penalty (RQ1), guided vs free (RQ2).
* **Secondary:** per-problem AOCC with Holm correction; held-out vs training; feasibility rate;
  evaluations to first feasible point; final score; sensitivity of conclusions to FEAS_TOL and to
  the score range.
* **Strategies (RQ3):** every valid candidate labelled with a fixed scheme (fixed / adaptive
  penalty, feasibility rules, ε-constrained, repair, internal multi-objective, constrained solver,
  hybrid) by an LLM classifier, validated by hand on 50–100 candidates (Cohen's κ). Frequencies
  over generations, performance conditioned on strategy, convergence of winning lineages
  (BLADE code-evolution graphs).

## 11. Secondary experiments (only after the main study)

* **Gradients (lens only):** penalty arm gets ∇(penalty); free arms get ∇f and the Jacobian of g;
  cost charged at the measured time ratio. Question: does gradient information change the
  representation effect?
* **Behavioural feedback:** a repeat of the best arm with constraint-related behaviour fed back to
  the LLM (fraction of evaluations spent infeasible, evaluations to first feasible, re-entries
  into the infeasible region). LLaMEA 1.4 also offers AST feature-guided mutation
  (`feature_guided_mutation`), off in the main study.

## 12. Validation of the harness (2026-10-18)

* 65 automated tests (`python -m pytest`), including: uniform coverage of integer values under
  uniform sampling (fails on purpose with the v5 mapping and with [0, n-1] bounds); hard budget
  (also against code that catches `Exception` or everything); identical evaluation in all arms;
  best-feasible record = brute force; NaN handling; timeouts (in func, SIGALRM, hard kill);
  syntax errors, missing class, crashes (`os._exit`); runs from scripts without a `__main__` guard;
  prompts differing only in the intended paragraphs; full LLaMEA loop with a fake LLM.
* End to end on the 10 problems × 3 seeds with real budgets (random search and DE, both arms):
  120/120 runs completed; max wall time per run 33 s (lens) and < 2 s (CEC), far below the limits.
  Random search gives identical results in both arms (same evaluations underneath). Mean AOCC:
  random search 0.45, DE + penalty 0.59, DE + Deb's rules 0.60.

* Baselines on the full test set (5 baselines × 10 problems × 15 seeds = 750 runs, all completed;
  `results/baselines/`). Mean AOCC: random 0.452, L-SHADE penalty 0.541 / Deb 0.544,
  CMA-ES penalty 0.522 / Deb 0.523. Within pairs (Mann-Whitney, Holm over 20 tests) the
  representation matters per problem but not on average: Deb better on RC12 (both algorithms,
  Cliff's δ ≈ +0.9) and RC16 (L-SHADE), penalty better on RC18 (both, δ ≈ −0.75), no difference
  on the lens and elsewhere. This is the reference the LLaMEA arms are compared against.
* Lens at the test budget (pilot, `results/lens50k_pilot/`, 5 seeds × 5 baselines at 50,000
  evaluations): lensgopt loss medians random 2.3e-1, CMA-ES 1.7–2.1e-1, L-SHADE 2.8e-2 (Deb) /
  7.4e-2 (penalty; Deb better, p = 0.016); consistent with the TFG (random ~5e-1, RLEMMO ~1.1e-2 at
  2.5M, LLaMEA's 642ace6e 2–6e-3 with gradient access, LDG-EA 3e-4). Lens AOCCs now 0.57–0.61.
  The lens rows of `results/baselines/` were removed (2,000-evaluation budget) and must be rerun at
  the test budget (150 runs × ~6 min: run on the DSlab).
* Lens caveat (resolved by D11): from random starts with 2,000 evaluations every baseline stays 2–3 orders of
  magnitude above the group template (median best RMS² 0.14–0.87 mm² vs 4.6e-4 mm²), so lens AOCCs
  are compressed in 0.50–0.53 and the lens contributes little to fitness differences. Options:
  larger lens budget, or a score range matched to reachable values; to decide before the main runs.

## 13. Known limitations

* The candidate code runs in a separate process but is not sandboxed (file system, network).
* Algorithms using their own unseeded RNG (`np.random.default_rng()` without a seed) are not
  reproducible; the harness seeds `random` and `np.random` only.
* A candidate that swallows `BaseException` (bare `except:`) in a loop is stopped by SIGALRM or by
  the hard kill; its record may be lost and the run scores 0 (status `timeout`).
* The lens worker caches the compiled simulator in `~/.cache/llamea_constraints_jax` (job start
  ~7 s instead of ~12 s).
* With `max_workers=1` LLaMEA ignores `eval_timeout`; the harness' own limits still apply.

## 14. Pending decisions (to fix before the main runs)

| id | decision | current default |
|---|---|---|
| D1 | realistic lens values | draft in `problems/lens.py` |
| D6 | lens instance for training | `lens/original` |
| D7 | CEC penalty weight ρ | 10·max(1, \|f*\|) |
| D8 | LLM and temperature | — |
| D10 | LDG-EA as lens baseline | needs code/results from K. Antonov |

Decided: D2 no gradients in the main study; D3 budgets as in §5; D4 score ranges as in §6;
D5 AOCC as fitness; D9 violation normalisation as in §6; D11 lens budgets 10k/50k; integers as typed indices with
[-0.5, n-0.5] bounds; FEAS_TOL = 1e-9; ray loss as graded constraint.

## Change log

* 2026-10-18 — first version (problems verified, harness implemented and tested). During testing:
  ray loss added as a graded constraint (lens: 23 → 24 constraints) and violation normalisation
  changed from per-tolerance (lens) / median |g| (CEC) to the common rule D9.
* 2026-10-19 — baselines implemented (L-SHADE and CMA-ES in penalty/Deb pairs, random search).
* 2026-10-19 — D11: lens budget 2,000 → 10,000 (train) / 50,000 (test), as in the TFG; with 2,000
  every baseline stayed 2–3 orders of magnitude above the template. Batch (vmap) lens evaluation
  added for GPU.
