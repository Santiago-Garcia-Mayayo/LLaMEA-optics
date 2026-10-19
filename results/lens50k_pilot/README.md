# Lens pilot at the test budget (50,000 evaluations), `lens/original`, seeds 1000–1004

Purpose: check the harness against the TFG (Yotam Lev) on the same instance and metric.
"Loss" = lensgopt's penalised scalar loss (= our penalty arm on `lens/original`), minimum over the run.

| algorithm | budget | loss (median over seeds) | source |
|---|---|---|---|
| random search | 50k | 2.3e-1 (range 4.5e-2 – 2.6e-1) | this pilot |
| CMA-ES + penalty / + Deb | 50k | 2.1e-1 / 1.7e-1 | this pilot |
| L-SHADE + penalty / + Deb | 50k | 7.4e-2 / 2.8e-2 (Deb better, MW p = 0.016, n = 5) | this pilot |
| random search | 50k | ~5e-1 (average) | TFG |
| RLEMMO | 2.5M | ~1.1e-2 | TFG |
| 642ace6e (best LLaMEA, gradient access) | 10k / 50k | ~1e-2 / 2–6e-3, 12/20 seeds < 1e-2 | TFG |
| LDG-EA | cluster, 1 h | 3e-4 (best) | LDG-EA paper via TFG |
| group template design | — | 1.06e-3 | lensgopt |

Ordering is consistent: random < CMA-ES < L-SHADE < RLEMMO (50x budget) < 642ace6e < LDG-EA.
Caveat: 642ace6e had gradient access (TFG "v4 True"); our baselines have none (DESIGN.md D2).
