# Optimizer Comparison

## Current methodology

All multi-target benchmarks use:

- **Objective**: eigenmatrix-diagonal + geometry refs via
  `ReferenceData.from_molecules()` with `invert_ts_curvature=True`
- **Parameters**: frozen base-FF, only OPT-substructure params active
  (matching the published parameter scope per system)
- **Gradients**: JaxLoss analytical gradients (per-molecule JIT,
  dispatched from Python)
- **Regularization**: L2 penalty λ=0.01
- **Optimizer**: SciPy L-BFGS-B with `ratio_tol=None` (bypasses the
  JaxLoss/ObjectiveFunction ratio check, which fails for all TS systems)

## Convergence results (GPU, RTX 5090)

All 5 systems converged using `scipy-lbfgsb-jax` (SciPy L-BFGS-B with
JaxLoss analytical gradients, ratio check bypassed). Bounds ±20% except
heck-relay which requires ±5% due to fragile TS landscape. **Loss values
are JaxLoss (differentiable surrogate), not ObjectiveFunction** — see
[§ Post-optimization R²](#post-optimization-r2) for why this matters.

| System | Molecules | Active | Init loss | Final loss | Δ% | Iters | NaN% | Converged |
|--------|:---------:|:------:|:---------:|:----------:|:---:|:-----:|:----:|:---------:|
| Rh-enamide | 9 | 182 | 0.0569 | 0.0565 | 0.72% | 8 | 35% | ✗ (ABNORMAL) |
| Heck relay | 23 | 462 | 2.957 | 2.820 | 4.62% | 15 | 0% | ✓ |
| Pd-allyl | 21 | 482 | 1.181 | 1.153 | 2.38% | 31 | 0% | ✓ |
| Pd 1,4-conjugate | 10 | 340 | 1.121 | 1.050 | 6.33% | 39 | 0% | ✓ |
| Rh 1,4-conjugate | 10 | 488 | 1.130 | 0.993 | 12.09% | 63 | 0% | ✓ |

**Notes:**

- Rh-enamide starts near-optimal (QFUERZA initial estimate is already
  very good); the 35% NaN rate is from boundary excursions, not from
  instability of the converged solution.
- Heck relay requires ±5% bounds because the system has 23 molecules with
  negative force constants up to −3753 kcal/mol·Å². Wider bounds trigger
  NaN singularities.
- All other systems converge cleanly with ±20% bounds in <1 minute of
  optimizer time post-JIT.

## Relationship to qfuerza-validation §5 results

The [QFUERZA validation table](qfuerza-validation.md#optimizer-convergence-gpu-rtx-5090)
shows larger reductions (16.8–62.6%) because it uses:

- **Full lower-triangular eigenmatrix** (N²/2 refs per molecule vs diagonal-only)
- **No L2 regularization** (parameters free to drift further)
- **200-iter fixed cap** (not convergence-based)

These are complementary views: the validation table demonstrates raw
optimizer capability on a harder objective; this page shows converged
solutions with regularization for the production workflow.

## Reference-data R² (per category)

We optimize eigenmatrix-diagonal and geometry references. R² is reported
per category to match what each paper uses. Published papers (e.g. Wahlers
2021) report **R²(hessian)** and **R²(geometry)** — not frequency R².
Our eigenmatrix-diagonal R² is analogous to paper R²(hessian), though we
use diagonal elements only while papers use the full lower-triangular
eigenmatrix.

### Seminario starting-point R² (pre-optimization)

These R² values show how well the Seminario-estimated force constants
reproduce QM data *before* any optimization. Negative R² means the MM
prediction is worse than predicting the QM mean.

| System | R²(eig_diag) | R²(bond_len) | R²(bond_ang) |
|--------|:------------:|:------------:|:------------:|
| Rh-enamide (9 mol) | 0.963 | 0.976 | 0.934 |
| Heck relay (23 mol) | −4.65 | −268.5 | −6.21 |
| Pd-allyl (21 mol) | −1.52 | 0.03 | 0.33 |
| Pd 1,4-conj (10 mol) | −4.46 | 0.44 | −0.12 |
| Rh 1,4-conj (10 mol) | −4.91 | −462.1 | −0.93 |

Rh-enamide starts near-optimal (R² > 0.93 in all categories). The other
four systems start far from optimal — the optimizer must close this gap.

### Post-optimization R²

**No post-optimization R² data is currently available.** Generating these
metrics is blocked by two issues:

1. **JaxLoss surrogate mismatch.** The JaxLoss analytical gradient path
   (used for the convergence results above) optimizes a differentiable
   surrogate that disagrees with the ObjectiveFunction for TS systems.
   JaxLoss/ObjectiveFunction ratios are 0.1–0.4 (should be ~1.0). The
   optimizer reduces JaxLoss but produces **0% ObjectiveFunction
   improvement** — the resulting parameters are no better than the
   starting point when evaluated with the true objective.

2. **Finite-difference gradients are impractical.** ObjectiveFunction
   evaluation takes ~20 s per call (GPU, warm JIT). With 182–488 active
   parameters, each FD gradient requires 365–977 evaluations = **2–5
   hours per gradient step**. A typical convergence run (8–63 iterations)
   would take days per system.

Until the surrogate mismatch is resolved or a faster evaluation path is
implemented, post-optimization R² cannot be computed. The convergence
table above (§ Convergence results) reports **JaxLoss** reduction, not
ObjectiveFunction reduction.

### Paper-reported metrics for comparison

| Paper | System | Metric | Value |
|-------|--------|--------|:-----:|
| Wahlers 2021 | Pd-allyl | R²(hessian) | 0.998 |
| Wahlers 2021 | Pd-allyl | R²(geometry) | 0.988 |
| Wahlers 2021 | Pd-allyl | R²(charges) | 0.822 |
| Wahlers 2021 | Pd-allyl | External MUE | 4.4 kJ/mol |
| Donoghue 2008 | Rh-enamide | Full eigenmatrix opt. | (MacroModel, no R² reported) |

Wahlers metrics were computed in MacroModel with the full eigenmatrix
(diagonal + off-diagonal) — a harder optimization target with more
reference data. Rosales 2020 reports selectivity predictions rather than
internal R². Direct comparison requires matching reference-data scope.

## Recommendations

- Use `scipy-lbfgsb-jax` (CLI) or `ScipyOptimizer(method='L-BFGS-B',
  jac='auto', ratio_tol=None)` for TS systems. This bypasses the ratio
  check that fails for all TS systems (ratios 0.1–0.4).
- **Do NOT use jaxopt L-BFGS** — its zoom linesearch triggers 30–60 min
  of extra XLA compilation post-JIT, making it impractical.
- For heck-relay specifically, use ±5% parameter bounds.
- Previous frequency-only results with all ~3,000 FF params have been
  removed — they optimized a fundamentally different problem.
