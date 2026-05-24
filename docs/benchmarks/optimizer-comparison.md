# Optimizer Comparison

## Methodology

All multi-target benchmarks use:

- **Objective**: eigenmatrix-diagonal + geometry refs via
  `ReferenceData.from_molecules()` with `invert_ts_curvature=True`
- **Parameters**: frozen base-FF, only OPT-substructure params active
  (matching the published parameter scope per system)
- **Gradients**: JaxLoss analytical gradients via implicit differentiation
  (per-molecule JIT, dispatched from Python). Falls back to finite-difference
  if the JaxLoss/ObjectiveFunction ratio check fails.
- **Optimizer**: SciPy L-BFGS-B with `ratio_tol=0.15` (validates
  JaxLoss agrees with ObjectiveFunction within ±15%)

## Convergence results (GPU)

The JaxLoss/ObjectiveFunction ratio check validates that JaxLoss is a
reliable surrogate before using its analytical gradients. Systems where
the ratio passes get fast JaxLoss-guided optimization; systems where it
fails fall back to finite-difference (impractical for 400+ active
parameters) or are skipped entirely.

The numbers below come from the committed regeneration script
`scripts/regenerate_convergence_results.py`; the raw JSON outputs (with
provenance: git SHAs, device, ratio_tol, timestamp) live in
[`ericchansen/q2mm-data/benchmarks/<system>/convergence/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks).

| System | Mols | Active | Ratio | Check |
|--------|:----:|:------:|:-----:|:-----:|
| Rh-enamide | 9 | 182 | 1.05 | ✓ |
| Pd-allyl | 21 | 482 | 1.11 | ✓ |
| Heck relay | 23 | 462 | ~10⁸¹ (out_of_band) | ✗ |
| Pd 1,4-conj | 10 | 340 | 1.20 | ✗ |
| Rh 1,4-conj | 10 | 488 | ~4 × 10³ | ✗ |

**Optimization results** (only systems that pass the ratio gate; rows
will be filled in as optimization runs are committed to q2mm-data):

| System | Init score | Final score | Δ% | Iters | jac |
|--------|:----------:|:-----------:|:--:|:-----:|:---:|
| Rh-enamide | 3.90 × 10⁵ | TBD | TBD | TBD | jax_loss |
| Pd-allyl | 8.00 × 10⁶ | TBD | TBD | TBD | jax_loss |

> Optimization artefacts for Rh-enamide and Pd-allyl from earlier code
> versions used to live in `q2mm/results/convergence/`.  They were
> deleted in the same change that introduced this regeneration script
> because no committed producer existed (violates AGENTS.md Rule 8 —
> every claim must be grounded in evidence).  Fresh optimization runs
> will populate the TBD cells above and write force fields to
> `q2mm-data/benchmarks/<system>/convergence/<system>_optimized.fld`.

**Notes:**

- **Rh-enamide and Pd-allyl** pass the ratio gate cleanly and are the
  two systems that can be optimized with JaxLoss analytical gradients
  out of the box.
- **Heck relay** has a catastrophically poor Seminario starting FF
  (bond_length R² ≈ −48, bond_angle R² ≈ −5).  Inside JaxLoss the
  inner geometry relaxation wanders into very-high-energy regions,
  producing a finite-but-astronomical surrogate value (~10⁸¹) that
  is classified `out_of_band`.  The absolute magnitude is
  non-deterministic across runs.  Optimization is impossible without
  first fixing the starting FF (see [heck-relay
  page](../systems/heck-relay.md)).
- **Pd 1,4-conj** misses the band by ~4 %.  This is the natural
  candidate for the experimental `ratio_tol=None` bypass — see
  [pd-conjugate](../systems/pd-conjugate.md).  Until that experiment
  has run and been validated against the real ObjectiveFunction, the
  default `ratio_tol=0.15` correctly skips this system.
- **Rh 1,4-conj** diverges by ~3 orders of magnitude.  Like Heck relay,
  the JaxLoss surrogate is unusable here without first improving the
  starting FF.  Previous sessions saw varying ratios (0.46–0.96) for
  this system depending on uncommitted state; the current committed
  baseline gives ~4 × 10³.  See [rh-conjugate](../systems/rh-conjugate.md)
  for the per-category R² explaining why the surrogate diverges.

### Why some systems fail the ratio check

JaxLoss and ObjectiveFunction both minimize molecular geometry to evaluate
bond lengths/angles. When the starting FF is good (R² > 0.9), both find
similar minima near the QM reference → ratio ≈ 1.0. When the starting FF
is poor (negative R²), unconstrained minimization can find different local
minima → ratio diverges.

This is not a bug — it is the ratio check correctly identifying that
JaxLoss is unreliable for the current parameter regime. As optimization
improves the FF, the ratio may converge toward 1.0 and JaxLoss may become
usable at later stages.

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
| Rh-enamide (9 mol) | 0.972 | 0.976 | 0.934 |
| Heck relay (23 mol) | −4.66 | −48.06 | −5.47 |
| Pd-allyl (21 mol) | −1.41 | 0.026 | 0.337 |
| Pd 1,4-conj (10 mol) | −4.47 | 0.435 | −0.118 |
| Rh 1,4-conj (10 mol) | −4.85 | −57.65 | −1.29 |

Rh-enamide starts near-optimal (R² > 0.93 in all categories). The other
four systems start far from optimal — the optimizer must close this gap.
These numbers come from `q2mm-data/benchmarks/<system>/convergence/paper_metrics.json`
and are reproducible via `scripts/regenerate_convergence_results.py`.

### Post-optimization R²

Post-optimization R² is available for systems where JaxLoss optimization
has been re-run against the regeneration script and committed to
`q2mm-data/benchmarks/<system>/convergence/paper_metrics.json`.  Run
`scripts/regenerate_convergence_results.py --system <name>` (no
`--skip-optimization`) to populate these rows.

| System | R²(eig_diag) | R²(bond_len) | R²(bond_ang) | Δ obj |
|--------|:------------:|:------------:|:------------:|:-----:|
| Rh-enamide (optimized) | TBD | TBD | TBD | TBD |
| Pd-allyl (optimized) | TBD | TBD | TBD | TBD |

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
reference data. Our current Pd-allyl R² is far below the published 0.998
because the Seminario starting point is poor and optimization was limited.
Rosales 2020 reports selectivity predictions rather than internal R².
Direct comparison requires matching reference-data scope.

## Recommendations

- Use `scipy-lbfgsb-jax` (CLI) or `ScipyOptimizer(method='L-BFGS-B',
  jac='auto')` with default `ratio_tol=0.15`. The ratio check validates
  JaxLoss reliability and falls back to FD if needed.
- **Do NOT use jaxopt L-BFGS** — its zoom linesearch triggers 30–60 min
  of extra XLA compilation post-JIT, making it impractical.
- For systems with poor Seminario starting points (negative R²), consider:
    - Running a short FD-gradient optimization first to improve the FF
      enough that JaxLoss geometry relaxation stabilizes
    - Tighter parameter bounds (±5%) to prevent geometry divergence
    - Using a restraint-based JaxLoss geometry relaxation (future work)
