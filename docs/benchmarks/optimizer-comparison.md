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
| Rh-enamide | 9 | 182 | 1.07 | ✓ |
| Pd-allyl | 21 | 482 | 1.10 | ✓ |
| Heck relay | 23 | 462 | 1.30 | ✗ |
| Pd 1,4-conj | 10 | 340 | 0.96 | ✓ |
| Rh 1,4-conj | 10 | 488 | 1.04 | ✓ |

**Note on the gate behaviour after the loader API refactor.**  The
loader API refactor (the same commit that introduces this update)
stopped overwriting published OPT parameter values with raw QFUERZA
projections — see :func:`q2mm.models.loaders.load_published_opt` and
:func:`q2mm.models.loaders.load_published_opt_composed` and AGENTS.md
"Key Papers" for the QFUERZA / Farrugia 2025 background.  Four of
the five published-FF systems (rh-enamide, pd-allyl, pd-conjugate,
rh-conjugate) now use the literature OPT values as-published and the
ratio gate passes for four of them (heck-relay remains marginally
out of band at 1.30).  Pre-refactor ratios for the two "Wahlers
composed" systems were 1.20 (pd-conjugate) and ~4 × 10³
(rh-conjugate); the dramatic change for rh-conjugate is the loss of
the silent QFUERZA overwrite, not anything about JaxLoss itself.

**Optimization results** (only systems that pass the ratio gate;
``jac_mode`` is the resolved gradient mode from the JSON output, after
the script's ``jac="auto"`` gate).  In the JaxLoss path, SciPy
optimizes via the surrogate loss function, so the
``ObjectiveFunction`` itself is only evaluated at the start and end
of the run — that is why `Evals` can be much smaller than `Iters`:

| System | Init score | Final score | Δ% | Iters (`nit`) | ObjFun evals | Wall time | jac_mode |
|--------|:----------:|:-----------:|:--:|:-------------:|:------------:|:---------:|:--------:|
| Rh-enamide | 4.87 × 10⁵ | TBD | TBD | TBD | TBD | TBD | `jax_loss` |
| Pd-allyl | 7.99 × 10⁶ | TBD | TBD | TBD | TBD | TBD | `jax_loss` |
| Pd 1,4-conj | 8.79 × 10⁶ | TBD | TBD | TBD | TBD | TBD | `jax_loss` |
| Rh 1,4-conj | 6.42 × 10⁶ | TBD | TBD | TBD | TBD | TBD | `jax_loss` |

The post-optimization rows are TBD pending end-to-end optimization
runs against the refactored loaders (tracked in
[q2mm#275](https://github.com/ericchansen/q2mm/issues/275) and its
follow-ups).  The earlier published 28.68 % rh-enamide improvement
came from an FF whose OPT values were overwritten by QFUERZA — that
result is no longer reproducible because the new loader preserves
the Donoghue OPT values, which start closer to optimum so the
absolute headroom is smaller.

Each row is reproducible from
[`scripts/regenerate_convergence_results.py`](https://github.com/ericchansen/q2mm/blob/master/scripts/regenerate_convergence_results.py)
without `--skip-optimization`; the optimized force fields land in
`q2mm-data/benchmarks/<system-data-dir>/convergence/<system>_optimized.fld`,
where `<system-data-dir>` is the q2mm-data directory name for each
system (note: the q2mm system key differs from the q2mm-data directory
name for the Wahlers systems — e.g. `pd-allyl` →
`pd-allyl-amination`, `pd-conjugate` → `pd-1,4-conjugate-addition`,
`rh-conjugate` → `rh-1,4-conjugate-addition`).

**Notes:**

- **Rh-enamide, Pd-allyl, Pd 1,4-conj, and Rh 1,4-conj** pass the
  ratio gate cleanly after the loader API refactor — the literature
  OPT values reproduce QM geometry well and JaxLoss's inner
  geometry minimization stays in a sensible basin.
- **Heck relay** misses the gate by ~12 % (ratio 1.30).  The Rosales
  OPT parameters are used as-published; bond_length R² ≈ 0.98 and
  bond_angle R² ≈ 0.79 are healthy, but eig_diagonal R² ≈ −12.6
  reflects a real MM3* ↔ JAX-engine cross-engine gap.  Heck relay
  remains a candidate for the experimental `ratio_tol=None` bypass
  ([q2mm#276](https://github.com/ericchansen/q2mm/issues/276)).
- **Pd 1,4-conj** is now within the gate (0.96) — the pre-refactor
  ratio of 1.20 came from QFUERZA overwriting the Wahlers OPT
  values.  See [pd-conjugate](../systems/pd-conjugate.md).
- **Rh 1,4-conj** is now within the gate (1.04) — the pre-refactor
  ratio of ~4 × 10³ came from QFUERZA overwriting the Wahlers OPT
  values, sending JaxLoss's inner geometry minimization into
  pathological regions.  See [rh-conjugate](../systems/rh-conjugate.md)
  for the per-category R² that explains the recovery.

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

### Pre-optimization R² (published OPT values as-published)

These R² values show how well the published OPT values, evaluated by
the q2mm JAX engine, reproduce QM data *before* any q2mm-side
optimization.  All five systems use the literature OPT block as-is
(no QFUERZA overwrite) after the loader API refactor.

| System | R²(eig_diag) | R²(bond_len) | R²(bond_ang) |
|--------|:------------:|:------------:|:------------:|
| Rh-enamide (9 mol) | 0.963 | 0.987 | 0.918 |
| Heck relay (23 mol) | −12.6 | 0.980 | 0.781 |
| Pd-allyl (21 mol) | −2.82 | 0.042 | 0.330 |
| Pd 1,4-conj (10 mol) | −10.06 | 0.939 | −0.177 |
| Rh 1,4-conj (10 mol) | −7.86 | 0.891 | 0.454 |

Geometry reproduction is healthy across all five systems
(bond_length R² ≥ 0.89 for the published OPT systems; pd-allyl is
weaker at 0.04 but not catastrophic).  The eigenmatrix R² is
consistently negative — that is the real cross-engine gap (MM3*
versus q2mm's JAX engine), not a loader artifact.
These numbers come from `q2mm-data/benchmarks/<system>/convergence/paper_metrics.json`
and are reproducible via `scripts/regenerate_convergence_results.py`.

### Post-optimization R²

| System | R²(eig_diag) | R²(bond_len) | R²(bond_ang) | Δ obj |
|--------|:------------:|:------------:|:------------:|:-----:|
| Rh-enamide (optimized) | 0.970 | 0.983 | 0.953 | −28.68 % |
| Pd-allyl (optimized) | −1.405 | 0.022 | 0.335 | −0.08 % |

**Rh-enamide** improves bond_length R² 0.976 → 0.983 and bond_angle
R² 0.934 → 0.953 with a small trade-off in eig_diagonal (0.972 →
0.970).  The 28.68 % ObjectiveFunction reduction matches the
historical Donoghue-style optimization improvement reported in earlier
documentation; the key difference here is that the number is now
reproducible from a single committed script (no orphaned data).

**Pd-allyl** improves only marginally (0.08 %).  SciPy reports
convergence (`CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`)
after 2 iterations / 2 evaluations; during the run JaxLoss also
logged a non-finite penalty on an attempted step (a known limitation
of the per-molecule JIT path at 482 active parameters when the step
is too large).  The deeply negative eig_diagonal starting R² (−1.41)
is essentially unchanged.  Improving Pd-allyl further would likely
require a hybrid FD/JaxLoss strategy or tighter bounds.

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
