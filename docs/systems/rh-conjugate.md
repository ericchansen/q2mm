# Rh 1,4-Conjugate

Rh 1,4-conjugate addition is the thesis-era Rh analogue of the Pd composed-force-field systems: a base MM3 field plus OPT overlay that does not transfer cleanly under our engine.

## Scope

- Type: Transition state (Rh-catalyzed 1,4-conjugate addition)
- Molecules: 10 TS structures
- Parameters: 488 (OPT substructure: 24 bonds, 46 angles, 348 torsions)
- QM reference: B3LYP-D3/6-31G(d)

## Publication

| Property | Value |
|----------|-------|
| **Thesis** | Wahlers, J. *Ph.D. Dissertation*, University of Notre Dame, 2022, Ch. 6 |
| **DOI** | — |
| **System** | Rh-catalyzed 1,4-conjugate addition |
| **Training set** | 10 transition-state structures |
| **Engine** | MacroModel MM3* |

## What the thesis reports

### What the original Q2MM workflow fitted

The Chapter 6 Rh systems continue the same Q2MM strategy: a MacroModel MM3* transition-state force field fit against multiple data types, not just eigenvalues.[^thesis]

- Structural targets
- Hessian/eigenvalue targets
- MacroModel MM3* optimization
- External selectivity validation on literature examples

### Reported outcomes

Wahlers reports separate internal-fit ranges for two ligand classes:[^thesis]

- **Bisphosphine systems:** slopes 0.94–1.01, R² 0.91–0.99
- **Diene systems:** slopes 1.0–1.07, R² 0.92–0.99
- **Bisphosphine selectivity validation:** MUE 4.1 kJ/mol, R² = 0.64, 67 structures
- **Diene selectivity validation:** MUE 5.3 kJ/mol, R² = 0.37, 69 structures

## Our reproduction

| Metric | Value |
|--------|:-----:|
| Overall eigenvalue R² | -3.90 |
| Per-molecule R² range | all negative |
| Positive R² values | 0 / 10 |
| Aggregate frequency RMSD | 645.7 cm⁻¹ (per-molecule avg: 228.3) |

**What this means:** A negative R² means our engine's reproduction of the
published eigenspectrum is worse than simply predicting the average — a
complete failure of cross-engine transfer, not a small miss.

!!! warning "Negative across the full training set"
    All per-molecule R² values are negative. Under our engine, the reproduced eigenspectrum is not preserving the literature fit at all.

## Benchmark results

!!! success "Executor-ratio gate now passes — loader API refactor + MM3 angle gradient fix"
    The pre-refactor loader silently overwrote the Wahlers OPT
    parameters with raw QFUERZA projections, sending the JAX executor's inner
    geometry minimization into pathological regions and producing
    ratios that varied wildly across runs (0.46 / 0.96 / ~4 × 10³ in
    successive sessions).  After the loader API refactor that
    preserves the published OPT values as-is, the ratio is 1.00 —
    comfortably inside the [0.85, 1.15] band.  JAX-executor-guided
    optimization is now possible.

Run with `--n-evals 10` so the verdict is statistically defensible
against the per-call engine noise documented in
[#284](https://github.com/ericchansen/q2mm/issues/284).  Numbers
below are from the post-fix (q2mm MM3 angle gradient-correctness
patch) run.

| Metric | Value |
|--------|:-----:|
| Ratio check | 0.996 (in_band) |
| Initial Python objective (n=10 mean) | 6.293 × 10⁶ ± 2.60 % CI₉₅ |
| Final Python objective (n=10 mean) | 5.160 × 10⁶ ± 1.92 % CI₉₅ |
| Improvement (mean Δ%) | **18.00 % (SIGNIFICANT — CI₉₅ ± 4.17 %)** |
| L-BFGS-B iterations / OF evaluations | 2 / 2 |
| Optimizer | L-BFGS-B (scipy) over JAX analytical executor gradients |
| Wall time | 691 s opt + ~13 min for 20 post-eval samples |

Per-category fit before and after optimization (single GPU calls;
per-category R² varies by ~1–2 % across calls — see the noise
context below):

| Category | n_refs | R² (optimized) |
|----------|-------:|---------------:|
| bond_length | 457 | 0.822 |
| bond_angle | 926 | 0.540 |
| eig_diagonal | 1,244 | −12.85 |

!!! success "Newly unlocked: 18 % real-OF reduction after MM3 angle gradient fix"
    A previous baseline (q2mm-data#9) reported −0.080 % ± 1.18 % CI₉₅
    for this system — "NOT SIGNIFICANT, FF sits at a JAX-executor local
    minimum".  That conclusion was wrong.  The optimizer wasn't
    finding a real local minimum; it was hitting a spurious stationary
    point caused by a gradient correctness bug in the JAX angle term
    (`arccos(clip(cos θ))` killed gradient information at near-collinear
    geometries — see [#284](https://github.com/ericchansen/q2mm/issues/284)
    for the diagnosis).  After the fix replaces the clip with a
    custom-VJP `atan2`-based angle function, L-BFGS-B finds a real
    descent direction worth **18.00 % ± 4.17 % CI₉₅** in the real
    Python objective (and the surrogate score itself drops by 18 %
    in tandem, confirming the fix made the JAX executor honest about the
    actual descent direction).  The 4602-ratio non-determinism
    reported in an earlier session is also resolved (ratio now stable
    at 1.00 across runs; closes
    [#278](https://github.com/ericchansen/q2mm/issues/278)).

The eigenmatrix R² remains negative — the optimizer improved the
real Python objective substantially but the cross-engine
MM3* ↔ JAX-backend eigenmatrix gap that affects all Wahlers systems
is a separate, deeper issue (see "Comparison and gap analysis"
below).  bond_length and bond_angle R² went slightly down from the
pre-optimization values (0.89 → 0.82, 0.45 → 0.54) because the
optimizer traded a fraction of geometry-target fit for the much
larger eigenmatrix improvement that drove the overall Python objective
down by 18 %.

The dramatic improvement vs the pre-refactor per-category numbers
(where bond_length R² was −58) is the loss of the QFUERZA overwrite
that was destroying the published Wahlers fit.  The eigenmatrix R²
is still negative, reflecting the same MM3* ↔ JAX-backend cross-engine
gap that affects all Wahlers systems.

See [Optimizer Comparison](../benchmarks/optimizer-comparison.md) for
the cross-system comparison.  Raw numbers (published-start baseline) are in the
[`from-published/` baseline](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/rh-1,4-conjugate-addition/from-published)
in `ericchansen/q2mm-data`, with full provenance (q2mm git SHA, JAX/OpenMM
device, executor_ratio_tol, timestamp).  Canonical QFUERZA-start results
(default since q2mm#290) live at
[`convergence/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/rh-1,4-conjugate-addition/convergence)
and are summarized in the [QFUERZA-recovery doc](../benchmarks/qfuerza-recovery.md).

## Comparison and gap analysis

### Comparison

The thesis reports respectable-to-strong internal fits across both ligand classes. Our reproduction does not transfer that quality.

As with the Pd systems, this is a **composed-force-field transfer problem**. The Rh 1,4-conjugate TSFF combines a base MM3 field with an OPT overlay, and that composition is sensitive to engine-specific semantics.

The optimizer story is mixed but still informative: L-BFGS achieves **331.3 cm⁻¹**, and Optax Adam reaches **307.0 cm⁻¹**, yet the reproduced eigenspectrum remains negative across the entire training set. Better optimizer robustness helps the benchmark objective; it does not remove the transfer gap.

### Gap analysis

To close the gap for Rh 1,4-conjugate addition, we would need:

1. **A verified composition path for the base MM3 field plus OPT overlay.**
2. **Closer parity for Rh-specific MM3* behavior** at the metal center.
3. **A re-fit against the original multi-target Q2MM objective** only after the composed starting field behaves as intended.

The negative R² reflects a real transfer gap in the composed FF workflow.

## Reproduce

Configure both `Q2MM_SUPPORTING_INFO` and `Q2MM_MM3_BASE` as described in
[External data for published systems](../getting-started.md#external-data-for-published-systems)
before running this command.

```bash
python -m q2mm.diagnostics.cli --system rh-conjugate --backend jax --optimizer optax-adam
```

Raw data:
[`q2mm-data/benchmarks/rh-1,4-conjugate-addition/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/rh-1,4-conjugate-addition).

[^thesis]: Wahlers, J. *Ph.D. Dissertation*, University of Notre Dame, 2022, Ch. 6. The chapter-level ranges are also summarized in [Published FF Validation](../benchmarks/published-ff-validation.md).
