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

!!! success "Ratio gate now passes — loader API refactor"
    The pre-refactor loader silently overwrote the Wahlers OPT
    parameters with raw QFUERZA projections, sending JaxLoss's inner
    geometry minimization into pathological regions and producing
    ratios that varied wildly across runs (0.46 / 0.96 / ~4 × 10³ in
    successive sessions).  After the loader API refactor that
    preserves the published OPT values as-is, the ratio is in the
    1.02–1.07 range (run-to-run variation reflects the ~2 % per-call
    GPU noise documented below) — comfortably inside the
    [0.85, 1.15] band.  JaxLoss-guided optimization is now possible.

| Metric | Value |
|--------|:-----:|
| Ratio check | 1.02 (in_band) |
| Initial ObjectiveFunction score | 6.48 × 10⁶ |
| Final ObjectiveFunction score | 6.38 × 10⁶ |
| Improvement | 0.00 % (reverted after surrogate-guided step worsened real OF by 1.0 %) — **within ~2.1 % per-call noise floor** |
| Iterations / Evaluations | 2 / 2 |
| Optimizer | L-BFGS-B (scipy) over JaxLoss analytical gradients |
| Wall time | 624 s (10 min) |

!!! warning "Noise floor caveat — both the proposed worsening and the result are within noise"
    Repeated GPU `ObjectiveFunction(x0)` calls on rh-conjugate vary
    by **~2.1 %** (5-call IQR/median; min 6.35e6, max 6.49e6).  The
    JaxLoss-guided step's 1.0 % "worsening" that triggered the
    revert is **smaller than the noise floor**, so we cannot say
    whether the optimizer actually moved in the wrong direction or
    whether the surrogate just landed on a different point in the
    noise cloud.  The 4602-ratio non-determinism reported in an
    earlier session is partially the same phenomenon — the post-refactor
    ratio is more stable around 1.02–1.07 across runs (closes
    [#278](https://github.com/ericchansen/q2mm/issues/278)).  Root cause
    traced to scipy `L-BFGS-B` Fortran internal state in the geometry
    minimizer plus MM3 non-smooth points; see the engine
    non-determinism issue for the full diagnosis.

Per-category fit before and after optimization, evaluated by the q2mm
JAX engine against the QM training data (single GPU calls; per-category
R² varies by ~1–2 % across calls):

| Category | n_refs | R² (published) | R² (optimized) |
|----------|-------:|---------------:|---------------:|
| bond_length | 457 | 0.891 | 0.888 |
| bond_angle | 926 | 0.454 | 0.443 |
| eig_diagonal | 1,254 | −7.86 | −7.86 |

The take-away for rh-conjugate: in this noise regime we cannot
distinguish "the optimizer found a true descent direction" from
"the optimizer ran a step inside the noise cloud and got lucky/unlucky".
Reliable optimization for this system requires either fixing the
engine non-determinism first or using a noise-robust optimization
strategy (median-of-N evaluations, larger trust region, etc.).

The dramatic improvement vs the pre-refactor per-category numbers
(where bond_length R² was −58) is the loss of the QFUERZA overwrite
that was destroying the published Wahlers fit.  The eigenmatrix R²
is still negative, reflecting the same MM3* ↔ JAX-engine cross-engine
gap that affects all Wahlers systems.

See [Optimizer Comparison](../benchmarks/optimizer-comparison.md) for
the cross-system comparison.  Raw numbers are in the
[convergence baseline](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/rh-1,4-conjugate-addition/convergence)
in `ericchansen/q2mm-data`, with full provenance (q2mm git SHA, JAX/OpenMM
device, ratio_tol, timestamp).

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

```bash
python -m q2mm.diagnostics.cli --system rh-conjugate --backend jax --optimizer optax-adam
```

Raw data:
[`q2mm-data/benchmarks/rh-1,4-conjugate-addition/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/rh-1,4-conjugate-addition).

[^thesis]: Wahlers, J. *Ph.D. Dissertation*, University of Notre Dame, 2022, Ch. 6. The chapter-level ranges are also summarized in [Published FF Validation](../benchmarks/published-ff-validation.md).
