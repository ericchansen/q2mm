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
    preserves the published OPT values as-is, the ratio is 1.04 —
    comfortably inside the [0.85, 1.15] band.  JaxLoss-guided
    optimization is now possible.

| Metric | Value |
|--------|:-----:|
| Ratio check | 1.04 (in_band) |
| Initial ObjectiveFunction score | 6.42 × 10⁶ |
| Optimization | TBD pending end-to-end run against the refactored loader |

Per-category fit of the published Wahlers force field against the QM
training data (no QFUERZA — these are the published OPT values
evaluated by the q2mm JAX engine):

| Category | n_refs | R² |
|----------|-------:|----:|
| bond_length | 457 | 0.891 |
| bond_angle | 926 | 0.454 |
| eig_diagonal | 1254 | −7.86 |

The dramatic improvement vs the pre-refactor numbers (where
bond_length R² was −58) is the loss of the QFUERZA overwrite that
was destroying the published Wahlers fit.  The eigenmatrix R² is
still negative, reflecting the same MM3* ↔ JAX-engine cross-engine
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
