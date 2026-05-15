# Pd-Allyl

Pd-allyl is a composed-force-field transfer case: the published Pd-catalyzed enantioselective allylic amination TSFF layers an OPT substructure (482 params) on top of an MM3 base field, and that composition does not survive cleanly under our engine. The benchmark still matters because optimizer refinement on this 21-structure system tests the frozen-parameter workflow at scale even though the literature-level internal fit does not transfer.

## Scope

- Type: Transition state (Pd-catalyzed allylic amination)
- Molecules: 21 TS structures
- Parameters: 482 (OPT substructure: 43 bonds, 88 angles, 220 torsions)
- QM reference: M06-D3/LANL2DZ/6-31+G*

## Publication

| Property | Value |
|----------|-------|
| **Paper** | Wahlers, J. et al. *Nat. Commun.* **2021**, *12*, 6508 |
| **DOI** | [10.1038/s41467-021-27065-2](https://doi.org/10.1038/s41467-021-27065-2) |
| **System** | Pd-catalyzed enantioselective allylic amination |
| **Training set** | 21 transition-state structures |
| **Engine** | MacroModel MM3* |

## What the paper fitted and reports

### What the original Q2MM workflow fitted

Like the other Notre Dame TSFF papers, this force field comes from the full Q2MM/MacroModel workflow rather than eigenvalue-only fitting.[^pdallyl]

- Simultaneous fitting of multiple target classes
- MacroModel MM3* throughout parameter refinement
- Internal validation reported separately for Hessian, geometry, and charges
- External validation reported on selectivity predictions

### What the paper reports

Wahlers et al. report:[^pdallyl]

- **Hessian R²:** 0.998
- **Geometry R²:** 0.988
- **Charges R²:** 0.822
- **External validation:** 77 selectivity predictions
- **Selectivity MUE:** 4.4 kJ/mol
- **Selectivity R²:** 0.41

## Our reproduction

| Metric | Value |
|--------|:-----:|
| Overall eigenvalue R² | -0.93 |
| Per-molecule R² range | -2.7 to +0.36 |
| Best molecule | Only mildly positive (+0.36) |
| Aggregate frequency RMSD | 1068.7 cm⁻¹ (per-molecule avg: 380.5) |

**What this means:** The overall negative R² means the published
eigenspectrum does not transfer cleanly into our engine. Even though a
few molecules are slightly positive, the system as a whole performs
worse than simply predicting the average.

!!! warning "Negative overall R²"
    The overall reproduction is still poor. A small number of molecules barely cross above zero, but the system as a whole remains negative. The paper reports Hessian R² = 0.998; our reproduction yields R² = −0.93.

## Benchmark results

SciPy L-BFGS-B with JaxLoss analytical gradients on RTX 5090 GPU.
The ratio check passed, confirming JaxLoss as a reliable surrogate
despite the poor Seminario starting FF.

| Metric | Value |
|--------|:-----:|
| Ratio check | 1.092 (pass) |
| Initial score | 7,998,071 |
| Final score | 7,993,193 |
| Reduction | 1.2% |
| Iterations | 3 |
| Gradient source | JaxLoss analytical |

The modest 1.2% improvement reflects the poor Seminario starting
point (negative eigenvalue R²) — the optimizer converges quickly to
a nearby local minimum with only marginal improvement.

See [Optimizer Comparison](../benchmarks/optimizer-comparison.md) for
cross-system comparison and methodology details.

## Comparison and gap analysis

### Comparison

The paper reports Hessian R² = 0.998 under MacroModel MM3*. Our reproduction yields R² = −0.93 under the JAX engine.

This TSFF is a **composed force field**: the published workflow layers an OPT substructure on top of an MM3 base field. That composition does not transfer cleanly into our engine — when the base field and the overlay do not interact with the same semantics they had in MacroModel, the eigenvalue structure collapses.

Frequency-only optimization still matters: the best Optax run lowers RMSD from **1068.7** to **214.0 cm⁻¹**, but the literature-transfer test remains negative overall. That means optimizer refinement can improve our benchmark metric without repairing the underlying engine-transfer gap.

### Gap analysis

To close the gap for Pd-allyl, we would need to:

1. **Validate base + OPT overlay composition exactly** against MacroModel MM3*.
2. **Confirm that metal-specific nonbonded and cross-term behavior** is transferred with the same conventions.
3. **Run a full Q2MM-style re-fit** only after the composed force field reproduces the intended starting eigenspectrum.

The negative R² reflects incomplete composed-force-field transfer, not a problem with the original TSFF.

## Reproduce

```bash
python -m q2mm.diagnostics.cli --system pd-allyl --backend jax --optimizer optax-adam-cosine
```

Raw data: [`q2mm-data/benchmarks/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks) → `pd-allyl-amination/`.

[^pdallyl]: Wahlers, J. et al. *Nat. Commun.* **2021**, *12*, 6508. [DOI: 10.1038/s41467-021-27065-2](https://doi.org/10.1038/s41467-021-27065-2)
