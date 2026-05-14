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
    The overall reproduction is still poor. A small number of molecules barely cross above zero, but the system as a whole remains negative. That is not paper-level behavior.

## Benchmark results

Converged using SciPy L-BFGS-B with JaxLoss analytical gradients
(ratio check bypassed) on RTX 5090 GPU.

| Metric | Value |
|--------|:-----:|
| Initial loss | 1.181 |
| Final loss | 1.153 |
| Reduction | 2.38% |
| Iterations | 31 |
| NaN rate | 0% |
| Convergence | ✓ (L-BFGS-B termination) |
| Bounds | ±20% |
| JIT compile | 227 s |
| Optimization | 13 s |

See [Optimizer Comparison](../benchmarks/optimizer-comparison.md) for
cross-system comparison and methodology details.

## Comparison and gap analysis

### Comparison

The paper reports a near-ideal internal Hessian fit. Our reproduction does not preserve that quality.

The most likely reason is not that the published chemistry is bad; it is that this TSFF is a **composed force field**. The published workflow layers an OPT substructure on top of an MM3 base field, and that composition does not transfer cleanly into our engine. When the base field and the overlay do not interact with the same semantics they had in MacroModel, the eigenvalue structure can collapse even if the original TSFF was good.

Frequency-only optimization still matters: the best Optax run lowers RMSD from **1068.7** to **214.0 cm⁻¹**, but the literature-transfer test remains negative overall. That means optimizer refinement can improve our benchmark metric without repairing the underlying engine-transfer gap.

### Gap analysis

To close the gap for Pd-allyl, we would need to:

1. **Validate base + OPT overlay composition exactly** against MacroModel MM3*.
2. **Confirm that metal-specific nonbonded and cross-term behavior** is transferred with the same conventions.
3. **Run a full Q2MM-style re-fit** only after the composed force field reproduces the intended starting eigenspectrum.

So the current result should be read as **"composed-force-field transfer is not yet faithful"**, not **"the literature TSFF was poor."**

## Reproduce

```bash
python -m q2mm.diagnostics.cli --system pd-allyl --backend jax --optimizer optax-adam-cosine
```

Raw data: [`q2mm-data/benchmarks/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks) → `pd-allyl-amination/`.

[^pdallyl]: Wahlers, J. et al. *Nat. Commun.* **2021**, *12*, 6508. [DOI: 10.1038/s41467-021-27065-2](https://doi.org/10.1038/s41467-021-27065-2)
