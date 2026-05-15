# Pd 1,4-Conjugate

Pd 1,4-conjugate addition is another composed-force-field transfer case: the published TSFF combines a base MM3 field with an OPT overlay, and that composition does not carry over faithfully under our engine. The paper reports internal R² > 0.99 under MacroModel MM3*; our reproduction yields R² = −4.94 under the JAX engine.

## Scope

- Type: Transition state (Pd-catalyzed 1,4-conjugate addition)
- Molecules: 10 TS structures
- Parameters: 340 (OPT substructure: 20 bonds, 32 angles, 236 torsions)
- QM reference: B3LYP-D3/6-31G(d)

## Publication

| Property | Value |
|----------|-------|
| **Paper** | Wahlers, J. et al. *J. Org. Chem.* **2021**, *86*, 5660–5667 |
| **DOI** | [10.1021/acs.joc.1c00136](https://doi.org/10.1021/acs.joc.1c00136) |
| **System** | Pd-catalyzed 1,4-conjugate addition |
| **Training set** | 10 transition-state structures |
| **Engine** | MacroModel MM3* |

## What the paper fitted and reports

### What the original Q2MM workflow fitted

This TSFF belongs to the same Notre Dame Q2MM program: MacroModel MM3* plus the standard multi-target Q2MM fitting logic, not eigenvalue-only projection.[^joc]

- Structural targets
- Hessian/eigenvalue targets
- Q2MM optimization under MacroModel MM3*
- External selectivity validation on literature reactions

### What the paper reports

The paper and thesis summary report:[^joc]

- **Internal slopes:** ~1.01
- **Internal R²:** > 0.99
- **External validation:** 82 predictions
- **Selectivity R²:** 0.88 with y = 0.86x
- **Selectivity MUE:** 1.8 kJ/mol

## Our reproduction

| Metric | Value |
|--------|:-----:|
| Overall eigenvalue R² | -4.94 |
| Per-molecule R² range | all negative |
| Positive R² values | 0 / 10 |
| Aggregate frequency RMSD | 764.4 cm⁻¹ (per-molecule avg: 226.5) |

**What this means:** A negative R² means our engine's reproduction of the
published eigenspectrum is worse than simply predicting the average — a
complete failure of cross-engine transfer, not a small miss.

!!! warning "All molecules fail the reproduction test"
    Every molecule is negative. The reproduced eigenspectrum is worse than a mean-value model across the whole training set.

## Benchmark results

!!! warning "Ratio check failed — optimization skipped"
    The JaxLoss/ObjectiveFunction ratio (1.200) falls outside the
    [0.85, 1.15] tolerance, indicating JaxLoss is not a reliable
    surrogate for this system.

| Metric | Value |
|--------|:-----:|
| Ratio check | 1.200 (fail) |
| Initial ObjectiveFunction score | 8,257,780 |
| Optimization | Skipped |

**Why it fails:** The Seminario starting FF has negative eigenvalue R²
for all 10 molecules (R² = −4.94 overall). Unconstrained geometry
relaxation finds different local minima in JaxLoss vs
ObjectiveFunction, producing divergent loss values.

See [Optimizer Comparison](../benchmarks/optimizer-comparison.md) for
cross-system comparison and methodology details.

## Comparison and gap analysis

### Comparison

The literature TSFF is reported as an excellent internal fit and a strong selectivity model. Our reproduction does not retain that internal fit.

Here the leading explanation is again the **composed-force-field workflow**. The published TSFF is built from a standard MM3 base plus an OPT overlay. If our engine does not interpret that composition exactly the way MacroModel MM3* does, the fitted eigenvalue structure is not preserved.

Frequency-only optimization improves the benchmark metric substantially — the best Adam+cosine run lowers RMSD from **764.4** to **305.3 cm⁻¹** — but the reproduced eigenspectrum is still negative for every molecule. The transfer problem comes before optimizer choice.

### Gap analysis

To close the gap for Pd 1,4-conjugate addition, we would need:

1. **MacroModel-faithful composition of the base MM3 field and OPT overlay.**
2. **System-specific validation of Pd nonbonded and coupling behavior** after composition.
3. **A full Q2MM re-fit under the original objective** once the composed starting field behaves correctly.

The negative R² reflects a transfer gap in the composed FF workflow.

## Reproduce

```bash
python -m q2mm.diagnostics.cli --system pd-conjugate --backend jax --optimizer optax-adam-cosine
```

Raw data:
[`q2mm-data/benchmarks/pd-1,4-conjugate-addition/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/pd-1,4-conjugate-addition).

[^joc]: Wahlers, J. et al. *J. Org. Chem.* **2021**, *86*, 5660–5667. [DOI: 10.1021/acs.joc.1c00136](https://doi.org/10.1021/acs.joc.1c00136)
