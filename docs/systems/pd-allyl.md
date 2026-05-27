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

SciPy L-BFGS-B with JaxLoss analytical gradients.  After the loader
API refactor that preserves the published Wahlers OPT values
as-published (no QFUERZA overwrite), the ratio gate passes for
pd-allyl.

| Metric | Value |
|--------|:-----:|
| Ratio check | 1.08 (pass) |
| Initial score | 8.04 × 10⁶ |
| Final score | 8.03 × 10⁶ |
| Reduction | 0.13 % (real OF) — **within ~0.65 % per-call noise floor; no statistically meaningful improvement** |
| Iterations / Evaluations | 1 / 2 |
| Gradient source | `jac="auto"` resolved to `jac_mode="jax_loss"` (JaxLoss analytical) |
| Wall time | 1,172 s (20 min) |

Per-category fit of the optimized force field (post-L-BFGS-B):

| Category | n_refs | R² |
|----------|-------:|----:|
| bond_length | 849 | 0.043 |
| bond_angle | 1,582 | 0.335 |
| eig_diagonal | 2,412 | −2.82 |

These numbers are reproducible from `scripts/regenerate_convergence_results.py`
(no `--skip-optimization`); raw JSON output with provenance lives at
[`q2mm-data/benchmarks/pd-allyl-amination/convergence/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/pd-allyl-amination/convergence).

L-BFGS-B converges after a single iteration: the published Wahlers
OPT values are already at (or extremely close to) a JaxLoss local
minimum, so there is no descent direction.

!!! warning "Noise floor caveat — this result is within measurement noise"
    Repeated GPU `ObjectiveFunction(x0)` calls on pd-allyl vary by
    ~0.65 % (5-call IQR/median: 8.026e6 ± 5.2e4, range 8.00e6–8.05e6).
    The reported 0.13 % "improvement" sits well inside this band, so
    we **cannot** scientifically claim any improvement vs the published
    starting point.  Root cause traced to scipy `L-BFGS-B` Fortran
    internal state combined with MM3 non-smooth points in the energy
    surface; see [#284](https://github.com/ericchansen/q2mm/issues/284) for the full
    diagnosis.

The take-away is the same regardless: the published Wahlers FF is
either at a JaxLoss local minimum or so close to one that L-BFGS-B
can't find a descent direction within the noise.  Improving on
pd-allyl requires either (a) closing the MM3* ↔ JAX-engine
functional-form gap (so JaxLoss's local minimum aligns with the real
ObjectiveFunction's), or (b) using a different optimizer / objective
that doesn't rely on the geometry-relaxation surrogate, and (c)
addressing the engine non-determinism so smaller improvements can be
measured at all.

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
