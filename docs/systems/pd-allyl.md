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
pd-allyl.  Run with `--n-evals 10` so the verdict is statistically
defensible against the per-call engine noise documented in
[#284](https://github.com/ericchansen/q2mm/issues/284).

| Metric | Value |
|--------|:-----:|
| Ratio check | 1.091 (pass) |
| Initial ObjectiveFunction (n=10 mean) | 8.036 × 10⁶ ± 0.173 % CI₉₅ |
| Final ObjectiveFunction (n=10 mean) | 8.037 × 10⁶ ± 0.229 % CI₉₅ |
| Improvement (mean Δ%) | **−0.010 % (NOT SIGNIFICANT — CI₉₅ ± 0.40 %)** |
| L-BFGS-B iterations / OF evaluations | 2 / 2 |
| Gradient source | `jac="auto"` → `jac_mode="jax_loss"` (JaxLoss analytical) |
| Wall time | 1,289 s opt + ~16 min for 20 post-eval samples |

Per-category fit of the optimized force field (post-L-BFGS-B):

| Category | n_refs | R² |
|----------|-------:|----:|
| bond_length | 849 | 0.046 |
| bond_angle | 1,582 | 0.331 |
| eig_diagonal | 2,412 | −2.82 |

These numbers are from the published-start baseline. Reproduce with
`scripts/regenerate_convergence_results.py --starting-point published --system pd-allyl --n-evals 10`;
raw JSON output with provenance lives at
[`q2mm-data/benchmarks/pd-allyl-amination/from-published/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/pd-allyl-amination/from-published).
The canonical QFUERZA-start results (current default since q2mm#290)
live at [`convergence/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/pd-allyl-amination/convergence)
and are summarized in the
[QFUERZA-recovery doc](../benchmarks/qfuerza-recovery.md).

!!! success "Confirmed: published Wahlers FF sits at a JaxLoss local minimum (post angle-grad fix)"
    With n=10 samples the 95 % CI on the improvement is **±0.40 %**,
    which excludes any improvement larger than ~0.4 %.  Unlike
    [rh-conjugate](rh-conjugate.md) and [heck-relay](heck-relay.md)
    — which were "newly unlocked" by the MM3 angle gradient
    correctness fix ([#284](https://github.com/ericchansen/q2mm/issues/284))
    — pd-allyl's verdict did not change after the fix: a true
    JaxLoss local minimum is exactly where the published OPT values
    already sit, so the fix had no descent direction to expose.

    The published FF was fit by a different objective (MacroModel MM3*
    multi-target).  The q2mm JAX engine's eigenmatrix-diagonal
    objective places its local minimum in the same location, but
    that location is not a *good* fit by either objective's full
    metric (bond_length R² ≈ 0.05, eig_diagonal R² ≈ −2.8 —
    see "Comparison and gap analysis" below).

Improving on pd-allyl requires either (a) closing the MM3* ↔
JAX-engine functional-form gap so JaxLoss's local minimum aligns with
a better point on the real objective surface, or (b) using a
different optimizer / objective that doesn't rely on the
geometry-relaxation surrogate.  [#284](https://github.com/ericchansen/q2mm/issues/284)
tracks the underlying engine-side problems (MM3 non-smooth points
in particular).

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

Configure both `Q2MM_SUPPORTING_INFO` and `Q2MM_MM3_BASE` as described in
[External data for published systems](../getting-started.md#external-data-for-published-systems)
before running this command.

```bash
python -m q2mm.diagnostics.cli --system pd-allyl --backend jax --optimizer optax-adam-cosine
```

Raw data: [`q2mm-data/benchmarks/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks) → `pd-allyl-amination/`.

[^pdallyl]: Wahlers, J. et al. *Nat. Commun.* **2021**, *12*, 6508. [DOI: 10.1038/s41467-021-27065-2](https://doi.org/10.1038/s41467-021-27065-2)
