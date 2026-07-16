# Benchmarks

!!! info "Reading guide"
    If you're **new to q2mm**, start with the [Tutorial](../tutorial.md) —
    this section is evidence and analysis, not a learning path. For
    **per-system results** and literature context, see the
    [Systems](../systems/rh-enamide.md) section.

q2mm's benchmark program answers three questions:

1. **Does the engine work?** Can q2mm load published force fields and
   reproduce their fit quality? → [Published FF Validation](published-ff-validation.md)
2. **Does the optimizer work?** Given QM reference data and a starting
   FF, can q2mm reduce the penalty function? →
   [Optimizer Comparison](optimizer-comparison.md),
   [Small Molecules](../systems/small-molecules.md)
3. **Is it fast enough?** When does GPU or the analytical-gradient path
   pay for itself? → [GPU Acceleration](gpu.md),
   [When Analytical Wins](crossover.md)

## Benchmark sections

| Section | What it covers | Key result |
|---------|---------------|------------|
| [Published FF Validation](published-ff-validation.md) | Load published FFs and document the MacroModel MM3* transfer boundary | Rh-enamide Check 1 reaches R² = 0.60 under q2mm; beating QFUERZA is out of scope without MacroModel parity |
| [QFUERZA Validation](qfuerza-validation.md) | Verify our QFUERZA implementation against the paper and Zenodo data; evaluate starting-point quality across 5 systems | Cisplatin matches Zenodo exactly for QFUERZA angles; all five TS systems have positive mass-weighted starting-point R² |
| [Optimizer Comparison](optimizer-comparison.md) | Compare the production SciPy L-BFGS-B + JAX executor path on literature-scale TS systems | 4 of 5 systems show significant q2mm-objective improvement; pd-allyl is a confirmed local minimum |
| [GPU Acceleration](gpu.md) | CPU-vs-GPU wall-clock comparisons | JAX GPU ~5.6× faster on Rh-enamide |
| [When Analytical Wins](crossover.md) | System-size crossover for analytical vs finite-difference gradients | Analytical gradients dominate above ~50 parameters |

## Systems

| System | Molecules | Parameters | Status |
|--------|:---------:|:----------:|--------|
| [Small Molecules](../systems/small-molecules.md) | 1 (CH₃F) | 6–12 | 82-combination matrix complete |
| [Rh-enamide](../systems/rh-enamide.md) | 9 | 182 | Validated against Donoghue *JCTC* 2008 |
| [Heck relay](../systems/heck-relay.md) | 23 | ~180 | QFUERZA evaluated; cross-engine gap |
| [Pd-allyl](../systems/pd-allyl.md) | 21 | ~200 | QFUERZA evaluated; cross-engine gap |
| [Pd 1,4-conjugate](../systems/pd-conjugate.md) | 10 | ~200 | QFUERZA evaluated; cross-engine gap |
| [Rh 1,4-conjugate](../systems/rh-conjugate.md) | 10 | ~200 | QFUERZA evaluated; cross-engine gap |

## What is not covered yet

- **Type-normalized penalties** — the upstream Q2MM divides each data
  type by its count (`score / N_type`). Our un-normalized eigenmatrix
  loss creates a flat landscape that limits L-BFGS-B convergence.
- **Selectivity prediction** — q2mm does not yet have the
  conformational-search infrastructure needed for end-to-end
  enantioselectivity benchmarks.

??? note "Metrics used in this section"
    | Metric | Meaning |
    |--------|---------|
    | **R²** | Coefficient of determination on eigenvalue scatter — 1.0 = perfect, 0 = no correlation, negative = worse than the mean. |
    | **RMSD** | Root-mean-square deviation between QM and MM frequencies (cm⁻¹). Lower is better. |
    | **TS** | Transition state — the highest-energy geometry along a reaction pathway. |
    | **FF** | Force field — the set of parameters defining how atoms interact. |
    | **TSFF** | Transition state force field — parameterized specifically for TS geometries. |

## Artifacts

Published-force-field re-evaluations are archived in
`test/fixtures/published_ff/`. Historical benchmark data (from earlier
frequency-only runs) is in the
[q2mm-data](https://github.com/ericchansen/q2mm-data) repo.
