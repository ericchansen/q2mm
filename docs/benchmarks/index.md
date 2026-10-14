# Benchmarks

!!! info "Reading guide"
    If you're **new to q2mm**, start with the [Tutorial](../tutorial.md) —
    this section is evidence and analysis, not a learning path. For
    **per-system results** and literature context, see the
    [Systems](../systems/rh-enamide.md) section.

Benchmarks help distinguish software-path coverage from scientific fit quality
and historical timing measurements. q2mm's benchmark program answers three
questions:

1. **Which publication problems can be built?** What does each objective
   include or omit? → [Publication Reproduction Coverage](published-ff-validation.md)
2. **Does the optimizer work?** Given QM reference data and a starting
   FF, can q2mm reduce the penalty function? →
   [Optimizer Comparison](optimizer-comparison.md),
   [Small Molecules](../systems/small-molecules.md)
3. **Is it fast enough?** When does GPU or the analytical-gradient path
   pay for itself? → [GPU Acceleration](gpu.md),
   [When Analytical Wins](crossover.md)

## Benchmark sections

| Section | What it covers |
|---------|---------------|
| [Publication Reproduction Coverage](published-ff-validation.md) | Current profile/start records, target completeness, and source-linked limitations; not numerical validation |
| [QFUERZA Validation](qfuerza-validation.md) | Implementation comparisons against the paper and Zenodo data, and starting-point diagnostics |
| [Optimizer Comparison](optimizer-comparison.md) | Historical mixed-source optimization measurements with endpoint provenance and local-basin limits |
| [GPU Acceleration](gpu.md) | Historical CPU-vs-GPU wall-clock measurements and their artifacts |
| [When Analytical Wins](crossover.md) | Historical analytical-vs-finite-difference timing measurements and their artifacts |

## Registered systems

This inventory is generated from registered-system metadata. Registration does
not imply that external inputs are installed or that a publication objective is
complete. Follow the profile/start records rather than treating a system as one
validated/not-validated label.

<!-- BEGIN GENERATED publication-inventory -->

| Registered system | Default functional forms | Publication coverage |
|---|---|---|
| [CH3F](../systems/small-molecules.md) (`ch3f`) | `harmonic`, `mm3` | No canonical publication record |
| [F⁻ + CH3F SN2 TS](../systems/small-molecules.md) (`ch3f-sn2`) | `harmonic`, `mm3` | No canonical publication record |
| [Rh-enamide](../systems/rh-enamide.md) (`rh-enamide`) | `mm3` | [Profile/start records](published-ff-validation.md#rh-enamide) |
| [Heck relay](../systems/heck-relay.md) (`heck-relay`) | `mm3` | [Profile/start records](published-ff-validation.md#heck-relay) |
| [Pd-allyl amination](../systems/pd-allyl.md) (`pd-allyl`) | `mm3` | [Profile/start records](published-ff-validation.md#pd-allyl) |
| [Pd 1,4-conjugate addition](../systems/pd-conjugate.md) (`pd-conjugate`) | `mm3` | [Profile/start records](published-ff-validation.md#pd-conjugate) |
| [Rh 1,4-conjugate addition](../systems/rh-conjugate.md) (`rh-conjugate`) | `mm3` | [Profile/start records](published-ff-validation.md#rh-conjugate) |
| [Ferrocene ground-state force field](../systems/ferrocene.md) (`ferrocene`) | `mm3` | [Profile/start records](published-ff-validation.md#ferrocene) |

<!-- END GENERATED publication-inventory -->

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

Published-force-field re-evaluation fixtures are archived in
[`test/fixtures/published_ff/`](https://github.com/ericchansen/q2mm/tree/master/test/fixtures/published_ff).
Historical measurements are separate from the current inventory: the linked
analysis pages identify their tracked artifacts in
[q2mm-data](https://github.com/ericchansen/q2mm-data).
