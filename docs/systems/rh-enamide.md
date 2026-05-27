# Rh-Enamide

Rh-enamide is the clearest literature-reproduction case in the repo: a complete Rh(I)-diphosphine asymmetric hydrogenation TSFF with 9 training structures.

## Scope

- Type: Transition state (Rh-catalyzed asymmetric hydrogenation)
- Molecules: 9 TS structures
- Parameters: 182 (OPT substructure: 8 bonds, 23 angles, 48 torsions)
- QM reference: B3LYP/LACVP**

## Publication

| Property | Value |
|----------|-------|
| **Paper** | Donoghue, P. J. et al. *J. Chem. Theory Comput.* **2008**, *4*, 1313–1323 |
| **DOI** | [10.1021/ct800132a](https://doi.org/10.1021/ct800132a) |
| **System** | Rh(I)-diphosphine asymmetric hydrogenation of enamides |
| **Training set** | 9 transition-state structures |
| **Engine** | MacroModel MM3* |

## What the paper fitted and reports

### What the original Q2MM workflow fitted

The original Q2MM workflow fit a multi-target penalty function, not just Hessian eigenvalues.[^donoghue]

- Bond lengths
- Bond angles
- Torsions
- The full Hessian matrix
- Partial charges
- Relative energies

The paper reports penalty-function tolerances of **0.01 Å** for bonds, **0.5°** for angles, **1°** for torsions, and **0.02 e** for charges, with Newton-Raphson plus Simplex refinement inside MacroModel MM3*.[^donoghue]

The authors also describe two fitted variants:

- **RhH** — the standard fit
- **RhH-E** — an energy-emphasized fit

### What the paper reports

Donoghue et al. report strong structural and energetic agreement between QM and MM for the fitted force field.[^donoghue]

- **Bond RMSD:** ≤ 0.03 Å (Table 5)
- **Angle RMSD:** < 2° (Table 6)
- **Relative-energy RMSD:** 0.3–0.5 kcal/mol (Table 7)
- **External selectivity validation:** MUE = 0.6 kcal/mol across 18 test points

The 2008 paper does **not** report an eigenvalue R² directly. It
reports structural and energetic agreement (bond RMSD, angle RMSD,
relative-energy RMSD).

## Our reproduction

| Metric | Value |
|--------|:-----:|
| Overall eigenvalue R² | 0.991 |
| Overall slope | 0.986 |
| Aggregate frequency RMSD | 259.9 cm⁻¹ (per-molecule avg: 85.5) |

Our analytical QFUERZA starting point reproduces 99.1% of the
variance in the QM eigenspectrum (R² = 0.991) with slopes near 1.0
across all 9 transition states, without any iterative optimization.

| TS | Atoms | Eig R² | Slope | Freq RMSD |
|----|:-----:|:------:|:-----:|:---------:|
| 1 | 36 | 0.990 | 1.000 | 93.9 |
| 2 | 38 | 0.995 | 0.986 | 86.0 |
| 3 | 38 | 0.993 | 0.987 | 87.1 |
| 4 | 62 | 0.985 | 0.975 | 97.5 |
| 5 | 62 | 0.985 | 0.975 | 98.2 |
| 6 | 58 | 0.994 | 0.989 | 76.6 |
| 7 | 58 | 0.993 | 0.989 | 76.3 |
| 8 | 58 | 0.994 | 0.990 | 77.2 |
| 9 | 58 | 0.993 | 0.989 | 76.9 |

## Benchmark results

### Multi-target objective

Eigenmatrix-diagonal + geometry refs, 182 frozen-scoped active params,
`invert_ts_curvature=True`, SciPy L-BFGS-B with JaxLoss analytical
gradients on RTX 5090 GPU:

| Metric | Value |
|--------|:-----:|
| Ratio check | 1.07 (pass) |
| Initial score | 4.86 × 10⁵ |
| Final score | 2.71 × 10⁵ |
| Reduction | **44.66 %** (vs ~0.6 % per-call ObjectiveFunction noise floor — 77× above noise) |
| Iterations (L-BFGS-B `nit`) | 15 |
| ObjectiveFunction evaluations | 2 |
| Gradient source | `jac="auto"` resolved to `jac_mode="jax_loss"` (JaxLoss analytical) |
| Wall time | 739 s (12 min) |

Per-category fit of the optimized force field (post-L-BFGS-B):

| Category | n_refs | R² |
|----------|-------:|----:|
| bond_length | 500 | 0.989 |
| bond_angle | 1,050 | 0.954 |
| eig_diagonal | 1,395 | 0.968 |

These numbers are reproducible from `scripts/regenerate_convergence_results.py`
(no `--skip-optimization`); raw JSON output with provenance lives at
[`q2mm-data/benchmarks/rh-enamide/convergence/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/rh-enamide/convergence).

The ratio check confirms JaxLoss is a reliable surrogate for this
system — the published Donoghue OPT values reproduce QM geometry well
so unconstrained geometry relaxation finds the correct local minima.
A 44.66 % real-objective improvement against the published starting
point — a 77× ratio over the per-call ObjectiveFunction noise floor
of ~0.6 % (see "Noise floor caveat" below) — is the largest reduction
of any published-FF system in the suite (the published values are good
but leave meaningful headroom under the q2mm JAX engine's
eigenmatrix-diagonal objective).

!!! warning "Noise floor caveat"
    Repeated GPU `ObjectiveFunction(x0)` calls on rh-enamide vary by
    ~0.6 % across calls (5-call IQR / median).  Root cause traced to a
    combination of scipy `L-BFGS-B` Fortran internal state and MM3
    non-smooth points; see
    [#284](https://github.com/ericchansen/q2mm/issues/284)
    for the full diagnosis.  The 44.66 % reduction reported here is
    77× the per-call noise band and therefore robust — single-call
    measurements at this magnitude are scientifically reliable.

### Historical frequency-only results

!!! note "Different objective"
    Earlier benchmarks used a **frequency-only** objective with all
    ~2,742 FF parameters (not the paper's multi-target penalty with
    182 OPT-substructure params). These numbers remain useful for
    optimizer comparison on that specific task, but do not represent
    literature reproduction.

Under the frequency-only objective, JaxOpt L-BFGS lowered frequency
RMSD from 259.9 to **187.7 cm⁻¹** and Optax Adam+cosine reached
**199.5 cm⁻¹** — but optimizer improvement on frequency RMSD can
trade away eigenspectrum quality (Adam+cosine dropped to R² = 0.843
even as RMSD improved).[^later]

## Comparison and gap analysis

### Comparison

- QFUERZA reaches R² = 0.991 with slopes near 1.0 across all 9 TS structures.
- The 2008 paper does not report eigenvalue R², so a direct comparison is not possible.
- The original force field was optimized for MacroModel MM3*; our reproduction uses a different engine (JAX). Cross-engine functional-form differences account for any remaining gap.

### What q2mm demonstrates

QFUERZA reaches R² = 0.991 in seconds without iteration. The
original Q2MM workflow required hours of gradient optimization in
MacroModel to reach its final fit.[^qfuerza]

The multi-target optimization pipeline runs end-to-end on this
9-molecule system (per-molecule JIT compilation, scipy L-BFGS-B with
JaxLoss analytical gradients) and lowers the real ObjectiveFunction
by 44.66 % against the published Donoghue OPT starting point
(4.86 × 10⁵ → 2.71 × 10⁵ in 15 L-BFGS-B iterations).  An earlier
baseline reported a 28.68 % reduction; that result depended on an FF
whose Donoghue OPT values had been overwritten by QFUERZA.  The
current loader API preserves the published OPT values, so the
optimizer starts from a better point and still finds meaningful
headroom.

### Gap analysis

To improve further:

1. **Type-normalized penalties** — divide each data type's contribution by its count, matching the upstream Q2MM weighting. This is the most impactful change for convergence.
2. **Closer MacroModel MM3* parity** — especially any remaining metal-center functional details that do not transfer cleanly to our engine.
3. **Off-diagonal eigenmatrix elements** — the current objective uses diagonal-only; the papers use the full lower triangle (weight 0.05).

## Reproduce

```bash
# Multi-target (correct methodology)
python -m q2mm.diagnostics.cli --system rh-enamide --backend jax --optimizer scipy-lbfgsb
```

Raw data: [`q2mm-data/benchmarks/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks) → `rh-enamide/`.

[^donoghue]: Donoghue, P. J. et al. *J. Chem. Theory Comput.* **2008**, *4*, 1313–1323. [DOI: 10.1021/ct800132a](https://doi.org/10.1021/ct800132a)
[^later]: See the later Q2MM/QFUERZA literature discussion in [QFUERZA Validation](../benchmarks/qfuerza-validation.md).
[^qfuerza]: [QFUERZA Validation](../benchmarks/qfuerza-validation.md) summarizes why the analytical start is valuable even before iterative optimization.
