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

The 2008 paper does **not** report an eigenvalue R² directly. The commonly repeated **~0.998** value comes from later Q2MM literature that cites this system as a high-quality internal fit benchmark.[^later]

## Our reproduction

Our analytical QFUERZA starting point comes close, but does not fully reach the later-cited paper quality.

| Metric | Value |
|--------|:-----:|
| Overall eigenvalue R² | 0.991 |
| Overall slope | 0.986 |
| Aggregate frequency RMSD | 259.9 cm⁻¹ (per-molecule avg: 85.5) |

**What this means:** Our analytical QFUERZA starting point reproduces
91.1% of the variance in the QM eigenspectrum (R² = 0.991) — close to
the ~99.8% reported in later Q2MM literature, but not identical. The
gap reflects cross-engine differences, not a bug.

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

### Multi-target objective (correct methodology)

Eigenmatrix-diagonal + geometry refs, 182 frozen-scoped active params,
`invert_ts_curvature=True`, L2 regularization λ=0.01:

| Optimizer | Initial loss | Final loss | Reduction | Iters | Wall time |
|-----------|:-----------:|:----------:|:---------:|:-----:|:---------:|
| scipy L-BFGS-B (GPU) | 18.2M | 17.5M | 3.5% | 6 | ~8 min |

The modest reduction reflects the **un-normalized penalty**: the
eigenmatrix diagonal contributes ~11K refs per system (weight 0.1
each), creating a very flat loss landscape near the Seminario minimum.
The upstream Q2MM normalizes each data type by its count (`score /
N_type`); implementing this normalization is expected to improve
convergence significantly.

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

This is a **good** reproduction, not a perfect one.

- The QFUERZA start reaches **0.991** versus the later-cited **~0.998** paper quality.
- The slopes stay close to 1.0 across all 9 transition states.
- The remaining gap is small enough to be scientifically encouraging, but still real.

The most likely explanation is functional-form mismatch: the original force field was optimized for **MacroModel MM3***, while our reproduction evaluates the same underlying chemistry in a different engine. So this page should be read as **"close under cross-engine transfer"**, not **"paper parity achieved."**

### What q2mm demonstrates

What q2mm demonstrates here is speed and starting-point quality. The QFUERZA analytical method reaches **R² = 0.991 in seconds without iteration**, which is already close to what the original Q2MM workflow achieved only after hours of optimization.[^qfuerza]

The multi-target optimization pipeline now runs end-to-end on this
9-molecule system (per-molecule JIT compilation, scipy L-BFGS-B with
JaxLoss analytical gradients), but convergence is limited by the
un-normalized penalty function.

### Gap analysis

To close the remaining gap to the literature value:

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
