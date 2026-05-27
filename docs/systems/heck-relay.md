# Heck Relay

Heck relay is a strong negative reproduction case: the published Pd-catalyzed asymmetric redox-relay Heck TSFF is reported as an excellent internal fit in MacroModel MM3*, but it does not transfer cleanly under our engine. It is also a useful optimizer benchmark — the published Rosales force field exposes **462 active OPT parameters** across 23 transition-state structures, but Seminario re-estimation of the standard MM3 backbone produces a catastrophically bad starting force field for our optimizer.

## Scope

- Type: Transition state (Pd-catalyzed asymmetric Heck reaction)
- Molecules: 23 TS structures (paper reports 24; one is excluded from our training set)
- Parameters: 462 active OPT parameters (overlay on a standard MM3 base of ~2,500 frozen parameters; total ~3,000)
- QM reference: M06-GD3/LANL2DZ/6-31+G*

## Publication

| Property | Value |
|----------|-------|
| **Paper** | Rosales, A. R. et al. *J. Am. Chem. Soc.* **2020**, *142*, 9700–9707 |
| **DOI** | [10.1021/jacs.0c01979](https://doi.org/10.1021/jacs.0c01979) |
| **System** | Pd-catalyzed asymmetric redox-relay Heck reaction |
| **Training set** | 23 transition-state structures |
| **Engine** | MacroModel MM3* |

## What the paper fitted and reports

### What the original Q2MM workflow fitted

Rosales follows the same Q2MM penalty-function logic introduced in the earlier Donoghue work: a multi-target fit under MacroModel MM3*, rather than eigenvalue matching alone.[^rosales]

- Geometries and structural targets
- Hessian/eigenvalue information
- The usual Q2MM penalty-function balancing across multiple data types
- MacroModel MM3* as the evaluation engine throughout fitting

The training set contains **23 transition-state structures**.[^rosales]

### What the paper reports

From the paper and the supporting dissertation summary:[^jacs]

- **Internal structural/eigenvalue fit:** R² > 0.998
- **Slopes:** 1.000 ± 0.004
- **External selectivity validation:** 151 predictions
- **Selectivity RMSD:** 2.3 kJ/mol
- **Selectivity MUE:** 1.8 kJ/mol
- **Correct assignments:** 98%

The JACS paper does **not** report an eigenvalue R² table in the same form used here; the high internal-fit numbers come from the Rosales thesis discussion of the same TSFF program.[^rosales]

## Our reproduction

| Metric | Value |
|--------|:-----:|
| Overall eigenvalue R² | -8.89 |
| Per-molecule R² range | -13.1 to -6.6 |
| Positive R² values | 0 / 23 |
| Aggregate frequency RMSD | 1592.4 cm⁻¹ (per-molecule avg: 429.5) |

**What this means:** A negative R² means our engine's reproduction of the
published eigenspectrum is worse than simply predicting the average — a
complete failure of cross-engine transfer, not a small miss.

!!! warning "Complete failure of reproduction"
    This is a **complete failure of reproduction**. A negative R² means the reproduced eigenvalue pattern is worse than predicting the mean eigenvalue. Every single molecule is negative.

## Benchmark results

!!! success "Loader bug fixed (ericchansen/q2mm#277) — Rosales FF now usable"
    The previous loader silently re-ran QFUERZA (the old
    ``estimate_force_constants(forcefield=ff)`` overload — since
    deleted in favour of separate ``qfuerza_fresh`` / ``qfuerza_into``
    APIs) on the Rosales FF, which overwrote the published OPT
    parameters with raw FUERZA projections.  In the three-baseline
    diagnostic
    (`q2mm-data/benchmarks/heck-relay/diagnostic/three_baseline_comparison.json`)
    the buggy combination collapsed bond_length R² to ≈ **−4787** for
    that run (the value is non-deterministic across runs because the
    inner geometry minimization diverges chaotically; a prior committed
    convergence baseline saw ≈ −48 for the same code).  Either way the
    fit is catastrophic.  The loader now keeps the Rosales OPT values
    as-is (it still calls `freeze_standard_params` so the active mask
    is correct for optimization, but no QFUERZA re-estimation).

| Metric | Value |
|--------|:-----:|
| Ratio check | 1.255 (out_of_band; upper bound 1.15) |
| Initial ObjectiveFunction score | 3.53 × 10⁶ (single GPU eval — see noise caveat below) |
| Final ObjectiveFunction score | 3.39 × 10⁶ (single GPU eval at *reverted* params; the difference vs initial is GPU noise, not optimization) |
| End-to-end optimization (`--ratio-tol none`) | 0.00 % (reverted, see below) |

The reverted parameters are bit-identical to the initial parameters —
the 3.53 × 10⁶ → 3.39 × 10⁶ apparent change in the table reflects
that two separate GPU evaluations of `ObjectiveFunction(x0)` differ
by ~4 % due to engine non-determinism, not actual optimization.  See
the noise caveat in "End-to-end optimization with `--ratio-tol none`"
below.

Per-category fit of the published Rosales force field against the QM
training data (no QFUERZA — these are the published OPT values
evaluated by the q2mm JAX engine):

| Category | n_refs | R² | RMSD | MAE |
|----------|-------:|----:|-----:|----:|
| bond_length | 1140 | **0.980** | 0.046 Å | — |
| bond_angle | 2157 | **0.786** | 7.96 ° | — |
| eig_diagonal | 3121 | −12.6 | 0.48 | — |

Geometry is now well-reproduced; the eigenmatrix remains the open
problem (the published Rosales FF was MM3*-fit and the residual
eigenmatrix gap reflects a real cross-engine MM3* ↔ JAX-engine
divergence for this chemistry, not a loader bug).

### End-to-end optimization with `--ratio-tol none`

Following the [pd-conjugate experiment in #276](https://github.com/ericchansen/q2mm/issues/276),
heck-relay was also run with `--ratio-tol none` to bypass the gate
and see whether JaxLoss-guided optimization would still produce a
useful descent step.  Unlike pd-conjugate (which post-refactor had
ratio = 0.985 and the bypass was a no-op), heck-relay's ratio is
genuinely outside band at 1.255.  L-BFGS-B took 4 iterations against
JaxLoss; the surrogate produced non-finite values 3 times during the
line search; the proposed parameter step was evaluated against the
real `ObjectiveFunction` and was 3.3 % *worse* than the starting
point (3,392,860 → 3,503,130); `ScipyOptimizer` reverted to the
initial parameters and the run reports 0.00 % improvement.

!!! warning "Noise floor caveat — the 3.3 % apparent worsening is within noise"
    Repeated GPU `ObjectiveFunction(x0)` calls on heck-relay vary by
    **~4.9 %** (5-call IQR/median; min 3.36e6, max 3.53e6).  This is
    the worst noise floor of any system in the published-FF suite.
    The "3.3 % worse" step that triggered the revert is **inside the
    noise floor**, so we cannot reliably say the JaxLoss-guided step
    was actually a bad direction; the comparison itself is noise-limited.
    What we **can** say is that the surrogate produced 3 non-finite
    values mid-optimization, indicating real numerical instability
    rather than just measurement noise.  Root cause traced to scipy
    `L-BFGS-B` Fortran state in the geometry minimizer plus MM3
    non-smooth points; see [#284](https://github.com/ericchansen/q2mm/issues/284) for the
    full diagnosis.

**Recommendation:** keep the default `ratio_tol=0.15`.  Heck relay
remains a case where the gate provides useful protection — the
non-finite values during line search are a genuine surrogate
breakdown signal, distinct from the noise floor.  Closing this gap
requires the engine-parity work in the gap analysis below plus the
engine non-determinism fix, not a relaxed tolerance.

The numbers above come from the committed regeneration script
`scripts/regenerate_convergence_results.py`; the raw JSON output and
the three-baseline diagnostic that diagnosed the loader bug live at
[`q2mm-data/benchmarks/heck-relay/convergence/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/heck-relay/convergence)
and [`q2mm-data/benchmarks/heck-relay/diagnostic/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/heck-relay/diagnostic).

See [Optimizer Comparison](../benchmarks/optimizer-comparison.md) for
cross-system comparison and methodology details.

## Comparison and gap analysis

### Comparison

The paper reports R² > 0.998 with slopes 1.000 ± 0.004 under MacroModel MM3*.[^jacs] Our reproduction yields R² = −8.89 under the JAX engine — a complete failure of cross-engine transfer.

This is **not** a composed-force-field problem. The Heck relay FF is already a complete Rosales force field. That makes this case especially important: it points to a more fundamental **cross-engine gap** for this system's chemistry rather than a simple overlay/composition artifact.

### Gap analysis

To close this gap, we would need to do more than rerun optimization:

1. **Match the relevant MacroModel MM3* behavior** for this Pd/Heck chemistry much more closely.
2. **Audit the metal-center and torsional functional details** that may transfer differently across engines.
3. **Re-fit under the original multi-target Q2MM objective** once engine parity is good enough to make that optimization meaningful.

Heck relay identifies a cross-engine boundary for literature transfer that cannot be resolved by optimizer choice alone.

## Reproduce

```bash
python -m q2mm.diagnostics.cli --system heck-relay --backend jax --optimizer optax-adam-cosine
```

Raw data: [`q2mm-data/benchmarks/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks) → `heck-relay/`.

[^jacs]: Rosales, A. R. et al. *J. Am. Chem. Soc.* **2020**, *142*, 9700–9707. [DOI: 10.1021/jacs.0c01979](https://doi.org/10.1021/jacs.0c01979)
[^rosales]: Rosales, A. R. *Ph.D. Dissertation*, University of Notre Dame, 2019, Ch. 2. The dissertation-level selectivity summary is also reflected in [Published FF Validation](../benchmarks/published-ff-validation.md).
