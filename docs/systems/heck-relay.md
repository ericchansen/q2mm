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

Run with `--n-evals 10 --ratio-tol none` so the verdict is
statistically defensible against the per-call engine noise documented
in [#284](https://github.com/ericchansen/q2mm/issues/284).  Heck-relay
has the worst noise floor of any system in the suite.

| Metric | Value |
|--------|:-----:|
| Ratio check | 1.378 (out_of_band; upper bound 1.15 — gate bypassed via `--ratio-tol none`) |
| Initial ObjectiveFunction (n=10 mean) | 3.415 × 10⁶ ± 1.84 % CI₉₅ |
| Final ObjectiveFunction (n=10 mean) | 3.435 × 10⁶ ± 1.41 % CI₉₅ |
| Improvement (mean Δ%) | **−0.59 % (NOT SIGNIFICANT — CI₉₅ ± 3.26 %)** |
| L-BFGS-B iterations / OF evaluations | 2 / 2 |
| Optimizer | L-BFGS-B (scipy) over JaxLoss analytical gradients |
| Wall time | 1,451 s opt + ~38 min for 20 post-eval samples |

Per-category fit of the optimized force field (R² varies by ~5 %
across single-call GPU evaluations):

| Category | n_refs | R² (optimized) | RMSD |
|----------|-------:|---------------:|-----:|
| bond_length | 1140 | **0.980** | 0.046 Å |
| bond_angle | 2157 | **0.787** | 7.96 ° |
| eig_diagonal | 3121 | −12.6 | 0.48 |

Geometry is well-reproduced; the eigenmatrix remains the open
problem (the published Rosales FF was MM3*-fit and the residual
eigenmatrix gap reflects a real cross-engine MM3* ↔ JAX-engine
divergence for this chemistry, not a loader bug).

### End-to-end optimization with `--ratio-tol none`

Following the [pd-conjugate experiment in #276](https://github.com/ericchansen/q2mm/issues/276),
heck-relay was run with `--ratio-tol none` to bypass the gate and
see whether JaxLoss-guided optimization can produce useful real-OF
descent in a regime where the surrogate has formally broken down
(ratio 1.378, well outside [0.85, 1.15]).  During optimization
JaxLoss produced non-finite values 2 times in the line search; the
optimizer eventually reported a 0.7 % surrogate reduction.  When
that step is statistically tested against the real ObjectiveFunction
with n=10 samples, the result is **−0.59 % ± 3.26 % CI₉₅** — within
noise, so we cannot claim either an improvement or a worsening.

!!! success "Confirmed: --ratio-tol none does not unlock useful optimization here"
    Heck-relay has the noisiest per-call ObjectiveFunction in the
    published-FF suite (CI₉₅ ≈ 3.3 % even with n=10).  Within that
    window we see neither a statistically real improvement nor a
    statistically real worsening from JaxLoss-guided L-BFGS-B with
    the gate disabled.  Combined with the visible surrogate
    breakdown (2 non-finite values during line search), this
    confirms that **the `ratio_tol=0.15` gate is providing useful
    protection** — bypassing it doesn't help and just risks
    accepting a bad step on a low-sample-count run.  Closing this
    gap needs the engine-parity work in the gap analysis below
    plus the MM3 non-smooth fix from
    [#284](https://github.com/ericchansen/q2mm/issues/284), not
    a relaxed tolerance.

**Recommendation:** keep the default `ratio_tol=0.15`.

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
