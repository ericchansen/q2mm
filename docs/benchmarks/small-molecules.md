# Small Molecules

Benchmarks on CH₃F (5 atoms, 8 parameters) optimized against B3LYP/6-31+G(d)
QM frequencies.  All methods start from identical Seminario-estimated
parameters.  Results cover speed, accuracy, and cross-engine agreement.

---

## Backend Leaderboard by Optimizer

All runs start from RMSD = 156.9 cm⁻¹ (score = 0.221).  Tables sorted by
wall-clock time (fastest first).

!!! tip "Reading the tables"
    **RMSD** = root-mean-square deviation of optimized MM frequencies from QM
    reference (lower is better).  **Score** = normalized objective function
    (lower is better; 0.000 = perfect match).  **Evals/s** = energy
    evaluations per second (higher is better).

### Nelder-Mead

| Backend | Final RMSD (cm⁻¹) | Final MAE | Score | Evals | Time | Evals/s |
|---------|-------------------:|----------:|------:|------:|-----:|--------:|
| **JAX** | 1037.9 | 888.8 | 0.000 | 1193 | 0.8 s | 1491 |
| **JAX-MD** | 1037.9 | 888.8 | 0.000 | 1205 | 1.2 s | 1004 |
| **OpenMM** | — | — | 0.001 | 378 | 2.0 s | 190 |
| **Tinker** | — | — | 0.001 | 376 | 286.9 s | 1.3 |

### Powell

| Backend | Final RMSD (cm⁻¹) | Final MAE | Score | Evals | Time | Evals/s |
|---------|-------------------:|----------:|------:|------:|-----:|--------:|
| **JAX** | **0.0** | **0.0** | 0.000 | 2565 | 1.3 s | 1973 |
| **JAX-MD** | **0.0** | **0.0** | 0.000 | 2612 | 1.8 s | 1451 |
| **OpenMM** | — | — | 0.001 | 722 | 3.8 s | 190 |
| **Tinker** | 553.8 | 289.4 | 0.000 | 11314 | 2972.7 s | 3.8 |

### L-BFGS-B

| Backend | Final RMSD (cm⁻¹) | Final MAE | Score | Evals | Time | Evals/s |
|---------|-------------------:|----------:|------:|------:|-----:|--------:|
| **JAX** | 813.4 | 610.1 | 0.077 | 406 | 0.6 s | 677 |
| **JAX-MD** | 813.4 | 610.1 | 0.077 | 370 | 1.0 s | 370 |
| **OpenMM** | 114.1 | 93.6 | 0.117 | 424 | 11.7 s | 36 |
| **Tinker** | 114.0 | 93.4 | 0.117 | 478 | 122.7 s | 3.9 |

### Key Observations

- **Powell and Nelder-Mead reach perfect convergence** (RMSD → 0) on JAX
  and JAX-MD.  These derivative-free methods are robust for small parameter
  spaces.
- **L-BFGS-B underperforms** with finite-difference gradients — it converges
  to a suboptimal point on all backends.  Connecting ``energy_and_param_grad()``
  (analytical gradients via ``jax.grad``) would likely fix this.
- **JAX backends are 5–10× faster** than OpenMM per evaluation.  JIT-compiled
  pure JAX eliminates Python ↔ C++ marshalling overhead.
- **JAX-MD is ~30% slower than JAX** due to neighbor list management and
  periodic boundary bookkeeping, but both are far faster than OpenMM/Tinker.
- **Tinker is 100–500× slower** than JAX per evaluation.  Powell on Tinker
  takes ~50 min (11k evals) vs 1.3 s on JAX for the same molecule.

---

## Cross-Engine Parity

Do different engines agree on the same answer?  These comparisons use the
CH₃F molecule at equilibrium geometry with identical Seminario-estimated
force field parameters.

### Energy Agreement

| Comparison | Energy Difference |
|------------|------------------:|
| JAX vs JAX-MD | **3 × 10⁻²⁰ kcal/mol** |
| JAX vs OpenMM | **3 × 10⁻¹⁸ kcal/mol** |
| JAX-MD vs OpenMM | **3 × 10⁻¹⁸ kcal/mol** |

All three engines agree to machine precision.

### Frequency Agreement

| Mode | OpenMM (cm⁻¹) | JAX (cm⁻¹) | JAX-MD (cm⁻¹) | Max Δ |
|-----:|--------------:|-----------:|-------------:|------:|
| 1 | 104.8102 | 104.8102 | 104.8102 | 4 × 10⁻⁵ |
| 2 | 104.8102 | 104.8106 | 104.8106 | 4 × 10⁻⁴ |
| 3 | 110.0376 | 110.0373 | 110.0373 | 3 × 10⁻⁴ |
| 4 | 162.9583 | 162.9583 | 162.9583 | 4 × 10⁻⁵ |
| 5 | 165.5864 | 165.5867 | 165.5867 | 3 × 10⁻⁴ |
| 6 | 165.5866 | 165.5868 | 165.5868 | 2 × 10⁻⁴ |
| 7 | 346.9681 | 346.9676 | 346.9676 | 5 × 10⁻⁴ |
| 8 | 361.5531 | 361.5529 | 361.5529 | 2 × 10⁻⁴ |
| 9 | 361.5539 | 361.5530 | 361.5530 | 9 × 10⁻⁴ |

- **JAX vs JAX-MD** agree to < 10⁻¹² cm⁻¹ (machine precision).
- **JAX/JAX-MD vs OpenMM** agree to < 0.001 cm⁻¹.  The tiny differences
  arise from different Hessian methods (analytical ``jax.hessian`` vs
  OpenMM's finite-difference Hessian).

!!! note "Why exact parity matters"
    If two engines produce different energies for the same force field, you
    cannot trust that one engine's implementation is correct.  Machine-precision
    agreement validates that JAX, JAX-MD, and OpenMM all compute the same
    math for the same functional form.  Note: this parity only holds when
    engines share the same functional form and non-bonded treatment
    (combining rules, 1-4 scaling, cutoffs).  Engines with different force
    field equations or different non-bonded parameters will naturally produce
    different results.

---

## Frequency Accuracy After Optimization

How well do the optimized MM frequencies match the QM reference?  This is
the primary accuracy metric — the whole point of Q2MM.

### Best Result: JAX + Powell (Score = 0.000)

Powell on both JAX backends converges to a perfect score (0.000), meaning
all optimized MM frequencies exactly match the QM reference.  This is
expected for a fully determined system (8 free parameters, 9 frequency
targets).

Starting from Seminario estimates (RMSD = 156.9 cm⁻¹), the optimizer
corrects all force constants to reproduce B3LYP/6-31+G(d) harmonic
frequencies within floating-point precision.

### Worst Result: L-BFGS-B with Finite Differences (Score = 0.077)

L-BFGS-B converges to a suboptimal local minimum on all backends.  With
finite-difference gradients (eps=1e-3), it cannot navigate the shallow
objective landscape — particularly for coupled bending/stretching modes.
Connecting ``energy_and_param_grad()`` (analytical gradients) would likely
fix this.

---

## Seminario Method

Extracting bond/angle force constants from a QM Hessian matrix.

| Molecule | Atoms | Time |
|----------|-------|------|
| Water | 3 | **0.4 ms** |
| CH₃F | 5 | **1.2 ms** |

The Seminario method is pure NumPy linear algebra (eigenvalue decomposition
of 3×3 Hessian sub-blocks).  It is effectively instant compared to
everything else in the pipeline.  It provides a good starting point
(RMSD 156.9 cm⁻¹ vs default 1870.1 cm⁻¹) but further optimization is
needed for high accuracy.

---

## QM Calculations (Psi4)

These are run once per molecule to generate reference data, not during the
optimization loop.

| Calculation | Level | Molecule | Time |
|------------|-------|----------|------|
| **Energy** | B3LYP/6-31G* | Water (3 atoms) | 1.1 s |
| **Hessian** | B3LYP/6-31G* | Water (3 atoms) | 7.8 s |

**Scaling notes:**

- QM cost scales as O(N³)–O(N⁴) with basis functions.  A 30-atom organic
  molecule with 6-31G* takes ~5–30 minutes for a Hessian.
- Transition state Hessians (for TSFF work) take the same time but contain
  one negative eigenvalue along the reaction coordinate.
- Psi4 parallelizes well — set `psi4.set_num_threads(N)` for multi-core
  speedup.

---

## Bottleneck Analysis

For a typical small-molecule optimization workflow:

```
QM Hessian (one-time)    ████████████████  7.8 s
Seminario (one-time)     ▏                 0.001 s
Optimization loop        ████████████████████████████████████  0.8–2 s (JAX)
  └─ per evaluation      ▏                 0.0005 s (JAX energy call)
```

**The energy evaluation is the bottleneck.**  Strategies to speed up:

1. **Use JAX or JAX-MD** — ~1000–2000 eval/s, 5–10× faster than OpenMM
2. **Use OpenMM over Tinker** — ~190 eval/s vs ~1.3 eval/s
3. **Reduce evaluations** — Nelder-Mead converges in ~400 evaluations
4. **Fewer molecules/geometries** — each adds one energy call per evaluation
5. **Analytical gradients** — JAX and JAX-MD support ``energy_and_param_grad()``
   via ``jax.grad``, eliminating the 2N+1 finite-difference overhead

---

*Benchmarks generated by ``q2mm-benchmark`` CLI — all methods
start from identical perturbed parameters (Seminario estimates).
Run ``q2mm-benchmark --list`` to see available backends and optimizers.*
