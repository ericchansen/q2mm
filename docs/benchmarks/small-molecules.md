# Small Molecules

Benchmarks on CH₃F (5 atoms, 8 parameters) optimized against B3LYP/6-31+G(d)
QM frequencies. The latest full supported matrix covers **24 combos**:
JAX and OpenMM on harmonic + MM3, JAX-MD on harmonic only, and Tinker on
MM3 only. Results cover speed, accuracy, and cross-engine agreement.

!!! info "Data"
    **Inputs:**
    [QM reference data](https://github.com/ericchansen/q2mm/tree/master/examples/sn2-test/qm-reference)
    (optimized geometry, Hessian, frequencies, normal modes)

    **Outputs:**
    [Benchmark results (JSON)](https://github.com/ericchansen/q2mm/tree/master/benchmark_results/ch3f/results) ·
    [Optimized force fields](https://github.com/ericchansen/q2mm/tree/master/benchmark_results/ch3f/forcefields) ·
    [Leaderboard](https://github.com/ericchansen/q2mm/blob/master/benchmark_results/ch3f/leaderboard.txt)

---

## Full Supported Matrix by Optimizer

JAX, JAX-MD, and OpenMM all start from Seminario RMSD = 156.9 cm⁻¹
(score ≈ 0.221). Tinker starts from 157.2 cm⁻¹ because its MM3 baseline is
evaluated through a separate backend. Tables include every supported
``(backend, form)`` combo and are sorted by wall-clock time within each
optimizer.

!!! tip "Reading the tables"
    **RMSD** = root-mean-square deviation of optimized MM frequencies from QM
    reference (lower is better).  **Score** = normalized objective function
    (lower is better; 0.000 = perfect match).  **Evals/s** = energy
    evaluations per second (higher is better).  Saved benchmark JSONs keep the
    full sorted MM real-mode list above the 50 cm⁻¹ cutoff, while RMSD and MAE
    compare the first ``min(len(qm), len(mm))`` sorted modes.  Failed runs can
    therefore store more optimized MM modes than the 3N-6 QM reference set.

### Nelder-Mead

| Backend | Form | Final RMSD (cm⁻¹) | Final MAE | Score | Evals | Time | Evals/s |
|---------|------|-------------------:|----------:|------:|------:|-----:|--------:|
| **JAX** | harmonic | 1037.9 | 888.8 | 0.0000 | 1202 | 3.0 s | 394.9 |
| **JAX-MD** | harmonic | 1037.9 | 888.8 | 0.0000 | 1204 | 3.1 s | 385.7 |
| **OpenMM** | MM3 | 564.2 | 299.9 | 0.0001 | 903 | 9.5 s | 95.5 |
| **OpenMM** | harmonic | 1040.5 | 892.1 | 0.0000 | 988 | 9.8 s | 100.4 |
| **JAX** | MM3 | 540.1 | 271.3 | 0.0001 | 14417 | 36.5 s | 394.7 |
| **Tinker** | MM3 | 563.3 | 299.5 | 0.0000 | 685 | 168.6 s | 4.1 |

### Powell

| Backend | Form | Final RMSD (cm⁻¹) | Final MAE | Score | Evals | Time | Evals/s |
|---------|------|-------------------:|----------:|------:|------:|-----:|--------:|
| **JAX-MD** | harmonic | **< 0.1** | **< 0.1** | 0.0000 | 2514 | 6.3 s | 398.2 |
| **JAX** | harmonic | **< 0.1** | **< 0.1** | 0.0000 | 2541 | 6.3 s | 401.1 |
| **JAX** | MM3 | 4.0 | 1.9 | 0.0001 | 12034 | 30.6 s | 393.6 |
| **OpenMM** | harmonic | **< 0.1** | **< 0.1** | 0.0000 | 4321 | 43.9 s | 98.4 |
| **OpenMM** | MM3 | 575.7 | 311.9 | 0.0003 | 13585 | 137.6 s | 98.7 |
| **Tinker** | MM3 | 555.7 | 291.2 | 0.0002 | 4843 | 1172.5 s | 4.1 |

### L-BFGS-B

| Backend | Form | Final RMSD (cm⁻¹) | Final MAE | Score | Evals | Time | Evals/s |
|---------|------|-------------------:|----------:|------:|------:|-----:|--------:|
| **JAX** | MM3 | 586.7 | 321.7 | 0.0007 | 77 | 3.3 s | 23.2 |
| **JAX-MD** | harmonic | 538.9 | 269.3 | 0.0001 | 126 | 5.4 s | 23.1 |
| **JAX** | harmonic | 540.2 | 271.4 | 0.0000 | 151 | 6.4 s | 23.6 |
| **OpenMM** | harmonic | 816.8 | 616.7 | 0.0873 | 80 | 13.7 s | 5.8 |
| **OpenMM** | MM3 | 30.4 | 25.7 | 0.0083 | 126 | 21.9 s | 5.7 |
| **Tinker** | MM3 | 114.1 | 93.4 | 0.1172 | 424 | 104.7 s | 4.1 |

### grad-simp

| Backend | Form | Final RMSD (cm⁻¹) | Final MAE | Score | Evals | Time | Evals/s |
|---------|------|-------------------:|----------:|------:|------:|-----:|--------:|
| **JAX** | harmonic | 811.8 | 579.1 | 0.0008 | 3948 | 9.9 s | 397.6 |
| **JAX-MD** | harmonic | 811.8 | 579.1 | 0.0008 | 4080 | 10.5 s | 390.2 |
| **JAX** | MM3 | 834.5 | 613.2 | 0.0024 | 4335 | 11.0 s | 393.4 |
| **OpenMM** | harmonic | 812.4 | 582.0 | 0.0014 | 2245 | 22.6 s | 99.3 |
| **OpenMM** | MM3 | 580.8 | 317.3 | 0.0004 | 2692 | 27.5 s | 97.9 |
| **Tinker** | MM3 | 833.4 | 612.3 | 0.0025 | 3271 | 779.6 s | 4.2 |

### Key Observations

- **Harmonic + Powell is the best overall combination** — JAX, JAX-MD, and
  OpenMM all land below 0.1 cm⁻¹ RMSD on the harmonic form. JAX and JAX-MD do it in
  6.3 s; OpenMM needs 43.9 s.
- **MM3 is harder to optimize than harmonic** — the best MM3 run is JAX +
  Powell (RMSD 4.0), followed by OpenMM + L-BFGS-B (30.4) and Tinker +
  L-BFGS-B (114.1).
- **grad-simp is not a win on CH₃F** — JAX/JAX-MD hit the 10-cycle cap, and
  OpenMM/Tinker do not beat the best single-shot optimizers on this small
  8-parameter problem.
- **JAX backends remain the fastest in-process engines** — derivative-free
  runs stay near 390–400 eval/s on JAX/JAX-MD, versus ~98 eval/s on OpenMM
  and ~4 eval/s on Tinker.
- **Tinker still provides a useful independent MM3 reference**, but its cost
  is much higher: 104.7 s for the best L-BFGS-B run and 1172.5 s for Powell.

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

### Best Harmonic Result: Powell on JAX, JAX-MD, and OpenMM (RMSD < 0.1)

Powell on the harmonic form reaches a near-exact frequency match on all three
in-process backends: JAX, JAX-MD, and OpenMM all finish below 0.1 cm⁻¹ RMSD.
This is expected for a fully determined system (8 free parameters, 9 frequency
targets) once the optimizer finds the right basin.

Starting from Seminario estimates (RMSD = 156.9 cm⁻¹), the optimizer
corrects all force constants to reproduce B3LYP/6-31+G(d) harmonic
frequencies to within a few hundredths of a wavenumber.

### Best MM3 Result: JAX + Powell (RMSD = 4.0)

The MM3 landscape is noticeably rougher than the harmonic one. JAX + Powell
is the closest MM3 fit in this matrix (RMSD 4.0), OpenMM + L-BFGS-B is the
strongest OpenMM MM3 run (30.4), and Tinker + L-BFGS-B is the strongest
Tinker MM3 run (114.1). On this system, optimizer choice matters more than
backend choice once the force field is constrained to MM3.

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

For the latest full supported CH₃F matrix:

```
QM Hessian (one-time)    ████████████████  7.8 s
Seminario (one-time)     ▏                 0.001 s
Optimization loop        ██████████        3.0–43.9 s (JAX/OpenMM harmonic Powell)
Tinker MM3 Powell        ████████████████████████████████████  1172.5 s
```

**The optimization loop is still the bottleneck.** Strategies to speed up:

1. **Use JAX or JAX-MD for fastest turnaround** — derivative-free CH₃F runs
   land around 390–400 eval/s on the GPU backends.
2. **Use OpenMM over Tinker for MM3** — OpenMM is ~98 eval/s on the
   derivative-free MM3 runs, versus ~4 eval/s for Tinker.
3. **Use Powell for harmonic CH₃F** — it reaches RMSD < 0.1 on JAX, JAX-MD,
   and OpenMM.
4. **Use JAX + Powell or OpenMM + L-BFGS-B for MM3** — those are the
   strongest MM3 results in the 24-combo matrix.
5. **Reserve grad-simp for larger systems** — it does not beat the best
   single-shot optimizers on CH₃F.

---

*Benchmarks generated by ``q2mm-benchmark`` CLI — all methods
start from identical perturbed parameters (Seminario estimates).
Run ``q2mm-benchmark --list`` to see available backends and optimizers.*
