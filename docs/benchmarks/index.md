# Benchmarks

Performance and validation benchmarks across molecules, QM reference sources,
and MM backends.  All times are wall-clock on an AMD/Intel desktop with 32 GB
RAM, Python 3.12.

---

## CH₃F (5 atoms, 8 parameters)

Best result per MM backend from the latest full supported matrix. The device is
shown because the fastest saved JAX, JAX-MD, and OpenMM runs are GPU/CUDA
results, while Tinker is CPU-only in this matrix.
QM reference: B3LYP/6-31+G(d).
See the [small-molecules](small-molecules.md) page for the full
backend × form × optimizer matrix.

**Data:**
[QM inputs](https://github.com/ericchansen/q2mm/tree/master/examples/sn2-test/qm-reference) ·
[Results](https://github.com/ericchansen/q2mm/tree/master/benchmark_results/ch3f/results) ·
[Force fields](https://github.com/ericchansen/q2mm/tree/master/benchmark_results/ch3f/forcefields) ·
[Leaderboard](https://github.com/ericchansen/q2mm/blob/master/benchmark_results/ch3f/leaderboard.txt)

| Backend | Device | Best form | Optimizer | RMSD₀ → RMSD | Time |
|---------|--------|-----------|-----------|--------------|-----:|
| **JAX** | GPU | harmonic | Powell | 156.9 → < 0.1 | 6.3 s |
| **JAX-MD** | GPU | harmonic | Powell | 156.9 → < 0.1 | 6.3 s |
| **OpenMM** | GPU (CUDA) | harmonic | Powell | 156.9 → < 0.1 | 43.9 s |
| **Tinker** | CPU | mm3 | L-BFGS-B | 157.2 → 114.1 | 104.7 s |

Supported combos: **24** total — JAX and OpenMM each run harmonic + MM3,
JAX-MD runs harmonic only, and Tinker runs MM3 only.

---

## Rh-Enamide (9 molecules, 94–182 parameters)

QM reference: Jaguar B3LYP/LACVP**.
See the [Rh-enamide](rh-enamide.md) page for the full matrix and analysis.

**Data:**
[QM inputs](https://github.com/ericchansen/q2mm/tree/master/examples/rh-enamide) ·
[Results](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/results) ·
[Force fields](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/forcefields) ·
[Timing evidence](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/logs)

### Overnight selected GPU matrix

| Backend | FF Form | Optimizer | Outcome | Wall clock |
|---------|---------|-----------|---------|-----------:|
| **OpenMM** | MM3 | grad-simp (CUDA) | **173.1 → 42.7 RMSD**, MAE 31.8 | **112,959.0 s** |
| **JAX** | harmonic | L-BFGS-B | 173.5 → 57.4 RMSD | 2,278.0 s |
| **JAX** | MM3 | L-BFGS-B | 173.1 → 61.5 RMSD | 1,975.2 s |
| **JAX** | MM3 | Nelder-Mead | 173.1 → 66.6 RMSD | 591.4 s |
| **JAX** | harmonic | Nelder-Mead | 173.5 → 81.2 RMSD | 559.9 s |
| **JAX-MD** | harmonic | L-BFGS-B | 41,409.7 → 86.7 RMSD | 3,261.1 s |

!!! note "Selected-matrix failure pattern"
    The remaining seven selected GPU combos all terminated with
    `Eigenvalues did not converge`. See the [Rh-enamide](rh-enamide.md)
    page for the full per-combo breakdown, archived result JSONs, and the
    raw timing log bundle.

### Earlier focused harmonic cycling study

| Backend | FF Form | Device | Cycles | Score Δ | Time |
|---------|---------|--------|-------:|---------|-----:|
| **JAX** | harmonic | GPU | 3 | 2,161 → 34.6 (↓98.4%) | 1,117 s |
| **JAX** | harmonic | CPU | 4 | 2,161 → 32.8 (↓98.5%) | 686 s |

---

## Key Takeaways

1. **Harmonic + Powell is the current CH₃F winner** — JAX, JAX-MD, and OpenMM
   all land below 0.1 cm⁻¹ RMSD on the harmonic form.  JAX and JAX-MD do it in
   6.3 s; OpenMM does it in 43.9 s.

2. **MM3 remains optimizer-sensitive** — the best MM3 CH₃F fit is JAX +
   Powell (RMSD 4.0), followed by OpenMM + L-BFGS-B (30.4) and Tinker +
   L-BFGS-B (114.1).  The "best backend" depends on which functional form
   and optimizer are allowed.

3. **GPU benefits are workload-dependent** — the dedicated GPU study shows
   CH₃F is still faster on CPU, rh-enamide JAX-MD gets strong per-evaluation
   GPU speedups, and the overnight selected GPU matrix shows that OpenMM CUDA
   can produce the best fit, but only at overnight-scale wall clock.  See the
   [GPU benchmark page](gpu.md) for details.

4. **All engines agree to machine precision when the math matches** — JAX, JAX-MD, and OpenMM
   produce identical energies (< 10⁻¹⁸ kcal/mol) and frequencies
   (< 0.001 cm⁻¹) for the same force field and functional form.  This
   validates implementation correctness across backends.  Note: parity
   only holds when engines share the same functional form and non-bonded
   treatment (combining rules, 1-4 scaling, cutoffs).

5. **Optimizer guidance depends on the system** — Powell is strongest on
   small harmonic CH₃F, JAX L-BFGS-B/Nelder-Mead are the most useful fast
   screening runs in the overnight rh-enamide sweep, and OpenMM CUDA MM3
   grad-simp currently gives the best selected-matrix fit when you can afford
   the overnight runtime. Finite-difference L-BFGS-B is still expensive on
   MM3 backends.

6. **JAX and JAX-MD provide analytical parameter gradients** via ``jax.grad``.
   The optimizer supports these through ``jac="auto"`` (auto-detects engine
   capability) or ``jac="analytical"`` (requires engine support).  For
   energy-based evaluators this eliminates the 2N+1 finite-difference
   overhead; frequency evaluators still use finite differences while
   differentiation through the Hessian eigendecomposition is in progress.

7. **The Seminario method is effectively free** — even 182-parameter
   organometallic systems complete in < 50 ms.

8. **Shared starting points make cross-form comparisons possible** — the
   same Seminario-derived bond/angle parameters can seed both MM3 and
   harmonic force fields, which is why CH₃F can be compared across JAX,
   JAX-MD, OpenMM, and Tinker despite different functional-form support.

---

## Detailed Results

- [**Small Molecules**](small-molecules.md) — CH₃F: combined speed + accuracy
  leaderboard, cross-engine parity, frequency accuracy analysis
- [**Rh-Enamide**](rh-enamide.md) — 9-structure organometallic training set
  with Jaguar B3LYP/LACVP** reference data
- [**GPU Acceleration**](gpu.md) — GPU vs CPU benchmarks, scaling analysis,
  and guidance on when GPU acceleration helps

---

*Benchmarks generated by ``q2mm-benchmark`` CLI. Run ``q2mm-benchmark --list``
to see available backends and optimizers, ``--system`` to select a benchmark
system, and ``--max-iter`` to control iteration count. The latest CH₃F
full-matrix artifacts live in ``benchmark_results/ch3f/``; rh-enamide archived
results, saved force fields, and raw timing evidence live in
``benchmarks/rh-enamide/``.*
