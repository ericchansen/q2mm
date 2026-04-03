# Benchmarks

Performance and validation benchmarks across molecules, QM reference sources,
and MM backends.  All times are wall-clock on an AMD/Intel desktop with 32 GB
RAM, Python 3.12.

---

## CH₃F (5 atoms, 8 parameters)

Best result per MM backend from the latest full supported matrix.
QM reference: B3LYP/6-31+G(d).
See the [small-molecules](small-molecules.md) page for the full
backend × form × optimizer matrix.

**Data:**
[QM inputs](https://github.com/ericchansen/q2mm/tree/master/examples/sn2-test/qm-reference) ·
[Results](https://github.com/ericchansen/q2mm/tree/master/benchmark_results/ch3f/results) ·
[Force fields](https://github.com/ericchansen/q2mm/tree/master/benchmark_results/ch3f/forcefields) ·
[Leaderboard](https://github.com/ericchansen/q2mm/blob/master/benchmark_results/ch3f/leaderboard.txt)

| Backend | Best form | Optimizer | RMSD₀ → RMSD | Time |
|---------|-----------|-----------|--------------|-----:|
| **JAX** | harmonic | Powell | 156.9 → 0.0 | 6.3 s |
| **JAX-MD** | harmonic | Powell | 156.9 → 0.0 | 6.3 s |
| **OpenMM** | harmonic | Powell | 156.9 → 0.0 | 43.9 s |
| **Tinker** | mm3 | L-BFGS-B | 157.2 → 114.1 | 104.7 s |

Supported combos: **24** total — JAX and OpenMM each run harmonic + MM3,
JAX-MD runs harmonic only, and Tinker runs MM3 only.

---

## Rh-Enamide (9 molecules, 94–182 parameters)

QM reference: Jaguar B3LYP/LACVP**.
See the [Rh-enamide](rh-enamide.md) page for the full matrix and analysis.

**Data:**
[QM inputs](https://github.com/ericchansen/q2mm/tree/master/examples/rh-enamide) ·
[Results](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/results)

### Grad-Simp Cycling (converged)

| Backend | FF Form | Device | Cycles | Score Δ | Time |
|---------|---------|--------|-------:|---------|-----:|
| **JAX** | harmonic | GPU | 3 | 2,161 → 34.6 (↓98.4%) | 1,117 s |
| **JAX** | harmonic | CPU | 4 | 2,161 → 32.8 (↓98.5%) | 686 s |

### Single-shot (2 iterations)

| Backend | FF Form | Optimizer | RMSD₀ → RMSD | Time |
|---------|---------|-----------|--------------|-----:|
| **JAX** | harmonic | L-BFGS-B | 18,177 → 36,105 | 50 s |
| **JAX-MD** | harmonic | Nelder-Mead | 24,727 → 68,857 | 39 s |
| **OpenMM** | MM3 | Nelder-Mead (CUDA) | 19,342 → 78,134 | 172 s |

!!! note "Single-shot vs cycling"
    Single-shot results are 2-iteration runs to assess scaling and
    convergence behavior.  RMSD increases because 2 iterations is not
    enough for 182-parameter systems.  grad-simp cycling results use full
    L-BFGS-B + Nelder-Mead alternation until convergence.
    See the [GPU page](gpu.md) for device comparison.

---

## Key Takeaways

1. **Harmonic + Powell is the current CH₃F winner** — JAX, JAX-MD, and OpenMM
   all reach RMSD = 0.0 on the harmonic form.  JAX and JAX-MD do it in
   6.3 s; OpenMM does it in 43.9 s.

2. **MM3 remains optimizer-sensitive** — the best MM3 CH₃F fit is JAX +
   Powell (RMSD 4.0), followed by OpenMM + L-BFGS-B (30.4) and Tinker +
   L-BFGS-B (114.1).  The "best backend" depends on which functional form
   and optimizer are allowed.

3. **GPU benefits are workload-dependent** — the dedicated GPU study shows
   CH₃F is still faster on CPU, rh-enamide JAX-MD gets strong per-evaluation
   GPU speedups, and the current converged rh-enamide grad-simp loop still
   finishes faster on CPU because float64 and sequential molecule evaluation
   dominate.  See the [GPU benchmark page](gpu.md) for details.

4. **All engines agree to machine precision when the math matches** — JAX, JAX-MD, and OpenMM
   produce identical energies (< 10⁻¹⁸ kcal/mol) and frequencies
   (< 0.001 cm⁻¹) for the same force field and functional form.  This
   validates implementation correctness across backends.  Note: parity
   only holds when engines share the same functional form and non-bonded
   treatment (combining rules, 1-4 scaling, cutoffs).

5. **Optimizer guidance depends on the system** — Powell is strongest on
   small harmonic CH₃F, Nelder-Mead is the most reliable derivative-free
   fallback in the rh-enamide 2-iteration sweep, and grad-simp remains the
   converged strategy for large frequency fits.  Finite-difference L-BFGS-B
   is still expensive on MM3 backends.

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

*Benchmarks generated by ``q2mm-benchmark`` CLI.  Run ``q2mm-benchmark --list``
to see available backends and optimizers, ``--system`` to select a benchmark
system, and ``--max-iter`` to control iteration count.  The latest CH₃F
full-matrix artifacts live in ``benchmark_results/ch3f/``; rh-enamide archived
results live in ``benchmarks/rh-enamide/``.*
