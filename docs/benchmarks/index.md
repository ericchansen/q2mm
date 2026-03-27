# Benchmarks

Performance and validation benchmarks across molecules, QM reference sources,
and MM backends.  All times are wall-clock on an AMD/Intel desktop with 32 GB
RAM, Python 3.12.

---

## CH₃F (5 atoms, 8 parameters)

Best result per MM backend, sorted by wall-clock time.
QM reference: B3LYP/6-31+G(d).
See the [small-molecules](small-molecules.md) page for the full
backend × optimizer matrix.

**Data:**
[QM inputs](https://github.com/ericchansen/q2mm/tree/master/examples/sn2-test/qm-reference) ·
[Results](https://github.com/ericchansen/q2mm/tree/master/benchmarks/ch3f/results) ·
[Force fields](https://github.com/ericchansen/q2mm/tree/master/benchmarks/ch3f/forcefields)

| Backend | Optimizer | Score Δ | Time |
|---------|-----------|---------|-----:|
| **JAX** | Nelder-Mead | 0.221 → 0.000 (100%) | 0.8 s |
| **JAX-MD** | Nelder-Mead | 0.221 → 0.000 (100%) | 1.2 s |
| **OpenMM** | Nelder-Mead | 0.221 → 0.000 (100%) | 85.1 s |
| **Tinker** | Nelder-Mead | 0.223 → 0.000 (100%) | 167.7 s |

---

## Rh-Enamide (9 molecules, 94–182 parameters)

QM reference: Jaguar B3LYP/LACVP**.
See the [Rh-enamide](rh-enamide.md) page for the full matrix and analysis.

**Data:**
[QM inputs](https://github.com/ericchansen/q2mm/tree/master/examples/rh-enamide) ·
[Results](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/results)

### GRAD→SIMP Cycling (converged)

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
    enough for 182-parameter systems.  GRAD→SIMP cycling results use full
    L-BFGS-B + Nelder-Mead alternation until convergence.
    See the [GPU page](gpu.md) for device comparison.

---

## Key Takeaways

1. **JAX backends are fastest** — JAX (~1500 eval/s) and JAX-MD (~1000 eval/s)
   are 5–10× faster than OpenMM and 500–1000× faster than Tinker.  Both
   JIT-compile energy functions as pure JAX, eliminating Python ↔ C++
   marshalling overhead.

2. **CPU beats GPU for current workloads** — on an RTX 5090, the JAX
   backend is 1.6× slower than CPU for rh-enamide (36–62 atoms) due to
   float64 overhead on consumer GPUs, small Hessian sizes, and sequential
   molecule evaluation.  See the [GPU benchmark page](gpu.md) for a
   detailed analysis and the path to making GPU viable.

3. **All engines agree to machine precision** — JAX, JAX-MD, and OpenMM
   produce identical energies (< 10⁻¹⁸ kcal/mol) and frequencies
   (< 0.001 cm⁻¹) for the same force field and functional form.  This
   validates implementation correctness across backends.  Note: parity
   only holds when engines share the same functional form and non-bonded
   treatment (combining rules, 1-4 scaling, cutoffs).

4. **Nelder-Mead is the most robust optimizer** — converges on both small
   (8-param) and large (182-param) systems.  Powell works well on small
   molecules but crashes on larger systems.  L-BFGS-B with finite-difference
   gradients diverges on high-dimensional problems.

5. **JAX and JAX-MD provide analytical parameter gradients** via ``jax.grad``.
   The optimizer supports these through ``jac="auto"`` (auto-detects engine
   capability) or ``jac="analytical"`` (requires engine support).  For
   energy-based evaluators this eliminates the 2N+1 finite-difference
   overhead; frequency evaluators still use finite differences while
   differentiation through the Hessian eigendecomposition is in progress.

6. **Speed matters for scaling** — with 182 parameters and 9 molecules,
   OpenMM CUDA takes ~172 s for Nelder-Mead (2 iterations) and L-BFGS-B
   exceeds 30 min due to finite-difference gradients.  Tinker is even slower.
   JAX backends complete the same work in under a minute.

7. **The Seminario method is effectively free** — even 182-parameter
   organometallic systems complete in < 50 ms.

8. **Functional form flexibility** — the same Seminario parameters work
   with both MM3 (OpenMM/Tinker) and harmonic (JAX/JAX-MD) functional
   forms, enabling cross-engine comparison on any training set.

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
system, and ``--max-iter`` to control iteration count.  Last updated: March 2026.*
