# Rh-Enamide

Full-loop validation on a real organometallic system using Jaguar
B3LYP/LACVP** QM reference data.

!!! info "Data"
    **Inputs:**
    [Rh-enamide training set](https://github.com/ericchansen/q2mm/tree/master/examples/rh-enamide)
    (structures, Jaguar QM data, MM3 FF template)

    **Outputs:**
    [Benchmark results](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide)
    (JSON results for all backend × optimizer combinations)

---

## System Description

The rh-enamide training set consists of **9 transition-state structures**
for a Rh(I)-diphosphine catalyzed enamide hydrogenation.  Each structure
has **36 atoms** including Rh, P, N, O, C, and H — a challenging test for
the Seminario method and MM force field optimization.

| Property | Value |
|----------|-------|
| **Structures** | 9 TS geometries |
| **Atoms per structure** | 36 |
| **Elements** | Rh, P, N, O, C, H |
| **QM level** | B3LYP/LACVP** (Hay-Wadt ECP for Rh) |
| **QM program** | Jaguar (Schrödinger) |
| **FF template** | MM3 (mm3.fld with Rh parameters) |
| **Parameters** | 182 (8 bond, 23 angle, 36 vdW types) |

---

## Pipeline

```mermaid
flowchart LR
    A[Jaguar QM data] --> B[Seminario]
    B --> C[Initial FF]
    C --> D[MM Frequencies]
    D --> E[Optimizer]
    E --> F[Optimized FF]
```

1. **Load**: 9 structures from MacroModel `.mmo` + Jaguar Hessians
2. **Seminario**: Estimate bond/angle force constants from QM Hessians using
   the MM3 template (preserves vdW parameters for all atom types including Rh)
3. **Reference**: Build multi-molecule frequency reference data — each
   molecule contributes its real vibrational frequencies (>50 cm⁻¹)
4. **Optimize**: Minimize weighted sum-of-squares between QM and MM
   frequencies across all 9 molecules simultaneously

!!! note "Functional forms"
    The MM3 force field template uses **MM3 functional forms** (cubic/quartic
    stretch, sextic bend), supported by OpenMM and Tinker.  For JAX and
    JAX-MD engines, which only support harmonic potentials, we use a
    **harmonic copy** of the same Seminario-estimated parameters.  Initial
    scores differ between functional forms because the energy expressions
    differ, but convergence behavior is comparable.

---

## Results (2 iterations)

Results generated with `q2mm-benchmark --system rh-enamide --max-iter 2`.
All results use the new `BenchmarkResult` JSON format.

### JAX Engines (harmonic FF)

| Backend | Optimizer | RMSD₀ | RMSD | Evals | Time |
|---------|-----------|------:|-----:|------:|-----:|
| JAX (harmonic) | L-BFGS-B | 18,177 | 36,105 | 1,648 | 50 s |
| JAX (harmonic) | **Nelder-Mead** | 18,177 | 82,597 | 186 | 8 s |
| JAX (harmonic) | Powell | — | LinAlgError | — | — |
| JAX-MD (OPLSAA) | L-BFGS-B | 24,727 | 67,391 | 1,099 | 229 s |
| JAX-MD (OPLSAA) | **Nelder-Mead** | 24,727 | 68,857 | 187 | 39 s |
| JAX-MD (OPLSAA) | Powell | — | LinAlgError | — | — |

### MM3 Engines (MM3 FF)

| Backend | Optimizer | RMSD₀ | RMSD | Evals | Time |
|---------|-----------|------:|-----:|------:|-----:|
| OpenMM (CUDA) | **Nelder-Mead** | 19,342 | 78,134 | 187 | 172 s |
| OpenMM (CUDA) | L-BFGS-B | — | too slow | — | >30 min |
| OpenMM (CUDA) | Powell | — | too slow | — | >10 min |

!!! success "Key findings"
    - **OpenMM CUDA now works on Blackwell (RTX 5090)** — install
      `OpenMM-CUDA-12` pip package; the engine falls back gracefully
      to CPU if CUDA context creation fails
    - **Nelder-Mead completes on all backends** including OpenMM CUDA
      (172 s for 9 molecules × 182 params)
    - **L-BFGS-B is impractical for OpenMM** — frequency gradients use
      finite differences (183 evals per gradient step), making each
      L-BFGS-B iteration extremely slow
    - **Powell crashes** with `LinAlgError` on JAX/JAX-MD — line-search
      drives parameters to physically unreasonable values
    - All results use `jac="auto"` which auto-enables analytical
      gradients for engines that support them (currently energy-only;
      frequency gradients still use finite differences)

!!! warning "RMSD increases with only 2 iterations"
    With `maxiter=2`, Nelder-Mead does not have enough iterations to
    converge for 182-parameter systems.  The RMSD *increases* because
    the simplex has barely started exploring.  Use GRAD→SIMP cycling
    (below) for converged results.

---

## GRAD→SIMP Cycling (converged)

Full optimization using L-BFGS-B (GRAD) → Nelder-Mead (SIMP) alternation
with up to 5 parameters per cycle.  Uses auto-generated harmonic FF from
molecular topology (94 parameters across 9 molecules, 1,273 frequency
reference values).

| Device | Cycles | Evals | Opt time | Final score | Improvement |
|--------|-------:|------:|---------:|------------:|:-----------:|
| GPU (RTX 5090) | 3 | 30,637 | 1,117 s | 34.56 | 98.40% |
| CPU | 4 | 30,936 | 686 s | 32.78 | 98.48% |

Both devices achieve >98% improvement from the Seminario starting point
with nearly identical eval counts (~30 k).  **CPU is 1.6× faster** due
to float64 overhead on consumer GPUs.  See the
[GPU acceleration page](gpu.md) for a detailed analysis of per-eval
throughput, why CPU wins, and the path to making GPU viable.

!!! success "Key result"
    GRAD→SIMP cycling reduces the score from **2,161 → ~33** (98.5%
    improvement) in 3–4 cycles.  This confirms that the cycling optimizer
    and JAX backend can handle real organometallic systems end-to-end.

---

*Data generated from Jaguar B3LYP/LACVP** reference data in
`examples/rh-enamide/`.  Full results archived in `benchmarks/rh-enamide/`.*
