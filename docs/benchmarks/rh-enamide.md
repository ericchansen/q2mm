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

## Results (2 iterations, preliminary)

These are preliminary results with only 2 optimizer iterations — enough to
reveal convergence direction and identify scaling problems.

### JAX Engines (harmonic FF)

| Backend | Optimizer | Score₀ | Score | Δ% | Evals | Time |
|---------|-----------|-------:|------:|---:|------:|-----:|
| JAX (harmonic) | **Nelder-Mead** | 385,737 | 341,407 | **↓ 11.5%** | 185 | 17 s |
| JAX (harmonic) | L-BFGS-B | 385,737 | 2,024,868 | ↑ 425% | 1,648 | 91 s |
| JAX (harmonic) | Powell | 385,737 | — | crash | — | 79 s |
| JAX-MD (OPLSAA) | **Nelder-Mead** | 663,711 | 316,575 | **↓ 52.3%** | 186 | 56 s |
| JAX-MD (OPLSAA) | L-BFGS-B | 663,711 | 5,234,397 | ↑ 689% | 1,099 | 248 s |
| JAX-MD (OPLSAA) | Powell | 663,711 | — | timeout | — | 300 s |

### MM3 Engines (MM3 FF)

| Backend | Optimizer | Score₀ | Status | Notes |
|---------|-----------|-------:|--------|-------|
| OpenMM | L-BFGS-B | 422,326 | ⏱ timeout | >5 min for 2 iterations |
| OpenMM | Nelder-Mead | 422,326 | ⏱ timeout | >5 min for 2 iterations |
| OpenMM | Powell | 422,326 | ⏱ timeout | >5 min for 2 iterations |
| Tinker | all | — | 🐛 bug | `_write_standalone_prm` element '00' error |

!!! success "Key findings"
    - **Nelder-Mead is the only viable optimizer** for 182-parameter systems
      with finite-difference gradients
    - **JAX-MD + Nelder-Mead** achieved **52.3% improvement in just 2
      iterations** (56 s) — the strongest early convergence of any combination
    - **L-BFGS-B diverges** on all backends — finite-difference gradients are
      unreliable at 182 parameters (each gradient step requires 365 function
      evaluations)
    - **Powell crashes** with `LinAlgError` — line-search drives parameters
      to physically unreasonable values

!!! warning "Scaling bottleneck"
    OpenMM and Tinker are too slow per function evaluation for 182-parameter
    optimization with 9 molecules.  Each Nelder-Mead iteration requires ~185
    evaluations, and each evaluation computes frequencies for all 9 molecules.
    At ~2.5 s/eval for OpenMM (vs ~0.1 s/eval for JAX), even 2 iterations
    exceed the 5-minute timeout.  See
    [issue #147](https://github.com/ericchansen/q2mm/issues/147) for planned
    improvements.

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
