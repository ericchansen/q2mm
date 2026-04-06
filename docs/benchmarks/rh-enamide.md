# Rh-Enamide

Full-loop validation on a real organometallic system using Jaguar
B3LYP/LACVP** QM reference data.

!!! info "Data"
    **Inputs:**
    [Rh-enamide training set](https://github.com/ericchansen/q2mm/tree/master/examples/rh-enamide)
    (structures, Jaguar QM data, MM3 FF template)

    **Archived outputs:**
    [Result JSONs](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/results)
    ·
    [Saved force fields](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/forcefields)
    ·
    [Raw timing evidence](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/logs)
    ·
    [Runner script](https://github.com/ericchansen/q2mm/blob/master/scripts/run_rh_enamide_selected_matrix.sh)

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

## Overnight selected GPU matrix (2026-04-03)

This overnight run executed combos **1-12 and 20** from the 24 supported
rh-enamide combinations, in fast-first order, on the RTX 5090. The run used
[`scripts/run_rh_enamide_selected_matrix.sh`](https://github.com/ericchansen/q2mm/blob/master/scripts/run_rh_enamide_selected_matrix.sh)
to issue explicit per-combo commands because
`q2mm-benchmark --system rh-enamide` defaults to **MM3-only** forms and would
otherwise skip the harmonic/JAX-MD cases.

### Successful runs

| Combo | Backend | FF form | Optimizer | RMSD₀ → RMSD | MAE | Wall clock |
|-------|---------|---------|-----------|-------------:|----:|-----------:|
| 20 | OpenMM (CUDA) | MM3 | **grad-simp** | **173.1 → 42.7** | **31.8** | **112,959.0 s** |
| 1 | JAX | harmonic | L-BFGS-B | 173.5 → 57.4 | 35.0 | 2,278.0 s |
| 5 | JAX | MM3 | L-BFGS-B | 173.1 → 61.5 | 48.5 | 1,975.2 s |
| 6 | JAX | MM3 | Nelder-Mead | 173.1 → 66.6 | 54.9 | 591.4 s |
| 2 | JAX | harmonic | Nelder-Mead | 173.5 → 81.2 | 50.7 | 559.9 s |
| 9 | JAX-MD (OPLSAA) | harmonic | L-BFGS-B | 41,409.7 → 86.7 | 48.2 | 3,261.1 s |

### Expected failures

| Combo | Backend | FF form | Optimizer | Wall clock | Outcome |
|-------|---------|---------|-----------|-----------:|---------|
| 3 | JAX | harmonic | Powell | 173.0 s | `Eigenvalues did not converge` |
| 7 | JAX | MM3 | Powell | 177.0 s | `Eigenvalues did not converge` |
| 11 | JAX-MD (OPLSAA) | harmonic | Powell | 211.0 s | `Eigenvalues did not converge` |
| 4 | JAX | harmonic | grad-simp | 861.0 s | `Eigenvalues did not converge` |
| 8 | JAX | MM3 | grad-simp | 836.0 s | `Eigenvalues did not converge` |
| 10 | JAX-MD (OPLSAA) | harmonic | Nelder-Mead | 351.0 s | `Eigenvalues did not converge` |
| 12 | JAX-MD (OPLSAA) | harmonic | grad-simp | 659.0 s | `Eigenvalues did not converge` |

!!! note "Timing provenance"
    Wall-clock values on this page come from the archived
    [raw CLI log](https://github.com/ericchansen/q2mm/blob/master/benchmarks/rh-enamide/logs/rh-enamide_selected_2026-04-03_0320.run.log)
    and derived
    [timings table](https://github.com/ericchansen/q2mm/blob/master/benchmarks/rh-enamide/logs/rh-enamide_selected_2026-04-03_0320.timings.tsv).
    The same log bundle also preserves the exact combo-order manifest and
    `SHA256SUMS` file for provenance.
    Successful JSON result files also store `optimized.elapsed_s`, but that is
    optimizer-local time rather than end-to-end wall clock. For example, the
    OpenMM CUDA MM3 grad-simp JSON records `33,223.1 s`, while the raw CLI log
    records `112,959.0 s` for the full overnight job.

!!! success "Key findings"
    - **OpenMM CUDA MM3 + grad-simp is the best fit in the selected GPU matrix**
      — it reduces RMSD from `173.1` to `42.7`, saves `.fld` / `.prm` / `.xml`
      force fields, and is now fully archived under `benchmarks/rh-enamide/`
    - **JAX single-shot runs are useful screening jobs** — both harmonic and
      MM3 L-BFGS-B/Nelder-Mead runs finish in `559.9–2,278.0 s`, making them
      realistic same-day checks before committing to an overnight OpenMM job
    - **JAX-MD harmonic L-BFGS-B can recover from a very poor starting point**
      — `41,409.7 → 86.7` RMSD is a large improvement, even though it still
      trails the best JAX and OpenMM results
    - **The unstable configurations are now well characterised** — Powell on
      JAX/JAX-MD and harmonic grad-simp on JAX/JAX-MD all terminated with the
      same `Eigenvalues did not converge` failure signature

---

## Earlier focused harmonic cycling study

This is a separate dedicated harmonic-only benchmark with an auto-generated
topology-driven force field (94 parameters across 9 molecules). It remains
useful context for GPU-vs-CPU scaling, but it is **not directly comparable**
to the 182-parameter overnight selected matrix above.

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
    grad-simp cycling reduces the score from **2,161 → ~33** (98.5%
    improvement) in 3–4 cycles.  This confirms that the cycling optimizer
    and JAX backend can handle real organometallic systems end-to-end.

---

*Data generated from Jaguar B3LYP/LACVP** reference data in
`examples/rh-enamide/`. Archived overnight results, saved force fields, and raw
timing evidence now live in `benchmarks/rh-enamide/`.*
