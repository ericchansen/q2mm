# Small Molecules

This page answers one question: how do the currently supported backend, form,
and optimizer combinations compare on a small, fully tractable benchmark? The
system is CH₃F (5 atoms, 8 fitted parameters) against B3LYP/6-31+G(d) QM
frequencies. Unlike the Rh-Enamide page, this page is the full supported
matrix, so it is the right place to compare combinations directly.

## Scope

- System: CH₃F (1 molecule, 5 atoms, 8 parameters)
- QM reference: B3LYP/6-31+G(d)
- Matrix size: 36 supported combos
- Backends/forms: JAX and OpenMM on harmonic + MM3, JAX-MD on harmonic, Tinker
  on MM3
- Optimizers: Powell, L-BFGS-B, Nelder-Mead, grad-simp — each gradient-using
  optimizer is run twice (once with analytical frequency gradients, once with
  pure FD)
- Starting point: QFUERZA initialization — JAX/JAX-MD begin at 192.0 cm⁻¹
  RMSD, OpenMM at 191.9 cm⁻¹, Tinker at 192.1 cm⁻¹

!!! note "Starting-point change (April 2026)"
    Prior benchmarks (commits before `69e4f7c4`) used the original Seminario
    method for initialization, which produced a starting RMSD of ~157 cm⁻¹.
    The default was changed to QFUERZA in commit `650e62e` — see
    [QFUERZA strategy](../how-it-works/theory.md#stage-1-qfuerza-estimation). QFUERZA starts at ~192 cm⁻¹
    but uses physically motivated force-constant estimates that provide a
    more robust basin for gradient-based optimizers. Results before and after
    this change are **not directly comparable** as a pure optimizer
    leaderboard, because local optimizers converge to different basins
    depending on the starting point.

## Full CH₃F matrix

Default rows are grouped by functional form and then by final RMSD. Use the
filters and sortable headers to narrow form/backend/device/optimizer
combinations, and compare like-with-like inside each form: harmonic and MM3
rows share the same benchmark system, but they do not represent the same
force-field model.

<div class="benchmark-table-anchor" data-benchmark-table="small-molecules"></div>

| Form | Backend | Device | Optimizer | E∇ | F∇ | Final RMSD (cm⁻¹) | Final MAE | Time | eval/s |
|------|---------|--------|-----------|:--:|:--:|-------------------:|----------:|-----:|--------:|
| harmonic | JAX | GPU | L-BFGS-B | — | A | 528.7 | 257.3 | 1.9 s | 41.1 |
| harmonic | JAX-MD | GPU | grad-simp | — | FD | 528.8 | 242.3 | 5.9 s | 142.5 |
| harmonic | JAX | GPU | grad-simp | — | A | 529.1 | 243.3 | 5.5 s | 243.1 |
| harmonic | JAX-MD | GPU | L-BFGS-B | — | FD | 531.1 | 254.6 | 4.3 s | 20.2 |
| harmonic | OpenMM | GPU | grad-simp | — | FD | 979.5 | 786.3 | 45.8 s | 91.5 |
| harmonic | JAX | GPU | grad-simp | — | FD | 981.4 | 790.1 | 13.1 s | 353.4 |
| harmonic | JAX-MD | GPU | grad-simp | — | FD | 981.4 | 790.1 | 13.8 s | 334.7 |
| harmonic | OpenMM | GPU | grad-simp | — | FD | 981.9 | 794.7 | 67.8 s | 31.0 |
| harmonic | JAX | GPU | Nelder-Mead | — | — | 987.4 | 795.0 | 34.2 s | 357.7 |
| harmonic | JAX-MD | GPU | Nelder-Mead | — | — | 987.5 | 795.0 | 34.2 s | 344.0 |
| harmonic | OpenMM | GPU | Powell | — | — | 1036.7 | 891.7 | 62.5 s | 97.8 |
| harmonic | JAX | GPU | Powell | — | — | 1041.5 | 899.0 | 10.1 s | 342.8 |
| harmonic | JAX-MD | GPU | Powell | — | — | 1041.5 | 899.0 | 10.4 s | 342.1 |
| harmonic | OpenMM | GPU | Nelder-Mead | — | — | 1043.6 | 868.8 | 9.2 s | 102.7 |
| harmonic | JAX | GPU | L-BFGS-B | — | FD | 1048.3 | 934.6 | 0.5 s | 336.0 |
| harmonic | JAX-MD | GPU | L-BFGS-B | — | FD | 1048.3 | 934.6 | 0.5 s | 334.8 |
| harmonic | OpenMM | GPU | L-BFGS-B | — | FD | 1048.3 | 934.7 | 3.5 s | 77.2 |
| harmonic | OpenMM | GPU | L-BFGS-B | — | FD | 1049.5 | 936.1 | 4.0 s | 5.5 |
| mm3 | OpenMM | GPU | L-BFGS-B | — | FD | 59.5 | 46.7 | 4.8 s | 104.4 |
| mm3 | OpenMM | GPU | L-BFGS-B | — | FD | 83.6 | 62.9 | 9.7 s | 5.4 |
| mm3 | Tinker | CPU | L-BFGS-B | — | FD | 83.8 | 63.4 | 152.5 s | 4.1 |
| mm3 | Tinker | CPU | L-BFGS-B | — | FD | 83.8 | 63.4 | 150.3 s | 4.2 |
| mm3 | JAX | GPU | L-BFGS-B | — | FD | 113.5 | 90.6 | 0.8 s | 347.2 |
| mm3 | Tinker | CPU | Powell | — | — | 542.5 | 275.2 | 2768.6 s | 4.3 |
| mm3 | Tinker | CPU | grad-simp | — | FD | 564.4 | 314.5 | 1094.9 s | 4.3 |
| mm3 | Tinker | CPU | grad-simp | — | FD | 564.4 | 314.5 | 1097.7 s | 4.3 |
| mm3 | OpenMM | GPU | grad-simp | — | FD | 566.2 | 306.6 | 36.1 s | 24.7 |
| mm3 | OpenMM | GPU | grad-simp | — | FD | 573.1 | 311.6 | 29.5 s | 97.1 |
| mm3 | Tinker | CPU | Nelder-Mead | — | — | 576.3 | 311.5 | 152.5 s | 4.3 |
| mm3 | JAX | GPU | L-BFGS-B | — | A | 579.0 | 313.9 | 2.2 s | 31.4 |
| mm3 | JAX | GPU | grad-simp | — | A | 579.0 | 313.9 | 3.4 s | 139.1 |
| mm3 | OpenMM | GPU | Nelder-Mead | — | — | 581.1 | 315.1 | 8.5 s | 97.0 |
| mm3 | JAX | GPU | Nelder-Mead | — | — | 608.1 | 334.2 | 25.6 s | 344.9 |
| mm3 | JAX | GPU | grad-simp | — | FD | 1050.0 | 910.4 | 8.2 s | 343.0 |
| mm3 | JAX | GPU | Powell | — | — | 1080.7 | 937.3 | 15.1 s | 339.0 |
| mm3 | OpenMM | GPU | Powell | — | — | 1090.5 | 950.4 | 124.2 s | 95.3 |

## Interpretation

**E∇** = energy gradient mode, **F∇** = frequency gradient mode.
**A** = analytical (autodiff), **FD** = finite-difference, **—** = not
applicable (derivative-free optimizer).

### Harmonic form

- The best harmonic results cluster around 528–531 cm⁻¹ RMSD, achieved by
  JAX and JAX-MD with L-BFGS-B or grad-simp using analytical frequency
  gradients. These combos benefit from QFUERZA's physically motivated
  starting parameters.
- Derivative-free optimizers (Powell, Nelder-Mead) perform poorly on the
  harmonic form from the QFUERZA starting point, landing in the 987–1049
  range. Under the previous Seminario initialization these reached near-zero
  RMSD, but that was an initialization-sensitive local optimum — the result
  was not robust across starting points.
- FD-only gradient combos (L-BFGS-B with FD) also perform poorly (~1048),
  suggesting that finite-difference frequency gradients are too noisy to
  guide L-BFGS-B from the QFUERZA basin.

### MM3 form

- The best MM3 result is OpenMM L-BFGS-B with FD gradients at 59.5 cm⁻¹.
- Tinker L-BFGS-B improved significantly from the prior run (114 → 84 cm⁻¹),
  showing that QFUERZA provides a better basin for gradient-based MM3
  optimization through the Tinker backend.
- Tinker grad-simp also improved dramatically (833 → 564 cm⁻¹), confirming
  the basin-quality effect.
- Powell and Nelder-Mead on MM3 remain mid-range (542–608) and are
  insensitive to the initialization change, as expected for derivative-free
  methods on a rugged landscape.

### Cross-cutting observations

- Analytical frequency gradients now matter more than energy gradients on
  this problem. The top four harmonic results all use analytical frequency
  gradients (A) or analytical-fallback (FD with JAX-MD). The energy gradient
  column shows "—" because QFUERZA initialization does not affect the
  gradient pipeline — the E∇ distinction is less important than F∇.
- On identical parameters, JAX, JAX-MD, and OpenMM agree to machine precision
  when the functional form matches: energy deltas stay at or below
  3 × 10⁻¹⁸ kcal/mol and frequency deltas stay below 0.001 cm⁻¹.
- The optimization loop dominates runtime; QFUERZA estimation is effectively
  free by comparison and serves mainly as a starting point, not as the
  expensive step.

## Artifacts and provenance

- Inputs:
  [QM reference data](https://github.com/ericchansen/q2mm/tree/master/examples/sn2-test/qm-reference)
- Outputs:
  [Benchmark results (JSON)](https://github.com/ericchansen/q2mm/tree/master/benchmarks/ch3f/results)
  and
  [optimized force fields](https://github.com/ericchansen/q2mm/tree/master/benchmarks/ch3f/forcefields)
- Git SHA: `69e4f7c4` (q2mm 5.0.0a3.dev223)
- GPU: NVIDIA GeForce RTX 5090

This page uses the current full-matrix artifact set in `benchmarks/ch3f/`.

## Reproducing

```bash
q2mm-benchmark --system ch3f --output benchmarks/ch3f --platform CUDA
q2mm-benchmark --load benchmarks/ch3f/results
q2mm-benchmark --list
```
