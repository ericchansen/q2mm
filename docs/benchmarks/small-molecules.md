# Small Molecules

This page answers one question: how do the currently supported backend, form,
and optimizer combinations compare on a small, fully tractable benchmark? The
system is CH₃F (5 atoms, 8 fitted parameters) against B3LYP/6-31+G(d) QM
frequencies. Unlike the Rh-Enamide page, this page is the full supported
matrix, so it is the right place to compare combinations directly.

## Scope

- System: CH₃F (1 molecule, 5 atoms, 8 parameters)
- QM reference: B3LYP/6-31+G(d)
- Matrix size: 24 supported combos
- Backends/forms: JAX and OpenMM on harmonic + MM3, JAX-MD on harmonic, Tinker
  on MM3
- Starting point: JAX/JAX-MD/OpenMM begin at 156.9 cm⁻¹ RMSD; Tinker begins at
  157.2 cm⁻¹ because its MM3 baseline is evaluated through a separate backend

## Full CH₃F matrix

Default rows are grouped by functional form and then by final RMSD. Use the
filters and sortable headers to narrow form/backend/device/optimizer
combinations, and compare like-with-like inside each form: harmonic and MM3
rows share the same benchmark system, but they do not represent the same
force-field model.

<div class="benchmark-table-anchor" data-benchmark-table="small-molecules"></div>

| Form | Backend | Device | Optimizer | Final RMSD (cm⁻¹) | Final MAE | Time | eval/s |
|------|---------|--------|-----------|-------------------:|----------:|-----:|--------:|
| harmonic | JAX-MD | GPU | Powell | < 0.1 | < 0.1 | 6.3 s | 398.2 |
| harmonic | JAX | GPU | Powell | < 0.1 | < 0.1 | 6.3 s | 401.1 |
| harmonic | OpenMM | GPU | Powell | < 0.1 | < 0.1 | 43.9 s | 98.4 |
| harmonic | JAX-MD | GPU | L-BFGS-B | 538.9 | 269.3 | 5.4 s | 23.1 |
| harmonic | JAX | GPU | L-BFGS-B | 540.2 | 271.4 | 6.4 s | 23.6 |
| harmonic | JAX | GPU | grad-simp | 811.8 | 579.1 | 9.9 s | 397.6 |
| harmonic | JAX-MD | GPU | grad-simp | 811.8 | 579.1 | 10.5 s | 390.2 |
| harmonic | OpenMM | GPU | grad-simp | 812.4 | 582.0 | 22.6 s | 99.3 |
| harmonic | JAX | GPU | Nelder-Mead | 1037.9 | 888.8 | 3.0 s | 394.9 |
| harmonic | JAX-MD | GPU | Nelder-Mead | 1037.9 | 888.8 | 3.1 s | 385.7 |
| harmonic | OpenMM | GPU | Nelder-Mead | 1040.5 | 892.1 | 9.8 s | 100.4 |
| mm3 | JAX | GPU | Powell | 4.0 | 1.9 | 30.6 s | 393.6 |
| mm3 | OpenMM | GPU | L-BFGS-B | 30.4 | 25.7 | 21.9 s | 5.7 |
| mm3 | Tinker | CPU | L-BFGS-B | 114.1 | 93.4 | 104.7 s | 4.1 |
| mm3 | JAX | GPU | Nelder-Mead | 540.1 | 271.3 | 36.5 s | 394.7 |
| mm3 | Tinker | CPU | Powell | 555.7 | 291.2 | 1172.5 s | 4.1 |
| mm3 | Tinker | CPU | Nelder-Mead | 563.3 | 299.5 | 168.6 s | 4.1 |
| mm3 | OpenMM | GPU | Nelder-Mead | 564.2 | 299.9 | 9.5 s | 95.5 |
| mm3 | OpenMM | GPU | Powell | 575.7 | 311.9 | 137.6 s | 98.7 |
| mm3 | OpenMM | GPU | grad-simp | 580.8 | 317.3 | 27.5 s | 97.9 |
| mm3 | JAX | GPU | L-BFGS-B | 586.7 | 321.7 | 3.3 s | 23.2 |
| mm3 | Tinker | CPU | grad-simp | 833.4 | 612.3 | 779.6 s | 4.2 |
| mm3 | JAX | GPU | grad-simp | 834.5 | 613.2 | 11.0 s | 393.4 |

## Interpretation

- Harmonic + Powell is the clear small-system winner: JAX, JAX-MD, and OpenMM
  all reach a near-exact fit, while JAX and JAX-MD do so in 6.3 s.
- MM3 remains optimizer-sensitive. The best MM3 result is JAX + Powell
  (RMSD 4.0), while OpenMM + L-BFGS-B is the strongest OpenMM MM3 run.
- On identical parameters, JAX, JAX-MD, and OpenMM agree to machine precision
  when the functional form matches: energy deltas stay at or below
  3 x 10^-18 kcal/mol and frequency deltas stay below 0.001 cm⁻¹.
- grad-simp is not the right default on this problem. It never beats the best
  single-shot result in either form.
- The optimization loop dominates runtime; Seminario is effectively free by
  comparison and serves mainly as a starting point, not as the expensive step.

## Artifacts and provenance

- Inputs:
  [QM reference data](https://github.com/ericchansen/q2mm/tree/master/examples/sn2-test/qm-reference)
- Outputs:
  [Benchmark results (JSON)](https://github.com/ericchansen/q2mm/tree/master/benchmark_results/ch3f/results),
  [optimized force fields](https://github.com/ericchansen/q2mm/tree/master/benchmark_results/ch3f/forcefields),
  and the
  [leaderboard](https://github.com/ericchansen/q2mm/blob/master/benchmark_results/ch3f/leaderboard.txt)

This page uses the current full-matrix artifact set in `benchmark_results/ch3f/`.

## Reproducing

```bash
q2mm-benchmark --system ch3f --output benchmark_results/ch3f
q2mm-benchmark --load benchmark_results/ch3f/results
q2mm-benchmark --list
```
