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
- Optimizers: Powell, L-BFGS-B, L-BFGS-B (FD), Nelder-Mead, grad-simp (FD),
  grad-simp (auto)
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
| harmonic | OpenMM | GPU | Powell | < 0.1 | < 0.1 | 41.1 s | 105.2 |
| harmonic | JAX | GPU | Powell | < 0.1 | < 0.1 | 7.7 s | 331.0 |
| harmonic | JAX-MD | GPU | Powell | < 0.1 | < 0.1 | 8.0 s | 323.8 |
| harmonic | JAX-MD | GPU | L-BFGS-B | 533.4 | 259.1 | 7.2 s | 20.4 |
| harmonic | JAX-MD | GPU | grad-simp (auto) | 537.8 | 267.4 | 14.2 s | 132.9 |
| harmonic | JAX | GPU | L-BFGS-B | 552.6 | 287.9 | 7.5 s | 20.8 |
| harmonic | OpenMM | GPU | L-BFGS-B (FD) | 804.5 | 633.8 | 2.9 s | 108.4 |
| harmonic | JAX | GPU | grad-simp (FD) | 811.8 | 579.1 | 12.0 s | 331.2 |
| harmonic | JAX-MD | GPU | grad-simp (FD) | 811.8 | 579.1 | 12.1 s | 320.6 |
| harmonic | OpenMM | GPU | grad-simp (auto) | 812.1 | 580.8 | 62.9 s | 34.6 |
| harmonic | JAX-MD | GPU | L-BFGS-B (FD) | 813.4 | 610.1 | 1.1 s | 323.2 |
| harmonic | JAX | GPU | L-BFGS-B (FD) | 813.4 | 610.1 | 1.1 s | 343.3 |
| harmonic | OpenMM | GPU | L-BFGS-B | 816.8 | 616.7 | 13.0 s | 6.2 |
| harmonic | OpenMM | GPU | grad-simp (FD) | 829.2 | 600.7 | 51.8 s | 102.9 |
| harmonic | JAX | GPU | grad-simp (auto) | 992.5 | 815.1 | 15.2 s | 138.1 |
| harmonic | JAX | GPU | Nelder-Mead | 1037.9 | 888.8 | 3.7 s | 325.8 |
| harmonic | JAX-MD | GPU | Nelder-Mead | 1037.9 | 888.8 | 3.7 s | 325.2 |
| harmonic | OpenMM | GPU | Nelder-Mead | 1040.5 | 892.1 | 9.0 s | 109.3 |
| mm3 | JAX | GPU | Powell | 4.0 | 1.9 | 37.1 s | 323.2 |
| mm3 | OpenMM | GPU | L-BFGS-B | 30.4 | 25.7 | 20.0 s | 6.3 |
| mm3 | OpenMM | GPU | L-BFGS-B (FD) | 56.1 | 39.0 | 3.6 s | 107.7 |
| mm3 | Tinker | CPU | L-BFGS-B (FD) | 114.1 | 93.4 | 98.2 s | 4.3 |
| mm3 | Tinker | CPU | L-BFGS-B | 114.1 | 93.4 | 98.5 s | 4.3 |
| mm3 | JAX | GPU | L-BFGS-B (FD) | 114.1 | 93.6 | 1.0 s | 329.4 |
| mm3 | JAX | GPU | Nelder-Mead | 547.5 | 281.8 | 25.0 s | 329.1 |
| mm3 | Tinker | CPU | Powell | 555.7 | 291.2 | 1106.3 s | 4.4 |
| mm3 | Tinker | CPU | Nelder-Mead | 563.3 | 299.5 | 157.3 s | 4.4 |
| mm3 | OpenMM | GPU | Nelder-Mead | 564.2 | 299.9 | 8.3 s | 109.1 |
| mm3 | OpenMM | GPU | Powell | 575.7 | 311.9 | 131.2 s | 103.6 |
| mm3 | OpenMM | GPU | grad-simp (auto) | 579.2 | 315.5 | 34.7 s | 23.0 |
| mm3 | OpenMM | GPU | grad-simp (FD) | 580.8 | 317.3 | 25.4 s | 106.1 |
| mm3 | JAX | GPU | grad-simp (auto) | 586.4 | 322.2 | 5.7 s | 119.9 |
| mm3 | JAX | GPU | L-BFGS-B | 586.7 | 321.7 | 3.8 s | 20.1 |
| mm3 | Tinker | CPU | grad-simp (auto) | 833.4 | 612.3 | 749.8 s | 4.4 |
| mm3 | Tinker | CPU | grad-simp (FD) | 833.4 | 612.3 | 745.3 s | 4.4 |
| mm3 | JAX | GPU | grad-simp (FD) | 834.5 | 613.2 | 13.6 s | 329.4 |

## Interpretation

- Harmonic + Powell is the clear small-system winner: JAX, JAX-MD, and OpenMM
  all reach a near-exact fit, while JAX and JAX-MD do so in under 8 s.
- MM3 remains optimizer-sensitive. The best MM3 result is JAX + Powell
  (RMSD 4.0), while OpenMM + L-BFGS-B is the strongest OpenMM MM3 run.
- Analytical gradients help L-BFGS-B significantly. On MM3, OpenMM L-BFGS-B
  with analytical gradients reaches RMSD 30.4 vs 56.1 with finite-difference
  (1.8× better). On harmonic, the gap is even larger: JAX L-BFGS-B analytical
  reaches 552.6 vs FD's 813.4.
- Tinker L-BFGS-B (analytical) and L-BFGS-B (FD) reach identical RMSD on MM3
  (114.1). Tinker's numerical gradients are accurate enough that analytical
  provides no benefit here.
- On identical parameters, JAX, JAX-MD, and OpenMM agree to machine precision
  when the functional form matches: energy deltas stay at or below
  3 × 10⁻¹⁸ kcal/mol and frequency deltas stay below 0.001 cm⁻¹.
- grad-simp is not the right default on this problem. Neither the FD nor the
  auto variant beats the best single-shot result in either form.
- The optimization loop dominates runtime; Seminario is effectively free by
  comparison and serves mainly as a starting point, not as the expensive step.

## Artifacts and provenance

- Inputs:
  [QM reference data](https://github.com/ericchansen/q2mm/tree/master/examples/sn2-test/qm-reference)
- Outputs:
  [Benchmark results (JSON)](https://github.com/ericchansen/q2mm/tree/master/benchmarks/ch3f/results),
  [optimized force fields](https://github.com/ericchansen/q2mm/tree/master/benchmarks/ch3f/forcefields),
  and the
  [leaderboard](https://github.com/ericchansen/q2mm/blob/master/benchmarks/ch3f/leaderboard.txt)

This page uses the current full-matrix artifact set in `benchmarks/ch3f/`.

## Reproducing

```bash
q2mm-benchmark --system ch3f --output benchmarks/ch3f
q2mm-benchmark --load benchmarks/ch3f/results
q2mm-benchmark --list
```
