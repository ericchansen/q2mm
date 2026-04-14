# Small Molecules

This page answers one question: how do the currently supported backend, form,
and optimizer combinations compare on a small, fully tractable benchmark? The
system is CH₃F (5 atoms, 8 fitted parameters) against B3LYP/6-31+G(d) QM
frequencies. Unlike the Rh-Enamide page, this page is the full supported
matrix, so it is the right place to compare combinations directly.

## Scope

- System: CH₃F (1 molecule, 5 atoms, 8 parameters)
- QM reference: B3LYP/6-31+G(d)
- Matrix size: 75 supported combos (71 single-shot + 4 composed workflows)
- Backends/forms: [JAX](https://jax.readthedocs.io/) and [OpenMM](https://openmm.org/) on harmonic + MM3, [JAX-MD](https://jax-md.readthedocs.io/) on harmonic, [Tinker](https://dasher.wustl.edu/tinker/)
  on MM3
- Optimizers: Powell, L-BFGS-B, Nelder-Mead, grad-simp,
  [optax](https://optax.readthedocs.io/) (Adam, AdaGrad, SGD),
  [basin-hopping](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html)
  (T=1.0, T=0.5), multi-start (n=5, n=10), and L2-regularized variants —
  each gradient-using optimizer is run twice (once with analytical frequency
  gradients, once with pure FD); optax optimizers use analytical gradients
  only (JAX backend); global optimizers (basin-hopping, multi-start) and L2
  variants run on fast GPU backends only (JAX, JAX-MD, OpenMM CUDA);
  composed workflows (multi-start → Adam, grad-simp with multi-start inner)
  on MM3 only
- Starting point: QFUERZA initialization — JAX/JAX-MD begin at 192.0 cm⁻¹
  RMSD, OpenMM at 191.9 cm⁻¹, Tinker at 192.1 cm⁻¹

## Full CH₃F matrix

Default rows are grouped by functional form and then by final RMSD. Use the
filters and sortable headers to narrow form/backend/device/optimizer
combinations, and compare like-with-like inside each form: harmonic and MM3
rows share the same benchmark system, but they do not represent the same
force-field model.

<div class="benchmark-table-anchor" data-benchmark-table="small-molecules"></div>

| Form | Backend | Device | Optimizer | F∇ | Final RMSD (cm⁻¹) | Final MAE | Time | eval/s |
|------|---------|--------|-----------|:--:|-------------------:|----------:|-----:|--------:|
| harmonic | JAX-MD | GPU | multi:L-BFGS-B (n=5) | FD | 525.9 | 241.5 | 20.6 s | 19.8 |
| harmonic | JAX | GPU | L-BFGS-B | A | 528.7 | 257.3 | 1.9 s | 41.1 |
| harmonic | JAX-MD | GPU | grad-simp | FD | 528.8 | 242.3 | 5.9 s | 142.5 |
| harmonic | JAX | GPU | grad-simp | A | 529.1 | 243.3 | 5.5 s | 243.1 |
| harmonic | JAX | GPU | multi:L-BFGS-B (n=10) | A | 529.5 | 246.2 | 7.9 s | 125.7 |
| harmonic | JAX | GPU | basinhopping (T=0.5) | A | 530.7 | 253.2 | 5.3 s | 117.5 |
| harmonic | JAX | GPU | basinhopping (T=1.0) | A | 530.9 | 253.3 | 6.4 s | 117.3 |
| harmonic | JAX-MD | GPU | L-BFGS-B | FD | 531.1 | 254.6 | 4.3 s | 20.2 |
| harmonic | JAX-MD | GPU | basinhopping (T=0.5) | FD | 531.2 | 254.1 | 31.0 s | 19.6 |
| harmonic | JAX-MD | GPU | basinhopping (T=1.0) | FD | 531.3 | 255.4 | 31.3 s | 19.5 |
| harmonic | JAX | GPU | multi:L-BFGS-B (n=5) | A | 531.7 | 254.2 | 4.3 s | 106.9 |
| harmonic | OpenMM | GPU | grad-simp | FD | 979.5 | 786.3 | 45.8 s | 91.5 |
| harmonic | JAX | GPU | grad-simp | FD | 981.4 | 790.1 | 13.1 s | 353.4 |
| harmonic | JAX-MD | GPU | grad-simp | FD | 981.4 | 790.1 | 13.8 s | 334.7 |
| harmonic | OpenMM | GPU | grad-simp | FD | 981.9 | 794.7 | 67.8 s | 31.0 |
| harmonic | OpenMM | GPU | multi:L-BFGS-B (n=5) | FD | 983.4 | 837.3 | 34.1 s | 6.3 |
| harmonic | OpenMM | GPU | multi:L-BFGS-B (n=10) | FD | 985.5 | 836.1 | 64.5 s | 6.4 |
| harmonic | JAX | GPU | Nelder-Mead | — | 987.4 | 795.0 | 34.2 s | 357.7 |
| harmonic | JAX-MD | GPU | Nelder-Mead | — | 987.5 | 795.0 | 34.2 s | 344.0 |
| harmonic | JAX | GPU | optax:sgd | A | 990.0 | 838.0 | 18.2 s | 109.9 |
| harmonic | JAX | GPU | L-BFGS-B + L2(λ=0.01) | A | 993.3 | 852.5 | 1.3 s | 20.3 |
| harmonic | JAX-MD | GPU | L-BFGS-B + L2(λ=0.01) | FD | 993.3 | 852.5 | 1.3 s | 20.6 |
| harmonic | JAX | GPU | optax:adam + L2(λ=0.01) | A | 993.4 | 852.5 | 6.0 s | 67.5 |
| harmonic | JAX-MD | GPU | multi:L-BFGS-B (n=10) | FD | 994.6 | 815.9 | 48.6 s | 19.6 |
| harmonic | OpenMM | GPU | L-BFGS-B + L2(λ=0.01) | FD | 995.8 | 857.6 | 19.5 s | 6.5 |
| harmonic | JAX | GPU | optax:adam+cosine | A | 999.4 | 831.7 | 29.5 s | 67.7 |
| harmonic | JAX | GPU | optax:adam | A | 1000.4 | 831.4 | 23.4 s | 85.4 |
| harmonic | JAX | GPU | optax:adagrad | A | 1000.9 | 868.0 | 19.7 s | 101.5 |
| harmonic | OpenMM | GPU | basinhopping (T=0.5) | FD | 1021.5 | 857.4 | 155.1 s | 6.1 |
| harmonic | OpenMM | GPU | Powell | — | 1036.7 | 891.7 | 62.5 s | 97.8 |
| harmonic | JAX | GPU | Powell | — | 1041.5 | 899.0 | 10.1 s | 342.8 |
| harmonic | OpenMM | GPU | basinhopping (T=1.0) | FD | 1041.4 | 872.4 | 140.0 s | 6.1 |
| harmonic | JAX-MD | GPU | Powell | — | 1041.5 | 899.0 | 10.4 s | 342.1 |
| harmonic | OpenMM | GPU | Nelder-Mead | — | 1043.6 | 868.8 | 9.2 s | 102.7 |
| harmonic | JAX | GPU | L-BFGS-B | FD | 1048.3 | 934.6 | 0.5 s | 336.0 |
| harmonic | JAX-MD | GPU | L-BFGS-B | FD | 1048.3 | 934.6 | 0.5 s | 334.8 |
| harmonic | OpenMM | GPU | L-BFGS-B | FD | 1048.3 | 934.7 | 3.5 s | 77.2 |
| harmonic | OpenMM | GPU | L-BFGS-B | FD | 1049.5 | 936.1 | 4.0 s | 5.5 |
| mm3 | OpenMM | GPU | multi:L-BFGS-B (n=10) | FD | 28.7 | 20.2 | 157.4 s | 6.4 |
| mm3 | JAX | GPU | optax:adam | A | 56.3 | 44.0 | 25.2 s | 79.4 |
| mm3 | OpenMM | GPU | L-BFGS-B | FD | 59.5 | 46.7 | 4.8 s | 104.4 |
| mm3 | JAX | GPU | optax:adam+cosine | A | 60.6 | 42.7 | 29.8 s | 67.2 |
| mm3 | OpenMM | GPU | L-BFGS-B | FD | 83.6 | 62.9 | 9.7 s | 5.4 |
| mm3 | Tinker | CPU | L-BFGS-B | FD | 83.8 | 63.4 | 152.5 s | 4.1 |
| mm3 | Tinker | CPU | L-BFGS-B | FD | 83.8 | 63.4 | 150.3 s | 4.2 |
| mm3 | JAX | GPU | L-BFGS-B | FD | 113.5 | 90.6 | 0.8 s | 347.2 |
| mm3 | OpenMM | GPU | L-BFGS-B + L2(λ=0.01) | FD | 133.3 | 108.8 | 12.7 s | 6.3 |
| mm3 | JAX | GPU | L-BFGS-B + L2(λ=0.01) | A | 133.5 | 109.5 | 1.8 s | 20.4 |
| mm3 | JAX | GPU | optax:adam + L2(λ=0.01) | A | 133.5 | 109.5 | 5.4 s | 56.8 |
| mm3 | JAX | GPU | optax:adagrad | A | 138.0 | 113.5 | 20.0 s | 100.0 |
| mm3 | JAX | GPU | optax:sgd | A | 192.0 | 177.5 | 1.7 s | 12.7 |
| mm3 | OpenMM | GPU | basinhopping (T=0.5) | FD | 513.8 | 263.9 | 179.3 s | 6.3 |
| mm3 | Tinker | CPU | Powell | — | 542.5 | 275.2 | 2768.6 s | 4.3 |
| mm3 | Tinker | CPU | grad-simp | FD | 564.4 | 314.5 | 1094.9 s | 4.3 |
| mm3 | Tinker | CPU | grad-simp | FD | 564.4 | 314.5 | 1097.7 s | 4.3 |
| mm3 | OpenMM | GPU | grad-simp | FD | 566.2 | 306.6 | 36.1 s | 24.7 |
| mm3 | OpenMM | GPU | grad-simp | FD | 573.1 | 311.6 | 29.5 s | 97.1 |
| mm3 | Tinker | CPU | Nelder-Mead | — | 576.3 | 311.5 | 152.5 s | 4.3 |
| mm3 | OpenMM | GPU | multi:L-BFGS-B (n=5) | FD | 578.1 | 341.1 | 59.8 s | 6.6 |
| mm3 | JAX | GPU | L-BFGS-B | A | 579.0 | 313.9 | 2.2 s | 31.4 |
| mm3 | JAX | GPU | grad-simp | A | 579.0 | 313.9 | 3.4 s | 139.1 |
| mm3 | JAX | GPU | multi:L-BFGS-B (n=5) | A | 579.0 | 313.9 | 4.6 s | 92.3 |
| mm3 | JAX | GPU | basinhopping (T=0.5) | A | 579.0 | 313.9 | 10.7 s | 125.4 |
| mm3 | OpenMM | GPU | Nelder-Mead | — | 581.1 | 315.1 | 8.5 s | 97.0 |
| mm3 | JAX | GPU | multi:L-BFGS-B (n=10) | A | 586.3 | 319.6 | 7.5 s | 112.3 |
| mm3 | JAX | GPU | Nelder-Mead | — | 608.1 | 334.2 | 25.6 s | 344.9 |
| mm3 | OpenMM | GPU | basinhopping (T=1.0) | FD | 842.6 | 636.2 | 168.0 s | 6.3 |
| mm3 | JAX | GPU | grad-simp | FD | 1050.0 | 910.4 | 8.2 s | 343.0 |
| mm3 | JAX | GPU | Powell | — | 1080.7 | 937.3 | 15.1 s | 339.0 |
| mm3 | OpenMM | GPU | Powell | — | 1090.5 | 950.4 | 124.2 s | 95.3 |
| mm3 | JAX | GPU | basinhopping (T=1.0) | A | 1105.0 | 978.2 | 11.7 s | 122.1 |
| | | | | | | | | |
| **Composed workflows** | | | | | | | | |
| mm3 | OpenMM | GPU | multi:L-BFGS-B (n=10) → optax:adam | FD | 46.1 | — | 604.0 s | 1084 |
| mm3 | JAX | GPU | multi:L-BFGS-B (n=10) → optax:adam | A | 563.8 | — | 24.7 s | 968 |
| mm3 | OpenMM | GPU | grad-simp (multi:L-BFGS-B inner) | FD | 592.1 | 341.1 | 450.2 s | 23586 |
| mm3 | JAX | GPU | grad-simp (multi:L-BFGS-B inner) | A | 526.7 | 238.2 | 31.6 s | 8970 |

## Interpretation

**F∇** = frequency gradient mode.
**A** = analytical (autodiff), **FD** = finite-difference, **—** = not
applicable (derivative-free optimizer). The energy gradient column (E∇) is
omitted because CH₃F benchmarks optimize on frequency data only.

### Harmonic form

- The best harmonic results cluster around 526–531 cm⁻¹ RMSD, achieved by
  JAX, JAX-MD, and OpenMM with [L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html), [grad-simp](../how-it-works/optimization-guide.md#workflow-c-medium-and-large-systems),
  [multi-start](../how-it-works/optimization-guide.md#workflow-b-small--rugged), or [basin-hopping](../how-it-works/optimization-guide.md#workflow-b-small--rugged)
  using analytical frequency gradients. These combos benefit from QFUERZA's
  physically motivated starting parameters.
- **[Multi-start](https://en.wikipedia.org/wiki/Multi-start_method) and
  [basin-hopping](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html)
  match plain L-BFGS-B** on the harmonic form (526–531 RMSD range). The
  harmonic landscape has fewer local minima, so random restarts and
  stochastic perturbations do not discover better basins than the QFUERZA
  starting point provides.
- **L2 regularization hurts harmonic performance** (993 cm⁻¹ vs 529 for
  unregularized L-BFGS-B). The penalty prevents parameters from reaching the
  deep basin that L-BFGS-B normally finds. L2 is counterproductive when the
  landscape is well-conditioned.
- [Optax](https://optax.readthedocs.io/) optimizers ([Adam](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam), [AdaGrad](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adagrad), [SGD](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.sgd)) perform poorly on the harmonic form
  (990–1001 cm⁻¹), comparable to derivative-free methods. The harmonic
  landscape from the QFUERZA starting point appears to favour quasi-Newton
  methods (L-BFGS-B) that use curvature information.
- Derivative-free optimizers ([Powell](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html), [Nelder-Mead](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)) perform poorly on the
  harmonic form from the QFUERZA starting point, landing in the 987–1049
  range. Under the previous Seminario initialization these reached near-zero
  RMSD, but that was an initialization-sensitive local optimum — the result
  was not robust across starting points.
- FD-only gradient combos (L-BFGS-B with FD) also perform poorly (~1048),
  suggesting that finite-difference frequency gradients are too noisy to
  guide L-BFGS-B from the QFUERZA basin.

### MM3 form

- **Multi-start n=10 on OpenMM achieves the best MM3 result at 28.7 cm⁻¹
  RMSD** — a 2× improvement over the previous best ([optax](https://optax.readthedocs.io/) Adam at 56.3) and a
  20× improvement over JAX [L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) with analytical gradients (579). Running
  10 independent L-BFGS-B optimizations from random starting points within
  the parameter bounds found a basin that no single-start optimizer reached.
- **L2 regularization dramatically improves L-BFGS-B on MM3**: 579 → 134 cm⁻¹
  (4× better), consistent across JAX and OpenMM backends. The λ=0.01 penalty
  prevents parameters from drifting too far from the QFUERZA starting point,
  steering L-BFGS-B away from the poor local minimum it normally finds. L2
  with optax Adam (134) does not improve over Adam alone (56), suggesting
  Adam already navigates the landscape well enough.
- **[Basin-hopping](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html) shows mixed results.** OpenMM basin-hopping T=0.5 found a
  better basin (514 RMSD) than the default L-BFGS-B minimum (579), but
  T=1.0 on both JAX (1105) and OpenMM (843) accepted too many uphill moves
  and wandered into worse regions. Basin-hopping is sensitive to the
  temperature parameter and the noise level of finite-difference gradients.
- Adam with cosine annealing (60.6 cm⁻¹) and [AdaGrad](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adagrad) (138.0 cm⁻¹) also
  outperform all scipy-based optimizers on the JAX backend.
- SGD fails to improve from the starting point (192 cm⁻¹), diverging early
  at the default learning rate — it needs careful LR tuning.
- OpenMM L-BFGS-B with FD gradients remains competitive at 59.5 cm⁻¹. The
  similarity between Adam (56.3) and OpenMM L-BFGS-B FD (59.5) suggests
  these are converging toward the same basin, but via very different paths.
- Tinker L-BFGS-B improved significantly from the prior run (114 → 84 cm⁻¹),
  showing that QFUERZA provides a better basin for gradient-based MM3
  optimization through the Tinker backend.
- Powell and Nelder-Mead on MM3 remain mid-range (542–608) and are
  insensitive to the initialization change, as expected for derivative-free
  methods on a rugged landscape.

### Composed workflows

Two composed strategies were benchmarked on CH₃F MM3 — the only landscape
where multi-start and global search methods show material differences:

- **Multi-start → Adam refinement** (Workflow B composition): On OpenMM CUDA,
  multi-start n=10 found a 46.2 RMSD basin. Running optax Adam from that
  result improved it to 46.1 — Adam added almost nothing. The FD gradient
  noise that helped multi-start find the basin limits Adam's ability to
  refine further. On JAX, multi-start found 563.8 and Adam left it
  unchanged — analytical gradients converge to the same local minimum
  regardless of starting approach. **Verdict:** Multi-start alone is
  sufficient; Adam refinement is not worth the extra cost.

- **Grad-simp with multi-start inner** (Workflow C composition): Using
  `full_method="multi:L-BFGS-B"` in the cycling loop gave 592 RMSD on
  OpenMM (worse than plain grad-simp at 586) and 527 on JAX (marginally
  better than 579). The random restarts within each gradient phase disrupt
  the cycling algorithm's inter-cycle convergence. **Verdict:** For rugged
  landscapes, use standalone multi-start ([Workflow B](../how-it-works/optimization-guide.md#workflow-b-small-rugged))
  rather than embedding it inside cycling.

### Cross-cutting observations

- **Optimizer choice matters more than backend choice on MM3.** The spread
  between the best (multi-start n=10, 28.7) and worst (basin-hopping T=1.0,
  1105) optimizer on MM3 is 39×, while the spread between backends using
  the same optimizer is typically < 2×.
- **Global optimization strategies are problem-dependent.** Multi-start n=10
  found the best-ever MM3 result (28.7), but basin-hopping T=1.0 found the
  *worst* (1105). On harmonic, neither strategy improves over plain L-BFGS-B.
  The value of global optimization depends on how rugged the landscape is.
- **L2 regularization acts as a safety net, not a global optimizer.** It
  dramatically helps L-BFGS-B on the rugged MM3 landscape (579 → 134) by
  preventing parameter drift, but actively hurts on the well-conditioned
  harmonic landscape (529 → 993). Use L2 when single-start optimizers are
  finding poor local minima.
- Analytical frequency gradients produce the best results on the harmonic
  problem. The top harmonic results all use analytical frequency
  gradients (A) or analytical-fallback (FD with JAX-MD).
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
- Git SHA: `690f81d4` (q2mm 5.0.0a3 + optax integration), updated with
  advanced optimizers in PR #239
- GPU: NVIDIA GeForce RTX 5090

This page uses the current full-matrix artifact set in `benchmarks/ch3f/`.

## Reproducing

```bash
# Run full matrix (all backends, all optimizers)
q2mm-benchmark --system ch3f --output benchmarks/ch3f --platform CUDA

# Run optax optimizers only (JAX backend)
q2mm-benchmark --system ch3f --output benchmarks/ch3f --backend jax \
  --optimizer "optax:adam" "optax:adam+cosine" "optax:adagrad" "optax:sgd" \
  --learning-rate 0.01 --optax-max-steps 2000

# Run global optimizers only (fast backends)
q2mm-benchmark --system ch3f --output benchmarks/ch3f --backend jax \
  --optimizer "basinhopping (T=1.0)" "basinhopping (T=0.5)" \
  "multi:L-BFGS-B (n=5)" "multi:L-BFGS-B (n=10)" \
  --platform CUDA

# Run L2-regularized optimizers
q2mm-benchmark --system ch3f --output benchmarks/ch3f --backend jax \
  --optimizer "L-BFGS-B + L2(λ=0.01)" "optax:adam + L2(λ=0.01)" \
  --platform CUDA

# Load and display results
q2mm-benchmark --load benchmarks/ch3f/results

# List available optimizers
q2mm-benchmark --list
```
