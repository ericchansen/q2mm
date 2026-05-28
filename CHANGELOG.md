# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `JaxOptOptimizer` for end-to-end differentiable optimization using jaxopt
  L-BFGS / L-BFGS-B — entire loss, gradient, and step computation runs inside
  JIT-compiled JAX with no Python callbacks (#152, #176)
- `JaxLoss` JIT-compiled loss function supporting energy, frequency, hessian
  element, and eigenmatrix reference types in a single differentiable graph
- `ObjectiveSpec` frozen dataclass as the shared contract between Python-side
  `ObjectiveFunction` and JAX-side `JaxLoss`
- `jaxopt:` prefix support in `OptimizationLoop.full_method` (e.g.
  `"jaxopt:lbfgs"`, `"jaxopt:lbfgsb"`) for using JaxOpt as the gradient phase
  in grad-simp cycling
- JAX-native frequency parameter sensitivity (`_jax_frequency_param_jacobian`)
  using closed-form eigenvalue derivatives — replaces finite-difference Jacobian
  for frequency objectives in the differentiable pipeline
- `BasinHoppingOptimizer` wrapping `scipy.optimize.basinhopping` with bounded
  perturbation steps and L-BFGS-B local minimization for global optimization
  (#232)
- `MultiStartOptimizer` meta-optimizer that runs any inner optimizer from N
  perturbed starting points and returns the best result (#231)
- L2 regularization on `ObjectiveFunction` via `regularization` and
  `reference_params` kwargs — penalizes parameter drift from QFUERZA starting
  values on under-determined systems (#229)
- Basin-hopping and multi-start dispatch in benchmark CLI
  (`--optimizer "basinhopping"`, `--optimizer "multi:L-BFGS-B"`)
- `multi:` prefix support in `OptimizationLoop.full_method` for multi-start
  as the gradient phase of grad-simp cycling
- Composed workflow benchmark script (`scripts/bench_composed.py`) for
  multi-start → optax Adam pipelines
- 75-combo CH₃F benchmark matrix (71 single-shot + 4 composed workflows)
  with full results, force fields, and history
- `OptaxOptimizer` for JAX-native Adam, AdaGrad, SGD, and AdamW optimization
  with analytical gradients and learning rate schedules
- Optax benchmark configs in CLI (`--optimizer "optax:adam"`, etc.)
- `--learning-rate` and `--optax-max-steps` CLI arguments for benchmark runner
- `--preflight` flag for benchmark CLI to verify GPU environment before running
- Platform support documentation (`docs/platform-support.md`)
- Published FF validation test harness (`test/integration/test_published_ff_validation.py`)
- Benchmark results saving to `results/` by default
- CHANGELOG.md (this file)

### Changed
- Per-molecule JIT compilation in `JaxLoss` — each molecule's loss and
  gradient compiles independently, preventing GPU OOM on multi-molecule
  systems
- Published-FF system loaders now preserve literature OPT values as
  published instead of silently overwriting them with QFUERZA projections;
  frozen base-FF parameters remain the optimization invariant for TS systems
- Optimizer-comparison documentation now treats MacroModel MM3* reproduction
  as an explicit cross-engine transfer boundary, not a release blocker for
  the q2mm JAX/OpenMM alpha line
- Benchmark data moved to `ericchansen/q2mm-data` repository
- Overhauled `AGENTS.md` for better AI agent guidance
- OpenMM-CUDA-12 platform gate: now excludes only macOS (was Linux-only)
- JAX engine now supports both harmonic and MM3 functional forms

### Fixed
- **MM3 angle gradient correctness** — replaced the JAX angle term's
  gradient-killing `arccos(clip())` path with a well-conditioned
  `atan2`-based custom VJP near collinear geometries. This moved the
  literature-scale TS systems back into the JaxLoss ratio gate and unlocked
  substantial real-objective improvements for Heck relay and Rh 1,4-conjugate.
- **Heck relay optimization** — after the MM3 angle-gradient fix, the
  JaxLoss/ObjectiveFunction ratio is 1.085 and SciPy L-BFGS-B over JaxLoss
  reduces the sampled real objective by 52.82% ± 1.54% CI95.
- **Rh 1,4-conjugate optimization** — the same gradient fix resolves the
  spurious stationary point seen in earlier runs; sampled real-objective
  reduction is 18.00% ± 4.17% CI95.
- **Pd 1,4-conjugate optimization** — after preserving the published Wahlers
  OPT values, the default ratio gate passes and the real objective improves
  by 16.1%.
- **Pd-allyl verdict** — n=10 sampled evaluation confirms the published
  Wahlers OPT values sit at a q2mm JaxLoss local minimum; any improvement is
  below the ±0.40% CI95 noise band.
- **JaxLoss harmonic restraint** — `_relax_coords()` previously added an
  artificial harmonic restraint (k=100 kcal/mol·Å²) to geometry
  relaxation, causing JaxLoss to optimize a different objective than
  `ObjectiveFunction` and producing 0% real improvement. Removed the
  restraint; systems with good Seminario starting FFs now show real
  improvement (28.7% for Rh-enamide).
- Re-enabled JaxLoss ratio check for `scipy-lbfgsb-jax` benchmark config
  (was bypassed via `ratio_tol: None`)
- Hessian unit conversion for Jaguar `.in` files (Hartree/Bohr² → kJ/mol/Å²)
- Golden fixture tolerance tightened from 2e-3 to 5e-4
- UTF-8 output encoding on Windows for benchmark CLI

### Removed
- `invert_ts_curvature` field from `MoleculeSpec` — curvature inversion
  now happens only during Seminario projection, before JAX
- `benchmarks/` directory — data moved to `q2mm-data` repository

## [5.0.0a3] - Pre-release

### Added
- Modern Python rewrite of Q2MM with clean architecture
- Format-agnostic data models (`ForceField`, `Q2MMMolecule`)
- OpenMM, JAX, JAX-MD, Tinker, Psi4 backend engines
- Seminario method for Hessian-based force constant estimation
- Scipy-based optimizers (L-BFGS-B, Nelder-Mead, trust-region)
- Batched Hessian evaluation via `jax.vmap`
- Comprehensive test suite (692+ tests)
- MkDocs documentation site

[Unreleased]: https://github.com/ericchansen/q2mm/compare/v5.0.0a3...HEAD
[5.0.0a3]: https://github.com/ericchansen/q2mm/releases/tag/v5.0.0a3
