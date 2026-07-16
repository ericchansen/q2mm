# Differences from v4

This page compares the current q2mm codebase with **v4** — the
Python 2 implementation at [q2mm/q2mm](https://github.com/q2mm/q2mm),
which is a fork of this repository.

Both implement the grad-simp cycling approach from
Norrby and co-workers — alternating gradient and simplex phases to fit force
field parameters against quantum mechanical reference data.

---

## What changed and why

The original Q2MM code drives external programs (MacroModel, [Tinker](https://dasher.wustl.edu/tinker/), Gaussian)
via subprocess calls, parses text output, and implements its own gradient
solvers. This works, but each force-field evaluation requires writing files,
spawning a process, and reading results back — a cycle that dominates runtime
on large systems.

q2mm (this repo) replaces the subprocess loop with in-process Python backends
([OpenMM](https://openmm.org/), [JAX](https://jax.readthedocs.io/), [JAX-MD](https://jax-md.readthedocs.io/)) where energy, gradient, and Hessian computations happen
inside the same process. [JAX backends are fully differentiable](../backends/jax-engine.md)
([`jax.grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html), [`jax.hessian`](https://jax.readthedocs.io/en/latest/_autosummary/jax.hessian.html), [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html)), which
eliminates finite-difference overhead for gradient and Hessian evaluation.

The practical result on committed benchmarks
([Rh-enamide](../systems/rh-enamide.md),
[CH₃F](../systems/small-molecules.md)):

- **Rh-enamide grad-simp**: OpenMM CUDA reached 42.7 cm⁻¹ RMSD in 33,223 s
  optimizer time; JAX MM3 GPU reached the same RMSD in 1,471 s — a
  [~23× speedup](../systems/rh-enamide.md).
- **GPU acceleration**: JAX and OpenMM CUDA backends run on GPU without
  leaving the Python process
  ([GPU benchmarks](../benchmarks/gpu.md)).

The codebase also replaces the hand-rolled solvers with [SciPy](https://scipy.org/) wrappers, adds
eigendecomposition safety checks (symmetrization, NaN/Inf detection, penalty
fallback), and introduces bound-aware sensitivity analysis. Details for each
area are linked in the table below.

---

## At a glance

| Capability | v4 (q2mm/q2mm) | q2mm (this repo) |
|---|---|---|
| Gradient solvers | 5 hand-rolled ([least-squares](https://github.com/Q2MM/q2mm/blob/b26404b/q2mm/gradient.py#L485), [Lagrange](https://github.com/Q2MM/q2mm/blob/b26404b/q2mm/gradient.py#L448), [LM](https://github.com/Q2MM/q2mm/blob/b26404b/q2mm/gradient.py#L467), [NR](https://github.com/Q2MM/q2mm/blob/b26404b/q2mm/gradient.py#L498), [SVD](https://github.com/Q2MM/q2mm/blob/b26404b/q2mm/gradient.py#L533)) | [SciPy wrapper](#optimizers) (L-BFGS-B, Nelder-Mead, Powell, trust-constr, least_squares) + [optax](#optimizers) (Adam, AdaGrad, SGD) |
| Global optimizers | None | [Basin-hopping](#optimizers) + [multi-start](#optimizers) global search |
| Regularization | None | [L2 penalty](#optimizers) to prevent parameter drift |
| Simplex | Custom Nelder-Mead (3 params) | [SciPy Nelder-Mead via subspace projection](#optimizers) |
| Cycling loop | Text command file | Dataclass-configured [`OptimizationLoop`](optimization-guide.md) |
| Sensitivity | Exception-based, one-sided FD fallback | [Symmetric step shrinking, bound-aware](#sensitivity-bounds) |
| Eigendecomposition | `np.linalg.eigh` | [Symmetrize + NaN/Inf check + penalty fallback](#eigendecomposition) |
| Backends | Subprocess (MacroModel, Tinker, Amber, Gaussian, Jaguar) | [API + differentiable](#backend-implementations) (OpenMM, Tinker, JAX, JAX-MD, Psi4) |
| Analytical ∂E/∂θ | Not evident | Yes ([`jax.grad`](../backends/jax-engine.md), OpenMM) |
| Batched evaluation | Not evident | Yes ([`jax.vmap`](../backends/jax-engine.md)) |
| GPU | Subprocess; depends on backend | [In-process CUDA](../benchmarks/gpu.md) (OpenMM, JAX, JAX-MD) |
| Functional forms | MM3 only (implicit) | [MM3 + Harmonic](architecture.md) (explicit enum) |
| Force field model | Format-coupled (`ParamMM3`) | [Format-agnostic `ForceField` dataclass](#evaluators-force-field-model) |
| Diagnostics | [Utility scripts](https://github.com/Q2MM/q2mm/tree/b26404b/tools) | [Full CLI benchmark suite](#diagnostics) |
| Test suite | [9 test files](https://github.com/Q2MM/q2mm/tree/b26404b/test) | 1,100+ unit tests |

---

## Optimizers

q2mm/q2mm implements five gradient solvers from scratch (least-squares,
Lagrange, Levenberg-Marquardt, Newton-Raphson, SVD) plus a custom
Nelder-Mead simplex. A text command file drives the `GRAD` → `SIMP`
cycling loop.

q2mm (this repo) delegates all optimization to SciPy (`scipy.optimize.minimize`
and `scipy.optimize.least_squares`), wrapping them in a `ScipyOptimizer` class.
The cycling loop is a dataclass-configured `OptimizationLoop` that:

1. Runs a full-space SciPy pass on all parameters
2. Ranks parameters by simplex suitability (lowest `simp_var`)
3. Runs Nelder-Mead on the least gradient-suitable subset
4. Repeats until convergence

Delegating to SciPy means the numerics are maintained by a large community and
benefit from ongoing improvements.

For JAX-based workflows, q2mm (this repo) also provides an `OptaxOptimizer`
that wraps [optax](https://optax.readthedocs.io/) adaptive optimizers (Adam,
AdaGrad, SGD, AdamW). These use JAX's analytical gradients directly and excel
on rugged potential energy surfaces like MM3, where Adam achieves 10× better
RMSD than L-BFGS-B on CH₃F
([benchmark results](../systems/small-molecules.md)). See the
[Optimization Guide](optimization-guide.md#workflow-b-small-rugged)
for details.

For global optimization, `BasinHoppingOptimizer` wraps
[`scipy.optimize.basinhopping`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html)
with bounded perturbation steps, and `MultiStartOptimizer` runs any inner
optimizer from N perturbed starting points.  `ObjectivePlan` also carries L2
regularization (`regularization` kwarg) to penalize parameter drift from
QFUERZA starting values — see
[Optimization Guide](optimization-guide.md#l2-regularization).

---

## Sensitivity & bounds

q2mm/q2mm catches out-of-bounds perturbations via exceptions and falls back to
one-sided finite differences — this gives O(h) truncation error instead of
O(h²), producing biased sensitivity estimates near bounds.

q2mm (this repo) shrinks the step size *before* evaluation to keep both
perturbations within bounds, preserving O(h²) central-difference accuracy.
Parameters at their bounds receive infinite sensitivity scores, ranking them
last and effectively excluding them from the simplex selection.

For the math, see [Theory & Methods — Optimization](theory.md#stage-4-optimization).

---

## Eigendecomposition

q2mm (this repo) adds Hessian symmetrization, NaN/Inf detection, and penalty
fallback to the eigendecomposition pipeline.

For details on Limé & Norrby's eigenvalue methods (A–E), see
[Theory & Methods — TS Eigenvalue Treatment](theory.md#stage-2-transition-state-eigenvalue-treatment).

---

## Backend implementations

q2mm/q2mm routes all computation through subprocess calls — write input files,
run shell commands, parse text output. Supported backends: MacroModel, Tinker,
Amber, Gaussian, Jaguar.

q2mm (this repo) uses in-process Python APIs with a registry pattern:

| Backend | Interface | Differentiable | GPU |
|--------|-----------|----------------|-----|
| OpenMM | Python API | FD Hessian; analytical energy grad | CUDA, OpenCL |
| Tinker | Subprocess | No | No |
| JAX | Pure Python | `jax.grad`, `jax.hessian`, `jax.vmap` | CUDA |
| JAX-MD | jax-md OPLSAA | `jax.grad`, `jax.hessian`, `jax.vmap` | CUDA |
| Psi4 | QM backend | N/A | N/A |

Key capabilities: analytical gradients via `jax.grad`, batched Hessians via
`jax.vmap`, JIT compilation, and runtime parameter substitution without
rebuilding the system.

---

## Evaluators & force field model

q2mm/q2mm computes the objective in two monolithic files (`compare.py` and
`calculate.py`) that interleave I/O and scoring. Parameters are stored in
`ParamMM3` objects tied to MM3 conventions.

q2mm (this repo) splits evaluation into five pluggable evaluator types
(`FrequencyEvaluator`, `EnergyEvaluator`, `GeometryEvaluator`,
`EigenmatrixEvaluator`, `HessianElementEvaluator`), each implementing a common
protocol. The `ForceField` dataclass is format-agnostic with typed parameter
collections, canonical units, and import/export for MM3, Tinker, OpenMM, AMBER,
and CHARMM.

---

## Published Force Field Comparison

q2mm v5 achieves lower frequency RMSD than the published FFs when both are
evaluated under our `JaxBackend`. However, this comparison has important caveats:
the published FFs were optimized for a different engine (MacroModel MM3*) and
a broader objective (geometries + full Hessian + charges + energies), while
our benchmark optimizes frequency RMSD only. On the papers' own metrics
(eigenvalue R², selectivity MUE), the published FFs perform well.

See the [QFUERZA starting-point quality](../benchmarks/qfuerza-validation.md#5-starting-point-quality-across-systems)
for honest context, then use the merged system pages for the detailed comparisons:

- [Rh-enamide](../systems/rh-enamide.md) — published eigenvalue R² ≈ 0.998 vs our 0.991 (QFUERZA)
- [Heck relay](../systems/heck-relay.md) — published selectivity RMSD = 2.3 kJ/mol (151 predictions)
- [Pd-allyl](../systems/pd-allyl.md), [Pd 1,4-conjugate](../systems/pd-conjugate.md), [Rh 1,4-conjugate](../systems/rh-conjugate.md) — literature-transfer gaps under our engine

---

## Diagnostics

q2mm/q2mm includes a [`tools/`](https://github.com/Q2MM/q2mm/tree/b26404b/tools)
directory with utility scripts (score graphing, FF formatting, structure
setup) and a test suite.

q2mm (this repo) provides a CLI diagnostics suite: benchmark matrix runner,
system registry (CH₃F, Rh-enamide), JSON result archival, leaderboard reports,
parameter comparison tables, Hessian quality checks, and PES distortion scans.
