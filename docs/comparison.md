# Q2MM v5 Improvements

This page compares two repositories:

- [**ericchansen/q2mm**](https://github.com/ericchansen/q2mm) — this
  repository, the actively developed codebase
- [**q2mm/q2mm**](https://github.com/q2mm/q2mm) — a fork of this repository

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

ericchansen/q2mm replaces the subprocess loop with in-process Python backends
([OpenMM](https://openmm.org/), [JAX](https://jax.readthedocs.io/), [JAX-MD](https://jax-md.readthedocs.io/)) where energy, gradient, and Hessian computations happen
inside the same process. [JAX backends are fully differentiable](backends/jax-engine.md)
([`jax.grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html), [`jax.hessian`](https://jax.readthedocs.io/en/latest/_autosummary/jax.hessian.html), [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html)), which
eliminates finite-difference overhead for gradient and Hessian evaluation.

The practical result on committed benchmarks
([Rh-enamide](benchmarks/rh-enamide.md),
[CH₃F](benchmarks/small-molecules.md)):

- **Rh-enamide grad-simp**: OpenMM CUDA reached 42.7 cm⁻¹ RMSD in 33,223 s
  optimizer time; JAX MM3 GPU reached the same RMSD in 1,471 s — a
  [~23× speedup](benchmarks/rh-enamide.md).
- **GPU acceleration**: JAX and OpenMM CUDA backends run on GPU without
  leaving the Python process
  ([GPU benchmarks](benchmarks/gpu.md)).

The codebase also replaces the hand-rolled solvers with [SciPy](https://scipy.org/) wrappers, adds
eigendecomposition safety checks (symmetrization, NaN/Inf detection, penalty
fallback), and introduces bound-aware sensitivity analysis. Details for each
area are linked in the table below.

---

## At a glance

| Capability | q2mm/q2mm | ericchansen/q2mm |
|---|---|---|
| Gradient solvers | 5 hand-rolled ([least-squares](https://github.com/Q2MM/q2mm/blob/b26404b/q2mm/gradient.py#L485), [Lagrange](https://github.com/Q2MM/q2mm/blob/b26404b/q2mm/gradient.py#L448), [LM](https://github.com/Q2MM/q2mm/blob/b26404b/q2mm/gradient.py#L467), [NR](https://github.com/Q2MM/q2mm/blob/b26404b/q2mm/gradient.py#L498), [SVD](https://github.com/Q2MM/q2mm/blob/b26404b/q2mm/gradient.py#L533)) | [SciPy wrapper](#optimizers) (L-BFGS-B, Nelder-Mead, Powell, trust-constr, least_squares) + [optax](#optimizers) (Adam, AdaGrad, SGD) |
| Global optimizers | None | [Basin-hopping](#optimizers) + [multi-start](#optimizers) global search |
| Regularization | None | [L2 penalty](#optimizers) to prevent parameter drift |
| Simplex | Custom Nelder-Mead (3 params) | [SciPy Nelder-Mead via subspace projection](#optimizers) |
| Cycling loop | Text command file | Dataclass-configured [`OptimizationLoop`](how-it-works/optimization-guide.md) |
| Sensitivity | Exception-based, one-sided FD fallback | [Symmetric step shrinking, bound-aware](#sensitivity-bounds) |
| Eigendecomposition | `np.linalg.eigh` | [Symmetrize + NaN/Inf check + penalty fallback](#eigendecomposition) |
| Backends | Subprocess (MacroModel, Tinker, Amber, Gaussian, Jaguar) | [API + differentiable](#backend-engines) (OpenMM, Tinker, JAX, JAX-MD, Psi4) |
| Analytical ∂E/∂θ | Not evident | Yes ([`jax.grad`](backends/jax-engine.md), OpenMM) |
| Batched evaluation | Not evident | Yes ([`jax.vmap`](backends/jax-engine.md)) |
| GPU | Subprocess; depends on backend | [In-process CUDA](benchmarks/gpu.md) (OpenMM, JAX, JAX-MD) |
| Functional forms | MM3 only (implicit) | [MM3 + Harmonic](how-it-works/architecture.md) (explicit enum) |
| Force field model | Format-coupled (`ParamMM3`) | [Format-agnostic `ForceField` dataclass](#evaluators-force-field-model) |
| Diagnostics | [Utility scripts](https://github.com/Q2MM/q2mm/tree/b26404b/tools) | [Full CLI benchmark suite](#diagnostics) |
| Test suite | [9 test files](https://github.com/Q2MM/q2mm/tree/b26404b/test) | 1,100+ unit tests |

---

## Optimizers

q2mm/q2mm implements five gradient solvers from scratch (least-squares,
Lagrange, Levenberg-Marquardt, Newton-Raphson, SVD) plus a custom
Nelder-Mead simplex. A text command file drives the `GRAD` → `SIMP`
cycling loop.

ericchansen/q2mm delegates all optimization to SciPy (`scipy.optimize.minimize`
and `scipy.optimize.least_squares`), wrapping them in a `ScipyOptimizer` class.
The cycling loop is a dataclass-configured `OptimizationLoop` that:

1. Runs a full-space SciPy pass on all parameters
2. Ranks parameters by simplex suitability (lowest `simp_var`)
3. Runs Nelder-Mead on the least gradient-suitable subset
4. Repeats until convergence

Delegating to SciPy means the numerics are maintained by a large community and
benefit from ongoing improvements.

For JAX-based workflows, ericchansen/q2mm also provides an `OptaxOptimizer`
that wraps [optax](https://optax.readthedocs.io/) adaptive optimizers (Adam,
AdaGrad, SGD, AdamW). These use JAX's analytical gradients directly and excel
on rugged potential energy surfaces like MM3, where Adam achieves 10× better
RMSD than L-BFGS-B on CH₃F
([benchmark results](benchmarks/small-molecules.md)). See the
[Optimization Guide](how-it-works/optimization-guide.md#workflow-b-small-rugged)
for details.

For global optimization, `BasinHoppingOptimizer` wraps
[`scipy.optimize.basinhopping`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html)
with bounded perturbation steps, and `MultiStartOptimizer` runs any inner
optimizer from N perturbed starting points.  `ObjectiveFunction` also supports
L2 regularization (`regularization` kwarg) to penalize parameter drift from
QFUERZA starting values — see
[Optimization Guide](how-it-works/optimization-guide.md#l2-regularization).

---

## Sensitivity & bounds

q2mm/q2mm catches out-of-bounds perturbations via exceptions and falls back to
one-sided finite differences — this gives O(h) truncation error instead of
O(h²), producing biased sensitivity estimates near bounds.

ericchansen/q2mm shrinks the step size *before* evaluation to keep both
perturbations within bounds, preserving O(h²) central-difference accuracy.
Parameters at their bounds receive infinite sensitivity scores, ranking them
last and effectively excluding them from the simplex selection.

For the math, see [Theory & Methods — Optimization](how-it-works/theory.md#stage-4-optimization).

---

## Eigendecomposition

ericchansen/q2mm adds Hessian symmetrization, NaN/Inf detection, and penalty
fallback to the eigendecomposition pipeline.

For details on Limé & Norrby's eigenvalue methods (A–E), see
[Theory & Methods — TS Eigenvalue Treatment](how-it-works/theory.md#stage-2-transition-state-eigenvalue-treatment).

---

## Backend engines

q2mm/q2mm routes all computation through subprocess calls — write input files,
run shell commands, parse text output. Supported backends: MacroModel, Tinker,
Amber, Gaussian, Jaguar.

ericchansen/q2mm uses in-process Python APIs with a registry pattern:

| Engine | Interface | Differentiable | GPU |
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

ericchansen/q2mm splits evaluation into five pluggable evaluator types
(`FrequencyEvaluator`, `EnergyEvaluator`, `GeometryEvaluator`,
`EigenmatrixEvaluator`, `HessianElementEvaluator`), each implementing a common
protocol. The `ForceField` dataclass is format-agnostic with typed parameter
collections, canonical units, and import/export for MM3, Tinker, OpenMM, AMBER,
and CHARMM.

---

## Diagnostics

q2mm/q2mm includes a [`tools/`](https://github.com/Q2MM/q2mm/tree/b26404b/tools)
directory with utility scripts (score graphing, FF formatting, structure
setup) and a test suite.

ericchansen/q2mm provides a CLI diagnostics suite: benchmark matrix runner,
system registry (CH₃F, Rh-enamide), JSON result archival, leaderboard reports,
parameter comparison tables, Hessian quality checks, and PES distortion scans.
