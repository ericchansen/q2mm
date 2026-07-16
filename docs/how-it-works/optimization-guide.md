# Optimization Guide

Force field optimization is hard because the objective landscape — the surface
defined by how well MM frequencies match QM reference data — is rarely smooth.
**MM3** landscapes are rugged with many local minima (Buckingham potentials,
coupled torsions). **Harmonic** landscapes are smoother but still
non-trivial from a poor starting point. No single optimizer handles both
cases well.

This guide tells you **what to do** for your system. It recommends concrete
workflows grounded in [CH₃F benchmark evidence](../systems/small-molecules.md),
then links to source code for API details.

---

## Quick-start: which workflow?

| Your system | Recommended workflow | Why |
|-------------|---------------------|-----|
| ≤ 10 params, harmonic form | [Workflow A](#workflow-a-small-smooth) | L-BFGS-B alone reaches the best basin (529 cm⁻¹ RMSD on CH₃F) |
| ≤ 10 params, MM3 form | [Workflow B](#workflow-b-small-rugged) | Multi-start finds basins single-start misses (28.7 vs 579 RMSD) |
| 10–50 params, any form | [Workflow C](#workflow-c-medium-and-large-systems) | Grad-simp cycling combines gradient speed with simplex robustness |
| 50+ params | [Workflow C](#workflow-c-medium-and-large-systems) + [L2](#l2-regularization) | Regularization prevents parameter drift in under-determined systems |
| JAX backend, multi-molecule TS systems | [Workflow D](#workflow-d-end-to-end-differentiable-jax) | Per-case JAX analytical gradients via scipy L-BFGS-B |

Every workflow assumes **QFUERZA initialization** — run
`qfuerza_fresh()` (single molecule) or `qfuerza_into()` (template-based,
multi-molecule averaging) before optimization. QFUERZA puts you in the
right neighbourhood; the optimizer refines from there.

The snippets below assume you have compiled an `ObjectivePlan` and selected
an executor explicitly:

- `PythonObjectiveExecutor(plan, backend, problem.starting_force_field)` for
  ordinary Python dispatch. Its default `GradientMode.NONE` lets SciPy use its
  internal finite differences.
- `JaxObjectiveExecutor(plan, backend, problem.starting_force_field)` for the
  analytical JAX path. This is what the CLI key `scipy-lbfgsb-jax` selects.

---

## Workflow A: Small + Smooth

**When:** ≤ 10 parameters, harmonic functional form, analytical gradients
available.

**Recipe:** One call to L-BFGS-B with analytical gradients. Done.

```python
from q2mm.objectives.jax import JaxObjectiveExecutor
from q2mm.objectives.plan import ObjectivePlan
from q2mm.optimizers.scipy_opt import ScipyOptimizer

plan = ObjectivePlan.from_problem(problem)
evaluator = JaxObjectiveExecutor(plan, backend, problem.starting_force_field)

optimizer = ScipyOptimizer(method="L-BFGS-B", maxiter=500)
result = optimizer.optimize(evaluator, plan.active_space)
print(result.summary())
```

**Why this works:** On smooth harmonic landscapes, L-BFGS-B's curvature
information reaches the best basin in the fewest evaluations. On CH₃F
harmonic, L-BFGS-B with analytical gradients achieves 529 cm⁻¹ RMSD in
~2 seconds. Basin-hopping, multi-start, and optax Adam all match or
slightly beat this (~526–531), but none find a meaningfully better basin —
the harmonic landscape simply doesn't have hidden minima worth searching
for.

**What doesn't work here:**

- **Optax Adam** (990–1001 RMSD) — adaptive learning rates can't exploit
  curvature the way L-BFGS-B can on a smooth surface
- **L2 regularization** — can hurt well-conditioned problems (see [L2 Regularization](#l2-regularization)).
- **FD-only gradients** (1048 RMSD) — finite-difference frequency gradients
  are too noisy to guide L-BFGS-B from the QFUERZA starting point

---

## Workflow B: Small + Rugged

**When:** ≤ 10 parameters, MM3 or other non-harmonic functional form.

**Recipe:** Multi-start L-BFGS-B to find the best basin.

```python
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.python import PythonObjectiveExecutor
from q2mm.optimizers.multistart import MultiStartOptimizer
from q2mm.optimizers.scipy_opt import ScipyOptimizer

plan = ObjectivePlan.from_problem(problem)
evaluator = PythonObjectiveExecutor(plan, backend, problem.starting_force_field)

# Multi-start global search
inner = ScipyOptimizer(method="L-BFGS-B", maxiter=500)
multi = MultiStartOptimizer(
    optimizer=inner,
    n_starts=10,
    perturbation_pct=0.1,
    seed=42,
)
result = multi.optimize(evaluator, plan.active_space)
```

**Why this works:** The MM3 landscape has many local minima. Single-start
L-BFGS-B gets trapped at 579 cm⁻¹ RMSD on CH₃F — a poor basin. Running
10 starts from perturbed initial points finds basins that single-start
misses:

| Strategy | CH₃F MM3 RMSD | Notes |
|----------|-------------:|-------|
| L-BFGS-B (single start) | 579.0 | Stuck in poor local minimum |
| L-BFGS-B + L2(λ=0.01) | 133.5 | Stabilized with [L2 regularization](#l2-regularization) |
| optax Adam | 56.3 | Momentum navigates rugged landscape |
| **multi-start n=10 (OpenMM)** | **28.7–46.2** | **Best strategy — stochastic, varies by run** |

Multi-start is the recommended approach because it's simple and consistently
finds the best basins on rugged landscapes.

**Does Adam refinement help after multi-start?** In
[composed benchmarks](../systems/small-molecules.md#composed-workflows),
running optax Adam from the multi-start winner barely changed the result:
46.2 → 46.1 on OpenMM (FD gradients limit Adam), 563.8 → 563.8 on JAX
(same local minimum). Multi-start alone finds the basin; refinement adds
diminishing returns.

**Alternatives that also work:**

- **[Basin-hopping](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html)**
  with T=0.5 improved on single-start L-BFGS-B (514 vs 579) but didn't
  match multi-start n=10 (28.7). Basin-hopping is more sensitive to
  temperature tuning — T=1.0 produced the *worst* result (1105 RMSD) by
  accepting too many uphill moves.
- **L2 regularization** can stabilize rugged single-start fits when you cannot afford multi-start (see [L2 Regularization](#l2-regularization)).

**What doesn't work here:**

- **Multi-start on JAX** (579–586 RMSD) — JAX's analytical gradients
  consistently converge to the same local minimum regardless of starting
  point. The OpenMM FD gradient noise actually helps escape this basin.
  This is a case where FD noise acts as implicit regularization.

---

## Workflow C: Medium and Large Systems

**When:** 10+ parameters, any functional form. This is the
production workflow.

**Recipe:** Grad-simp cycling — alternating L-BFGS-B full-space passes
with Nelder-Mead on the most sensitive parameters.

```python
from q2mm.optimizers.cycling import OptimizationLoop

loop = OptimizationLoop(
    evaluator,
    plan.active_space,    # ActiveParameterSpace over plan.layout
    max_params=3,         # simplex on top 3 params per cycle
    max_cycles=10,        # up to 10 grad-simp cycles
    convergence=0.01,     # stop when <1% improvement per cycle
    full_method="L-BFGS-B",
    simp_method="Nelder-Mead",
    full_maxiter=200,
    simp_maxiter=200,
    verbose=True,
)
result = loop.run()
print(result.summary())
```

**How it works:** Grad-simp cycling alternates between gradient-based full-space optimization and Nelder-Mead simplex refinement of the most sensitive parameters. See [Theory & Methods — Stage 4](theory.md#stage-4-optimization) for the full mechanics. In practice, this keeps fast global progress from L-BFGS-B while simplex focuses on the few parameters gradients handle poorly.


**Why this works:** L-BFGS-B quickly converges most parameters, then
Nelder-Mead polishes the stubborn ones that gradients can't handle.
On Rh-enamide (182 parameters, MM3), grad-simp achieves the best
observed score (3.29 with OpenMM) — better than either L-BFGS-B (5.81) or
Nelder-Mead (5.11) alone.

!!! info "Why only 3 parameters per simplex pass?"
    Nelder-Mead creates an (N+1)-vertex simplex. With 3 parameters that's
    4 vertices; with 20 it's 21 — and convergence slows significantly.
    `max_params=3` keeps simplex passes fast while addressing the most
    problematic parameters each cycle.

### Composing with global search

You can use multi-start as the gradient phase inside grad-simp cycling
via the `full_method="multi:L-BFGS-B"` parameter:

```python
from q2mm.optimizers.cycling import OptimizationLoop

loop = OptimizationLoop(
    evaluator,
    plan.active_space,
    max_params=3,
    max_cycles=5,
    full_method="multi:L-BFGS-B",  # multi-start each gradient phase
    simp_method="Nelder-Mead",
)
result = loop.run()
```

!!! warning "Evidence: this composition doesn't outperform plain cycling"
    In [CH₃F benchmarks](../systems/small-molecules.md#composed-workflows),
    grad-simp with multi-start inner achieved **592 RMSD on OpenMM** and
    **527 RMSD on JAX** — comparable to or worse than plain grad-simp
    (586 / 579). The random restarts within each cycle disrupt inter-cycle
    convergence. For rugged landscapes, multi-start alone
    ([Workflow B](#workflow-b-small-rugged)) is more effective than
    embedding multi-start inside cycling.

### Adding L2 regularization to cycling

For under-determined systems (more parameters than independent
observations), add L2 to the objective:

```python
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.python import PythonObjectiveExecutor

plan = ObjectivePlan.from_problem(
    problem,
    regularization=0.01,  # keeps params near QFUERZA values
)
evaluator = PythonObjectiveExecutor(plan, backend, problem.starting_force_field)

loop = OptimizationLoop(evaluator, plan.active_space, max_params=3, max_cycles=10)
result = loop.run()
```

L2 regularization can stabilize under-determined cycling runs (see [L2 Regularization](#l2-regularization)).

---

## Workflow D: End-to-End Differentiable (JAX)

**When:** JAX backend with multi-molecule TS systems, eigenmatrix or
geometry references — you want analytical gradients without
finite-difference overhead.

**Recipe:** Build a `JaxObjectiveExecutor` and pass it to `ScipyOptimizer`.
Gradient selection is explicit: the optimizer uses the evaluator's declared
`GradientMode.ANALYTICAL`; there is no auto-detection or silent
finite-difference fallback.

```python
from q2mm.objectives.jax import JaxObjectiveExecutor
from q2mm.objectives.plan import ObjectivePlan
from q2mm.optimizers.scipy_opt import ScipyOptimizer

plan = ObjectivePlan.from_problem(problem)
evaluator = JaxObjectiveExecutor(plan, backend, problem.starting_force_field)

optimizer = ScipyOptimizer(method="L-BFGS-B", maxiter=200)
result = optimizer.optimize(evaluator, plan.active_space)
print(result.summary())
```

**How it works:** `JaxObjectiveExecutor` compiles one JIT loss+gradient
fragment per training case. Each case's Hessian, eigenmatrix, geometry, and
energy terms are compiled into a small XLA program. SciPy calls the executor
from Python at each iteration (treating it as a black-box function returning
`(loss, grad)`), so no single XLA program needs to contain all molecules.

**Supported reference types:**

| Reference type | Supported | Notes |
|----------------|:---------:|-------|
| Energy | ✅ | Weighted residuals |
| Frequency | ✅ | Closed-form sensitivity via eigenvalue derivatives |
| Hessian elements | ✅ | Raw Cartesian Hessian entries |
| Eigenmatrix (Q^T H Q) | ✅ | Diagonal and off-diagonal projection terms |
| Geometry | ✅ | Bond lengths, angles, torsions via `jaxopt.LBFGS(implicit_diff=True)` inner minimization |

**When to use this vs SciPy FD:**

- `JaxObjectiveExecutor` provides analytical gradients for all reference types
  including geometry — no finite-difference overhead.
- First evaluation is slow (~5 min) due to per-molecule JIT
  compilation. Subsequent evaluations are fast (~7 s for 9 molecules).
- SciPy L-BFGS-B with FD gradients works with any backend but is
  O(n_params) evaluations per step.

**Benchmark results (CH₃F, see [Small Molecules](../systems/small-molecules.md)):**

| Form | Device | Optimizer | RMSD (cm⁻¹) | Time | eval/s |
|------|--------|-----------|:-----------:|-----:|-------:|
| harmonic | CPU | jaxopt:lbfgsb | 528.3 | 4.8s | 45.4 |
| harmonic | GPU | jaxopt:lbfgs | 532.0 | 16.3s | 9.9 |
| harmonic | GPU | SciPy L-BFGS-B (A) | 528.7 | 1.9s | 41.1 |
| mm3 | GPU | jaxopt:lbfgs | 578.7 | 16.2s | 18.3 |
| mm3 | GPU | SciPy L-BFGS-B (A) | 579.0 | 2.2s | 31.4 |

!!! note "CH₃F is a single-molecule system"
    On single-molecule systems, `JaxOptOptimizer` with monolithic JIT
    compilation still works well (no OOM risk). For multi-molecule TS
    systems, use `ScipyOptimizer` with a `JaxObjectiveExecutor`, which
    provides per-case JIT dispatch without one monolithic XLA graph.

---

## Modifiers

These cross-cutting options layer onto any workflow.

### L2 Regularization

Penalizes parameter drift from the starting values (QFUERZA estimates).
The total loss becomes:

$$\text{loss}_\text{total} = \text{loss}_\text{data} + \lambda \cdot \| \mathbf{p} - \mathbf{p}_\text{ref} \|^2$$

```python
from q2mm.objectives.plan import ObjectivePlan

plan = ObjectivePlan.from_problem(
    problem,
    regularization=0.01,  # λ — penalty strength
    # reference_params=...  # defaults to the active-space baseline
)
```

**When it helps:** Rugged MM3 landscapes where single-start optimizers
find poor local minima. On CH₃F MM3, L2 improved L-BFGS-B by 4×
(579 → 134 RMSD). Also useful for under-determined systems (more
parameters than observations).

**When it hurts:** Well-conditioned problems. On CH₃F harmonic, L2
*doubled* the RMSD (529 → 993) by preventing parameters from reaching
the optimal basin.

!!! tip "Choosing λ"
    Start with 0.001–0.01. Increase until parameters stay close to
    QFUERZA values. Aim for the penalty term to be ~1–10% of data loss
    at the optimum. Too large = can't improve; too small = no effect.

L2 works with **every** optimizer — [SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.html),
[optax](https://optax.readthedocs.io/), basin-hopping, multi-start, and
grad-simp — because it modifies the
[`ObjectivePlan`](https://github.com/ericchansen/q2mm/blob/master/q2mm/objectives/),
not the optimizer.

### Sensitivity Analysis

A diagnostic tool that ranks parameters by how the objective responds to
perturbation, using the ratio `simp_var = d2/d1²`. Low `simp_var` means
the parameter strongly affects the objective relative to its curvature —
these are parameters where simplex outperforms gradient methods.

```python
from q2mm.optimizers.cycling import compute_sensitivity

sens = compute_sensitivity(evaluator, plan.active_space.baseline, metric="simp_var")
labels = [kind.value for kind in plan.layout.kinds]
for rank, idx in enumerate(sens.ranking):
    print(f"  {rank+1}. {labels[idx]:12s}  "
          f"d1={sens.d1[idx]:+.4f}  simp_var={sens.simp_var[idx]:.4f}")
```

Use this to understand your landscape before choosing a workflow, or to
debug why optimization stalls.

!!! note "Cost"
    Sensitivity analysis requires **2N + 1** objective evaluations in the
    worst case (one baseline plus two perturbations per parameter).

---

## Optimizer Reference

For constructor parameters, return types, and full API details, see the
[API Reference](../reference/q2mm/optimizers/index.md).

### Gradient modes

Gradient behavior is declared by the evaluator, not probed by the optimizer:

| Evaluator | Gradient mode | What it does | When to use |
|-----------|---------------|-------------|-------------|
| `PythonObjectiveExecutor(...)` | `GradientMode.NONE` | Scalar values only; SciPy supplies internal finite differences | Default Python path; works with every backend |
| `PythonObjectiveExecutor(..., gradient_mode=GradientMode.FINITE_DIFFERENCE)` | `FINITE_DIFFERENCE` | Executor-owned central finite differences | When you need FD evaluations counted by the executor |
| `PythonObjectiveExecutor(..., gradient_mode=GradientMode.ANALYTICAL)` | `ANALYTICAL` | Backend analytical derivatives for supported categories | Supported energy/Hessian-like categories; raises if unsupported |
| `JaxObjectiveExecutor(...)` | `ANALYTICAL` | Per-case JIT + `jax.value_and_grad` | Recommended for JAX multi-case TS workflows |

There is no automatic `jac` mode and no silent finite-difference fallback. If
an analytical executor cannot support a requested category, it raises
`ObjectiveGradientError` so you can choose a different executor deliberately.

---

## Tips and Pitfalls

!!! warning "L-BFGS-B may not fully converge on rugged landscapes"
    On CH₃F MM3, L-BFGS-B gets trapped at 579 cm⁻¹ RMSD — a poor local
    minimum. Use multi-start or optax Adam on MM3 forms. On smooth
    harmonic forms, L-BFGS-B is the best choice.

!!! tip "QFUERZA initialization matters"
    Starting from QFUERZA-estimated parameters puts you close to the
    optimum. Run `qfuerza_fresh()` or `qfuerza_into()` before
    optimization when QM data is available.

!!! tip "Monitor convergence"
    Plot `result.history` (single-shot) or `result.cycle_scores` (cycling)
    to visualize convergence. If the score plateaus early, the optimizer
    may be stuck — try multi-start or switch the sensitivity metric to
    `"abs_d1"`.

!!! info "Backend speed"
    Per-evaluation cost on CH₃F (8 parameters), from derivative-free
    methods:

    | Backend | Per-eval | vs Tinker |
    |---------|---------|-----------|
    | JAX (GPU) | ~2.5 ms | 96× faster |
    | OpenMM (GPU) | ~10 ms | 24× faster |
    | Tinker (CPU) | ~240 ms | baseline |

!!! info "FD noise as implicit regularization"
    On CH₃F MM3, OpenMM multi-start n=10 (FD gradients) achieved 28.7
    RMSD while JAX multi-start n=10 (analytical gradients) stayed at 586.
    The FD gradient noise helped OpenMM escape the local minimum that
    JAX's precise gradients consistently converge to. This is an unusual
    case where lower-quality gradients produced a better result.

---

## GPU Optimizer Recommendations

For multi-molecule TS systems with the multi-target objective
(eigenmatrix + geometry, frozen base-FF params):

| Optimizer | Use case | Notes |
|-----------|----------|-------|
| **ScipyOptimizer L-BFGS-B + `JaxObjectiveExecutor`** | Default for TS systems | Explicit per-case JAX analytical gradients. ~8 s/eval on 9-molecule Rh-enamide (GPU). |
| JaxOptOptimizer L-BFGS | Single-molecule systems | Monolithic JIT is fast for small systems; pathologically slow on multi-molecule due to internal line search compilation. |
| OptaxOptimizer Adam | Exploration | First-order; useful for rugged landscapes where L-BFGS-B gets stuck. |

!!! tip "Start with ScipyOptimizer"
    Use `ScipyOptimizer(method="L-BFGS-B")` with a
    `JaxObjectiveExecutor` as the default for JAX-backend workflows. The
    executor builds per-case JIT fragments and provides analytical gradients
    to SciPy's battle-tested L-BFGS-B implementation.

See [Optimizer Comparison](../benchmarks/optimizer-comparison.md) for
detailed results and timing data.

---

## Further Reading

- [Tutorial: Step 6 — Optimize](../tutorial.md#step-6-optimise-the-force-field) — walkthrough of a single-shot optimization
- [CH₃F Benchmarks](../systems/small-molecules.md) — full 75-combo comparison matrix with RMSD, timing, and per-eval costs
- [Rh-Enamide Benchmarks](../systems/rh-enamide.md) — large-system case study (182 parameters)
- [References](../references.md) — academic papers describing the Q2MM methodology
