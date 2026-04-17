# Geometry references via implicit differentiation — spike report

## What this page is

Geometry reference kinds (`bond_length`, `bond_angle`, `torsion_angle`)
are the last ❌ in the [differentiability matrix][matrix]. Their gradient
is non-trivial because the observed bond length, angle, or dihedral is
measured on a *relaxed* geometry `x*(p) = argmin_x E(x; p)` — the output
of an inner optimization. Computing `∂L/∂p` therefore requires `∂x*/∂p`,
which is not a plain autodiff problem.

This page documents the short exploration ("spike") that chose an
approach before writing the full Phase 4b implementation. The script
that produced the numbers in the tables below lives at
[`scripts/spike_geom_implicit_diff.py`][script].

## The problem, stated explicitly

For a geometry reference kind, the pipeline is:

1. Fix the force-field parameters `p`.
2. Run an inner minimizer to find `x*(p) = argmin_x E(x; p)`.
3. Compute the observable on that relaxed geometry,
   e.g. `b(x*) = ||x_2* − x_1*||`.
4. Compare to the reference: `L(p) = (b(x*(p)) − b_ref)²`.

Computing `∂L/∂p` by simply autodiffing through the inner minimizer
would mean differentiating through every step of LBFGS — memory- and
compile-time-expensive, and brittle.

The standard trick is the **implicit function theorem** applied to the
stationarity condition `∇_x E(x*; p) = 0`:

```
∂x*/∂p = −[∂²E/∂x²(x*)]⁻¹ · [∂²E/∂x∂p(x*)]
```

This lets autodiff skip the solver's internals entirely and compute the
gradient from two second-derivative objects evaluated once at the
converged `x*`.

## Two realistic ways to implement it

### Option A — `jaxopt.LBFGS(..., implicit_diff=True)`

```python
from jaxopt import LBFGS

solver = LBFGS(fun=energy, tol=1e-10, maxiter=200, implicit_diff=True)

def relax(p):
    x0 = initial_guess(p)
    return solver.run(x0, p).params
```

`jaxopt` installs a `custom_vjp` behind the scenes that implements the
implicit-function formula. `relax` is then a regular JAX function:
JIT-compilable, `vmap`-able, composable with `jax.value_and_grad`.

### Option B — hand-rolled `jax.custom_vjp`

```python
@jax.custom_vjp
def relax(p):
    x0 = initial_guess(p)
    return run_inner_minimizer(energy, x0, p).x   # any minimizer

def relax_fwd(p): x = relax(p); return x, (x, p)

def relax_bwd(saved, g):
    x, p = saved
    H = jax.hessian(energy, argnums=0)(x, p)                 # ∂²E/∂x²
    M = jax.jacfwd(jax.grad(energy, argnums=0), argnums=1)(x, p)  # ∂²E/∂x∂p
    lam = jnp.linalg.solve(H.T, g)
    return (-(lam @ M),)

relax.defvjp(relax_fwd, relax_bwd)
```

The math is identical — just more surface area to own.

## Empirical comparison

Run on a 1D triatomic A–B–C system with two harmonic bonds and an
external force on atom C. The relaxed geometry has a closed form:
`b_12* = r1 + F/k1`, `b_23* = r2 + F/k2`, so we can compute the exact
loss gradient by hand and measure errors against it. Parameters for the
test: `p = (k1=2, r1=1.5, k2=3, r2=1.2)`, `refs = (1.6, 1.3)`,
`F = 0.1`. All runs on CPU in float64.

### Gradient accuracy vs inner solver tolerance

| inner tol | method     |  max `|err|` | rel err   |
| :-------: | :--------- | :----------: | :-------: |
| `1e-3`    | A (jaxopt) | `9.7e-17`    | `3.4e-15` |
| `1e-3`    | B (custom) | `4.7e-16`    | `3.5e-15` |
| `1e-12`   | A (jaxopt) | `9.7e-17`    | `3.4e-15` |
| `1e-12`   | B (custom) | `4.7e-16`    | `3.5e-15` |
| `1e-3`    | FD (`ε=1e-3`)  | `8.2e-11`  | `5.6e-8`  |
| `1e-3`    | FD (`ε=1e-5`)  | `1.8e-12`  | `1.2e-9`  |

Both implicit-diff options hit machine precision. Inner tolerance has
no effect on outer gradient accuracy for this well-conditioned convex
system — because the implicit-function formula only needs
`∇_x E(x*) ≈ 0`, which LBFGS satisfies quickly. Finite differences, by
contrast, inherit truncation + cancellation error and are at best
~9 orders of magnitude worse.

### Wall time

| method     | inner tol | ms / gradient (CPU, float64) |
| :--------- | :-------: | :--------------------------: |
| A (jaxopt) | `1e-6`    | 0.019                        |
| B (custom) | `1e-6`    | 0.033                        |
| A (jaxopt) | `1e-10`   | 0.033                        |
| B (custom) | `1e-10`   | 0.029                        |

Both are well under 0.1 ms per gradient for the 4-parameter toy
system, and the differences are within run-to-run jitter. Neither is
the bottleneck in any realistic q2mm objective.

### Hessian ill-conditioning

As `k1 → 0`, the Hessian `∂²E/∂x²` becomes nearly singular. Both
approaches depend on the same `H⁻¹ g` solve, so both degrade together.

| `k1`    | `cond(H)` | A err    | B err    |
| :------ | :-------: | :------: | :------: |
| `2e+0`  | `8.6`     | `9.7e-17`| `4.7e-16`|
| `1e-1`  | `1.2e+2`  | `9.3e-13`| `1.5e-14`|
| `1e-3`  | `1.2e+4`  | `1.5e-5` | `8.5e-6` |
| `1e-5`  | `1.2e+6`  | `2.0e+13`| `2.0e+13`|

Neither option has an advantage — ill-conditioning is a mathematical
property of the test system, not an implementation choice.

## Decision: Option A

- **Already a dependency.** `jaxopt` is used by `JaxOptOptimizer` and
  `JaxMultiStartOptimizer`; adopting it for the inner loop adds zero
  new dependencies.
- **Less code to own.** The custom VJP in Option B is ~20 lines that do
  the same thing `jaxopt`'s `implicit_diff` does.
- **Same accuracy.** At machine precision, both match the closed-form
  exactly.
- **Same performance.** Both are well under 0.1 ms per gradient on the
  toy system, within run-to-run jitter of each other.
- **Composable.** `jax.jit(jax.vmap(jax.grad(loss)))` works out of the
  box, as it already does for the existing `JaxOptOptimizer`.

## Implications for Phase 4b

1. `q2mm/optimizers/jaxloss.py` — add a `geometry` category that calls
   an inner `jaxopt.LBFGS(..., implicit_diff=True)` over the energy
   function (`energy_fn(p, coords)` is already available on the
   backend handle).
2. Drop the `to_jax_spec()` guard that raises on geometry-only
   objectives; wire `has_geometry_refs` to return `True`.
3. Use `jax.jacrev` to differentiate the observable (bond length,
   angle, dihedral) with respect to the relaxed coordinates —
   straightforward JAX, no new math.

## Watch-outs carried forward to Phase 4b

- **Non-convergence.** The spike used a convex quadratic where LBFGS
  always converges. Real MM energies are anharmonic and can have flat
  or multi-basin landscapes; an inner solver can time out or get stuck.
  Phase 4b should surface an `ok` flag alongside `x*` and fall back to
  the initial coordinates + a penalty residual when the inner solver
  fails to converge.
- **Ill-conditioning.** When `H` is near-singular, the implicit-diff
  Jacobian blows up exactly as the table above shows. A small Tikhonov
  regularization `H ← H + ε·I` (with `ε` chosen by residual heuristics)
  keeps the gradient finite at the cost of a small bias. Decide during
  Phase 4b whether to surface this as a knob or apply it silently for
  `cond(H) > 10⁴`.
- **Jacobian of observables.** Bond length and angle use `arccos`, which
  has infinite derivative at `±1`. Collinear three-atom geometries will
  produce NaN gradients. Clip the cosine input to `[-1 + 1e-12,
  1 - 1e-12]` before `arccos`.

## Reproducing the numbers

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/spike_geom_implicit_diff.py
```

Runs in under a minute on any developer laptop and prints all three
tables above.

[matrix]: ./architecture.md#differentiability-status
[script]: https://github.com/ericchansen/q2mm/blob/master/scripts/spike_geom_implicit_diff.py
