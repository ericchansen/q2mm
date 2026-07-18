# Analytical vs Finite-Difference

This page answers one question: when is the JAX analytical-gradient path
worth its compile-time overhead, and when does single-shot scipy on
OpenMM still win? It is interpretive — pulling from the
[Small Molecules](../systems/small-molecules.md), [GPU Acceleration](gpu.md), and
[Rh-Enamide](../systems/rh-enamide.md) result tables — and is intended as a
decision aid, not a new benchmark.

## TL;DR

- **Single-shot, one molecule, frequency-only**: scipy on OpenMM is faster.
  The JAX analytical path pays a one-time JIT compile cost (~1–2 s on CPU,
  more on GPU first-touch) that a 5-atom system cannot amortize.
- **Multi-start (n ≥ 5)**: the analytical path crosses over. JIT compile
  amortizes across all starts inside one XLA kernel; per-start cost
  collapses.
- **Many parameters, organometallic-scale (Rh-enamide, 182 params)**: GPU
  helps even for single-shot, because per-evaluation cost is now
  large enough to dominate kernel-launch overhead.
- **TS curvature inversion (QFUERZA) inside JIT**: now lives inside the
  analytical path (Phase 5), so SN2/rh-enamide benchmarks no longer pay
  a Python round-trip per evaluation.

## What "analytical" means here

The analytical path = JAX-traced loss with `jax.value_and_grad`, JIT
compiled per case, and aggregated from Python by `JaxObjectiveExecutor`.
No finite-difference gradients. See
[Architecture](../how-it-works/architecture.md) for the residual-kind ×
analytical-gradient × in-JIT matrix.

The non-analytical baselines are scipy `L-BFGS-B` with finite-difference
gradients on top of OpenMM (CUDA), Tinker (CPU), or JAX/JAX-MD as a
function evaluator (no gradients propagated back).

## Crossover, in numbers

These are the rows that bracket the question. For the full matrix see
[Small Molecules](../systems/small-molecules.md).

### CH₃F (1 mol, 8 params, harmonic + MM3)

| Path | Configuration | RMSD (cm⁻¹) | Wall time |
|------|--------------|------------:|----------:|
| Scipy / OpenMM CUDA | L-BFGS-B (FD) | 59.5 | 4.8 s |
| JAX analytical | L-BFGS-B (analytical) | 579.0 | 2.2 s |
| JAX analytical | optax:adam (analytical) | 56.3 | 25.2 s |
| JAX analytical | jaxopt:lbfgsb | 579.5 | 6.8 s |
| JAX analytical | multi:L-BFGS-B (n=10) | 586.3 | 7.5 s |
| Scipy / OpenMM CUDA | multi:L-BFGS-B (n=10) | 28.7 | 157.4 s |

**Reading this table:** on CH₃F, the *cheapest* good answer is OpenMM
single-shot L-BFGS-B (4.8 s, RMSD 59.5). The analytical path matches
that RMSD only with optax:adam (25.2 s) — slower in absolute terms.
But the multi-start row crosses over: jaxopt multi(n=10) finishes in
7.5 s versus OpenMM's 157.4 s for the same n. That is the regime where
analytical wins: **OpenMM's per-start cost is fixed; JAX's per-start
cost approaches zero as n grows because all starts share one
JIT-compiled gradient kernel.**

What's missing from the table — and from the analytical pipeline today
— is the JAX `multi:L-BFGS-B (n=10)` row reaching the *same* low RMSD
as the OpenMM sequential row (28.7). Two reasons it does not:

1. The analytical gradient path with `jaxopt.LBFGSB` converges to a
   different local minimum than scipy's `L-BFGS-B` on the MM3
   landscape. Both are valid optima of the chosen loss; the basins
   they prefer differ. This is a parameterization issue, not a
   correctness gap.
2. The JAX path now uses per-molecule JIT splitting — each molecule's
   Hessian + eigenmatrix loss is compiled independently and dispatched
   from Python. This prevents compilation OOM on multi-molecule systems
   (e.g., Rh-enamide's 9 molecules across 4 topology groups).

### Rh-enamide (9 mols, 182 params, organometallic)

!!! note "Frequency-only historical data"
    The timing data below used a **frequency-only** objective with the
    old full-FF parameter scope. These numbers are valid for
    CPU-vs-GPU comparison but do not represent the paper's multi-target
    methodology. See [Rh-enamide](../systems/rh-enamide.md) for
    multi-target results.

| Path | Configuration | s / eval | Wall time |
|------|--------------|---------:|----------:|
| JAX-MD (OPLSAA) | L-BFGS-B (FD), CPU | 75.4 | 23,819 s |
| JAX-MD (OPLSAA) | L-BFGS-B (FD), GPU | 13.4 | 6,009 s (4.0× faster) |
| JAX (harmonic) | L-BFGS-B (FD), CPU | 26.2 | 550 s |
| JAX (harmonic) | L-BFGS-B (FD), GPU | 12.6 | 391 s (1.4× faster) |
| JAX MM3 grad-simp (analytical) | GPU | — | ~25 min |

Source: [GPU Acceleration](gpu.md), [Rh-Enamide](../systems/rh-enamide.md).

For the realistic case study, **GPU helps because the per-evaluation
arithmetic is finally large enough to dominate kernel launch overhead.**
The 4.0× JAX-MD speedup is the strongest extant device-level
acceleration in the project.

## When you should reach for the analytical path

- You are running a parameter sweep or multi-start with `n ≥ 5`.
- Your training set is dominated by frequency or eigenmatrix residuals
  (the analytical Hessian/Jacobian path makes these cheap; the
  finite-difference baseline pays an `O(n_param)` cost per outer step).
- You are on GPU with ≥ ~100 parameters or ≥ ~50 atoms total across the
  training set.
- You are iterating on TS systems and need QFUERZA inversion inside the
  inner loss (Phase 5; pure-JAX path).

## When OpenMM single-shot still wins

- One small molecule (≤ ~10 atoms), single-shot, frequency-only, 8–20
  parameters. The JIT compile is not amortized.
- You need the *cheapest* answer above some quality bar and you do not
  need the full Pareto frontier of optima.
- You are CPU-only and the system is small enough that vectorization
  doesn't materially help.

## What is not on the path

The "analytical path" today does not include:

- **`vmap` over molecules.** Replaced by **per-molecule JIT splitting**
  — each molecule's loss + gradient is compiled into its own small XLA
  program, dispatched from Python, and summed.  This avoids compilation
  OOM on multi-molecule systems (where the old monolithic graph exceeded
  32 GB VRAM).  The per-molecule approach trades vmap parallelism for
  compilation safety; each cached evaluation takes ~7 s for 9 molecules.
- **Geometry references (bond_length, bond_angle, torsion_angle) via
  implicit differentiation.** **Done.**
  `jaxopt.LBFGS(implicit_diff=True)` relaxes coordinates at the current
  parameters; the outer `jax.grad` gets exact `∂x*/∂p` via the implicit
  function theorem. The inner solve uses a 4,000-iteration cap and a
  `1e-5` maximum Cartesian-force tolerance. Unconverged trial points receive
  a differentiable barrier toward the known-converged baseline, and
  observable extraction raises a typed convergence error rather than
  treating a non-stationary geometry as valid.
- **Stretch-bend cross-term in OpenMM/Tinker.** The JAX engine now
  computes stretch-bend energy; OpenMM and Tinker do not yet.
  `StretchBendParam` is in `ForceField`, and the MM3 `.fld` loader
  populates it.  Wildcard atom type `00` matching is not yet supported.
- **Parameter equivalences / linked params** as JIT-traceable
  projections. Box bounds work today via `jaxopt.LBFGSB`. Anything
  more — atom-type equivalence groups, linear constraints — would
  require model-side infrastructure that does not yet exist in q2mm.

These are explicit non-goals for the current analytical pipeline; they
will be revisited when a real workflow makes them necessary.

## Where the numbers come from

- CH₃F rows: [`q2mm-data/benchmarks/ch3f/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/ch3f)
  (golden fixtures archived in the data repo).
- Rh-enamide rows: [`q2mm-data/benchmarks/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks) → `rh-enamide/`.
- Architectural background: [Architecture](../how-it-works/architecture.md).
- Theory of analytical-gradient observables:
  [Theory & Methods](../how-it-works/theory.md).
