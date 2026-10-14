# Q2MM

Q2MM fits molecular mechanics force fields to quantum-mechanical reference
data. A force field is an inexpensive energy model whose parameters describe
bonds, angles, and other interactions. Use Q2MM when standard parameters do not
cover your chemistry and you need to control which parameters are fitted and
which stay frozen.

## Start without your own QM calculation

The [first-run guide](getting-started.md#first-success-on-cpu) uses a small
CH3F molecule with an approved, precomputed geometry and Hessian distributed
inside q2mm. A Hessian describes the energy curvature as atoms move.
There is no publication archive to recover and no GPU or QM installation to
configure.

From the documented source installation, with `JAX_PLATFORMS=cpu` set:

```powershell
python examples\ch3f\run.py --optimizer.maxiter 2 --output-root results\ch3f
```

This runs genuine JAX molecular mechanics on CPU and a budget-limited SciPy
optimizer, then explicitly saves `ch3f.frcmod` and its provenance manifest in
your chosen directory. **It is a first execution success, not a convergence
or force-field quality claim.**

The separate `--bounded-ci` mode is only a software-path proof: zero energy,
an echoed reference Hessian/geometry, and no parameter update. Do not use it as
native-backend or scientific validation.

!!! warning "Use this source revision or its local wheel"
    This breaking alpha API/CLI is unreleased. An arbitrary PyPI version may
    not contain it. Examples are source-only; approved small-system inputs are
    installed package data. Use Python **3.15 final** and the
    [installation instructions](getting-started.md).

## Then choose your fitting problem

The [tutorial](tutorial.md) progresses from the executable example to your own
inputs: one FCHK, XYZ plus an explicit Hessian, or several structures sharing
a complete force field and an OPT/custom parameter subset. Stationary-point
kind (minimum or transition state), force-field form, and runtime components
are explicit choices, never inferred from filenames.

There is one public lifecycle: **prepare → evaluate → optimize → assemble →
save**. Settings construction loads no engines; `prepare` starts no hidden QM
job. `optimize` returns a numerical result, while
`OptimizationRun.from_result` builds the final force field and provenance.
Read the [canonical lifecycle](how-it-works/optimization-guide.md#canonical-application-lifecycle)
for the full API and breaking migration notes.

## Continue

[Getting started](getting-started.md) covers installation, outputs, and a local
wheel. [The tutorial](tutorial.md) explains inputs and settings in Python.
[Publication coverage](benchmarks/published-ff-validation.md) gives the current
source-linked availability and blockers; it does not claim exact reproduction.
[Theory and methods](how-it-works/theory.md) explains Hessian-informed
initialization and transition-state handling.
