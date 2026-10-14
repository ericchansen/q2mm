# Q2MM

**Quantum-guided molecular mechanics force-field optimization.**

[![CI](https://github.com/ericchansen/q2mm/actions/workflows/ci.yml/badge.svg)](https://github.com/ericchansen/q2mm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/q2mm)](https://pypi.org/project/q2mm/)
[![Python](https://img.shields.io/pypi/pyversions/q2mm)](https://pypi.org/project/q2mm/)

Q2MM turns quantum-mechanical structures and Hessians into a molecular
mechanics fitting problem. It supports fresh force fields and
literature-template workflows while keeping the scientific choices—stationary
point, functional form, backend, optimizer, and bounds—explicit.

**[Documentation](https://ericchansen.github.io/q2mm/)** ·
**[Bring-your-own tutorial](https://ericchansen.github.io/q2mm/tutorial/)** ·
**[Publication coverage](validation/published_ffs/README.md)**

## Install

```bash
pip install q2mm                 # models, preparation, I/O, and application API
pip install "q2mm[jax]"          # JAX MM backend + optional SciPy optimizer
```

Q2MM is an alpha release. Add `--pre` if no stable PyPI release is available.
SciPy is optional; core installation does not silently choose or install an
optimizer.

Python **3.15** is the primary runtime; packaging permits **>=3.14** for a
**BEST-EFFORT native/CPU fallback**. OpenMM 8.6.1 has no cp315 wheels and is
never installed by the `dev` extra. See
[runtime installation notes](https://ericchansen.github.io/q2mm/getting-started/#native-dependency-availability)
for separate native environments and PyYAML's compiler-free source build.

## Bring your own calculation

This complete path reads a Gaussian formatted checkpoint containing geometry
and Cartesian force constants, prepares a fresh harmonic transition-state
problem, evaluates it with JAX, runs the documented JAX/SciPy workflow, and
saves an AMBER force-field file plus provenance manifest.

```python
from pathlib import Path

import q2mm
from q2mm.backends.registry import load_backend
from q2mm.io import load_fchk_molecule
from q2mm.objectives.plan import ObjectivePlan
from q2mm.settings import BackendSettings, PreparationSettings, TransitionStateOptimizationSettings

molecule = load_fchk_molecule(
    Path("my-transition-state.fchk"),
    bond_tolerance=1.4,
)

problem = q2mm.prepare(
    molecule,
    settings=PreparationSettings(stationary_point="transition_state", functional_form="harmonic"),
)
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
settings = TransitionStateOptimizationSettings(backend=BackendSettings(key="jax"))
backend = load_backend(settings.backend.key, **settings.backend.options)
optimizer = q2mm.build_optimizer(settings.optimizer)
workflow = q2mm.build_workflow(settings.workflow)
configuration = q2mm.capture_execution_configuration(
    settings, backend=backend, optimizer=optimizer, workflow=workflow
)
make_evaluator = q2mm.make_evaluator_factory(
    backend, problem.starting_force_field, settings=settings.objective
)
plan = ObjectivePlan.from_problem(problem, regularization=settings.objective.regularization)
baseline = q2mm.evaluate(make_evaluator(plan), problem.active_space.baseline)
result = q2mm.optimize(
    problem,
    optimizer=optimizer,
    workflow=workflow,
    make_evaluator=make_evaluator,
    regularization=settings.objective.regularization,
    n_evals=settings.n_evals,
)
run = q2mm.OptimizationRun.from_result(problem, result, configuration=configuration)
saved = q2mm.save(run, output_dir / "my-transition-state.frcmod")

print(baseline.total)
print(saved.force_field_path, saved.manifest_path)
```

The choices above are intentional:

- `stationary_point="transition_state"` drives TS curvature inversion;
- `functional_form="harmonic"` is compatible with JAX and `.frcmod`;
- `BackendSettings(key="jax")` declares the MM backend to load explicitly;
- `TransitionStateOptimizationSettings` declares the JAX/SciPy TS defaults;
- capture records the actual components **before** numerical execution.

`optimize` returns a numerical `OptimizationResult`, not a run or force field.
`OptimizationRun.from_result` materializes the final field and its provenance.
See the [canonical lifecycle and breaking API notes](https://ericchansen.github.io/q2mm/how-it-works/optimization-guide/#canonical-application-lifecycle).

An XYZ file carries symbols and coordinates, **not a Hessian**. If XYZ is your
geometry source, attach a separately generated canonical Hartree/Bohr² Hessian
with `molecule.with_hessian(...)` before calling `q2mm.prepare`.

## Multi-structure template workflow

For a production fitting set, supply one complete force field and a smaller
OPT/custom field identifying the scalar slots that may change:

```python
problem = q2mm.prepare(
    molecules,
    settings=PreparationSettings(stationary_point="transition_state", initialize="qfuerza"),
    force_field=complete_force_field,
    active_parameters=opt_force_field,
    observations=observations,
    case_ids=case_ids,
)
```

Then choose settings, load/build components, capture, optimize and assemble a
run exactly as above, using this problem and an `.fld` output filename.

The [Rh-enamide tutorial](https://ericchansen.github.io/q2mm/tutorial/#first-full-case-rh-enamide)
shows the nine-structure form of this workflow and clearly labels its current
geometry/eigenmatrix objective as a partial repository reproduction. The
governing source is
[Donoghue et al. 2008](https://doi.org/10.1021/ct800132a).

## Source-only examples

- `examples/ch3f/` — fresh ground-state and caller-owned FCHK workflow
- `examples/ch3f-sn2/` — fresh transition-state workflow
- `examples/publication/` — six real multi-molecule case studies
- `examples/backend-plugin/` — independently installable backend API v1 example

Examples and Rh-enamide source inputs are excluded from wheel and sdist
artifacts. Rh-enamide is tracked in the source repository; its
redistribution/licensing is not established. Dissertation archives and the
standalone MM3 base are never copied into q2mm artifacts.

## License

The q2mm software is MIT licensed; see [LICENSE](LICENSE). This statement does
not grant rights to external scientific inputs or third-party force fields.
