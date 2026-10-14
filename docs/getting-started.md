# Getting started with your own files

Q2MM prepares and optimizes a molecular mechanics force field against
quantum-mechanical reference data. The shortest safe workflow starts from a
file that contains both an optimized geometry and a Cartesian Hessian, then
requires you to choose the stationary-point kind, force-field form, and MM
backend explicitly.

## Install only what you use

Python **3.15** is the primary development and core-test runtime. Package
metadata permits **>=3.14** for native backends that do not yet provide 3.15
builds; that fallback is **BEST-EFFORT**, not a guarantee for every backend.

```bash
pip install q2mm
pip install "q2mm[jax]"       # JAX backend and its optimization dependencies
pip install "q2mm[openmm]"    # OpenMM backend, Python 3.14 environment
pip install "q2mm[qcengine]"  # optional reference-property generation
pip install "q2mm[ase]"       # optional ASE reference calculators
```

SciPy is in optimizer extras such as `q2mm[jax]` and `q2mm[optimize]`; it is not
an automatic core dependency. Q2MM never chooses a backend or runs a QM job
inside `prepare`.

For an alpha release that is not yet stable on PyPI, add `--pre`. Developers
can instead clone the repository and run `pip install -e ".[dev]"`. The
development extra does not install OpenMM, ParmEd, ASE, QCEngine, or
QCElemental; request their individual extras only when using those integrations.

### Native dependency availability

OpenMM 8.6.1 has no CPython 3.15 wheels. Use a separate Python **3.14**
environment for its **BEST-EFFORT native/CPU lane**, rather than adding it to
the 3.15 development environment. Extras request real packages and are not
silently disabled on unsupported Python versions. CI may report an unavailable
native environment; if provisioning succeeds, correctness tests are required.
See the [runtime and environment contracts](https://github.com/ericchansen/q2mm/blob/master/AGENTS.md#4-platform-guide).

PyYAML 6.0.3 also has no cp315 wheel, but supports a pure-Python source build.
Set `PYYAML_FORCE_LIBYAML=0` before installing to avoid requiring a native C
compiler. In PowerShell:

```powershell
$env:PYYAML_FORCE_LIBYAML = "0"
python -m pip install -e ".[dev]"
```

On Linux or macOS:

```bash
PYYAML_FORCE_LIBYAML=0 python -m pip install -e ".[dev]"
```

Pydantic v2 and pydantic-settings v2 are base dependencies, independent of
optional computational backends.

## First complete problem

A Gaussian formatted checkpoint (`.fchk`) can carry the geometry and Cartesian
force constants needed by QFUERZA:

```python
from pathlib import Path

import q2mm
from q2mm.backends.registry import load_backend
from q2mm.io import load_fchk_molecule
from q2mm.objectives.plan import ObjectivePlan
from q2mm.settings import BackendSettings, PreparationSettings, TransitionStateOptimizationSettings

input_file = Path("/data/project/ts.fchk")
output_dir = Path("/data/project/q2mm-output")
output_dir.mkdir(parents=True, exist_ok=True)

molecule = load_fchk_molecule(input_file, bond_tolerance=1.4)
problem = q2mm.prepare(
    molecule,
    settings=PreparationSettings(stationary_point="transition_state", functional_form="harmonic"),
)
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
initial = q2mm.evaluate(make_evaluator(plan), problem.active_space.baseline)
result = q2mm.optimize(
    problem, optimizer=optimizer, workflow=workflow, make_evaluator=make_evaluator,
    regularization=settings.objective.regularization, n_evals=settings.n_evals,
)
run = q2mm.OptimizationRun.from_result(problem, result, configuration=configuration)
saved = q2mm.save(run, output_dir / "optimized.frcmod")
```

Choose `TransitionStateOptimizationSettings` for a declared TS, or
`GroundStateOptimizationSettings` for a declared minimum. These immutable
models hold final settings; loading a backend and constructing the
optimizer/workflow are separate explicit actions. Capture the actual objects
before execution; `optimize` returns only numerical results.
`OptimizationRun.from_result` builds the final field and provenance without
executing again. The package-root functions are lazy exports of those same
canonical services, not a separate convenience execution API. Partial nested
constructor input, such as
`TransitionStateOptimizationSettings(optimizer={"maxiter": 100})`, uses native
Pydantic validation and retains the other TS defaults. No resolver or
requested/effective settings pipeline is involved.

Backend selection does not implicitly change the objective executor. For a
non-JAX backend, supply an explicit typed objective, for example
`TransitionStateOptimizationSettings(backend=BackendSettings(key="openmm"),
objective=ObjectiveSettings(executor="python"))`. Import `ObjectiveSettings`
from `q2mm.settings`. The explicit executor selects its native gradient
default (Python/scalar or JAX/analytical); solver tolerances and TS bounds
remain the named preset's values. Incompatible JAX-objective/backend settings
are rejected rather than silently switched.

The output manifest records the resolved backend, optimizer, workflow, bounds,
input fingerprints, active slots, preparation audit, and objective result.
Saving never overwrites an existing file unless `overwrite=True` is explicit.
The numerical result retains the regularization and endpoint-sampling controls
actually used. Run assembly rejects a capture that disagrees with those
controls or either endpoint's sample count; direct algorithm results without
canonical execution-control capture are not ready-to-save run records.

Omitting `observations` selects the stationary-point-aware generic recipe.
See [Fitting objectives and reference data](how-it-works/fitting-objectives.md)
for mode selection, near-linear overrides, and the distinct frozen recipe
used by publication case loaders.

!!! warning "XYZ is geometry only"
    XYZ contains element labels and coordinates. It does not carry a Cartesian
    Hessian. Load the Hessian from its actual source and attach it with
    `molecule.with_hessian(hessian, provenance=...)`; never imply that
    `load_xyz` supplied it.

## Input bridges are explicit

Q2MM does not guess file formats or which structure in a trajectory is intended.

```python
from q2mm.io import (
    load_fchk_molecule,
    load_gaussian_molecules,
    load_jaguar_molecules,
    load_macromodel_molecules,
)

one = load_fchk_molecule("minimum.fchk")
gaussian = load_gaussian_molecules(
    ["case-1.log", "case-2.log"],
    structure_index=-1,
    require_hessian=True,
)
jaguar = load_jaguar_molecules(
    ["case-1.in", "case-2.in"],
    structure_index=0,
    require_hessian=True,
)
macromodel = load_macromodel_molecules(
    ["training-set.mmo"],
    structure_index=0,
)
```

Gaussian, Jaguar, and MacroModel bridges require `structure_index`; the SDK
does not silently choose first or last. Batch loading is all-or-nothing.

## Template-backed multi-structure problems

Use a complete field for evaluation and a smaller OPT/custom field to identify
the rows that can change:

```python
import q2mm
from q2mm.io import load_gaussian_molecules, load_mm3_fld
from q2mm.settings import PreparationSettings

paths = ["TS1.log", "TS2.log", "TS3.log"]
molecules = load_gaussian_molecules(paths, structure_index=-1)
case_ids = ("TS1", "TS2", "TS3")
full_ff = load_mm3_fld("complete.fld")
opt_ff = load_mm3_fld("custom-opt.fld", include_standard=False)

problem = q2mm.prepare(
    molecules,
    settings=PreparationSettings(stationary_point="transition_state", initialize="qfuerza"),
    force_field=full_ff,
    active_parameters=opt_ff,
    case_ids=case_ids,
)
```

Only active scalar slots are re-estimated. Frozen slots remain bitwise equal to
the supplied baseline. QFUERZA methodology is governed by
[Farrugia et al.](https://doi.org/10.1021/acs.jctc.5c01751); transition-state
curvature handling follows
[Limé and Norrby](https://doi.org/10.1002/jcc.23797).

To retain supplied values, use
`PreparationSettings(stationary_point="transition_state", initialize="provided")`.
Its `qfuerza` controls default to `None`; passing `qfuerza=None` explicitly is
also valid. Other initialization choices default to `QFuerzaSettings()`.
Validation checks the actual initialization/control combination, and the same
settings remain valid after JSON serialization and reconstruction.

## Publication data roots

Publication examples never download scientific data. Supply the required roots
as CLI arguments or through the loader's `ExternalDataRoots` object.

| Root | Systems | Distribution status |
|---|---|---|
| `rh_enamide` / `Q2MM_RH_ENAMIDE` | Rh-enamide | Source-tracked at `examples/publication/rh-enamide`, excluded from wheel/sdist; redistribution/licensing not established. |
| `supporting_info` / `Q2MM_SUPPORTING_INFO` | Heck, Pd/Rh conjugate, Pd-allyl, Ferrocene | Caller-supplied recovered/source archive; never packaged. |
| `mm3_base` / `Q2MM_MM3_BASE` | Pd/Rh conjugate, Pd-allyl, Ferrocene | Caller-supplied MM3 base; never packaged. |

PowerShell example:

```powershell
$env:Q2MM_RH_ENAMIDE = "C:\path\to\q2mm\examples\publication\rh-enamide"
$env:Q2MM_SUPPORTING_INFO = "C:\path\to\publication-data"
$env:Q2MM_MM3_BASE = "C:\path\to\mm3_base.fld"
```

The canonical source-linked status and blocker table is
[Publication Force-Field Coverage](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md).

## Run fast executable examples

From a source checkout, while q2mm is installed:

```bash
python examples/ch3f/run.py --bounded-ci --output-root ./results/ch3f
python examples/ch3f-sn2/run.py --bounded-ci --output-root ./results/ch3f-sn2
```

`--bounded-ci` constructs the real problem and enters its optimizer once. It is
deliberately separate from the default scientific workflow and makes no
convergence claim.

Next, follow the [tutorial](tutorial.md) for the first full nine-structure
Rh-enamide case, bring-your-own substitutions, reference backends, manual
problem construction, blocked publication records, and an external plugin.
