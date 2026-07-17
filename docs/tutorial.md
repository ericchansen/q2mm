# Tutorial: from a published case to your own system

Q2MM is useful when you have several QM training structures, a complete
molecular mechanics force field, and a smaller custom region that may change.
This tutorial first runs that full pattern on Rh-enamide, then shows exactly
which paths and objects to replace for a new project.

The examples use the package-root application API:
`q2mm.prepare`, `q2mm.evaluate`, `q2mm.optimize`, and `q2mm.save`. Publication
loaders are repository teaching helpers; internally they delegate problem
construction to the same `q2mm.prepare` service.

## Prerequisites

```bash
pip install "q2mm[jax]"
git clone https://github.com/ericchansen/q2mm.git
```

SciPy arrives through the JAX/optimizer extra, not through q2mm core. Use Linux
or WSL2 for JAX CUDA; follow [Platform Support](platform-support.md) before any
long optimization.

## First full case: Rh-enamide

The governing source is Donoghue, Helquist, Norrby, and Wiest,
[*J. Chem. Theory Comput.* **2008**, 4, 1313–1323](https://doi.org/10.1021/ct800132a).
It uses nine transition-state structures. The source objective also used ESP
charges and relative enthalpies.

The executable repository profile is
`repository-geometry-eigenmatrix-v1`: nine-case geometry and full-eigenmatrix
targets with the frozen repository weights. It is a
**partial repository reproduction**, not an exact reproduction of the paper.
Relative enthalpy is represented by a canonical type but remains blocked until
an MM backend supplies thermochemical enthalpy rather than potential energy.

The Rh scientific files are tracked under
`examples/publication/rh-enamide`, excluded from wheel and sdist artifacts, and
have no established redistribution/licensing statement. Supply that source
root explicitly:

```python
from pathlib import Path

import q2mm
from q2mm.benchmarks.systems import load_system
from q2mm.benchmarks.systems._paths import ExternalDataRoots

source_root = Path("/path/to/q2mm/examples/publication/rh-enamide")
output_root = Path("/path/to/output/rh-enamide")
output_root.mkdir(parents=True, exist_ok=True)

case = load_system(
    "rh-enamide",
    data_roots=ExternalDataRoots(rh_enamide=source_root),
    starting_point="qfuerza",
    objective_profile="repository-geometry-eigenmatrix-v1",
    functional_form="mm3",
)
problem = case.problem

print(problem.case_ids)
print(problem.active_space.n_active, problem.active_space.n_full)
print(problem.preparation_provenance.qfuerza_settings)

initial = q2mm.evaluate(problem, backend="jax", executor="jax")
run = q2mm.optimize(problem, backend="jax", recipe="recommended")
saved = q2mm.save(run, output_root / "rh-enamide-qfuerza.fld")

print(initial.total, dict(initial.category_scores))
print(run.configuration.optimizer.key, dict(run.configuration.optimizer.settings))
print(saved.force_field_path, saved.manifest_path)
```

For a fast software-path check rather than a scientific convergence run:

```bash
python examples/publication/rh-enamide/run.py \
  --rh-enamide /path/to/q2mm/examples/publication/rh-enamide \
  --output-root /path/to/output \
  --bounded-ci
```

`--bounded-ci` still parses all nine structures, composes the real field,
preserves the OPT active/frozen partition, evaluates the actual observations,
enters the optimizer, and writes a field plus manifest. It deliberately makes
no convergence claim.

### Published versus QFUERZA start

`starting_point="published"` keeps the source OPT values. The QFUERZA row
retains literature/base values and projects supported active bond/angle
scalars from the QM Hessians, with TS inversion derived from the stationary
point. Both starts use the same named observations; changing the start does not
change the objective.

QFUERZA is governed by
[Farrugia et al.](https://doi.org/10.1021/acs.jctc.5c01751). Transition-state
curvature inversion follows
[Limé and Norrby](https://doi.org/10.1002/jcc.23797).

## Replace Rh-enamide with your files

For Gaussian training logs and a complete/custom MM3 pair:

```python
from pathlib import Path

import q2mm
from q2mm.io import load_gaussian_molecules, load_mm3_fld
from q2mm.models.observations import ObservationSet

paths = tuple(sorted(Path("/data/my-project/qm").glob("*.log")))
case_ids = tuple(path.stem for path in paths)
molecules = load_gaussian_molecules(
    paths,
    structure_index=-1,
    require_hessian=True,
    bond_tolerance=1.4,
)
full_ff = load_mm3_fld("/data/my-project/complete.fld")
opt_ff = load_mm3_fld("/data/my-project/custom-opt.fld", include_standard=False)
observations = ObservationSet.from_molecules(molecules, case_ids=case_ids)

problem = q2mm.prepare(
    molecules,
    stationary_point="transition_state",
    force_field=full_ff,
    active_parameters=opt_ff,
    observations=observations,
    case_ids=case_ids,
    initialize="qfuerza",
)
initial = q2mm.evaluate(problem, backend="jax")
run = q2mm.optimize(problem, backend="jax")
q2mm.save(run, "/data/my-project/output/optimized.fld")
```

Decide case order yourself; filenames do not become scientific identities
automatically. Use `initialize="provided"` when supplied values must remain
unchanged. Jaguar and MacroModel inputs have explicit
`load_jaguar_molecules(..., structure_index=...)` and
`load_macromodel_molecules(..., structure_index=...)` bridges.

## Fresh force field for one molecule

When no shared template exists, one FCHK molecule can start a fresh field:

```python
import q2mm
from q2mm.io import load_fchk_molecule

molecule = load_fchk_molecule("/data/my-minimum.fchk")
problem = q2mm.prepare(
    molecule,
    stationary_point="ground_state",
    functional_form="harmonic",
)
baseline = q2mm.evaluate(problem, backend="jax")
run = q2mm.optimize(problem, backend="jax")
q2mm.save(run, "/data/output/my-minimum.frcmod")
```

Fresh preparation accepts one molecule because QFUERZA creates one topology.
For multiple molecules, provide an explicit shared template. XYZ is not a
Hessian format.

The installed-data smoke scripts are
[`examples/ch3f/`](https://github.com/ericchansen/q2mm/tree/master/examples/ch3f)
and
[`examples/ch3f-sn2/`](https://github.com/ericchansen/q2mm/tree/master/examples/ch3f-sn2).

## Other publication studies

Every comparison row links to its detailed source/status page.

| Study | What it adds beyond Rh-enamide | Executable claim |
|---|---|---|
| [Heck relay](systems/heck-relay.md) | 23 deposited cases, a separately blocked 24-case row, and the explicit `fc_fraction=0.05` scientific bound | 23-case executable archive with a partial objective |
| [Pd-allyl](systems/pd-allyl.md) | OPT-only composition over an external MM3 base; 21 primary plus four auxiliary cases | 21-case partial repository reproduction |
| [Pd 1,4-conjugate](systems/pd-conjugate.md) | Independent ten-case Pd composition and six physical OPT blocks | Partial repository reproduction |
| [Rh 1,4-conjugate](systems/rh-conjugate.md) | Staged eight-bisphosphine/two-diene source workflow | 2021 developmental SDK demonstration |
| [Ferrocene](systems/ferrocene.md) | Seven ground-state structures and published-only initialization | Seven-case partial profile; four scans and QFUERZA remain blocked |

Run the matching script under `examples/publication/<system>/run.py`. Wahlers
systems require both `--supporting-info` and `--mm3-base`; Heck requires
`--supporting-info`.

## Generate a reference Hessian with QCEngine

Reference generation is an explicit action. `q2mm.prepare` never starts a QM
calculation.

```python
import q2mm
from q2mm.backends import load_backend
from q2mm.io.xyz import load_xyz

molecule = load_xyz("/data/optimized.xyz", charge=-1, bond_tolerance=1.4)
reference = load_backend(
    "qcengine",
    program="psi4",
    method="b3lyp",
    basis="6-31g*",
)
hessian_result = q2mm.evaluate(
    molecule,
    backend=reference,
    property="hessian",
)
molecule = molecule.with_hessian(
    hessian_result.hessian,
    provenance=hessian_result.hessian_provenance,
)
problem = q2mm.prepare(
    molecule,
    stationary_point="transition_state",
    functional_form="harmonic",
)
```

QCEngine also exposes atomic energy and Cartesian coordinate gradient when the
selected program supports them. See the
[QCEngine documentation](https://qcengine.readthedocs.io/en/latest/).

## Evaluate an ASE calculator

ASE is also a lazy reference backend. It does not assign q2mm force-field
parameters:

```python
import q2mm
from ase.calculators.emt import EMT
from q2mm.backends import load_backend

reference = load_backend("ase", calculator=EMT())
energy = q2mm.evaluate(molecule, backend=reference, property="energy")
gradient = q2mm.evaluate(
    molecule,
    backend=reference,
    property="coordinate_gradient",
)
```

The v1 ASE adapter is non-periodic and reference-only. Periodic cells are
rejected instead of silently losing state. Calculator details are documented by
[ASE](https://ase-lib.org/ase/calculators/calculators.html).

## Construct an `OptimizationProblem` manually

Mixed stationary points or custom observation blocks use the immutable core:

```python
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import OptimizationProblem, StationaryPointKind, TrainingCase

layout = ParameterLayout.from_force_field(force_field)
space = ActiveParameterSpace.all_active(layout, force_field)
observations = ObservationSet.from_molecule(molecule, case_id="case-1")
problem = OptimizationProblem(
    cases=(
        TrainingCase(
            case_id="case-1",
            molecule=molecule,
            stationary_point=StationaryPointKind.GROUND_STATE,
        ),
    ),
    starting_force_field=force_field,
    layout=layout,
    active_space=space,
    observations=observations,
)

baseline = q2mm.evaluate(problem, backend=backend)
run = q2mm.optimize(
    problem,
    backend=backend,
    recipe="explicit",
    optimizer=optimizer,
    workflow="single-stage",
    executor="python",
)
```

Manual construction changes no evaluator or persistence contract.

## Provenance and blocked records

For a publication problem:

```python
metadata = problem.publication_metadata
print(metadata.status.value)
print(metadata.objective_profile.identifier)
for target in metadata.targets:
    print(target.category.value, target.disposition.value, target.details)
```

Atomic charge, direct ESP, relative enthalpy, constrained scan, and parameter
tether categories are never silently replaced. Unsupported selected targets
raise typed errors. The canonical blocker inventory—including missing Heck
`prrts1`, Pd-allyl auxiliary Hessians, Ferrocene scans/D1 topology, and the
unverified Os/Ru/sulfone mappings—is
[Publication Force-Field Coverage](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md).

Numerical publication claims belong in
[`ericchansen/q2mm-data`](https://github.com/ericchansen/q2mm-data), not in
example output committed to this repository.

## Author an external backend

The canonical independent plugin is
[`examples/backend-plugin/`](https://github.com/ericchansen/q2mm/tree/master/examples/backend-plugin).
Install it beside q2mm, list descriptors without importing implementation
modules, then explicitly load it:

```bash
pip install --no-deps ./examples/backend-plugin
q2mm-benchmark list
```

```python
from q2mm.backends import load_backend

backend = load_backend("harmonic-reference")
```

Continue with [Authoring a Plugin](backends/authoring.md) for the stable backend
API v1 manifest and conformance contract.
