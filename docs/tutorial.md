# Tutorial: from a first run to your own chemistry

Q2MM fits an inexpensive molecular mechanics (MM) energy model against
quantum-mechanical (QM) reference observations. Start here after the
[CPU first-run installation](getting-started.md#install-this-revision): first
use the packaged CH3F inputs, then replace them with your own molecule or a
multi-structure fitting set. A completed run is not evidence of force-field
accuracy; the observations and independent checks define what you can claim.

## The same example from Python

The source-only runner is the canonical small example implementation, not a
second tutorial algorithm. Set `JAX_PLATFORMS=cpu` before launching Python.
From the checkout, the following executes it with native Pydantic settings:

```python
from pathlib import Path
import runpy

from q2mm.settings import GroundStateOptimizationSettings

example = runpy.run_path(str(Path("examples") / "ch3f" / "run.py"))
settings = GroundStateOptimizationSettings(
    optimizer={"maxiter": 2, "maxls": 5},
    n_evals=0,
)
restored = GroundStateOptimizationSettings.model_validate_json(settings.model_dump_json())
assert restored.model_dump() == settings.model_dump()
result = example["run"](settings=restored, output_root=Path("results") / "ch3f")
assert result["execution_mode"] == "budget_limited_native"
assert result["optimization"]["convergence_claim"] is False
print(result["saved"])
```

For an arbitrary working directory or installed wheel, replace the relative
`Path("examples") / "ch3f" / "run.py"` with the absolute source-script path.
The [local-wheel instructions](getting-started.md#use-a-local-wheel) keep the
source script separate from the installed package.

`GroundStateOptimizationSettings` defaults to a JAX analytical objective and
SciPy L-BFGS-B. The partial dictionary updates the existing typed optimizer
default; it does not discard its other controls. JSON reconstruction validates
the same values. `n_evals=0` disables optional endpoint sampling, not the
optimizer's evaluations. The source command exposes the same controls as
`--optimizer.maxiter 2 --optimizer.maxls 5 --n-evals 0`.

The example defaults to a two-iteration, five-line-search-step, single-stage
budget even when no settings are supplied. An iteration cap is not an energy
evaluation count or wall-time guarantee. All these onboarding runs report
**no convergence claim**, regardless of the solver flag.

### What happens inside?

Read [`examples/_small_runner.py`](https://github.com/ericchansen/q2mm/blob/master/examples/_small_runner.py)
alongside the [canonical lifecycle](how-it-works/optimization-guide.md#canonical-application-lifecycle):

1. Load the packaged geometry and explicitly attach the packaged Hessian with
   units and source provenance.
2. `q2mm.prepare` creates an immutable problem: declared ground-state/TS
   cases, starting force field, parameter layout, active/frozen partition and
   reference observations. It launches no QM calculation.
3. Build the backend, optimizer and workflow from actual settings; capture
   their configuration **before** execution.
4. Create an objective evaluator and evaluate the baseline. `q2mm.optimize`
   uses the supplied optimizer, workflow and evaluator factory, returning
   an `OptimizationResult`, not a force field.
5. `OptimizationRun.from_result` assembles the final field and provenance;
   explicit `q2mm.save` writes the field and manifest.

The default observations compare geometry and the Hessian in a reference
eigenvector basis ("eigenmatrix"). This is not frequency matching and not the
same as every publication objective. See
[fitting objectives](how-it-works/fitting-objectives.md).
When deliberately fitting matched frequencies, prepare with an empty
`ObservationSet()`, then call the explicit runtime `q2mm.match_frequencies`
step with a loaded MM backend and `FrequencyMatchingSettings`.

## Bring your own FCHK

A Gaussian formatted checkpoint (`.fchk`) must contain element identities,
coordinates, and **Cartesian force constants** (the Hessian). An XYZ file is
not a substitute. Decide whether the calculation describes a minimum
(`ground_state`) or a transition state (`transition_state`); Q2MM does not
infer this from the filename or Hessian eigenvalues.

For a declared minimum, use the existing source runner, still with CPU selected:

```powershell
python examples\ch3f\run.py --fchk inputs\minimum.fchk --stationary-point ground_state --optimizer.maxiter 2 --output-root results\my-molecule
```

The example's filenames remain `ch3f.frcmod` and
`ch3f.frcmod.manifest.json` in your chosen directory even when the molecule is
caller-supplied. The JSON identifies `input.kind="caller_fchk"` and the input
name; the manifest fingerprints the actual input. Do not interpret these
filenames as a scientific identity.

For a declared TS, use `examples\ch3f-sn2\run.py` and
`--stationary-point transition_state`; its filenames are `ch3f-sn2.frcmod`
and `ch3f-sn2.frcmod.manifest.json`. This selects the TS settings and
curvature inversion, not the GS settings. The approved packaged SN2 Hessian
can emit a multiple-negative-mode diagnostic; that warning is not suppressed
or repaired for a demo, and this is not a TS scientific validation.

Missing files, missing force constants, or conflicting stationary-point
options fail explicitly. Do not supply both `--fchk` and `--input-root`.
The small scripts intentionally support only a fresh harmonic field.

For direct **preparation only** (no backend, optimizer or output):

```python
import q2mm
from q2mm.io import load_fchk_molecule
from q2mm.settings import PreparationSettings

molecule = load_fchk_molecule("inputs/minimum.fchk", bond_tolerance=1.5)
problem = q2mm.prepare(
    molecule,
    settings=PreparationSettings(stationary_point="ground_state", functional_form="harmonic"),
)
```

The [`prepare` API](reference/q2mm/preparation.md) describes input
validation. To execute your own prepared problem without the source runner,
use the [canonical lifecycle](how-it-works/optimization-guide.md#canonical-application-lifecycle)
with an explicitly budgeted GS or TS settings model and a chosen output path.
Do not reintroduce an execution wrapper or settings resolver.

## XYZ plus an explicit Hessian

XYZ provides element labels and coordinates, **not** charge, stationary-point
kind or a Hessian. The following preparation-only snippet expects your
optimized geometry and a separately obtained Hessian in canonical
**Hartree/Bohr²** units:

```python
import numpy as np
import q2mm
from q2mm.io.xyz import load_xyz
from q2mm.models.hessian import HessianProvenance, HessianUnits
from q2mm.settings import PreparationSettings

molecule = load_xyz("inputs/minimum.xyz", charge=0, bond_tolerance=1.5)
molecule = molecule.with_hessian(
    np.load("inputs/minimum-hessian.npy", allow_pickle=False),
    HessianProvenance(
        units=HessianUnits.ATOMIC,
        source="caller-quantum-calculation",
        path="inputs/minimum-hessian.npy",
    ),
)
problem = q2mm.prepare(
    molecule,
    settings=PreparationSettings(stationary_point="ground_state", functional_form="harmonic"),
)
```

For N atoms the matrix must be 3N by 3N. Labeling a matrix with these units
does not convert values from other units. Supply a real Hessian, not a
frequency list or a guessed identity matrix. Preparation without a Hessian
fails; it does not silently generate one.

Reference-property generation is a separate explicit
`q2mm.evaluate_reference(molecule, backend=..., property="hessian")` action.
Optional [QCEngine](https://qcengine.readthedocs.io/en/latest/) and
[ASE](https://ase-lib.org/) integrations require their extras and an actual
supported program/calculator. See [reference services](backends/index.md);
they are not prerequisites for onboarding.

## Many structures: complete field plus OPT/custom subset

Several molecules must share an explicit template topology. Supply a
**complete** MM3 field for evaluation, and a smaller **OPT/custom** field
that identifies which scalar parameters are allowed to change. It is a
selection mask, not a second independent force field for evaluation.

This preparation-only example expects your own ordered QM logs and field
files. It does not launch a full publication optimization:

```python
import q2mm
from q2mm.io import load_gaussian_molecules, load_mm3_fld
from q2mm.settings import PreparationSettings

case_ids = ("TS1", "TS2")
molecules = load_gaussian_molecules(
    ["inputs/TS1.log", "inputs/TS2.log"],
    structure_index=-1,
    require_hessian=True,
    bond_tolerance=1.4,
)
full_ff = load_mm3_fld("inputs/complete.fld")
opt_ff = load_mm3_fld("inputs/custom-opt.fld", include_standard=False)
problem = q2mm.prepare(
    molecules,
    settings=PreparationSettings(stationary_point="transition_state", initialize="qfuerza"),
    force_field=full_ff,
    active_parameters=opt_ff,
    case_ids=case_ids,
)
```

`case_ids` define identity and order explicitly. `structure_index=-1` selects
the last structure; Gaussian/Jaguar/MacroModel bridges require you to choose
an index instead of silently taking one. Batch loading is all-or-nothing.

With `initialize="qfuerza"`, Hessian-informed initialization re-estimates
supported active bond/angle scalars; frozen values stay equal to the baseline.
With `initialize="provided"`, supplied values are retained and the native
`qfuerza` default becomes `None`. Both settings JSON-round-trip without a
resolver. See [the initialization methodology](how-it-works/theory.md).
Declared TS preparation inverts negative QM curvature so the target TS
geometry can be an MM **minimum**, not an MM saddle.

Pass explicit `observations=` when your scientific target set differs from
the generic recipe. Neither this preparation nor a successful optimizer
flag establishes completeness of charges, energy differences, torsions or
other scientific targets. See
[observation categories](how-it-works/fitting-objectives.md).
Assemble and save an MM3 run to `.fld`, not harmonic `.frcmod`.

## Publication workflows are a separate branch

Current system/profile/start availability and blockers are generated on the
[publication coverage page](benchmarks/published-ff-validation.md), not
maintained as another tutorial table. Follow its per-system source links
before provisioning files or making a reproduction claim.

Rh-enamide source inputs at `examples/publication/rh-enamide` are
source-tracked but excluded from wheel/sdist;
recovered dissertation archives and the MM3 base must be supplied by the
caller. Publication examples never download them. Canonical relaxed-geometry
publication optimization remains **methodology-blocked**: limit work to
preparation/evaluation and bounded optimizer-entry/save proofs, not full
convergence runs.

### First full case: Rh-enamide

Here "full case" means a multi-structure template problem, **not** exact
publication reproduction or a full convergence run. The named Rh-enamide
loader prepares the ordered nine-case, frozen
`repository-geometry-eigenmatrix-v1` compatibility objective. That objective
does not supply all targets in the
[governing source](https://doi.org/10.1021/ct800132a); consult the
[source/status details](systems/rh-enamide.md) before using it.

Preparation only, with the source checkout explicitly selected:

```python
from pathlib import Path
from q2mm import ExternalDataRoots
from q2mm.benchmarks.systems import load_system

case = load_system(
    "rh-enamide",
    data_roots=ExternalDataRoots(rh_enamide=Path("examples") / "publication" / "rh-enamide"),
    starting_point="qfuerza",
    objective_profile="repository-geometry-eigenmatrix-v1",
    functional_form="mm3",
)
problem = case.problem
print(problem.case_ids)
print(problem.active_space.n_active, problem.active_space.n_full)
```

`starting_point="published"` retains the supplied OPT values instead of
re-estimating them; it does not change the named objective. No engine is
loaded here. Use the publication script's explicitly labeled `--bounded-ci`
only for a software-entry/save proof, never to claim a native fit.

The source-only
[`publication examples`](https://github.com/ericchansen/q2mm/tree/master/examples/publication)
document required roots and their software-only `--bounded-ci` mode. That
mode echoes references and updates no parameters, exactly as in the small
software proof; it is not a physical MM engine.

The benchmark CLI records candidates under your selected
`<output>/candidates/`. Only accepted candidates are promoted to
`<output>/accepted/` and `<output>/forcefields/`; a rejected/skipped candidate
does **not** promise a saved force field. The small example instead calls
`q2mm.save` explicitly, including when its numerical budget expires, without
changing benchmark acceptance policy. See [CLI persistence](benchmarks/cli.md).

Historical numerical claims require tracked evidence in
[`ericchansen/q2mm-data`](https://github.com/ericchansen/q2mm-data).
Keep generated example outputs in your chosen results directory, not in a
code commit.

For custom observation blocks and mixed stationary points, construct explicit
`TrainingCase` and `OptimizationProblem` values using the
[data model guide](how-it-works/data-types.md).
For an independent backend, reuse the source-only
[`backend plugin example`](https://github.com/ericchansen/q2mm/tree/master/examples/backend-plugin)
and [backend API v1 authoring guide](backends/authoring.md).
