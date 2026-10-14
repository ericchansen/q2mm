# Getting started

Q2MM builds and fits molecular mechanics force fields against quantum-mechanical
(QM) reference data. This guide gets you to a saved first result before you
need your own QM files. The small CH3F example uses a precomputed geometry and
Hessian (energy curvature as atoms move), then performs real CPU molecular
mechanics with a deliberately small optimizer budget.

## Install this revision

Python **3.15** is the primary development, docs, release and JAX CPU runtime.
Python **3.14** is also supported and has required core checks. OpenMM and
JAX-MD use separate Python 3.14 CPU environments; the remaining documented
native and lightweight integration lanes use Python 3.15.

The breaking alpha API and generated CLI described here are **unreleased**.
Install the source revision containing this guide, or a wheel built from it.
`pip install q2mm` from PyPI is not evidence that this API is available.

Use **Python 3.15 final**, not a 3.15 prerelease. From the source checkout:

=== "Windows PowerShell"

    ```powershell
    python --version
    python -c "import sys; assert sys.version_info[:2] == (3, 15) and sys.version_info.releaselevel == 'final'"
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    $env:PYYAML_FORCE_LIBYAML = "0"
    python -m pip install -e ".[jax]"
    ```

    If activation is disallowed, call `.\.venv\Scripts\python.exe` instead of
    `python` in subsequent commands; no execution-policy change is needed.

=== "Linux / macOS"

    ```bash
    python3.15 --version
    python3.15 -c "import sys; assert sys.version_info.releaselevel == 'final'"
    python3.15 -m venv .venv
    source .venv/bin/activate
    PYYAML_FORCE_LIBYAML=0 python -m pip install -e ".[jax]"
    ```

The `jax` extra installs JAX/jaxlib, SciPy, and the JAX optimizer dependencies.
It does not require OpenMM or a QM program. Core dependencies include Pydantic
and pydantic-settings. PyYAML's supported pure-Python source build
(`PYYAML_FORCE_LIBYAML=0`) avoids a C compiler when no cp315 wheel is available.
SciPy is optional in core; it is explicitly installed by this extra.

## First success on CPU

Select CPU **before starting Python**. Standalone scripts do not inherit
pytest's CPU policy.

=== "Windows PowerShell"

    ```powershell
    $env:JAX_PLATFORMS = "cpu"
    python -c "import jax; assert all(d.platform == 'cpu' for d in jax.devices()); print(jax.devices())"
    python examples\ch3f\run.py --optimizer.maxiter 2 --output-root results\ch3f
    ```

=== "Linux / macOS"

    ```bash
    export JAX_PLATFORMS=cpu
    python -c "import jax; assert all(d.platform == 'cpu' for d in jax.devices()); print(jax.devices())"
    python examples/ch3f/run.py --optimizer.maxiter 2 --output-root results/ch3f
    ```

The JAX CPU path refuses an unset or different `JAX_PLATFORMS` and checks the
selected backend's CPU platform before creating an evaluator. It loads
the approved installed `ch3f-optimized.xyz` and `ch3f-hessian.npy` resources,
prepares a fresh harmonic **ground-state** field with QFUERZA, evaluates the
generic geometry/eigenmatrix objective using JAX, and enters SciPy L-BFGS-B.
No data download, reference QM job, or publication convergence run occurs.
The [resource manifest](https://github.com/ericchansen/q2mm/blob/master/q2mm/data/sn2/manifest.json)
records source generation, licensing, sizes and hashes.

### Read the result

One JSON document appears on stdout; diagnostics/warnings appear on stderr.
The only output files are:

```text
results/ch3f/ch3f.frcmod
results/ch3f/ch3f.frcmod.manifest.json
```

`ch3f.frcmod` contains the harmonic parameters in AMBER format.
The manifest records input/layout fingerprints, the baseline and active
slots, preparation provenance, actual backend/optimizer/workflow settings,
and the numerical result. Its `provenance.example` records the software/native
mode and `convergence_claim=false`, while retaining the raw solver result
unchanged. It is not an acceptance certificate.

| Control or output | Meaning |
| --- | --- |
| `--output-root` | Your chosen results directory; all saved files go here. Existing files are not overwritten. |
| `--optimizer.maxiter 2` | At most two outer SciPy iterations, **not** two energy evaluations or a wall-time limit. The example also defaults to `maxls=5` and a single-stage workflow. |
| `--n-evals 0` (default) | No optional endpoint sampling; it does **not** disable optimization or its objective evaluations. |
| `initial` / `final` | Objective totals and category scores from the selected executor, not an independent accuracy measurement. |
| `optimization.success` / `message` | The raw solver status. An iteration-limit stop and `success=false` are valid budgeted outcomes. |
| `optimization.convergence_claim` | Always `false` for this onboarding example, even if the solver reports success. |
| `optimization.parameter_update_norm` | Size of the actual returned parameter change; the software-only mode reports zero. |

**First success means the native execution/save path ran, not that the force
field is converged, accurate, accepted by benchmark policy, or a publication
reproduction.** Do not increase the budget and infer quality from the solver's
success flag alone. See [fitting objectives](how-it-works/fitting-objectives.md)
before designing a scientific fit.

To inspect policy without calculation:

```powershell
python examples\ch3f\run.py --output-root results\ch3f --optimizer.maxiter 2 --show-config
q2mm-benchmark single --system ch3f --backend.key jax --functional-form harmonic --optimizer.maxiter 2 --show-config
```

Both use native Pydantic CLI fields. `--show-config` loads no engines or
scientific data, writes no files, and does **not** verify runtime availability,
input completeness, or convergence. Settings read explicit constructor/CLI
input, not ambient environment/dotenv/secret sources. CPU selection is a
separate explicit JAX runtime environment choice.

### Software-only proof without JAX

In a core-only environment installed with `python -m pip install -e .`:

```powershell
python examples\ch3f\run.py --bounded-ci --output-root results\ch3f-software
```

This proves parsing, preparation, objective dispatch, optimizer entry, run
assembly, and saving only. `BoundedEchoBackend` returns **zero energy** and
the **reference geometry/Hessian**; `BoundedExampleOptimizer` returns the
unchanged parameter vector. It is **not native MM**, optimizer convergence,
force-field quality, or scientific reproduction. It records
`execution_mode="software_path_only"`, versus `"budget_limited_native"` above.
Numerical policy flags in this mode do not configure the echo/no-update proof;
the saved manifest identifies the actual software components.

## Use a local wheel

The example scripts are **source-only and excluded from wheels and sdists**.
The approved small-system resources are **installed package data**. To verify
the same first-run path against an isolated wheel, keep the source checkout
for the script but install only the wheel and its documented extra into a
new environment. PowerShell, starting in the checkout:

```powershell
python -m pip install build
python -m build --wheel
$wheels = @(Get-ChildItem dist\q2mm-*.whl)
if ($wheels.Count -ne 1) { throw "Expected exactly one locally built wheel in dist." }
$wheel = $wheels[0]
$example = (Resolve-Path examples\ch3f\run.py).Path
python -m venv .wheel-venv
.\.wheel-venv\Scripts\python.exe -m pip install "$($wheel.FullName)[jax]"
$wheelPython = (Resolve-Path .wheel-venv\Scripts\python.exe).Path
$env:JAX_PLATFORMS = "cpu"
```

Use a directory outside the checkout, such as a newly created empty project
directory, as the working directory. Keep the absolute `$example` and
`$wheelPython` paths captured above, then run:

```powershell
& $wheelPython -I -c "import q2mm; print(q2mm.__file__)"
& $wheelPython -I $example --optimizer.maxiter 2 --output-root results\ch3f
```

The import must resolve under the new environment's `site-packages`, not the
source tree. `-I` prevents accidental import from the working directory or
`PYTHONPATH`. The script resolves its shared runner relative to itself, and
inputs relative to the installed package, so an arbitrary working directory
is supported. Outputs remain relative to your explicit `--output-root`.
Add `--bounded-ci` for the distinct installed-wheel software-path proof.

## Other installation contexts

### Native dependency availability

OpenMM 8.6.1 has no CPython 3.15 wheels. Use a separate Python **3.14**
environment for its required CPU lane rather than adding it to the primary
development environment. The available JAX-MD dependency chain likewise
requires Python 3.14. Extras request real packages; dependency conflicts,
missing distributions, image failures and numerical failures are not silently
converted into successful checks.
See the [runtime and environment contracts](https://github.com/ericchansen/q2mm/blob/master/AGENTS.md#4-platform-guide).

| Responsibility | CPU lane | Dependency owner |
| --- | --- | --- |
| [Core](https://github.com/ericchansen/q2mm/blob/master/.github/workflows/ci.yml) | Python 3.14 and 3.15 | PyPI |
| [JAX, QCElemental v2, QCEngine, ASE, ParmEd IO](https://github.com/ericchansen/q2mm/blob/master/pyproject.toml) | Python 3.15 | PyPI |
| [OpenMM 8.6.1](https://github.com/ericchansen/q2mm/blob/master/.github/envs/openmm.yml) | Python 3.14 | PyPI |
| [JAX-MD](https://github.com/ericchansen/q2mm/blob/master/.github/envs/jax-md.yml) | Python 3.14 | PyPI owns JAX, jaxlib and JAX-MD together |
| [Tinker 8.11.3](https://github.com/ericchansen/q2mm/blob/master/.github/envs/tinker.yml) | Python 3.15 | Conda plus matching verified runtime parameters |
| [Psi4 1.11](https://github.com/ericchansen/q2mm/blob/master/.github/envs/psi4.yml) | Python 3.15 | Conda |

CI runs small, explicitly selected native and pairwise contracts, not broad
research workloads. Unaffected native jobs report `not affected` without
pulling images. Affected jobs require their pinned image, consistent packages,
real imports and nonzero executed tests. These checks do not establish full
scientific convergence, GPU performance or publication reproduction.

JAX CUDA/JAX-MD are not available on native Windows. No GPU is needed here.
See [platform support](platform-support.md) for other contexts.

Install only the extras you need **from this revision**: `.[optimize]` for
SciPy without JAX, `.[qcengine]` or `.[ase]` for explicit reference-property
generation, and `.[openmm]` in its separate supported environment.
`.[dev]` supplies development tools, not all optional backends.

Backend selection does not change the objective executor implicitly. In
Python, supply `BackendSettings(key="openmm")` together with a complete typed
`ObjectiveSettings(executor="python")` from `q2mm.settings`. Its native gradient
default is scalar (`none`); solver tolerances and TS bounds remain the named
preset's values. In the small example's CLI, explicitly pass
`--backend.key openmm --objective.executor python --objective.gradient-mode none`
because nested CLI updates retain the other existing fields.
Incompatible JAX-objective/backend settings are rejected before backend
construction, never silently switched.
The small example uses `null` for optional unset values so the literal
gradient mode `none` stays distinct; inspect its generated `--help`.

**OpenMM needs its own CPU selection**: `JAX_PLATFORMS` has no effect on it.
Add `--backend.options '{"platform_name":"CPU"}'` to the OpenMM options above
(in Python, `BackendSettings(key="openmm", options={"platform_name": "CPU"})`).
Alternatively, explicitly set `OPENMM_DEFAULT_PLATFORM=CPU` before starting
Python. A GPU option overrides that environment setting and is rejected.
Absent or non-CPU OpenMM selection fails before the backend factory runs;
the loaded backend's selected platform is checked again before any context
preparation or evaluation. No backend/executor/option is silently rewritten.
The selected CPU platform is recorded in stdout and the saved provenance;
other custom native backends must report CPU platform provenance too.

The numerical result retains the regularization and endpoint-sampling controls
actually used. Run assembly rejects a capture that disagrees with those
controls or either endpoint's sample count; direct algorithm results without
canonical execution-control capture are not ready-to-save run records.

Next, the [tutorial](tutorial.md) uses the same runner from Python, then
branches into your own FCHK, XYZ plus Hessian, and multi-structure complete/OPT
inputs. Current publication provisioning and methodology blockers live on the
generated [publication coverage page](benchmarks/published-ff-validation.md);
they are not the onboarding starting point.
