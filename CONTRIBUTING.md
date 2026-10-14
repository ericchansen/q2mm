# Contributing to Q2MM

Q2MM fits molecular-mechanics force fields to quantum-mechanical reference
calculations. This guide helps you choose where to make a change and how to
check it without launching an expensive scientific experiment. A force field
is the collection of parameters defining the molecular energy model; fitting
adjusts selected parameters to reproduce reference observations.

For your first calculation, use [Getting Started](docs/getting-started.md)
and the [Tutorial](docs/tutorial.md). For development, read
[AGENTS.md](AGENTS.md), then follow the
[canonical input-to-save walkthrough](docs/how-it-works/architecture.md#input-to-save-walkthrough).
The architecture page owns the module map, dependency boundaries, and unit
conventions; this page is the task-oriented entry point, not another tutorial.

## Set up a development checkout

Use **Python 3.15** in a checkout-local virtual environment:

```bash
git clone https://github.com/ericchansen/q2mm.git
cd q2mm
python -m venv .venv
```

Activate `.venv` with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install -e ".[dev]"
```

The [dev extra](pyproject.toml) supplies pytest, Ruff, mypy, SciPy, and build
tools. It does **not** install OpenMM, ParmEd, ASE, QCEngine, or QCElemental.
Select the extra needed by your change, such as `.[dev,amber]`, rather than
installing every backend. Core imports and settings must work without them.
Python 3.14 is also supported and has required core checks. Ruff and mypy
enforce that language/type floor while running on the primary Python 3.15
development interpreter.

See [native dependency availability](docs/getting-started.md#native-dependency-availability)
for runtime selection. OpenMM and JAX-MD use separate Python 3.14 CPU
environments; the other documented native and lightweight integration lanes
use Python 3.15. JAX CUDA and JAX-MD are unavailable on native Windows.

If PyYAML 6.0.3 needs a source build on Python 3.15 without a C compiler,
set `PYYAML_FORCE_LIBYAML=0` before installation
(PowerShell: `$env:PYYAML_FORCE_LIBYAML = "0"`). This selects upstream's
pure-Python build; it does not remove the dependency.

## Choose your task

Each row identifies the implementation owner, a consumer to trace, the
invariant to preserve, and a small default-tier check. An **invariant** is a
property that must remain true after your change. Run commands from the
repository root with the activated environment. These checks are a starting
point: add a focused regression for your change and run its numerical/backend
checks when that implementation is available.

| Change | Canonical owner and consumer to follow | Invariant to preserve | Smallest useful starting check |
| --- | --- | --- | --- |
| File format | A format module in [q2mm/io](q2mm/io), its [public exports](q2mm/io/__init__.py), and, for molecular input, [molecule bridges](q2mm/io/molecules.py). Follow [prepare](q2mm/preparation.py); for output follow [save](q2mm/application/persistence.py). | [Boundary units and coefficient conventions](docs/how-it-works/architecture.md#2-canonical-internal-units), topology, Hessian provenance, and lossless supported output. Models must not import parsers. Add a saver to the application format registry only when its scientific representation is supported; do not add parser methods to `ForceField`. | `python -m pytest test/test_io_type_boundaries.py test/test_io_fchk.py -q` |
| Observation or objective category | [Observation types](q2mm/models/observations.py), [plan compilation](q2mm/objectives/plan.py), and [shared metrics](q2mm/objectives/metrics.py). Follow the [Python executor](q2mm/objectives/python.py) and [JAX executor](q2mm/objectives/jax.py). | [Case bindings, weights, residuals, and explicit gradient support](docs/how-it-works/fitting-objectives.md). Unknown/unsupported targets fail explicitly; a new target is not silently omitted. Keep the numerical oracle independent of production residual formulas. | `python -m pytest test/test_observations.py test/test_publication_observations.py -q -m "not jax"` |
| Backend or external plugin | [Typed requests/sessions](q2mm/backends/contracts.py), [manifest validation](q2mm/backends/discovery.py), [lazy registry](q2mm/backends/registry.py), and [conformance](q2mm/backends/conformance.py). Follow [create_evaluator](q2mm/objectives/__init__.py) or the separately typed [reference service](q2mm/application/evaluation.py). | [Backend API v1](docs/backends/authoring.md): descriptor-only discovery, isolated failures, truthful capability/form ceilings, exact runtime role, immutable requests, canonical result units. Equal units do not prove equal physical models. | `python -m pytest test/test_backend_discovery.py test/test_backend_conformance.py -q` |
| Optimizer or workflow | [Typed controls](q2mm/settings.py), [explicit optimizer builder/catalog](q2mm/optimizers/catalog.py), and the selected [algorithm](q2mm/optimizers). For stages, follow [workflow construction/factory](q2mm/workflows/base.py), then the shared [optimize](q2mm/application/optimization.py). | [Numerical execution boundaries](docs/how-it-works/architecture.md#input-to-save-walkthrough): `optimize(evaluator, space)` returns the canonical result, preserves frozen slots and accounting, and does not construct a backend. `CyclingOptimizer` uses configured factories for fresh phase state on every cycle/call; it is not a bound-input loop adapter. | `python -m pytest test/test_optimizer_catalog.py test/test_multistart_composition.py -q` |
| Publication system or profile | The [per-system loader/registry](q2mm/benchmarks/systems), [shared assembly](q2mm/benchmarks/systems/_assembly.py), [publication records](q2mm/benchmarks/publications.py), and [typed profiles](q2mm/benchmarks/profiles.py). Follow [runner acceptance/persistence](q2mm/benchmarks/runner.py) and the [page projection](scripts/gen_publication_pages.py). | [Source-linked reproduction evidence](validation/published_ffs/README.md): declared GS/TS, source case membership, frozen/active partition, starting-point identity, objective profile, and explicit blockers. Read the governing papers before changing methodology; do not call a partial repository objective an exact publication reproduction. | `python -m pytest test/test_benchmarks_systems.py test/test_publication_compatibility.py test/test_publication_docs.py -q` |
| Execution setting or CLI field | [Strict settings](q2mm/settings.py), [neutral component capture](q2mm/configuration.py), [application capture](q2mm/application/configuration.py), and [native command models](q2mm/benchmarks/cli.py). Follow the [typed RunProfile](q2mm/benchmarks/profiles.py) into explicit builders and [optimize](q2mm/application/optimization.py). | [Final settings and generated CLI](docs/how-it-works/architecture.md#settings-and-command-line-ownership): strict finite/deep-immutable values, applicable `None`/zero/false controls, effective-value round trips, actual component identity, and preserved exception causes/interrupts. No custom parser, generic override resolver, or runtime setup container. | `python -m pytest test/test_settings.py test/test_object_configuration.py test/test_benchmarks_cli.py -q` |

For a different existing format or algorithm, select its corresponding test
module rather than treating the examples above as complete coverage.
[Ordinary, advanced, and internal APIs](docs/how-it-works/architecture.md#api-boundaries)
explain which interfaces contributors can extend without reaching into a
private implementation.

### Backend authors: verify an installed plugin

Use the single [backend author guide](docs/backends/authoring.md) and the
independently installable [reference plugin](examples/backend-plugin/README.md).
Install it into your activated development environment:

```bash
python -m pip install ./examples/backend-plugin
q2mm-benchmark list
```

Discovery alone is not numerical conformance. Follow the example's
`MMConformanceCase`/`run_mm_conformance` check with a small deterministic
molecule and force field; reference adapters use `ReferenceConformanceCase`.
The [release-artifact verifier](scripts/check_release_artifacts.py) additionally
installs this plugin against a rebuilt wheel in a clean environment and checks
its public conformance (`external-plugin=ok`). For packaging changes:

```bash
python -m build
python scripts/check_release_artifacts.py --dist-dir dist
```

This is an installed-artifact check, not a dataset download or GPU benchmark.

### Publication documentation has one source

The [system inventory](docs/benchmarks/index.md#registered-systems) and
[profile/start records](docs/benchmarks/published-ff-validation.md#current-matrix)
are projections, not manually maintained status tables.
After an authorized metadata change, `python scripts/gen_publication_pages.py --write`
refreshes marked sections; `python scripts/gen_publication_pages.py` checks
them without writing. Preserve the [source evidence](validation/published_ffs/README.md),
scientific fingerprints, and independent golden fixtures. Dataset artifacts
belong in [q2mm-data](https://github.com/ericchansen/q2mm-data), not this code
repository; licensed third-party data must not enter release artifacts.

## Choose the right kind of test

Test purpose and execution cost are different. A **numerical oracle** is an
independent expected answer, not a second call to the code under test.

| Test role | What it proves | Existing starting point |
| --- | --- | --- |
| Scientific independent oracle | Equations, units, mode selection, and frozen/active scientific semantics match independently derived expectations. Do not generate expected values with the production formula. | [Unit conversions](test/test_units.py), [scientific contracts](test/test_scientific_contracts.py), [publication compatibility](test/test_publication_compatibility.py) |
| Public contract | Inputs, outputs, object identity, error causes, full vectors, and result/run separation remain usable by callers. | [Facade](test/test_facade.py), [application](test/test_application.py), [problem](test/test_problem.py) |
| Import and ownership guard | Canonical owners stay unique; full module paths and local/relative imports obey layer boundaries. | [Architecture/module map](test/test_architecture_doc.py), [benchmark ownership](test/test_benchmark_owners.py) |
| Optional-dependency isolation | Imports, settings, help, and discovery work without computational runtimes; inspection does not construct engines or load data. | [Facade](test/test_facade.py), [settings](test/test_settings.py), [CLI](test/test_benchmarks_cli.py), [discovery](test/test_backend_discovery.py) |
| Persistence fault injection | Simulated failures, interrupts, concurrent writers, path aliases, and cleanup exercise rollback and retained-backup behavior while preserving original causes; sequential replacements are not crash-atomic. Do not replace these with a happy-path save test. | [Save transactions](test/test_persistence_transactions.py), [benchmark transactions](test/test_benchmarks_runner.py) |
| Opt-in scientific workload | Real pipelines, publication comparisons, or convergence experiments exercise scientific behavior beyond bounded software proofs. | [Integration/validation workloads](test/integration), [tier policy](test/conftest.py) |

Default tests are small software and numerical checks. `--run-integration`
adds real pipelines; `--run-validation` includes integration and scientific
comparisons; `--run-nightly` includes both plus full convergence workloads.
These flags do not schedule a run. Backend markers (`openmm`, `tinker`, `jax`,
`jax_md`, `psi4`) describe required runtimes, not test tiers.

CPU is the default unless the environment already selects a platform.
Do not enable `--gpu` or `Q2MM_USE_GPU=1` for ordinary development.
A skip means the runtime/data/tier was not exercised, **not** that it passed.
Collect an expensive suite before deciding to run it:

```bash
python -m pytest test/integration/test_published_ff_validation.py --collect-only -q
```

Canonical relaxed-geometry publication optimization remains subject to the
[documented methodology blockers](docs/benchmarks/published-ff-validation.md).
Preparation, evaluation, and bounded optimizer-entry/save proofs are not a
license to launch the full convergence matrix. Consult the
[benchmark pre-flight checklist](AGENTS.md#11-benchmark-pre-flight-checklist)
before any authorized long run.

## Verify and submit a focused change

Before committing, run the same checks as the primary CI lane:

```bash
python -m ruff check q2mm/ test/ scripts/
python -m ruff format --check q2mm test scripts examples
python -m mypy q2mm/
python scripts/check_env_dep_parity.py
python -m pytest test/ -x -q -m "not (openmm or tinker or jax or jax_md or psi4)"
```

The [CI workflow](.github/workflows/ci.yml) requires core checks on Python 3.14
and 3.15. Lint/types, docs, release, JAX CPU and the independent
QCElemental-v2/QCEngine/ASE/ParmEd adapters use Python 3.15.
Native backend and pairwise jobs are path-gated: unaffected jobs report
`not affected`, while affected jobs require pinned prebuilt images, consistent
packages, native imports, and nonzero executed tests. Pull, installation, and
correctness failures fail closed, not as best-effort availability.
Dependency parity checks that declared extras and CI environments agree;
it does not prove numerical backend equivalence.

For the same bounded backend responsibilities locally:

```bash
bash scripts/ci_local.sh --job openmm-cpu
bash scripts/ci_local.sh --job tinker
bash scripts/ci_local.sh --job parity-openmm-jax
bash scripts/ci_local.sh --all
```

These use input-pinned images and reject zero executed tests. JAX CPU uses
the primary Python environment directly; there is no standalone JAX or
kitchen-sink image. `--all` is not a broad integration/validation/nightly/GPU
suite proxy. [Environment files](.github/envs) are the image inventory, and
[ci_image.py](scripts/ci_image.py) derives their input-pinned references:

```bash
python scripts/ci_image.py openmm --field reference
IMAGE="$(python scripts/ci_image.py openmm --field reference)"
docker build -f .github/docker/Dockerfile --build-arg ENV_FILE=openmm.yml -t "$IMAGE" .github
```

For documentation changes, install `.[docs]` into the same environment, then
use the project documentation runner, not a global `properdocs`:

```bash
uv run properdocs build -f properdocs.yml
python -m pytest test/test_architecture_doc.py test/test_publication_docs.py -q
node test/test_benchmark_tables.js
```

Check rendered anchors/API links as well as source links. The existing module
map guard must stay strong: when a module moves, update the canonical map and
verify the dependency direction rather than deleting the assertion.

Work on a feature branch, use focused conventional commits (`feat`, `fix`,
`docs`, `refactor`, `test`, `ci`, or `chore`), and open a pull request against
the intended base (`master` for independent work, the parent branch for a
stacked change). Describe the affected contract, evidence, and actual checks;
state optional-runtime/data limitations explicitly. Do not regenerate golden
data to conceal a changed scientific result.
