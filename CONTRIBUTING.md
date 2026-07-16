# Contributing to Q2MM

Thanks for your interest in contributing! This guide covers setup, testing, and
submitting changes.

## Development Setup

```bash
git clone https://github.com/ericchansen/q2mm.git
cd q2mm
pip install -e ".[dev]"
```

The `[dev]` extras include pytest, ruff, other development tools, and core
backend/optimizer dependencies such as `openmm` and `scipy`.

If you need *all* optional backends for local testing, you can install the
broader `all` extra:

```bash
pip install -e ".[dev,all]"               # everything (OpenMM, JAX, scipy, etc.)
```

## Running Tests

### Test tiers

Tests are organized into four tiers so you can iterate quickly:

```bash
pytest                        # core only (~13s, no backends)
pytest --run-integration      # core + integration (~49s)
pytest --run-validation       # core + integration + validation (~80s)
pytest --run-nightly          # everything (~330s+, optimizer loops)
```

By default, all tests run on **CPU only** — no GPU memory is allocated.
Use `--gpu` or set `Q2MM_USE_GPU=1` to opt into GPU execution for
benchmarks or GPU-specific tests.

### Backend markers

Tests requiring specific backends are auto-skipped when the dependency is
missing. Use `-m` to filter:

```bash
pytest -m openmm           # only OpenMM tests
pytest -m tinker           # only Tinker tests
pytest -m "not tinker"     # skip Tinker tests
```

Available markers: `openmm`, `tinker`, `jax`, `jax_md`, `psi4`.

### Running backend tests with Docker

If you don't have OpenMM, Tinker, or other backends installed locally, you can
use the pre-built CI images from GitHub Container Registry:

```bash
# OpenMM tests
docker run --rm -v $PWD:/workspace -w /workspace \
    ghcr.io/ericchansen/q2mm/ci-openmm:latest \
    pytest -m openmm --run-nightly

# Tinker tests
docker run --rm -v $PWD:/workspace -w /workspace \
    ghcr.io/ericchansen/q2mm/ci-tinker:latest \
    pytest -m tinker --run-nightly

# JAX tests
docker run --rm -v $PWD:/workspace -w /workspace \
    ghcr.io/ericchansen/q2mm/ci-jax:latest \
    pytest -m jax --run-nightly

# All backends (full image)
docker run --rm -v $PWD:/workspace -w /workspace \
    ghcr.io/ericchansen/q2mm/ci-full:latest \
    pytest --run-nightly
```

On Windows (PowerShell), replace `$PWD` with `${PWD}`.

Images are built from `.github/docker/Dockerfile` using environment files in
`.github/envs/`. To rebuild locally from the repository root:

```bash
docker build -f .github/docker/Dockerfile --build-arg ENV_FILE=openmm.yml -t q2mm-openmm .
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Install pre-commit hooks so formatting and lint errors are caught before they
reach CI:

```bash
# Option A: prek (fast, Rust-based — https://github.com/j178/prek)
prek install

# Option B: Python pre-commit framework
pip install pre-commit
pre-commit install
```

Either tool reads the same `.pre-commit-config.yaml` in the repo root. Once
installed, hooks run automatically on every `git commit`.

The ruff hook auto-fixes simple lint issues during commit. When fixes are
applied, the hook exits with an error so you can review the changes. Stage the
updated files with `git add` and commit again.

To check the entire codebase manually:

```bash
ruff check q2mm/ test/ scripts/
ruff format --check q2mm test scripts examples
```

## Benchmarks

When running benchmarks with `q2mm-benchmark`, the runner **always** saves
its output — every requested candidate is written to `<output>/candidates/`
and each accepted candidate is promoted to `<output>/accepted/` and
`<output>/forcefields/`.  Benchmark runs can take minutes to hours depending
on the system and optimizer.  With SciPy L-BFGS-B + the JAX objective
executor on GPU, all 5 TS systems complete in ~15 minutes total (dominated
by JIT compilation, not optimization).  Results are not reproducible
bit-for-bit due to floating-point reduction-order differences between
devices.

```bash
# Convergence run for one system (writes candidate + promoted artifacts)
q2mm-benchmark single --system rh-enamide --backend jax --optimizer scipy-lbfgsb-jax \
  --workflow method-e2 --output results/rh-enamide

# Convergence batch across all systems
q2mm-benchmark batch --output results

# Backend x form x optimizer matrix for one system
q2mm-benchmark matrix --system rh-enamide --backend jax --optimizer scipy-lbfgsb \
  --output results/rh-enamide
```

**Rules:**

1. **Output is always persisted** — there is no `--no-save`; pass `--output`
   to choose the location (defaults to `./results`).
2. **Commit output force fields** to the separate `q2mm-data` repo under
   `benchmarks/<system>/forcefields/` so they are tracked in version control.
3. **Commit result JSON files** to `q2mm-data/benchmarks/<system>/` for
   reproducibility and future analysis.
4. If running GPU vs CPU comparisons, run them **sequentially** on an idle
   system — parallel benchmark runs produce invalid timing data.

## Submitting Changes

1. Fork and create a feature branch from `master`
2. Make your changes with clear, focused commits
3. Ensure `ruff check` and `ruff format --check` pass
4. Ensure `pytest` passes (at minimum the core tier; `--run-integration` recommended)
5. Open a pull request against `master`

## CI Pipeline

Pull requests trigger CI automatically across six jobs:

- **Lint** — ruff check + format + dependency parity (`check_env_dep_parity.py`)
- **Core** — pure-Python tests on Python 3.10–3.13 (no backends)
- **OpenMM** — OpenMM backend tests in Docker
- **Tinker** — Tinker backend tests in Docker
- **JAX** — JAX backend tests in Docker
- **JAX-MD** — JAX-MD backend tests in Docker

All jobs must pass before merging.

## Extending Q2MM

### Adding a New Backend

To support a new MM backend:

1. **Implement the prepared-session backend contract** in
   `q2mm.backends.contracts`: the backend exposes `info` and
   `prepare(PreparationRequest(...))`, and each prepared session handles typed
   requests such as `EnergyRequest`, `HessianRequest`, and `FrequencyRequest`.
2. **Handle unit conversion** between canonical units (kcal/mol, Å,
   hartree/bohr² as appropriate) and whatever the underlying engine expects
   internally.
3. **Declare capabilities** in the backend `BackendInfo` so objective executors
   can fail clearly when a requested operation is unsupported.
4. **Register the backend** in `q2mm.backends.registry` if it should be loaded
   by name.
5. **Add tests** with the appropriate backend marker (e.g., `@pytest.mark.mybackend`).

Optimizers consume objective executors, so a backend that satisfies the typed
contract works with `ObjectivePlan`, `PythonObjectiveExecutor`, and any
optimizer using `optimize(evaluator, space)`.

### Adding a New Force Field Format

To support a new file format:

1. **Add a module** in `q2mm/io/` (e.g., `q2mm/io/charmm.py`) with a loader
   (e.g., `load_charmm_rtf()`) that reads the file and returns a `ForceField`
   with parameters in canonical units.
2. **Add a saver** in the same module (e.g., `save_charmm_rtf()`) that converts
   canonical → format units on write.
3. **Re-export** the public functions from `q2mm/io/__init__.py`.
4. **Add convenience methods** on `ForceField` (e.g., `from_charmm_rtf()`,
   `to_charmm_rtf()`).
