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

Tests are organized into speed tiers so you can iterate quickly:

```bash
pytest                     # fast only (~13s)
pytest --run-medium        # fast + medium (~49s)
pytest --run-slow          # everything (~330s)
```

### Backend markers

Tests requiring specific backends are auto-skipped when the dependency is
missing. Use `-m` to filter:

```bash
pytest -m openmm           # only OpenMM tests
pytest -m tinker           # only Tinker tests
pytest -m "not tinker"     # skip Tinker tests
```

Available markers: `openmm`, `tinker`, `jax`, `psi4`.

### Running backend tests with Docker

If you don't have OpenMM, Tinker, or other backends installed locally, you can
use the pre-built CI images from GitHub Container Registry:

```bash
# OpenMM tests
docker run --rm -v $PWD:/workspace -w /workspace \
    ghcr.io/ericchansen/q2mm/ci-openmm:latest \
    pytest -m openmm --run-slow

# Tinker tests
docker run --rm -v $PWD:/workspace -w /workspace \
    ghcr.io/ericchansen/q2mm/ci-tinker:latest \
    pytest -m tinker --run-slow

# JAX tests
docker run --rm -v $PWD:/workspace -w /workspace \
    ghcr.io/ericchansen/q2mm/ci-jax:latest \
    pytest -m jax --run-slow

# All backends (full image)
docker run --rm -v $PWD:/workspace -w /workspace \
    ghcr.io/ericchansen/q2mm/ci-full:latest \
    pytest --run-slow
```

On Windows (PowerShell), replace `$PWD` with `${PWD}`.

Images are built from `.github/docker/Dockerfile` using environment files in
`.github/envs/`. To rebuild locally from the repository root:

```bash
docker build -f .github/docker/Dockerfile --build-arg ENV_FILE=openmm.yml -t q2mm-openmm .
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Pre-commit hooks run automatically if you have them installed:

```bash
ruff check q2mm/ test/ scripts/
ruff format q2mm/ test/ scripts/
```

## Submitting Changes

1. Fork and create a feature branch from `master`
2. Make your changes with clear, focused commits
3. Ensure `ruff check` and `ruff format --check` pass
4. Ensure `pytest` passes (at minimum the fast tier)
5. Open a pull request against `master`

## CI Pipeline

Pull requests trigger two CI tiers automatically:

- **Fast tier** — lint on Python 3.12 + pure-Python tests on Python 3.10–3.13 (no backends)
- **Backend tier** — OpenMM, Tinker, JAX, and Psi4 tests in Docker containers

Both tiers must pass before merging.
