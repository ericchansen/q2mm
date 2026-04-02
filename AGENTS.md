# Agent Instructions

> **This is the primary reference for any AI agent working on this repo.**
> Read it fully before making changes. Everything here was learned the hard way.

---

## 1. Project Overview

q2mm is a modern Python rewrite of Q2MM (Quantum-guided Molecular Mechanics).
It optimizes molecular mechanics force field parameters to match quantum
mechanical reference data. The codebase supports multiple computational backends
(OpenMM, Tinker, JAX, JAX-MD, Psi4) and provides optimizers, objective
functions, and evaluation tools for force field development.

---

## 2. Before Every Commit

Run the **exact same** lint and format checks that CI runs. If either fails,
fix the issues before committing.

### Lint

```bash
python -m ruff check q2mm/ test/ scripts/
```

### Format

```bash
python -m ruff format --check q2mm test scripts examples
```

### Core Tests (no backends required)

```bash
python -m pytest test/ -x -q -m "not (openmm or tinker or jax or jax_md or psi4)"
```

### Backend Tests (require Docker)

```bash
scripts/ci_local.sh --all
```

This runs the full CI matrix locally inside Docker containers.

### GPG Signing

GPG signing is broken (expired key). **Always** use:

```bash
git -c commit.gpgsign=false commit
```

---

## 3. Platform Guide

> **This is where agents keep failing.** Read this section carefully.

### Windows (native) — good for development

- Editing, linting, formatting, and non-GPU tests all work.
- OpenMM **CPU** works.
- JAX **CPU** works.
- **JAX CUDA and JAX-MD are NOT available on Windows** — they are excluded in
  `pyproject.toml`. Do not attempt to install or use them.

### WSL2 Ubuntu — recommended for benchmarks and GPU work

- Full GPU stack is available: OpenMM CUDA, JAX CUDA (5.6× speedup), JAX-MD.
- All verified GPU benchmarks were run here.
- **To enter the WSL2 GPU environment:**

  ```bash
  wsl -d Ubuntu-24.04
  source /home/eric/repos/q2mm/.venv/bin/activate
  ```

### Verify GPU Before Running Benchmarks

**Always** run these checks before any benchmark or GPU-dependent work:

```bash
# Must show "CUDA" in the platform list
python -c "import openmm; [print(openmm.Platform.getPlatform(i).getName()) for i in range(openmm.Platform.getNumPlatforms())]"

# Must show CudaDevice (not CpuDevice)
python -c "import jax; print(jax.devices())"
```


> ⛔ **If OpenMM shows OpenCL instead of CUDA, STOP.** Do not run benchmarks on
> OpenCL. Install `openmm-cuda-12` or switch to WSL2.

---

## 4. Git Workflow

- **Never push directly to `main` or `master`** — always use a feature
  branch + PR.
- Branch naming: `<type>/<short-description>` (e.g., `feat/jax-optimizer`,
  `fix/openmm-parity`).
- Conventional commit prefixes: `feat`, `fix`, `docs`, `refactor`, `chore`,
  `test`, `ci`, `perf`.
- GPG signing is broken — see §2 above.

---

## 5. Benchmark Runbook

1. **Verify GPU platform first** — see §3. No exceptions.
2. **Use WSL2** for all GPU benchmarks.
3. **Never use `--no-save`** — always save results and force fields so they
   can be reviewed and compared.
4. **Save outputs** to `benchmark_results/` or `benchmarks/`.
5. **Run sequentially on an idle system** for consistent timing.

### Expected Runtimes

| Benchmark                         | Approximate Time |
|-----------------------------------|------------------|
| JAX CPU — Rh-enamide L-BFGS-B    | ~9 min           |
| JAX GPU — Rh-enamide L-BFGS-B    | ~6 min           |
| OpenMM CUDA — Rh-enamide         | Varies by optimizer |
| OpenMM OpenCL                     | **DO NOT USE** — 14% GPU utilization, hours of wasted compute |

---

## 6. Active Workstreams

### Check 1 — Published Force Field Evaluation

Load published force fields, evaluate them with q2mm engines, and compare to
literature values. Rh-enamide is in progress; issue **#197** tracks the parity
gap between q2mm and published results. Golden fixture lives at
`test/fixtures/published_ff/`.

### Check 2 — Force Field Re-derivation

Re-derive published force fields from scratch using q2mm optimizers. **Not
started yet** — blocked on resolving Check 1 first.

### Validation Roadmap

Issue **#198** is the umbrella tracker for the overall published-validation
program.

### GPU Benchmarks

Issue **#194** tracks re-running benchmarks with CUDA and saving all artifacts.

---

## 7. Key Open Issues

| Issue  | Title                          | Status  | Next Action                              |
|--------|--------------------------------|---------|------------------------------------------|
| **#198** | Published validation roadmap | Active  | Umbrella tracker                         |
| **#197** | Check 1: OpenMM parity gap   | Blocked | Debug MM3 functional-form differences    |
| **#194** | Re-run GPU benchmarks        | Active  | Run with CUDA, save artifacts            |

---

## 8. Diagnostic Commands

```bash
# Check OpenMM platforms (must show CUDA for GPU work)
python -c "import openmm; [print(openmm.Platform.getPlatform(i).getName()) for i in range(openmm.Platform.getNumPlatforms())]"

# Check JAX GPU (must show CudaDevice)
python -c "import jax; print(jax.devices())"

# Check GPU utilization
nvidia-smi

# Run core tests (no backends)
python -m pytest test/ -x -q -m "not (openmm or tinker or jax or jax_md or psi4)"

# Run lint + format checks
python -m ruff check q2mm/ test/ scripts/
python -m ruff format --check q2mm test scripts examples

# Generate golden fixture (opt-in, slow)
Q2MM_UPDATE_GOLDEN=1 python -m pytest test/integration/test_published_ff_validation.py --run-slow -v
```

---

## 9. Common Pitfalls

| Pitfall | What Happens | Fix |
|---------|-------------|-----|
| **OpenCL ≠ CUDA** | Benchmark shows `OpenMM (OpenCL)` — 14% GPU utilization, hours wasted | Install `openmm-cuda-12` or use WSL2 |
| **JAX on Windows** | JAX CPU works but JAX CUDA is excluded in `pyproject.toml` | Use WSL2 for GPU |
| **`--no-save`** | Benchmark results and force fields are lost | Never use `--no-save` — always save artifacts |
| **Long benchmarks** | OpenMM L-BFGS-B can take hours | Check CPU/GPU utilization periodically with `nvidia-smi` |
| **GPG signing** | Commits fail with signing error | Always use `git -c commit.gpgsign=false commit` |
