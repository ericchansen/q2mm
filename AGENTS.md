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

> ⚠️ **This project is in alpha.** The rules below are non-negotiable.

---

## 2. Alpha Discipline

This is a pre-release codebase aiming for lean perfection. Zero tolerance for
bloat, duplication, or deprecated artifacts.

### Rules

1. **One canonical location for each kind of data.** Before creating a new
   directory or file, check if one already exists. If it does, use it.
2. **No duplicate directories.** If two directories serve the same purpose,
   merge them immediately. Do not leave both "until later."
3. **No deprecated artifacts.** If something is superseded, delete the old
   version in the same commit. Do not keep it "just in case."
4. **Every file earns its place.** If you can't explain why a file exists and
   what would break without it, it probably shouldn't be there.
5. **Data that is documented must be tracked.** If a number appears in
   documentation, the data backing it must be committed to the repo. Untracked
   data does not exist for publication purposes.
6. **Know before you write.** Before rewriting a page or creating a file, gather
   the full picture — check all related directories, issues, PRs, and prior
   work. A rewrite based on partial context will introduce errors.
7. **No one-off or timestamped directories.** Output goes to the canonical
   location. If the canonical location doesn't exist yet, create it with a
   permanent name.
8. **Every claim must be grounded in evidence.** Treat all documentation as
   if it will be published in a scientific journal. Every number needs a
   traceable source (a JSON file, a log, a paper). Do not embellish, glorify,
   or editorialize — a fork is a fork, not "a fork that preserves the earlier
   implementation." If you cannot cite evidence for a claim, do not make it.
9. **Every comparison claim must link to its substantiation.** In tables or
   lists that compare features, each row must link to the documentation page
   or section that provides the full details. A claim without a link to
   supporting detail has no value — the reader cannot verify it.
10. **Write for a first-year graduate student.** The target audience is a
    22-year-old who has never used Q2MM or force field parameterization tools.
    Every page must open with *what* this is and *why* anyone should care,
    before diving into details. Avoid jargon without explanation. If a page
    references concepts from other pages (e.g., "Check 1"), define them
    inline or link to the definition. Never assume the reader has context
    you haven't provided on the page itself.

## 3. Before Every Commit

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

By default, all tests run on **CPU only** — no GPU memory is allocated.
This prevents JAX CUDA initialization (~25 GiB VRAM) and OpenMM CUDA
platform selection from locking up the developer's machine.

To opt into GPU execution for benchmarks or GPU-specific tests:

```bash
python -m pytest --gpu                 # CLI flag
Q2MM_USE_GPU=1 python -m pytest        # environment variable
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

## 4. Platform Guide

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

## 5. Git Workflow

- **Never push directly to `main` or `master`** — always use a feature
  branch + PR.
- Branch naming: `<type>/<short-description>` (e.g., `feat/jax-optimizer`,
  `fix/openmm-parity`).
- Conventional commit prefixes: `feat`, `fix`, `docs`, `refactor`, `chore`,
  `test`, `ci`, `perf`.
- GPG signing is broken — see §3 above.

---

## 6. Benchmark Runbook

1. **Verify GPU platform first** — see §4. No exceptions.
2. **Use WSL2** for all GPU benchmarks.
3. **Never use `--no-save`** — always save results and force fields so they
   can be reviewed and compared.
4. **Save outputs** to `benchmarks/<system>/` (e.g.,
   `benchmarks/ch3f/`, `benchmarks/rh-enamide/`). Never create
   one-off or timestamped directories — keep one canonical location per system.
5. **Benchmark data is tracked in git** — `benchmarks/` is committed,
   not gitignored. Any data referenced in documentation **must** be in the
   repo. If it's not tracked, it doesn't exist for publication purposes.
6. **Run sequentially on an idle system** for consistent timing.

### Expected Runtimes

| Benchmark                         | Approximate Time |
|-----------------------------------|------------------|
| JAX CPU — Rh-enamide L-BFGS-B    | ~9 min           |
| JAX GPU — Rh-enamide L-BFGS-B    | ~6 min           |
| OpenMM CUDA — Rh-enamide         | Varies by optimizer |
| OpenMM OpenCL                     | **DO NOT USE** — 14% GPU utilization, hours of wasted compute |

---

## 7. Active Workstreams

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

## 8. Key Open Issues

| Issue  | Title                          | Status  | Next Action                              |
|--------|--------------------------------|---------|------------------------------------------|
| **#198** | Published validation roadmap | Active  | Umbrella tracker                         |
| **#197** | Check 1: OpenMM parity gap   | Blocked | Debug MM3 functional-form differences    |
| **#194** | Re-run GPU benchmarks        | Active  | Run with CUDA, save artifacts            |

---

## 9. Diagnostic Commands

```bash
# Check OpenMM platforms (must show CUDA for GPU work)
python -c "import openmm; [print(openmm.Platform.getPlatform(i).getName()) for i in range(openmm.Platform.getNumPlatforms())]"

# Check JAX GPU (must show CudaDevice)
python -c "import jax; print(jax.devices())"

# Check GPU utilization
nvidia-smi

# Run core tests (no backends)
python -m pytest test/ -x -q -m "not (openmm or tinker or jax or jax_md or psi4)"

# Run integration tests (backend contracts, parity)
python -m pytest --run-integration -q

# Run validation tests (Seminario parity, published FF eval)
python -m pytest --run-validation -q

# Run nightly tests (optimizer loops, full loops — slow)
python -m pytest --run-nightly -q

# Run lint + format checks
python -m ruff check q2mm/ test/ scripts/
python -m ruff format --check q2mm test scripts examples

# Generate golden fixture (opt-in, requires --run-validation)
Q2MM_UPDATE_GOLDEN=1 python -m pytest test/integration/test_published_ff_validation.py --run-validation -v
```

---

## 10. Publication Metadata

**Always validate publication years and metadata via Zotero MCP**, not
raw CrossRef API fields. CrossRef exposes multiple date fields
(`issued`, `published-print`, `published-online`, `created`) that can
disagree. The Zotero library is the authoritative source for citation
metadata in this project.

- Use `zotero_search_items` or `zotero_get_item_metadata` to look up
  papers.
- If a paper is not in Zotero, add it first (`zotero_add_by_doi`), then
  use the Zotero-resolved metadata.
- **Never use CrossRef `published-print` as the citation year** — use
  the `issued` date or, better, the year already recorded in Zotero.

---

## 11. Common Pitfalls

| Pitfall | What Happens | Fix |
|---------|-------------|-----|
| **Duplicate directories** | Data gets fragmented, docs reference wrong source, context is lost | Check if a canonical location exists before creating anything new (§2) |
| **Untracked data in docs** | Numbers in docs can't be verified from the repo | Commit all data that documentation references (§2) |
| **One-off / timestamped dirs** | `grad_simp_jax_fix_test/`, `results_2026-04-03/` — impossible to find later | Use the canonical directory; delete one-offs immediately after merging |
| **OpenCL ≠ CUDA** | Benchmark shows `OpenMM (OpenCL)` — 14% GPU utilization, hours wasted | Install `openmm-cuda-12` or use WSL2 |
| **JAX on Windows** | JAX CPU works but JAX CUDA is excluded in `pyproject.toml` | Use WSL2 for GPU |
| **`--no-save`** | Benchmark results and force fields are lost | Never use `--no-save` — always save artifacts |
| **Long benchmarks** | OpenMM L-BFGS-B can take hours | Check CPU/GPU utilization periodically with `nvidia-smi` |
| **GPG signing** | Commits fail with signing error | Always use `git -c commit.gpgsign=false commit` |
| **Rewriting without full context** | Page rewrite introduces errors because not all data sources were checked | Gather ALL related dirs, issues, PRs, and prior work before rewriting (§2) |
| **Wrong publication year** | CrossRef has multiple date fields that disagree; using the wrong one corrupts citations | Always validate via Zotero MCP (§10) |
