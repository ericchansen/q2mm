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

### Key Papers — Read Before Touching Methodology

Q2MM has a specific scientific lineage.  Before changing anything in
`q2mm/models/seminario.py`, `q2mm/models/forcefield.py`, the
parametrization loaders in `q2mm/benchmarks/systems/`, or the
optimizers in `q2mm/optimizers/`, look at the right paper.  Agents have
repeatedly cited the wrong reference here.

| Paper | DOI | What it governs |
|-------|-----|-----------------|
| **Farrugia, Helquist, Norrby & Wiest 2025** — "Rapid FF Generation via Hessian-Informed Initial Parameters and Automated Refinement", *J. Chem. Theory Comput.* **22**, 469. | [10.1021/acs.jctc.5c01751](https://doi.org/10.1021/acs.jctc.5c01751) | **QFUERZA** — the methodology of record for `q2mm/models/seminario.py`. Defines the FUERZA + H-angle-substitution variant we ship; documents that torsions are intentionally zeroed at initial-parameter time; explains the "mixing literature + custom params" workflow our loaders implement. |
| Seminario 1996 — "Calculation of intramolecular force fields from second-derivative tensors", *Int. J. Quantum Chem.* **60**, 1271. | [10.1002/(SICI)1097-461X(1996)60:7%3C1271::AID-QUA8%3E3.0.CO;2-W](https://doi.org/10.1002/(SICI)1097-461X(1996)60:7%3C1271::AID-QUA8%3E3.0.CO;2-W) | Original **FUERZA** projection.  Background only; QFUERZA above supersedes it for us. |
| Limé & Norrby 2014 (vol. 36, 2015) — "Improving the Q2MM method for transition state force field modeling", *J. Comput. Chem.* **36**, 244–250. | [10.1002/jcc.23797](https://doi.org/10.1002/jcc.23797) | TS Hessian inversion (`invert_ts_curvature=True`) and Method E2.  See the "TS Hessian Inversion" warning below. |
| Donoghue, Helquist, Norrby & Wiest 2008 — *J. Chem. Theory Comput.* **4**, 1313. | [10.1021/ct800132a](https://doi.org/10.1021/ct800132a) | Rh-enamide TSFF reference paper.  Governs `examples/rh-enamide/`, `load_rh_enamide`, and the 9-molecule benchmark system. |
| Rosales, Helquist, Norrby & Wiest 2020 — *J. Am. Chem. Soc.* **142**, 9700. | [10.1021/jacs.0c01979](https://doi.org/10.1021/jacs.0c01979) | Heck-relay TSFF reference paper.  Governs `load_heck_relay` and the 23-molecule benchmark system. |
| Wahlers, *Ph.D. Dissertation*, University of Notre Dame, 2022. | (dissertation, no DOI) | pd-allyl, pd 1,4-conjugate-addition, rh 1,4-conjugate-addition TSFFs.  Governs `load_pd_allyl`, `load_pd_conjugate`, `load_rh_conjugate`. |

**If you propose changes to the parametrization or loader code without
having consulted Farrugia 2025, stop and read it first.** Many of the
loader-API decisions (frozen params, OPT-only re-estimation, the
distinction between literature and custom parameter values) are
deliberate implementations of the QFUERZA workflow and will not make
sense otherwise.

### TS Hessian Inversion — DO NOT GET THIS WRONG

Q2MM handles **transition state (TS) systems** by inverting the QM Hessian's
negative eigenvalues before Seminario projection (`invert_ts_curvature=True`).
This produces a force field with **all-positive force constants** where the TS
geometry is a **proper minimum** on the MM potential energy surface — not a
saddle point.

**Consequences that agents repeatedly get wrong:**

1. **Unconstrained geometry relaxation IS correct for TS systems.**
   `_relax_coords()` uses unconstrained L-BFGS to find the MM minimum.  After
   Hessian inversion, that minimum should be near the QM saddle-point geometry.
   Do NOT claim that "relaxation is wrong for TS" or "TS structures sit at
   saddle points on the MM PES" — that is false after inversion.

2. **When relaxation diverges, the problem is the starting FF, not the
   method.**  If Seminario produces deeply negative R² (e.g., heck-relay
   bond_length R² in the tens of negatives), the force constants are so
   wrong that the MM PES has its minimum far from the QM geometry.  The
   fix is to improve the starting FF, not to change the relaxation
   approach.

3. **Reference:** Limé & Norrby, *J. Comput. Chem.* **2014** (vol. **36**, 244–250, 2015). [10.1002/jcc.23797](https://doi.org/10.1002/jcc.23797)

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

### Types (mypy)

```bash
python -m mypy q2mm/
```

mypy is enforced repo-wide in the CI Lint job via a grandfathered ratchet
(see the `[[tool.mypy.overrides]]` block in `pyproject.toml`): the modules
listed there with pre-existing type errors are exempt, so any **new** module —
or any module removed from that grandfather list — must be mypy-clean.
`[tool.mypy]` pins `python_version = "3.12"` (the CI Lint job's Python) as a
single source of truth, so the command above matches CI with no extra flags.
3.12 is required because numpy ≥2.5 ships PEP 695 `type` aliases in its stubs
that mypy cannot parse under a 3.10 target; the ≥3.10 runtime floor is still
enforced by ruff (`target-version=py310`) and the 3.10 test-core matrix.

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

### Docs (properdocs)

```bash
uv sync --extra docs                                   # install docs deps once
uv run properdocs serve -f properdocs.yml              # live preview
uv run properdocs build -f properdocs.yml              # static build to site/
```

> ⚠️ **Always use `uv run properdocs`, not a globally-installed one.** A
> standalone install at `~/.local/bin/properdocs` may be missing transitive
> deps (notably `yaml_env_tag` → `pyyaml-env-tag`) and will fail on import.
> The `docs` extra pulls the full toolchain (`properdocs`, `mkdocs-material`,
> `mkdocstrings[python]`, `mkdocs-gen-files`, `mkdocs-literate-nav`,
> `mkdocs-section-index`, and transitive deps).

### Backend Tests (require Docker)

```bash
scripts/ci_local.sh --all
```

This runs the full CI matrix locally inside Docker containers.

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


---

## 6. Benchmark Runbook

1. **Verify GPU platform first** — see §4. No exceptions.
2. **Use WSL2** for all GPU benchmarks.
3. **Output is always persisted** — the `q2mm-benchmark` runner has no
   `--no-save`; every candidate lands under `<output>/candidates/` and each
   accepted candidate is promoted to `<output>/accepted/` and
   `<output>/forcefields/`. Pass `--output` to choose the location.
4. **Save outputs** to a local results directory such as `results/<system>/`
   (e.g., `results/ch3f/`, `results/rh-enamide/`). Archive the canonical
   benchmark artifacts in [`ericchansen/q2mm-data`](https://github.com/ericchansen/q2mm-data)
   under `benchmarks/<system>/`. Never create one-off or timestamped
   directories — keep one canonical location per system.
5. **Benchmark data is tracked in git** — commit benchmark artifacts to the
   separate [`ericchansen/q2mm-data`](https://github.com/ericchansen/q2mm-data)
   repo, not this code repo. Any data referenced in documentation **must** be
   tracked there. If it's not tracked, it doesn't exist for publication
   purposes.
6. **Run sequentially on an idle system** for consistent timing.

### Reference Data and Objective Function

For **publication reproduction** benchmarks (TS systems from the Q2MM
literature), use `ObservationSet.from_molecules()` (`q2mm.models.observations`)
to build the objective. This auto-populates bond lengths, bond angles, and
Hessian eigenmatrix references from the QM training structures — matching
the multi-target penalty function used in the published papers (Donoghue
2008, Rosales 2020, Wahlers 2021/2022).

`_build_frequency_reference()` builds frequency-only references. This is
appropriate for the CH₃F full-matrix benchmark and ground-state force
fields, but **not** for TS publication reproduction. The papers use a
multi-target penalty (geometry + Hessian + charges + energies), not
frequency RMSD.

### Starting Point (`starting_point` kwarg / `--starting-point` CLI flag)

`q2mm.benchmarks.systems.load_system()` and the `q2mm-benchmark` runner
accept a `starting_point` of either `"qfuerza"` (canonical default) or
`"published"`:

- **`"qfuerza"` (default, Farrugia 2025)** — the FF skeleton (atom types,
  OPT-substructure topology, frozen/active partition, vdW, stretch-bend)
  comes from the literature `.fld` file; QFUERZA fills in the bond and
  angle force constants and equilibria from the QM Hessian. This is the
  standard QFUERZA workflow: the chemist provides the skeleton (no tool
  can automate the decisions of which atom types to use or which
  substructure rows need OPT parameters); QFUERZA fills in the
  Hessian-derivable scalars.
- **`"published"`** — the literature OPT values are retained verbatim;
  no QFUERZA overwrite. Pass this to reproduce historical
  publication-baseline runs.

The runner is invoked as `q2mm-benchmark single`/`batch`/`matrix` with the
same `--starting-point` flag; the `RunProfile.starting_point` field records
which path a run used, and the resolved run provenance embeds it.

### TS Curvature Inversion

All TS system loaders **must** pass `invert_ts_curvature=True` when
building the QFUERZA starting force field (`q2mm.models.seminario.qfuerza_fresh`/
`qfuerza_into`). Without this, the reaction-coordinate eigenvalue stays
negative, producing a saddle-point force field where geometry
minimization blows up. If you see "negative FC (TS reaction
coordinate?)" warnings, TS inversion is missing.

### Frozen Parameters

Published papers only optimize the OPT substructure parameters near
the metal center (~182–488 params), not the full composed force field
(~2,700–3,200). Use `q2mm.models.parameters.opt_substructure_membership(ff, opt_ff)`
+ `ActiveParameterSpace.from_membership(layout, ff, membership)` to
derive which base-FF parameters are frozen — frozen/active state lives
entirely in `ActiveParameterSpace`, never on the (immutable) `ForceField`
itself. All three optimizers (SciPy, Optax, JaxOpt) take this
`ActiveParameterSpace` as their required second `optimize()` argument
and respect `space.n_active`.

### Objective Executors and Gradient Selection

q2mm separates the immutable fitting plan from the backend-specific executor:
build an `ObjectivePlan`, attach either a `PythonObjectiveExecutor` or a
`JaxObjectiveExecutor`, then pass that evaluator plus the
`ActiveParameterSpace` to `optimizer.optimize(evaluator, space)`. Gradient
selection is explicit — there is no `jac` auto-detection and no silent
finite-difference fallback.

| Pipeline | Executor and optimizer use | Geometry gradients | When to use |
|----------|----------------------------|:------------------:|-------------|
| **SciPy + JAX objective executor** (recommended) | `JaxObjectiveExecutor(plan, JaxBackend(), ff)` with `ScipyOptimizer(method="L-BFGS-B").optimize(evaluator, space)` | Analytical (per-case JIT) | **All TS systems** — fastest, most robust |
| **Python executor, scalar-only** | `PythonObjectiveExecutor(plan, backend, ff, gradient_mode=GradientMode.NONE)` with SciPy's internal finite differences | SciPy FD (slow) | Frequency-only objectives, small systems, non-JAX backends |
| **Python executor, explicit FD** | `PythonObjectiveExecutor(..., gradient_mode=GradientMode.FINITE_DIFFERENCE)` | Executor FD (slow) | When FD evaluations must be counted by the executor |
| **JAX-native optimizer path** | `JaxObjectiveExecutor` with JaxOpt or Optax | Analytical (implicit diff / autodiff) | ⚠️ See jaxopt warning below |

`JaxObjectiveExecutor` compiles each case's loss function independently
(per-case JIT split), then dispatches them from Python and sums — no single
XLA program contains all molecules. This prevents compilation OOM on
multi-molecule systems.

- **SciPy + JAX objective executor** (recommended): build
  `JaxObjectiveExecutor(plan, backend, starting_ff)` yourself and pass it to
  `ScipyOptimizer(method="L-BFGS-B").optimize(evaluator, space)`. The
  optimizer sees `GradientMode.ANALYTICAL` and feeds analytical gradients to
  `scipy.optimize.minimize(..., jac=True)`.
- **Python executor**: build `PythonObjectiveExecutor(plan, backend,
  starting_ff, gradient_mode=GradientMode.NONE)` to let SciPy compute internal
  finite differences, or choose `GradientMode.FINITE_DIFFERENCE` for
  executor-owned finite differences.
- **JaxOpt** uses `jit=False` + `value_and_grad=True` so
  `solver.run()` executes a Python while-loop, dispatching
  per-molecule compiled functions at each step.
- **Optax** calls `JaxObjectiveExecutor.value_and_grad_jax()` directly
  (Python dispatch) instead of wrapping in `jax.jit(jax.value_and_grad(...))`.

⚠️ **jaxopt ≥ 0.8.5 is not recommended for multi-molecule systems.**
The default `linesearch="zoom"` triggers 30–60 minutes of additional
XLA compilation beyond the JAX executor JIT phase. This makes jaxopt
impractical for convergence runs. Use `scipy-lbfgsb-jax` instead —
it completes in seconds post-JIT.

### Expected Runtimes

Runtimes on RTX 5090 GPU (WSL2). Each system's time is JIT + optimization.

| Benchmark | JIT | Optimization | Total |
|-----------|-----|:------------:|-------|
| **SciPy L-BFGS-B + JAX objective executor (GPU, recommended)** |||
| Rh-enamide (9 mol, 182 params) | 95 s | 6 s | ~2 min |
| Heck relay (23 mol, 462 params) | 249 s | 18 s | ~5 min |
| Pd-allyl (21 mol, 482 params) | 227 s | 13 s | ~4 min |
| Pd 1,4-conjugate (10 mol, 340 params) | 120 s | 9 s | ~2 min |
| Rh 1,4-conjugate (10 mol, 488 params) | 122 s | 10 s | ~2 min |
| **All 5 systems sequentially** | — | — | **~15 min** |
| **jaxopt L-BFGS (DO NOT USE)** | 95–250 s | 30–60 min zoom linesearch JIT | hours |
| **OpenMM OpenCL** | — | — | **DO NOT USE** — 14% GPU utilization |

---

## 7. Diagnostic Commands

```bash
# Check OpenMM platforms (must show CUDA for GPU work)
python -c "import openmm; [print(openmm.Platform.getPlatform(i).getName()) for i in range(openmm.Platform.getNumPlatforms())]"

# Check JAX GPU (must show CudaDevice)
python -c "import jax; print(jax.devices())"

# Check GPU utilization
nvidia-smi

# Run core tests (no backends)
python -m pytest test/ -x -q -m "not (openmm or tinker or jax or jax_md or psi4)"

# Run integration tests (local only — too slow for GitHub-hosted CI)
python -m pytest --run-integration -q

# Run validation tests (local only — not in CI)
python -m pytest --run-validation -q

# Run nightly tests (local only — optimizer loops, full loops, slow)
python -m pytest --run-nightly -q

# Run lint + format checks
python -m ruff check q2mm/ test/ scripts/
python -m ruff format --check q2mm test scripts examples

# Generate golden fixture (opt-in, requires --run-validation)
Q2MM_UPDATE_GOLDEN=1 python -m pytest test/integration/test_published_ff_validation.py --run-validation -v
```

---

## 8. Publication Metadata

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

## 9. Common Pitfalls

| Pitfall | What Happens | Fix |
|---------|-------------|-----|
| **Duplicate directories** | Data gets fragmented, docs reference wrong source, context is lost | Check if a canonical location exists before creating anything new (§2) |
| **Untracked data in docs** | Numbers in docs can't be verified from the repo | Commit all data that documentation references (§2) |
| **One-off / timestamped dirs** | `grad_simp_jax_fix_test/`, `results_2026-04-03/` — impossible to find later | Use the canonical directory; delete one-offs immediately after merging |
| **OpenCL ≠ CUDA** | Benchmark shows `OpenMM (OpenCL)` — 14% GPU utilization, hours wasted | Install `openmm-cuda-12` or use WSL2 |
| **JAX on Windows** | JAX CPU works but JAX CUDA is excluded in `pyproject.toml` | Use WSL2 for GPU |
| **Assuming output is optional** | Expecting a `--no-save` flag that no longer exists | The `q2mm-benchmark` runner always persists every candidate; choose the location with `--output` (default `./results`) |
| **Long benchmarks** | OpenMM L-BFGS-B can take hours | Check CPU/GPU utilization periodically with `nvidia-smi` |
| **Rewriting without full context** | Page rewrite introduces errors because not all data sources were checked | Gather ALL related dirs, issues, PRs, and prior work before rewriting (§2) |
| **Wrong publication year** | CrossRef has multiple date fields that disagree; using the wrong one corrupts citations | Always validate via Zotero MCP (§10) |
| **JIT-wrapping `JaxObjectiveExecutor` internals** | Wrapping the full multi-case objective in another `jax.jit()` or `jax.vmap()` re-inlines all molecules into one XLA program → compilation OOM | Use `JaxObjectiveExecutor.value_and_grad_jax()` (Python dispatch) or `JaxObjectiveExecutor.value_and_gradient()` instead. See §6 Objective Executors and Gradient Selection. |
| **Frequency-only refs for TS** | Misses geometry + eigenmatrix targets that the papers actually used → wrong optimization | Use `ObservationSet.from_molecules()` + `invert_ts_curvature=True` for TS systems. See §6 Reference Data. |
| **jaxopt zoom linesearch** | `jaxopt ≥ 0.8.5` LBFGS default `linesearch="zoom"` triggers 30–60 min extra XLA compilation post-JIT | Use `scipy-lbfgsb-jax` instead — same L-BFGS-B algorithm, completes in seconds post-JIT |
| **TS executor-ratio gate** | Literature-scale TS systems can have JAX/Python executor ratios of 0.1–0.4 from poor starts; the optimizer no longer falls back silently | Pass `--executor-ratio-tol none` for documented TS runs when intentionally bypassing the gate. CLI key: `scipy-lbfgsb-jax` |
| **Heck relay bounds** | ±20% bounds cause 35–92% NaN rate due to fragile TS landscape with large negative FCs (−3753) | Use ±5% bounds for heck-relay specifically |
| **`n_iterations<=2` silent exit** | L-BFGS-B "converges" after 0–2 iterations with negligible objective-executor score change — the optimizer didn't optimize. Common with from-poor-start runs and loose `ftol`. | Tighten `--ftol` (e.g. `1e-12`); apply `--fc-fraction`/`--eq-fraction` to keep optimizer in starting basin; check the `executor_ratio`. The diagnostic warning in `scipy_opt._run_minimize` flags this. |
| **Default sanity bounds for from-poor-start runs** | `DEFAULT_BOUNDS` (bond_k ±3600, bond_eq 0.5–3.0 Å) let L-BFGS-B escape the QFUERZA / random-default starting basin → final FF unrelated to start | Use `ScipyOptimizer(fc_fraction=0.20, eq_fraction=0.05)` (or CLI flags `--fc-fraction --eq-fraction`) to bound each param to a ± fraction of its current value. |
| **Benchmark batch never optimized** | All systems exit at `nfev≤2` with no OF change but the batch reports "success" | The runner rejects no-progress candidates and `run_profiles` returns `outcome.ok is False` when optimizations ran but none was accepted (`q2mm/benchmarks/runner.py`). Re-tune `ftol` or bounds and re-run. See §11. |

---

## 10. Validation Data & External Datasets

### Data location

`validation/supporting-info/` (gitignored, ~1.9 GB) contains supporting data
from the Wahlers and Rosales Notre Dame dissertations: 996 Gaussian DFT logs,
12 published MM3* force fields, and 134 MacroModel structures across 10
reaction systems. See `validation/supporting-info/README.md` for provenance,
file inventories, and how to obtain the zip files.

### MM3 base force field

`validation/published_ffs/mm3_base.fld` is the **standard, unmodified MM3
force field** (1,850 lines). It provides backbone parameters (bonds, angles,
torsions, vdW) that all Q2MM systems build on.

- **Not committed** — the header says "Copyright Columbia University 1990 —
  All rights reserved", which does not grant redistribution rights.
- **Source:** download from [atlas-nano/ATLAS_toolkit](https://github.com/atlas-nano/ATLAS_toolkit)
  `ff/macromodel/mm3.fld` or extract from a MacroModel installation.
- **Identical** to the base section of `examples/rh-enamide/mm3.fld`
  (the first ~1,850 lines before Rh-specific additions).

### Force field composition

Some published FFs (Wahlers dissertation) are **standalone OPT-substructure-only
files** (71–238 lines) containing only custom parameters, not the full MM3
backbone. To build a complete, evaluable force field:

1. Start with `mm3_base.fld` (gitignored, see above for how to obtain)
2. Insert metal-specific vdW entries before "END OF NONBONDED INTERACTIONS"
3. Append the OPT substructure blocks from the standalone `.fld` file

Rosales FFs are already complete (full MM3 base + OPT substructures in one
file, ~2,000 lines) and do not need composition.

### Adding a new benchmark system

1. Write a `load()` function in a new module under
   `q2mm/benchmarks/systems/` (e.g. `q2mm/benchmarks/systems/my_system.py`)
   following the pattern of `rh_enamide.py` — load molecules, build an
   `OptimizationProblem` (via `q2mm.benchmarks.systems._shared`'s
   `assemble_published_case`/`assemble_qfuerza_fresh_case` helpers), and
   return a `BenchmarkCase`.
2. Register the key -> module mapping in `q2mm/benchmarks/systems/__init__.py`'s
   `_REGISTRY` dict.
3. Create a validation test under `test/integration/` following the pattern
   of `test_published_ff_validation.py` — load published FF via
   `q2mm.benchmarks.systems.load_system(...)`, evaluate with `JaxBackend`,
   compare QM vs MM, pin results in a golden fixture JSON.
4. Mark with `@pytest.mark.validation` so it runs with `--run-validation`.
5. If the system depends on external (gitignored) data, the loader must
   skip gracefully when files are absent.
6. Document in `validation/published_ffs/README.md`.

### Published FF validation status

See `validation/published_ffs/README.md` for the full table. As of April 2026:
- **1 system validated** (Rh-enamide)
- **4 systems ready to implement** (Heck, Pd-allyl, Pd 1,4-conj, Ferrocene)
  — QM data available from dissertation supporting info
- **3 systems blocked** (OsO₄, Ru ketone, Sulfone) — no QM training data

---

## 11. Benchmark Pre-Flight Checklist

> **Before launching any q2mm batch >30 min (e.g. `q2mm-benchmark batch`/`matrix` on >1 system, or any from-scratch FF generation), walk through every step below.**

Many hours of GPU time have been wasted on batches where the optimizer never actually optimized. The pattern is silent — scipy reports `success=True`, the runner writes its JSON, and the misleading result is only caught during post-hoc analysis. This checklist prevents that.

1. **Write a measurable success spec** in your plan/PR description before launching:
   - What metric defines "the optimizer worked"? (e.g. `n_iterations > 5 AND real-OF improvement > 10%`)
   - What does the comparison table look like? Mock the rows/columns now.
   - If the user asked a specific scientific question (e.g. "do params end up near published?"), restate it verbatim and map each metric back to the question.

2. **Sanity-check optimizer config** for the starting point:
   - Canonical default (QFUERZA-start, `starting_point="qfuerza"`): use `--ftol 1e-12` and `--fc-fraction 0.20 --eq-fraction 0.05` (or `--fc-fraction 0.05` for heck-relay).
   - Publication-baseline (`--starting-point published`): `ftol=1e-8`, sanity bounds → fine (starting FF is already in the published basin).
   - Always pass `--executor-ratio-tol none` for TS systems when intentionally bypassing the executor-ratio gate (ratios can be 0.1–0.4).

3. **Verify GPU + device** (§4 + §7): `python -c "import jax; print(jax.devices())"` → must show `CudaDevice`.

4. **Run the FIRST system alone.** Do NOT launch all systems sequentially in one call.

5. **AUDIT GATE — read the first candidate's persisted record (`<output>/candidates/<stem>.json`, its `summary`) before launching the rest:**
   - `n_iterations > 5` (not just `n_evaluations`; that counter can be misleading on the JAX executor path — see `scipy_opt._run_minimize`).
   - `|improvement_pct| > 1%` on the Python objective-executor score of record (the `improvement_pct` field, not `final_optimizer_score`).
   - `executor_ratio = initial_jax_score / initial_obj_score` in `[0.1, 10]` (or document why it's outside).
   - Per-category R² (`seminario` → `optimized`) improves on at least one of bond_length, bond_angle, eigenmatrix.
   - The candidate `status` is `accepted` (not `rejected`/`skipped`/`error`).
   - If ANY of these fail, **STOP**. Re-tune and re-run the single system. Do not launch the batch.

6. **Launch remaining systems.** Check `nvidia-smi` periodically (>50% utilization expected).

7. **Post-batch validation** — read every candidate record under `<output>/candidates/`, not just the console summary. `run_profiles` logs a batch-level failure and returns `outcome.ok is False` when optimizations ran but none was accepted; treat that as a hard failure even if individual files exist.

### Source-code safety nets

These exist to back up the checklist; do not rely on them alone:
- `q2mm/optimizers/scipy_opt.py::_run_minimize` — WARNING when `n_iterations<=2` and `|delta|/init<0.01`
- `q2mm/benchmarks/acceptance.py::NoProgressPolicy.made_progress` — the single no-progress decision that rejects a stalled candidate
- `q2mm/benchmarks/runner.py::run_profiles` — logs BATCH FAILURE and reports `outcome.ok is False` when optimizations ran but none was accepted
- `.copilot/skills/q2mm-benchmark/SKILL.md` — agent skill that walks through this checklist automatically before launching any batch
- `.copilot/skills/q2mm-analysis-design/SKILL.md` — agent skill that forces design-first analysis methodology before writing comparison docs

## 12. Backend Plugin Discovery (internal, unstable)

> ⚠️ **This is an internal, unstable API — not a public plugin API.** It is
> documented as internal until Milestone PR 3 and carries **no compatibility
> promise**. Do not describe it as a stable/public plugin API v1, and do not
> write a migration guide for it.

`q2mm.backends.registry` discovers backends lazily through one validator. Both
in-tree built-ins and out-of-tree plugins are declared as **JSON-safe manifest
mappings** (`api_version`/`name`/`role`/`capabilities`/`forms`/`factory`/`probe`)
and validated by `q2mm.backends.discovery.validate_manifest`, which produces a
`BackendDescriptor`. There is exactly one construction path — built-ins are not
privileged.

- **Entry-point group:** out-of-tree plugins advertise one entry point in the
  `q2mm.backends` group whose value targets a *lightweight descriptor module*
  (the manifest mapping or a zero-arg provider returning it). Enumeration and
  provider loading import **only** that descriptor module; the backend
  implementation is imported solely when the manifest's `factory` string is
  resolved by an explicit `load_backend`/`BackendDescriptor.load`.
- **Lazy + cached:** importing `q2mm.backends.registry` does not enumerate entry
  points or import descriptor/implementation modules. The first
  `descriptors`/`catalog`/`registered_backends`/`available_*`/`get_descriptor`/
  `load_backend` call builds one deterministic snapshot; `registry.refresh()`
  rebuilds it (tests, newly-installed plugins).
- **Failure isolation:** missing dependency, descriptor import error,
  incompatible API version, duplicate name, invalid descriptor/capability/form,
  and broken factory become typed `DiscoveryRecord`s (see
  `DiscoveryIssueKind`) and never hide a healthy built-in or external plugin.
  Built-in names win conflicts; two externals claiming one name are **all**
  rejected. A broken factory is discovered only on explicit load.
- **Fixture + tests:** the out-of-tree fixture lives at
  `test/fixtures/backend_plugin/` (repository test code — it must **never** ship
  in the `q2mm` wheel/sdist). `test/test_backend_discovery.py` covers the
  discovery/isolation/conformance matrix without `pip install`ing anything;
  `test/_conformance.py` executes only a backend's declared capabilities.
  `scripts/check_release_artifacts.py` performs the only real install and prints
  `external-plugin=ok`.
