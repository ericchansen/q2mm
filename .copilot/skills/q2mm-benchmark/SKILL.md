---
name: q2mm-benchmark
description: 'MANDATORY before launching any q2mm batch optimization run >30 minutes (benchmark.py, multi-system convergence runs, from-QFUERZA runs, etc.). Forces success-spec write, optimizer config sanity check, FIRST-system audit gate before launching the rest, and post-batch validation. Use when user says "run benchmarks", "regenerate convergence results", "run all systems", "QFUERZA recovery", or any variant of launching a multi-system q2mm optimization batch.'
license: MIT
allowed-tools: Bash, Read, Edit, Grep, Glob
---

# q2mm Benchmark Pre-Flight & Audit

A multi-hour GPU benchmark batch that produces useless results is worse than one that fails fast. This skill exists because a previous agent shipped a flawed batch (5 systems, ~4 GPU-hours) where the optimizer exited at `n_evaluations=2` on every system without actually optimizing, and then wrote a PR comparing the wrong metric. Don't repeat that.

**The cardinal rule**: AUDIT THE FIRST SYSTEM BEFORE LAUNCHING THE REST.

## Step 1 — Write the success spec (mandatory)

Before any other work, write a paragraph in the session plan answering:

1. **What is the user's literal question?** Quote it verbatim.
2. **What deliverable artifacts will exist when this is done?** Be specific: "a markdown page with three tables: (a) per-system R² of q2mm-optimized FF vs published-paper R², (b) per-parameter abs deviation tables, (c) physical-chemistry walkthrough of top-5 deviations per system."
3. **What numerical results count as 'success'?** Examples: "QFUERZA-start final OF within 20% of published-start final OF on at least 3/5 systems" or "per-param mean abs deviation < 10% on bond_eq."
4. **What would make me declare the batch failed and stop?** Examples: "All systems exit at `n_evals<=2`" or "R² is negative on the optimized FF."

If you cannot answer (1)–(4) in five sentences, stop and ask the user to clarify. Do not launch the batch.

## Step 2 — Sanity-check the optimizer config

Read these defaults in the q2mm source and confirm they're appropriate for the starting point:

### Bounds (`q2mm/models/forcefield.py` → `DEFAULT_BOUNDS`)

The defaults are **physical sanity bounds**, not "stay near starting FF":
- `bond_k: (-3600, 3600)` kcal/mol/Å²
- `angle_k: (-720, 720)` kcal/mol/rad²
- `bond_eq: (0.5, 3.0)` Å
- `angle_eq: (30, 180)` deg

For **canonical-default QFUERZA-start** runs (the default; or any from-poor-start run), sanity bounds let L-BFGS-B escape the QFUERZA basin into a worse local minimum. Use fractional bounds around the starting value:
- Typical: `bounds_fraction_fc=0.20`, `bounds_fraction_eq=0.05`
- Fragile systems (heck-relay): `bounds_fraction_fc=0.05`, `bounds_fraction_eq=0.02`

For **publication-baseline (`--starting-point published`)** runs, sanity bounds are usually fine — the starting FF is already in the published basin.

Pass via `--fc-fraction` / `--eq-fraction` CLI flags on `scripts/benchmark.py`.

### Convergence tolerance (`scipy_opt.py` → `_run_minimize`)

The script default L-BFGS-B `ftol=1e-8` is loose for from-poor-start runs — `nfev` will often be ≤ 5 with no real optimization. Override with `--ftol 1e-12` (or tighter) for any run where you actually want the optimizer to work.

### Executor-ratio gate (`benchmark_runner.py` → executor-ratio check)

For TS systems with a poor starting FF, the JAX/Python executor ratio can be 0.1–0.4 or even diverge to 1e74 (heck-relay from QFUERZA). The default executor-ratio check rejects these. Two options:
- `--executor-ratio-tol -1` to bypass entirely (use for from-QFUERZA TS runs)
- Document the explosion honestly in the analysis instead of pretending it didn't happen

## Step 3 — Pre-flight checklist

Walk through this LITERALLY before running. Each item must be checked.

- [ ] Goal restated as a success spec (Step 1)
- [ ] Bounds strategy chosen and matches starting-point quality (Step 2)
- [ ] `ftol` / `gtol` tight enough for real optimization
- [ ] Per-system overrides documented if any system needs special handling
- [ ] GPU verified: `python -c "import jax; print(jax.devices())"` shows CudaDevice
- [ ] Output directory chosen (`q2mm-data/benchmarks/<system>/convergence/` for the canonical QFUERZA-start default; `q2mm-data/benchmarks/<system>/from-published/` for opt-in publication-baseline runs)
- [ ] PYTHONPATH set if running from a worktree (editable install points to master)

## Step 4 — Run the FIRST system in isolation

Do NOT launch all systems in a batch. Run **only the first system**:

```bash
# Canonical default is --starting-point qfuerza; pass --starting-point
# published only when reproducing publication-baseline benchmarks.
PYTHONPATH=/path/to/worktree python scripts/benchmark.py \
    --system <first-system> \
    --ftol 1e-12 \
    --fc-fraction 0.20 \
    --eq-fraction 0.05 \
    --executor-ratio-tol <value> \
    --output-dir /path/to/q2mm-data/benchmarks
```

## Step 5 — AUDIT GATE (do not skip)

After the first system completes, inspect `<output-dir>/validation_results.json`:

```python
import json
with open("<first-system>/convergence/validation_results.json") as f:
    r = json.load(f)["result"]
print("n_evaluations:", r["n_evaluations"])
print("n_iterations:", r["n_iterations"])
print("real OF:      ", r["initial_obj_score"], "→", r["final_obj_score"])
print("improvement%: ", r["improvement_pct"])
print("executor ratio:", r["executor_ratio"])
print("Seminario R²: ", r["seminario"])
print("Optimized R²: ", r["optimized"])
```

**Pass criteria** (all must hold or you must explicitly justify deviation):
- `n_evaluations > 5` (the optimizer actually evaluated the gradient)
- `n_iterations > 3` (it took real steps)
- `|improvement_pct| > 1` (the real OF actually changed)
- Optimized R² ≥ Seminario R² on each metric (the run didn't make things worse)

**Fail criteria** (stop immediately if any holds):
- `n_evaluations ≤ 2` AND `|improvement_pct| < 1` → **optimizer did not optimize**. Likely `ftol` too loose or bounds too wide. Do NOT launch the remaining systems. Diagnose and re-run.
- `executor_ratio > 100` AND `improvement_pct < 0` → JAX objective executor diverged AND the optimizer made the FF worse. Diagnose: tighter bounds, FC clamping, different starting point.
- Optimized R² < Seminario R² → the run degraded the FF. Diagnose before continuing.

## Step 6 — Launch the rest (only if Step 5 passes)

Launch the remaining systems, one at a time or in a small batch. Continue to spot-check intermediate results; if a system shows the same `nfev<=2` failure mode that didn't appear in the first system, stop and diagnose.

## Step 7 — Post-batch validation

Before writing the analysis doc:
- All systems have `n_evaluations > 5` (or each exception is documented with a chemical/physical reason)
- All systems have `validation_results.json` with full provenance
- Spot-check at least two `validation_results.json` files for sanity

If a benchmark batch ends with multiple systems exiting at `n_evals=2`, that's a silent protocol failure — even if all the JSON files were written.

## Quick reference: things that should make you stop and ask

- "How do I know if the optimizer actually optimized?" → check `n_evaluations` and real OF delta
- "Should I use sanity bounds or fractional bounds?" → fractional for the canonical QFUERZA-start default; sanity is fine for `--starting-point published` runs
- "Why does `nfev=2` happen on every system?" → default `ftol` is 1e-8, way too loose for from-poor-start; use `--ftol 1e-12`
- "Heck-relay's executor ratio is 1e74, is that OK?" → no, the JAX objective executor exploded; document honestly and consider tighter bounds or FF pre-conditioning
- "Should I just bypass the executor-ratio gate with `--executor-ratio-tol -1`?" → only if you understand why it's failing; the gate exists for a reason

## Anti-patterns to refuse

- Launching all 5 systems in a batch without auditing the first one
- Shipping results where `n_evaluations <= 2` on every system as if optimization happened
- Comparing only `final_obj_score` when the user asked about parameter values or R²
- Bypassing the executor-ratio gate with `--executor-ratio-tol -1` without diagnosing why it's failing
- Writing the analysis doc before re-reading the user's literal question
