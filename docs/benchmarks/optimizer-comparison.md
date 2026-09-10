# Optimizer Comparison

## What this page answers

This page compares q2mm's current production optimizer path on five
transition-state force-field systems from the Q2MM literature.  The question is
not "can we reproduce MacroModel MM3* exactly?"  The answer to that is no: the
published TSFFs were optimized under MacroModel-specific MM3* semantics, and
q2mm does not include a licensed MacroModel compatibility layer.  The question
here is narrower and testable:

> Given the published OPT-substructure parameters as a starting point, can
> q2mm's JAX backend and analytical-gradient optimizer reduce q2mm's own
> multi-target objective without corrupting the force field?

The historical records below report reductions for four of the five systems
and little movement for Pd-allyl. The displayed fields mix legacy sample
means and single-call endpoints, identified per row. Repeated samples do not
by themselves establish uncertainty for the independent Python objective
of record or prove that a starting point is a local minimum.

---

## Methodology

The historical runs use the following setup, with recorded exceptions
identified below:

- **Objective:** eigenmatrix-diagonal + geometry observations built by
  `ObservationSet.from_molecules()` from the QM structures/Hessians.
- **Parameter scope:** an `ActiveParameterSpace` keeps the base force field
  inactive while exposing only OPT-substructure parameters, matching the
  published Q2MM workflow.
- **Starting force field:** the literature OPT values are preserved as
  published (`starting_point="published"`).  The loader does not
  overwrite them with QFUERZA projections.  This page is the
  published-start baseline; for the canonical QFUERZA-start results
  (default since q2mm#290) see the
  [QFUERZA-recovery doc](qfuerza-recovery.md).
- **Optimizer:** SciPy L-BFGS-B driven by `JaxObjectiveExecutor`
  analytical gradients.
- **Gradient source:** the `scipy-lbfgsb-jax` CLI path builds the JAX
  executor explicitly when the JAX/Python executor ratio check is within the
  default ±15% band, or when that guard is explicitly bypassed.
- **Validation:** the Python executor supplies the independent before/after
  scores of record. Workflow repeats sample the configured optimization
  executor; for a JAX workflow those are JAX samples, not Python repeats.
  Their means and intervals must not be attached to the Python endpoint
  improvement as if both described the same series.

Current summaries name sample statistics `initial_optimizer_score_mean`,
`initial_optimizer_score_ci95`, `final_optimizer_score_mean`,
`final_optimizer_score_ci95`, `optimizer_improvement_pct_mean`, and
`optimizer_improvement_significant`. They also record
`optimizer_samples_executor`, both sample counts, and
`optimizer_sample_statistics_version=1`. The interval arithmetic and
acceptance policy are unchanged. `initial_obj_score`, `final_obj_score`,
and `improvement_pct` still describe the independent Python endpoints.
No statistics are emitted when either sample series is absent.

Multi-stage results retain their individual endpoint means, intervals,
sample counts, and raw samples, but omit paired optimizer-improvement and
significance fields. They instead record
`optimizer_sample_comparison_omitted="multiple_workflow_stages"`.
Method E2 can change the observation plan between rounds, so a ratio of
those sample means would mix optimization with an objective change.
The final optimizer/Python executor ratio is also omitted, with
`final_executor_ratio_omitted="multiple_workflow_stages"`, because its
numerator may use the modified plan while its denominator uses the original
plan of record. Publication audits receive an unavailable ratio rather
than evidence of agreement between different objectives.
Single-stage results, including Method E2 that stops after one round,
retain their paired statistics. Independent Python endpoint scores and
the acceptance policy are unchanged.

The raw JSON outputs and optimized force fields for these published-start
runs live in
[`ericchansen/q2mm-data/benchmarks/<system>/from-published/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks).
They include provenance such as q2mm git SHA, device, executor-ratio tolerance, and
run timestamp.  (Sibling `convergence/` directories hold the canonical
QFUERZA-start runs covered by the QFUERZA-recovery doc.)

---

## Executor-ratio gate

Before using JAX analytical gradients, q2mm compares the JAX executor value
with the Python executor value. Ratios inside the default `[0.85, 1.15]` band
are accepted; outside the band, the analytical surrogate is considered
unreliable for that parameter regime.

After the loader API refactor and the MM3 angle-gradient fix, every system in
this table is inside the default band. This numerical comparison does not
prove that a gate executed: the pinned Heck relay and Pd 1,4-conj runs record
`ratio_tol: null` and `ratio_status: "ok_bypassed"`.

| System | Mols | Active params | Executor ratio | Within default band |
|--------|:----:|:-------------:|:-----:|:----:|
| [Rh-enamide](../systems/rh-enamide.md) | 9 | 182 | 1.07 | ✓ |
| [Heck relay](../systems/heck-relay.md) | 23 | 462 | 1.085 | ✓ |
| [Pd-allyl](../systems/pd-allyl.md) | 21 | 482 | 1.091 | ✓ |
| [Pd 1,4-conj](../systems/pd-conjugate.md) | 10 | 340 | 0.985 | ✓ |
| [Rh 1,4-conj](../systems/rh-conjugate.md) | 10 | 488 | 0.996 | ✓ |

Two fixes changed the interpretation of this table:

1. **Loader API refactor:** published OPT values are now used as published;
   QFUERZA no longer silently overwrites them during system loading.
2. **MM3 angle-gradient fix:** the JAX angle term now uses a custom-VJP
   `atan2`-based angle function instead of gradient-killing `arccos(clip())`
   near collinear geometries.

Heck relay is the clearest example: its ratio moved from outside the default
band to 1.085 after the angle-gradient fix, and JAX-executor-guided
optimization now transfers to the real objective.

---

## Optimization results

| System | Initial historical field | Final historical field | Field provenance | Derived change | Legacy interval | L-BFGS-B iters | Legacy `n_evaluations` | Wall time |
|--------|--------------:|------------:|-----------------|-------:|------------:|---------------:|--------------:|----------:|
| [Rh-enamide](../systems/rh-enamide.md) | 4.885 × 10⁵ | 2.700 × 10⁵ | [Sample means; n=5 requested](https://github.com/ericchansen/q2mm-data/blob/8724a128a22e9867ba937999026224f075762884/benchmarks/rh-enamide/from-published/validation_results.json) | **−44.73%** | ±0.29% | 13 | 2 | 710 s opt + post-evals |
| [Heck relay](../systems/heck-relay.md) | 3.098 × 10⁶ | 1.461 × 10⁶ | [Sample means; n=10 requested](https://github.com/ericchansen/q2mm-data/blob/8724a128a22e9867ba937999026224f075762884/benchmarks/heck-relay/from-published/validation_results.json) | **−52.82%** | ±1.54% | 7 | 2 | 1,825 s opt + post-evals |
| [Pd-allyl](../systems/pd-allyl.md) | 8.036 × 10⁶ | 8.037 × 10⁶ | [Sample means; n=10 requested](https://github.com/ericchansen/q2mm-data/blob/8724a128a22e9867ba937999026224f075762884/benchmarks/pd-allyl-amination/from-published/validation_results.json) | **+0.010%** | ±0.40% | 2 | 2 | 1,289 s opt + post-evals |
| [Pd 1,4-conj](../systems/pd-conjugate.md) | 8.608 × 10⁶ | 7.235 × 10⁶ | [Single-call endpoints; no mean fields](https://github.com/ericchansen/q2mm-data/blob/8724a128a22e9867ba937999026224f075762884/benchmarks/pd-1,4-conjugate-addition/from-published/validation_results.json) | **−15.96%** | not sampled | 3 | 2 | 700 s |
| [Rh 1,4-conj](../systems/rh-conjugate.md) | 6.293 × 10⁶ | 5.160 × 10⁶ | [Sample means; n=10 requested](https://github.com/ericchansen/q2mm-data/blob/8724a128a22e9867ba937999026224f075762884/benchmarks/rh-1,4-conjugate-addition/from-published/validation_results.json) | **−18.00%** | ±4.17% | 4 | 2 | 691 s opt + post-evals |

Each row links its historical `validation_results.json` at a pinned revision.
Sample-mean rows use `result.initial_obj_score_mean` and
`result.final_obj_score_mean`; the endpoint row instead uses
`result.initial_obj_score` and `result.final_obj_score`. The requested repeat
counts come from `provenance.n_evals`, not a newly reconstructed sample series.
Derived change is `100 * (final / initial - 1)` using the unrounded fields:
negative means a reduction, while Pd-allyl's positive value is an increase.
The Pd 1,4-conj record's separate `improvement_pct` reports 16.09%, which
differs from the 15.96% reduction derived from its endpoint fields; this
table does not substitute that inconsistent stored percentage.
The evaluation-count column is the historical `result.n_evaluations` field,
not a reconstructed sample count or a count of all native optimizer
callbacks. It must not be compared directly with the current
optimizer-executor accounting contract. Current independent Python endpoint
evaluation is a separate operation; the legacy counter name alone does not
establish its execution provenance.

These artifacts were refreshed
under [#288](https://github.com/ericchansen/q2mm/pull/288) /
[q2mm-data#10](https://github.com/ericchansen/q2mm-data/pull/10) after the MM3
angle-gradient fix; the canonical/opt-out subdir rename in
[q2mm-data#11](https://github.com/ericchansen/q2mm-data/pull/11) moved
these published-start files from `convergence/` to `from-published/`). The legacy interval formula is
`(initial_obj_score_ci95 + final_obj_score_ci95) / initial_obj_score_mean × 100`
— the same combination used by those records' `improvement_significant`
flag. These legacy names alone do not identify the sampled executor; they
must not be treated as verified Python-objective confidence bounds without
matching sample provenance. The historical artifacts and numbers above have
not been recomputed or relabeled in place. They predate the current
preparation gates and do not certify current full-field publication runs.

Interpretation:

- **The historical records report substantial reductions for Rh-enamide,
  Heck relay, Pd 1,4-conj, and Rh 1,4-conj**, with the mixed field provenance
  stated above.
- **Pd-allyl's reported sample means show little movement.** Its legacy
  sample interval does not establish a hidden-improvement bound for a
  different executor or prove a local minimum. Local-basin methodology and
  canonical publication convergence remain separate scientific questions.
- **Small L-BFGS-B iteration counts are expected.**  In the JAX executor path,
  SciPy evaluates the surrogate many times internally; the Python executor is
  called only for the initial baseline and final validation.

---

## Per-category fit after optimization

The objective combines geometry references and eigenmatrix-diagonal references.
R² is reported by category so geometry improvements are not hidden by the much
larger eigenmatrix term.

| System | R²(bond_length) | R²(bond_angle) | R²(eig_diag) | Takeaway |
|--------|----------------:|---------------:|-------------:|----------|
| [Rh-enamide](../systems/rh-enamide.md) | 0.989 | 0.954 | 0.968 | Strong fit across all target classes |
| [Heck relay](../systems/heck-relay.md) | 0.983 | 0.909 | −14.28 | Geometry excellent; eigenmatrix gap remains |
| [Pd-allyl](../systems/pd-allyl.md) | 0.046 | 0.331 | −2.82 | Geometry and eigenmatrix agreement remain limited |
| [Pd 1,4-conj](../systems/pd-conjugate.md) | 0.950 | 0.037 | −9.642 | Bond geometry strong; eigenmatrix gap remains |
| [Rh 1,4-conj](../systems/rh-conjugate.md) | 0.822 | 0.540 | −12.85 | Real objective improves; eigenmatrix gap remains |

These R² values should not be read as claims about the original papers'
performance.  The papers used MacroModel MM3* and often the full lower-triangle
eigenmatrix, charges, and/or selectivity validation.  The table reports how the
same published OPT values and q2mm-optimized descendants behave under q2mm's
current JAX backend and objective.

---

## MacroModel MM3* transfer boundary

The published TSFFs remain scientifically valid in their original setting, but
several do not transfer their internal Hessian/eigenmatrix quality into q2mm's
JAX backend.  This is not a release blocker for q2mm because exact MacroModel
MM3* reproduction is outside the current alpha scope.

Known transfer gaps include:

- metal-center torsion behavior that may be suppressed or attenuated by
  MacroModel-specific rules,
- wildcard MM3 atom-type matching such as `00`,
- cross terms beyond the currently implemented JAX stretch-bend term,
- composed-force-field semantics for base MM3 + OPT overlays,
- the absence of a licensed MacroModel validation loop for confirming any
  compatibility-layer guesses.

q2mm's supported path is therefore:

1. load the published or QFUERZA starting force field without corrupting it,
2. build an `ActiveParameterSpace` that keeps non-OPT parameters inactive for literature-scale TS systems,
3. optimize under the q2mm backend/objective being used,
4. report the remaining cross-engine gap honestly.

---

## Recommendations

- Use `scipy-lbfgsb-jax` on the CLI or build a `JaxObjectiveExecutor` and pass
  it to `ScipyOptimizer(method="L-BFGS-B")` in Python for multi-molecule TS
  systems.
- Keep the default executor-ratio gate enabled.  It now admits all five benchmark
  systems after the loader and angle-gradient fixes, and it remains useful as
  a guard against future surrogate/objective divergence.
- Do not use `JaxOptOptimizer` as the default for multi-molecule TS systems.
  Its monolithic optimization path is useful on small systems, but the
  per-case JAX executor + SciPy L-BFGS-B path is the production route for the
  literature-scale benchmarks.
- Do not treat failure to beat a MacroModel-published FF under q2mm as a bug by
  itself.  Treat it as evidence of the documented MM3* transfer boundary unless
  a q2mm-native invariant or parity test fails.

---

## Reproduce

```bash
# Full convergence regeneration for all systems; writes results under results/
q2mm-benchmark batch

# Example: statistically sampled pd-allyl verdict
q2mm-benchmark single --system pd-allyl --n-evals 10
```

Archive any result JSON or optimized force field used in documentation in the
separate [`q2mm-data`](https://github.com/ericchansen/q2mm-data) repository;
local `results/` output is intentionally gitignored in this code repo.
