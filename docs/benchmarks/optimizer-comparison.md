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

For four of the five systems the answer is yes.  Pd-allyl is the exception: it
passes the executor-ratio gate, but the published Wahlers parameters already
sit at a local minimum for the current q2mm objective.

---

## Methodology

All multi-target benchmarks use the same production setup:

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
  default ±15% band.
- **Validation:** the Python executor is evaluated before and
  after the JAX-executor-guided optimization.  For noisy systems, the reported
  improvement is the mean over 10 initial and 10 final evaluations with a
  95% confidence interval.

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
this table is inside the default band.

| System | Mols | Active params | Executor ratio | Gate |
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

| System | Initial score | Final score | Mean Δ | 95% CI on Δ | L-BFGS-B iters | Real OF evals | Wall time |
|--------|--------------:|------------:|-------:|------------:|---------------:|--------------:|----------:|
| [Rh-enamide](../systems/rh-enamide.md) | 4.885 × 10⁵ | 2.700 × 10⁵ | **−44.73%** | ±0.29% | 13 | 2 | 710 s opt + post-evals |
| [Heck relay](../systems/heck-relay.md) | 3.098 × 10⁶ | 1.461 × 10⁶ | **−52.82%** | ±1.54% | 7 | 2 | 1,825 s opt + post-evals |
| [Pd-allyl](../systems/pd-allyl.md) | 8.036 × 10⁶ | 8.037 × 10⁶ | **−0.010%** | ±0.40% | 2 | 2 | 1,289 s opt + post-evals |
| [Pd 1,4-conj](../systems/pd-conjugate.md) | 8.608 × 10⁶ | 7.235 × 10⁶ | **−15.96%** | not sampled | 3 | 2 | 700 s |
| [Rh 1,4-conj](../systems/rh-conjugate.md) | 6.293 × 10⁶ | 5.160 × 10⁶ | **−18.00%** | ±4.17% | 4 | 2 | 691 s opt + post-evals |

Score and CI values come from `benchmarks/<system>/from-published/validation_results.json`
in [ericchansen/q2mm-data](https://github.com/ericchansen/q2mm-data) (refreshed
under [#288](https://github.com/ericchansen/q2mm/pull/288) /
[q2mm-data#10](https://github.com/ericchansen/q2mm-data/pull/10) after the MM3
angle-gradient fix; the canonical/opt-out subdir rename in
[q2mm-data#11](https://github.com/ericchansen/q2mm-data/pull/11) moved
these published-start files from `convergence/` to `from-published/`).  `95% CI on Δ` is the conservative bound
`(initial_obj_score_ci95 + final_obj_score_ci95) / initial_obj_score_mean × 100`
— the same combination used by the JSON's `improvement_significant` flag.
Rh-enamide and ch3f were re-evaluated with `--n-evals 5`; the others with
`--n-evals 10`.  Pd 1,4-conj is a single-call run (no CI sampled).

Interpretation:

- **Rh-enamide, Heck relay, Pd 1,4-conj, and Rh 1,4-conj improve
  substantially** under the q2mm JAX-backend objective.
- **Pd-allyl does not improve in a statistically meaningful way.**  The
  optimizer converges quickly, the executor-ratio gate is healthy, and the 10-sample
  confidence interval excludes any hidden >0.4% improvement.  This is a local
  minimum of the current objective, not a failed run.
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
| [Pd-allyl](../systems/pd-allyl.md) | 0.046 | 0.331 | −2.82 | Published values are a q2mm local minimum but not a good transfer fit |
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
