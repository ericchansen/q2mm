# Q2MM

Q2MM builds molecular mechanics force fields from quantum-mechanical reference
structures and Hessians. It matters when standard force fields do not describe
your chemistry—especially transition states and metal-centered custom regions—
and you need explicit control over what is fitted and what stays frozen.

## One public workflow

```python
import q2mm
from q2mm.io import load_fchk_molecule
from q2mm.settings import PreparationSettings

molecule = load_fchk_molecule("my-transition-state.fchk", bond_tolerance=1.4)
problem = q2mm.prepare(
    molecule,
    settings=PreparationSettings(stationary_point="transition_state", functional_form="harmonic"),
)
```

Continue with the complete [execution lifecycle](getting-started.md#first-complete-problem):
choose final immutable settings, load/build components, capture the actual
configuration, create the evaluator, and optimize. The root exports are the
same canonical objects as their owning modules, loaded only when accessed.

1. **`prepare`** builds an immutable problem and records stationary point,
   functional form, observations, active/frozen slots, and QFUERZA audit.
2. **`evaluate(evaluator, parameters)`** scores an explicit full parameter vector.
   Reference-property generation uses the separate `evaluate_reference` service.
3. **`optimize`** executes the supplied optimizer, workflow and evaluator factory,
   returning a numerical `OptimizationResult`. Build the final field and
   provenance with `OptimizationRun.from_result`.
4. **`save`** writes a semantic force-field format and, for a run, a provenance
   manifest.

No function guesses whether a structure is a minimum or transition state.
`prepare` does not launch hidden QM calculations. SciPy and all computational
backends are optional dependencies.

## Two fitting patterns

- **Fresh one-molecule field:** bring one Hessian-bearing molecule and select a
  functional form. QFUERZA creates the starting field.
- **Multi-structure template:** bring many molecules, one complete field, and a
  smaller OPT/custom field. Only the selected scalar slots change; the base
  remains frozen.

The [tutorial](tutorial.md) starts with the real nine-structure Rh-enamide
template workflow from
[Donoghue et al. 2008](https://doi.org/10.1021/ct800132a), labels the current
objective as a partial repository reproduction, and then swaps in user paths.

## Where to go next

- [Getting Started](getting-started.md) — installation and your first FCHK
- [Tutorial](tutorial.md) — Rh-enamide, BYO templates, QCEngine, ASE, and manual problems
- [Publication Coverage](benchmarks/published-ff-validation.md) — provisionable and blocked source rows
- [Backend authoring](backends/authoring.md) — stable external backend API v1
- [Theory & Methods](how-it-works/theory.md) — QFUERZA and TS curvature handling
