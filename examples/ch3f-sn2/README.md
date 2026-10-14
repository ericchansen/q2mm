# CH3F identity-SN2: explicit transition-state semantics

This source-only small example prepares a harmonic transition-state force
field from the approved installed SN2 geometry/Hessian. Use it after the
[ground-state first run](../ch3f/README.md) to see how a declared TS drives
curvature inversion rather than guessing from filenames.

With this revision installed as `.[jax]` on Python 3.15 final, from the
checkout in PowerShell:

```powershell
$env:JAX_PLATFORMS = "cpu"
python examples\ch3f-sn2\run.py --optimizer.maxiter 2 --output-root results\ch3f-sn2
```

The native CPU path uses JAX, a two-iteration SciPy budget, and the shared
TS defaults. It explicitly saves `ch3f-sn2.frcmod` and
`ch3f-sn2.frcmod.manifest.json`, **not a convergence or scientific validation
claim**. The reference Hessian can emit a multiple-negative-mode warning;
it is not hidden or modified to make the demonstration look successful.
Use `--help` for actual-model options and `--show-config` for side-effect-free
policy inspection.

```powershell
python examples\ch3f-sn2\run.py --bounded-ci --output-root results\ch3f-sn2-software
```

This separate mode returns zero energy, echoes the reference geometry/Hessian
and changes no parameters. It is only a software-path proof, not native MM.
Both paths report `convergence_claim=false` regardless of a raw solver flag.

A caller-owned TS can use `--fchk inputs\ts.fchk --stationary-point transition_state`.
A ground-state override is rejected. Scripts are excluded from distributions;
the approved inputs are installed package data. The source-only regeneration
utility is `scripts/generate_sn2_reference.py`; the sole resource location is
`q2mm/data/sn2/`. See [onboarding](../../docs/getting-started.md) and
[the tutorial](../../docs/tutorial.md) for installation contexts and inputs.

For another supported MM backend, explicitly select its executor too:
`--backend.key openmm --objective.executor python --objective.gradient-mode none`
and `--backend.options '{"platform_name":"CPU"}'`.
Install that backend in the appropriate environment first. TS tolerance,
fractional bounds and small budget remain unchanged; no automatic
backend-dependent executor fallback occurs. OpenMM requires its own explicit
CPU option (or `OPENMM_DEFAULT_PLATFORM=CPU`); `JAX_PLATFORMS` cannot constrain
it. The selected platform is checked before any context preparation or
evaluation. No GPU execution is supported by this example.
