# CH3F: first native CPU run

This source-only example fits a fresh harmonic ground-state force field to
the approved CH3F geometry/Hessian installed with q2mm. It gets you from known
inputs to saved parameters without a QM installation or publication archive.
The script and its shared runner are excluded from wheels and sdists; the
reference data are installed package resources.

Use Python 3.15 final and this unreleased source revision, installed with
`python -m pip install -e ".[jax]"` from the checkout. See the
[first-run guide](../../docs/getting-started.md) for environment setup,
PyYAML's compiler-free build and an isolated local-wheel proof.

From the checkout in PowerShell:

```powershell
$env:JAX_PLATFORMS = "cpu"
python examples\ch3f\run.py --optimizer.maxiter 2 --output-root results\ch3f
```

This performs real JAX MM evaluation and enters SciPy L-BFGS-B with a
two-iteration budget (`maxls=5`, single-stage, no optional endpoint sampling).
It prints JSON and explicitly saves `ch3f.frcmod` plus
`ch3f.frcmod.manifest.json` under the chosen output root.
An iteration-limit stop is valid; `convergence_claim` is always false, even
if the raw solver reports success. This is execution evidence, **not
convergence, force-field quality, or publication reproduction**.

For a **software-only** proof in a core-only installation:

```powershell
python examples\ch3f\run.py --bounded-ci --output-root results\ch3f-software
```

The echo backend returns zero energy and reference geometry/Hessian; its
optimizer does not update parameters. This is not a native MM calculation.

Use `--fchk inputs\minimum.fchk --stationary-point ground_state` for your own
minimum; geometry and Cartesian force constants must be present.
`--fchk` and `--input-root` are mutually exclusive. Output names still use
`ch3f`, while the manifest fingerprints your actual input.
See the [tutorial](../../docs/tutorial.md) for FCHK, explicit XYZ/Hessian,
shared-template preparation and Python settings JSON reuse.

`--help` and `--show-config` use the actual shared Pydantic models, including
`--backend.key`, `--optimizer.maxiter`, and objective controls. Inspection
loads no scientific data/engines and saves nothing.

Backend and executor remain independent explicit choices. With a supported
non-JAX backend, install its dependencies in the appropriate environment and
pass `--backend.key openmm --objective.executor python --objective.gradient-mode none`
and `--backend.options '{"platform_name":"CPU"}'`.
This selects Python/scalar evaluation with SciPy finite differences while
retaining the small GS budget. `--backend.key openmm` alone is rejected,
not silently switched. `JAX_PLATFORMS` does not control OpenMM: its own
explicit CPU option (or `OPENMM_DEFAULT_PLATFORM=CPU`) is required before
construction, and the selected platform is verified before evaluation.
No GPU execution is supported by this example.
