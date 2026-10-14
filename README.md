# Q2MM

**Fit molecular mechanics force fields to quantum-mechanical reference data.**

[![CI](https://github.com/ericchansen/q2mm/actions/workflows/ci.yml/badge.svg)](https://github.com/ericchansen/q2mm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/q2mm)](https://pypi.org/project/q2mm/)
[![Python](https://img.shields.io/pypi/pyversions/q2mm)](https://pypi.org/project/q2mm/)

A force field is a set of parameters for an inexpensive molecular energy
model. Q2MM helps when standard parameters do not describe your chemistry:
it builds starting parameters from a quantum-mechanical (QM) geometry and
Hessian, then fits selected parameters against reference observations. A
Hessian describes how energy curves when atoms move.

**Alpha, breaking API:** the API and generated CLI below are unreleased.
Use this source revision or a wheel built from it, not an arbitrary PyPI
version. Python **3.15 final** is the primary runtime.

**[First-run guide](docs/getting-started.md)** ·
**[Tutorial](docs/tutorial.md)** ·
**[Documentation](https://ericchansen.github.io/q2mm/)** ·
**[Publication coverage](validation/published_ffs/README.md)**

## First success: a small CPU run

No QM installation, Hessian file of your own, publication archive, or GPU is
needed. The source-only CH3F example reads the approved geometry and Hessian
distributed as **installed q2mm package data**.

In PowerShell, from a checkout of this revision:

```powershell
python --version  # must be Python 3.15 final
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYYAML_FORCE_LIBYAML = "0"
python -m pip install -e ".[jax]"
$env:JAX_PLATFORMS = "cpu"
python examples\ch3f\run.py --optimizer.maxiter 2 --output-root results\ch3f
```

On Linux/macOS, activation is `source .venv/bin/activate`; use
`PYYAML_FORCE_LIBYAML=0 python -m pip install -e ".[jax]"`, then
`export JAX_PLATFORMS=cpu` and run `python examples/ch3f/run.py` with the same
options. PyYAML's supported pure-Python build avoids a C compiler on 3.15.

The script performs **real JAX molecular mechanics on CPU**, enters SciPy
L-BFGS-B with a two-iteration budget, prints JSON, and explicitly writes:

```text
results/ch3f/ch3f.frcmod
results/ch3f/ch3f.frcmod.manifest.json
```

This proves input loading, preparation, native evaluation, budget-limited
optimization, and saving work together. **It does not prove convergence,
force-field quality, or publication reproduction.** An iteration-limit
message and `optimization.success=false` are valid budgeted outcomes;
`convergence_claim` is always false. Existing files are not overwritten.
The [first-run guide](docs/getting-started.md#read-the-result) explains the
settings and output.

For a **software-path-only** check, install core with `pip install -e .`
and add `--bounded-ci`. That mode returns zero energy, echoes the reference
Hessian and geometry, and makes **no parameter update**. It is not a native
MM calculation. Use a different chosen output directory for this proof.

## Your chemistry comes next

After the packaged example, the [tutorial](docs/tutorial.md) branches into:

- **One molecule:** a Gaussian FCHK containing geometry and Cartesian force
  constants, or XYZ plus an explicitly supplied Hessian with units/provenance.
- **Many structures:** a complete force field plus an OPT/custom field that
  identifies the parameters allowed to change; the remaining parameters stay
  frozen.

The lifecycle has one owner per operation: `prepare` creates an immutable
problem; `evaluate` scores an evaluator and parameter vector; `optimize`
returns a numerical result using explicitly supplied components;
`OptimizationRun.from_result` assembles the final field and provenance;
`save` writes it. There is no hidden QM job or configuration resolver.

Settings are native Pydantic models. The generated CLI uses their typed fields,
not a second option schema. `q2mm-benchmark list` is a cheap catalog;
`single --show-config` only inspects requested policy, without loading engines,
data, or saving. See [CLI documentation](docs/benchmarks/cli.md).
Benchmark candidates are separate from these explicitly saved example files:
rejected/skipped candidates are **not** automatically promoted to force fields.

## Installation and scientific boundaries

Examples are **source-only**, excluded from wheels and sdists. The approved
CH3F/SN2 resources are installed package data; see their
[integrity/provenance manifest](q2mm/data/sn2/manifest.json).
The [local-wheel proof](docs/getting-started.md#use-a-local-wheel) runs the same
source script against an isolated installed wheel.

Python **3.14** is also supported and has required core checks. OpenMM 8.6.1
and the available JAX-MD dependency chain use Python 3.14 CPU lanes; neither
is required for this Python 3.15 JAX CPU path. Affected native and lightweight
integration checks fail closed. JAX CUDA/JAX-MD are not available on native
Windows. Other integrations are explicit extras; see
[installation contexts](docs/getting-started.md#other-installation-contexts).

Current publication availability, missing targets, and methodology blockers
are projected from canonical metadata on the
[publication coverage page](validation/published_ffs/README.md).
Do not infer exact reproduction from an executable example. Historical
measurements and publication artifacts remain separate.

## License

The q2mm software is MIT licensed; see [LICENSE](LICENSE). This does not grant
rights to external scientific inputs or third-party force fields. Rh-enamide
source inputs are excluded from wheel and sdist artifacts;
redistribution/licensing is not established. Recovered dissertation archives
are never distributed in q2mm wheels or sdists.
