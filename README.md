# Q2MM

**Quantum-guided molecular mechanics force field optimization.**

[![CI](https://github.com/ericchansen/q2mm/actions/workflows/ci.yml/badge.svg)](https://github.com/ericchansen/q2mm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/q2mm)](https://pypi.org/project/q2mm/)
[![Python](https://img.shields.io/pypi/pyversions/q2mm)](https://pypi.org/project/q2mm/)

Q2MM optimizes molecular mechanics (MM) force field parameters by minimizing
the difference between MM-calculated properties and quantum mechanics (QM)
reference data. It is designed for building **transition state force fields
(TSFFs)** that enable rapid virtual screening of enantioselective catalysts.

**📖 [Documentation](https://ericchansen.github.io/q2mm/)**

## Why Q2MM?

- **Hessian-informed initialization** — QFUERZA extracts bond and angle force
  constants directly from QM Hessians, providing excellent starting parameters
  before optimization begins.
- **Open-source backends** — first-class support for [OpenMM](https://openmm.org/)
  and [Psi4](https://psicode.org/) alongside commercial packages (Gaussian,
  Schrödinger, Tinker).
- **Clean, modular architecture** — format-agnostic data models (`ForceField`,
  `Molecule`) decouple algorithms from file formats.
- **Modern optimization** — powered by `scipy.optimize` with L-BFGS-B,
  Nelder-Mead, trust-region, and Levenberg-Marquardt methods.
- **Transition state support** — negative force constants, torsion parameters,
  and proper eigenvalue handling for saddle-point geometries.

## Quick Start

```bash
pip install "q2mm[openmm,optimize]"   # OpenMM backend + scipy optimizer
```

> **Pre-release:** the current version is an alpha. Add `--pre` to any
> install command (e.g. `pip install --pre "q2mm[openmm,optimize]"`) if a
> stable release hasn't been published yet.

> **GPU acceleration:** OpenMM CUDA works on Linux, WSL2, and native
> Windows. For the full GPU stack (JAX CUDA + JAX-MD), use Linux or WSL2.
> See the [Platform Support](https://ericchansen.github.io/q2mm/platform-support/)
> guide for details.

For development, clone the repo and install in editable mode:

```bash
pip install -e ".[dev]"
```

```python
from q2mm.io.fchk import load_fchk_reference
from q2mm.models.forcefield import FunctionalForm
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.models.seminario import qfuerza_fresh
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.python import PythonObjectiveExecutor
from q2mm.optimizers.scipy_opt import ScipyOptimizer
from q2mm.backends.mm.openmm import OpenMMBackend

# 1. Load QM reference data and molecule from a Gaussian checkpoint
ref, mol = load_fchk_reference("ts-optimized.fchk", bond_tolerance=1.4)

# 2. Build the initial force field from the QM Hessian (QFUERZA)
ff = qfuerza_fresh(mol, functional_form=FunctionalForm.MM3, au_hessian=True)

# 3. Compile the objective plan, attach a backend, and optimize
backend = OpenMMBackend()
layout = ParameterLayout.from_force_field(ff)
space = ActiveParameterSpace.all_active(layout, ff)
plan = ObjectivePlan(
    case_ids=("0",),
    molecules=(mol,),
    stationary_points=(StationaryPointKind.TRANSITION_STATE,),
    observations=ref,
    layout=layout,
    active_space=space,
)
obj = PythonObjectiveExecutor(plan, backend, ff)
result = ScipyOptimizer(method="L-BFGS-B").optimize(obj, space)

print(result.summary())
```

`load_fchk_reference()` auto-extracts bond lengths and angles from the
QM geometry. You can also use `load_gaussian_reference()` for `.log` files, or
`ObservationSet.from_molecule()` for maximum control. See the
[Tutorial](https://ericchansen.github.io/q2mm/tutorial/) for the full
workflow including frequencies, eigenmatrix data, and multi-molecule fits.

## Supported Backends

| Backend | Type | License |
|---------|------|---------|
| **OpenMM** | MM | MIT |
| **JAX** | MM | Apache 2.0 |
| **JAX-MD** | MM | Apache 2.0 |
| **Tinker** | MM | Free (academic) |
| **Psi4** | QM | BSD-3 |
| **Gaussian** | QM | Commercial |
| **Schrödinger** | QM/MM | Commercial |

## License

MIT. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing with
Docker, and submitting changes.

