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

- **Hessian-informed initialization** — the Seminario method extracts bond and
  angle force constants directly from QM Hessians, providing excellent starting
  parameters before optimization begins.
- **Open-source backends** — first-class support for [OpenMM](https://openmm.org/)
  and [Psi4](https://psicode.org/) alongside commercial packages (Gaussian,
  Schrödinger, Tinker).
- **Clean, modular architecture** — format-agnostic data models (`ForceField`,
  `Q2MMMolecule`) decouple algorithms from file formats.
- **Modern optimization** — powered by `scipy.optimize` with L-BFGS-B,
  Nelder-Mead, trust-region, and Levenberg-Marquardt methods.
- **Transition state support** — negative force constants, torsion parameters,
  and proper eigenvalue handling for saddle-point geometries.

## Quick Start

```bash
pip install q2mm                   # from PyPI
pip install "q2mm[openmm]"         # with OpenMM backend
```

> **Pre-release:** the current version is an alpha. Add `--pre` to any
> install command (e.g. `pip install --pre q2mm` or
> `pip install --pre "q2mm[openmm]"`) if a stable release hasn't been
> published yet.

For development, clone the repo and install in editable mode:

```bash
pip install -e ".[dev]"
```

```python
import numpy as np
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import estimate_force_constants

# Load QM data (coordinates + Hessian from your QM package)
mol = Q2MMMolecule.from_xyz("ts-optimized.xyz")
mol.hessian = np.load("ts-hessian.npy")  # Hartree/Bohr²

# Estimate force constants from the QM Hessian (Seminario method)
ff = estimate_force_constants(mol, au_hessian=True)

print(f"Bonds: {len(ff.bonds)}, Angles: {len(ff.angles)}")
for b in ff.bonds:
    print(f"  {b.elements}: k={b.force_constant:.3f} mdyn/Å")
```

See the [Tutorial](https://ericchansen.github.io/q2mm/tutorial/) for a
complete end-to-end workflow.

## Supported Backends

| Backend | Type | License |
|---------|------|---------|
| **OpenMM** | MM | MIT |
| **Psi4** | QM | BSD-3 |
| **Tinker** | MM | Free (academic) |
| **Gaussian** | QM | Commercial |
| **Schrödinger** | QM/MM | Commercial |

## License

BSD-3-Clause. See [LICENSE](LICENSE).

