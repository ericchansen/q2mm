# Getting Started

## Installation

!!! note "Requirements"
    Python **3.10** or newer is required.

### From PyPI (recommended)

```bash
pip install q2mm                   # core package
pip install "q2mm[openmm]"         # with OpenMM backend
pip install "q2mm[jax]"            # with JAX backend (gas-phase)
pip install "q2mm[jax-md]"         # with JAX-MD backend (periodic, PBC)
pip install "q2mm[amber]"          # with parmed (AMBER support)
pip install "q2mm[all]"            # all optional dependencies
```

> **Pre-release:** the current version is an alpha. Add `--pre` to any
> install command (e.g. `pip install --pre q2mm` or
> `pip install --pre "q2mm[openmm]"`) if a stable release hasn't been
> published yet.

### From source (for development)

```bash
git clone https://github.com/ericchansen/q2mm.git
cd q2mm
pip install -e ".[dev]"            # editable install with dev tools
```

---

## QM/MM Backends

Q2MM can interface with several quantum-mechanical and molecular-mechanics
engines. Install the ones your workflow requires:

| Backend          | Type  | License              | Install                                          |
| ---------------- | ----- | -------------------- | ------------------------------------------------ |
| **OpenMM**       | MM    | MIT                  | `pip install openmm`                             |
| **JAX-MD**       | MM    | Apache-2.0           | `pip install "q2mm[jax-md]"` (Linux/macOS/WSL2)  |
| **Psi4**         | QM    | BSD-3 (open source)  | `conda install psi4 -c conda-forge`              |
| **Tinker**       | MM    | Free (academic)      | [download](https://dasher.wustl.edu/tinker/)     |
| **Gaussian**     | QM    | Commercial           | Site license                                     |
| **Jaguar** (Schrödinger)  | QM | Commercial      | Site license (Schrödinger Suite)                 |

!!! tip
    You only need the backends relevant to your project — Q2MM will skip
    unavailable engines gracefully.

---

## Quick Example

A minimal script that reads QM reference data, loads a structure, and inspects
an MM3 force field:

```python
from q2mm.parsers import GaussLog, Mol2, MM3

# Parse a Gaussian log for the Hessian
log = GaussLog("ethane.log")
hessian = log.structures[0].hess

# Read a MOL2 structure
mol2 = Mol2("ethane.mol2")
print(f"Atoms: {len(mol2.structures[0].atoms)}")

# Load an MM3 force field
ff = MM3("mm3.fld")
ff.import_ff()
print(f"Parameters: {len(ff.params)}")
```

---

## Package Structure

```
q2mm/
├── backends/      # QM/MM engine integrations (OpenMM, Tinker, JAX, Psi4)
├── diagnostics/   # Benchmarking and convergence analysis
├── models/        # Clean molecule/force-field models + Seminario estimation
├── optimizers/    # Objective functions, scoring, and scipy-based optimization
└── parsers/       # File format parsers (Gaussian, Jaguar, MM3, MOL2, etc.)
```

---

## Development

Install the dev extras, then run the linter and test suite:

```bash
pip install -e ".[dev]"
pytest -v
ruff check q2mm/ test/ scripts/
ruff format --check q2mm test scripts examples
```

!!! tip
    Run `ruff check --fix` to auto-fix simple lint issues before committing.
