# Getting Started

## Installation

!!! note "Requirements"
    Python **3.9** or newer is required.

```bash
pip install -e .            # Basic
pip install -e ".[dev]"     # With pytest + ruff
pip install -e ".[openmm]"  # With OpenMM backend
pip install -e ".[amber]"   # With parmed (AMBER support)
pip install -e ".[docs]"    # MkDocs Material for docs
```

---

## QM/MM Backends

Q2MM can interface with several quantum-mechanical and molecular-mechanics
engines. Install the ones your workflow requires:

| Backend          | Type  | License              | Install                                          |
| ---------------- | ----- | -------------------- | ------------------------------------------------ |
| **OpenMM**       | MM    | MIT                  | `pip install openmm`                             |
| **Psi4**         | QM    | BSD-3 (open source)  | `conda install psi4 -c conda-forge`              |
| **Tinker**       | MM    | Free (academic)      | [download](https://dasher.wustl.edu/tinker/)     |
| **Gaussian**     | QM    | Commercial           | Site license                                     |
| **Schrödinger**  | QM/MM | Commercial           | Site license                                     |

!!! tip
    You only need the backends relevant to your project — Q2MM will skip
    unavailable engines gracefully.

---

## Quick Example

A minimal script that reads QM reference data, loads a structure, and inspects
an MM3 force field:

```python
from q2mm.io import GaussLog, Mol2
from q2mm.forcefields import MM3

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
├── backends/      # QM/MM engine integrations (Psi4, Tinker, etc.)
├── forcefields/   # Force field types (MM3, AMBER, Tinker)
├── io/            # Convenience re-exports from parsers
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
