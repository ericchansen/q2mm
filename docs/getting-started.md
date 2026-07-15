# Getting Started

This page covers installation and a quick sanity check. For a full
parameterization walkthrough, see the [Tutorial](tutorial.md).

## Installation

!!! note "Requirements"
    Python **3.10** or newer is required.

### From PyPI (recommended)

```bash
pip install q2mm                   # core package
pip install "q2mm[openmm]"         # with OpenMM backend
pip install "q2mm[jax]"            # with JAX backend + optax optimizers
pip install "q2mm[jax-md]"         # with JAX-MD backend (periodic, PBC)
pip install "q2mm[amber]"          # with parmed (AMBER support)
pip install "q2mm[all]"            # all optional dependencies
```

> **Pre-release:** the current version is an alpha. Add `--pre` to any
> install command (e.g. `pip install --pre q2mm` or
> `pip install --pre "q2mm[openmm]"`) if a stable release hasn't been
> published yet.

### GPU setup

For GPU setup instructions (CUDA, WSL2, verification commands), see
[Platform Support](platform-support.md#gpu-setup).

### From source (for development)

```bash
git clone https://github.com/ericchansen/q2mm.git
cd q2mm
pip install -e ".[dev]"            # editable install with dev tools
```

### External data for published systems

Q2MM does not distribute the licensed or third-party datasets used by the
published transition-state systems. Configure their locations before running
those systems:

| Variable | Required by | Value |
|----------|-------------|-------|
| `Q2MM_RH_ENAMIDE` | Rh-enamide | Directory containing `mm3.fld` and `rh_enamide_training_set/` |
| `Q2MM_SUPPORTING_INFO` | Heck relay; Pd/Rh conjugate; Pd-allyl | Root of the extracted Wahlers/Rosales supporting information |
| `Q2MM_MM3_BASE` | Pd-allyl; Pd/Rh conjugate | Licensed `mm3_base.fld` file |

For example, in Bash:

```bash
export Q2MM_RH_ENAMIDE=/path/to/q2mm/examples/rh-enamide
export Q2MM_SUPPORTING_INFO=/path/to/q2mm/validation/supporting-info
export Q2MM_MM3_BASE=/path/to/mm3_base.fld
```

In PowerShell, use `$env:Q2MM_RH_ENAMIDE = "..."` (and likewise for the
other variables). Missing or invalid roots raise an error naming the exact
variable or `ExternalDataRoots` field to configure; Q2MM never searches above
the installed package.

---

## QM/MM backends

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

## Quick example

A minimal script that reads QM reference data from the included example files.
Clone the repository first to access the example data:

```bash
git clone https://github.com/ericchansen/q2mm.git
cd q2mm
```

```python
from q2mm.io import GaussLog, load_mm3_fld
from q2mm.io.xyz import load_xyz
from q2mm.resources import sn2_reference_dir

# Parse a Gaussian log for the QM Hessian (matrix of energy second derivatives)
log = GaussLog("examples/ethane/TS.log")
mol_from_log = log.molecules[-1]
print(f"Gaussian molecule: {mol_from_log.n_atoms} atoms, Hessian shape: {mol_from_log.hessian.shape}")

# Load an XYZ geometry into the unified Molecule model
mol = load_xyz(sn2_reference_dir() / "ch3f-optimized.xyz")
print(f"Molecule atoms: {mol.n_atoms}")

# Load an MM3 force field
ff = load_mm3_fld("examples/rh-enamide/mm3.fld")
print(f"Bonds: {len(ff.bonds)}, Angles: {len(ff.angles)}")
```

---

## Package structure

```
q2mm/
├── io/            # File format I/O (Gaussian, Jaguar, MM3, MOL2, AMBER, etc.)
├── backends/      # QM/MM engine integrations (OpenMM, Tinker, JAX, Psi4)
├── diagnostics/   # Benchmarking and convergence analysis
├── models/        # Molecule/force-field models + QFUERZA estimation
└── optimizers/    # Objective functions, scoring, and scipy-based optimization
```

## Next steps

1. Follow the [Tutorial](tutorial.md) for a complete parameterization walkthrough
2. Read [Theory & Methods](how-it-works/theory.md) to understand the pipeline
3. See [Platform Support](platform-support.md) for GPU and backend setup
