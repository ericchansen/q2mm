# Psi4 Backend

The `Psi4Engine` wraps the [Psi4](https://psicode.org/) Python API for
quantum mechanical calculations: single-point energy, Hessian, geometry
optimization, and vibrational frequencies. It is the only QM backend and
is used to generate the reference data that drives force field
optimization.

---

## Installation

Psi4 is available via conda-forge:

```bash
conda install psi4 -c conda-forge
```

!!! tip "Verify installation"
    ```python
    import psi4
    print(psi4.__version__)
    ```

!!! note "Psi4 is a Python library, not a standalone binary"
    Unlike Gaussian (which produces a `.log` file you parse after the
    fact), Psi4 runs inside Python. You call functions that return NumPy
    arrays directly — no file parsing needed.

---

## Supported methods

Psi4 supports any method string accepted by `psi4.energy()` — including
HF, DFT functionals (B3LYP, M06, ωB97X-D, etc.), and post-HF methods
(MP2, CCSD). The method is configured via the `method` parameter when
creating the engine. Basis sets are set via `basis`.

---

## Configuration

```python
from q2mm.backends.qm import Psi4Engine

engine = Psi4Engine(
    method="b3lyp",          # DFT functional or QM method
    basis="6-31+G(d)",       # basis set
    memory="2 GB",           # memory allocation
    n_threads=4,             # parallel threads
    charge=0,                # molecular charge
    multiplicity=1,          # spin multiplicity
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"b3lyp"` | DFT functional or method (e.g. `"mp2"`, `"hf"`) |
| `basis` | `str` | `"6-31+G(d)"` | Basis set |
| `memory` | `str` | `"2 GB"` | Memory allocation string |
| `n_threads` | `int` | `4` | Number of threads for parallel computation |
| `charge` | `int` | `0` | Molecular charge |
| `multiplicity` | `int` | `1` | Spin multiplicity (1 = singlet, 2 = doublet, …) |

---

## Capabilities

| Method | Supported | Notes |
|--------|:---------:|-------|
| `energy()` | ✅ | Returns Hartrees |
| `optimize()` | ✅ | Minimization or TS search (`opt_type="ts"`) |
| `hessian()` | ✅ | Returns Hartree/Bohr², shape (3N, 3N) |
| `frequencies()` | ✅ | Returns cm⁻¹ |
| Context manager | ✅ | Auto-cleans temp files on exit |

### Input formats

Structures can be passed as:

- **XYZ file path** — reads standard `.xyz` format
- **Tuple** — `(atoms, coords)` where `atoms` is a list of element
  symbols and `coords` is a NumPy array of shape `(N, 3)` in Å

All methods accept optional `method` and `basis` overrides to run a
single calculation with different settings without reconfiguring the
engine.

---

## Limitations

- **CPU only** — Psi4 does not use GPU acceleration.
- **No minimize() method** — uses `optimize()` instead (Psi4's own
  geometry optimizer with geom_maxiter=100).
- **No analytical MM gradients** — this is a QM engine, not an MM
  engine. It generates reference data, not force field evaluations.
- **Conda required** — `pip install psi4` does not work; must use
  conda-forge.
- **Temporary files** — each engine instance creates a temp directory
  for Psi4 output. Use the context manager or call `close()` to clean up.

---

## Example

```python
from q2mm.backends.qm import Psi4Engine
import numpy as np

with Psi4Engine(method="b3lyp", basis="6-31+G(d)") as engine:
    # Single-point energy
    e = engine.energy("molecule.xyz")
    print(f"Energy: {e:.6f} Hartree")

    # Geometry optimization (transition state)
    energy, atoms, coords = engine.optimize("ts-guess.xyz", opt_type="ts")
    print(f"TS energy: {energy:.6f} Hartree")

    # Hessian for QFUERZA estimation
    hess = engine.hessian("ts-optimized.xyz")
    print(f"Hessian shape: {hess.shape}")

    # Vibrational frequencies
    freqs = engine.frequencies("ts-optimized.xyz")
    print(f"Frequencies: {freqs[:5]} cm⁻¹")
```

---

## Role in the Q2MM pipeline

Psi4 is typically used in **Stage 0** of the Q2MM workflow — generating
QM reference data before any force field optimization begins:

1. **Optimize** the transition state geometry (`opt_type="ts"`)
2. **Compute the Hessian** at the optimized geometry
3. **Extract frequencies** for validation
4. Feed the Hessian into [QFUERZA estimation](../how-it-works/theory.md)
   for initial force constant estimation

The MM engines (OpenMM, JAX, Tinker, JAX-MD) then handle the iterative
force field optimization against this QM reference data.

---

## See also

- [Engine comparison table](index.md#engine-overview)
- [Tutorial: Generating QM Reference Data](../tutorial.md)
- [API Reference: Psi4Engine](../reference/q2mm/backends/qm/psi4.md)
