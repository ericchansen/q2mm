# Psi4 Backend

The `Psi4Backend` wraps the [Psi4](https://psicode.org/) Python API as a
`BackendRole.REFERENCE` backend for single-point energy, Hessian, geometry
optimization, and vibrational frequencies. It generates reference data that
drives force-field optimization.

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
creating the backend. Basis sets are set via `basis`.

---

## Configuration

```python
from q2mm.backends.qm.psi4 import Psi4Backend

backend = Psi4Backend(
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

| Prepared-session operation | Supported | Notes |
|--------|:---------:|-------|
| `energy(ReferenceEnergyRequest)` | ✅ | Returns Hartrees |
| `optimize_geometry(ReferenceGeometryOptimizationRequest)` | ✅ | Minimization or TS search (`opt_type="ts"`) |
| `hessian(ReferenceHessianRequest)` | ✅ | Returns Hartree/Bohr², shape (3N, 3N) |
| `frequencies(ReferenceFrequencyRequest)` | ✅ | Returns cm⁻¹ |
| Context manager | ✅ | Auto-cleans temp files on exit |

### Input model

Prepare a Psi4 session with a `q2mm.models.molecule.Molecule`. For file-based
workflows, load the structure first (for example with `q2mm.io.xyz.load_xyz`)
and pass that molecule in `PreparationRequest`.

Method and basis are fixed on the backend instance. To run with different
settings, construct a second `Psi4Backend`.

---

## Limitations

- **CPU only** — Psi4 does not use GPU acceleration.
- **No MM minimization method** — uses `optimize_geometry()` instead (Psi4's own
  geometry optimizer with geom_maxiter=100).
- **No analytical MM gradients** — this is a reference backend, not an MM
  backend. It generates quantum-mechanical reference data, not force-field
  evaluations.
- **Conda required** — `pip install psi4` does not work; must use
  conda-forge.
- **Temporary files** — each backend instance creates a temp directory
  for Psi4 output. Use the context manager or call `close()` to clean up.

---

## Example

```python
from q2mm.backends.contracts import (
    PreparationRequest,
    ReferenceEnergyRequest,
    ReferenceFrequencyRequest,
    ReferenceGeometryOptimizationRequest,
    ReferenceHessianRequest,
)
from q2mm.backends.qm.psi4 import Psi4Backend
from q2mm.io.xyz import load_xyz

mol = load_xyz("molecule.xyz")

with Psi4Backend(method="b3lyp", basis="6-31+G(d)") as backend:
    session = backend.prepare(PreparationRequest(case_id="example", molecule=mol))

    # Single-point energy
    e = session.energy(ReferenceEnergyRequest()).energy
    print(f"Energy: {e:.6f} Hartree")

    # Geometry optimization (transition state)
    ts = session.optimize_geometry(ReferenceGeometryOptimizationRequest(opt_type="ts"))
    print(f"TS energy: {ts.energy:.6f} Hartree")

    # Hessian for QFUERZA estimation
    hess = session.hessian(ReferenceHessianRequest()).hessian
    print(f"Hessian shape: {hess.shape}")

    # Vibrational frequencies
    freqs = session.frequencies(ReferenceFrequencyRequest()).frequencies
    print(f"Frequencies: {freqs[:5]} cm⁻¹")
```

---

## Role in the Q2MM pipeline

Psi4 is typically used in **Stage 0** of the Q2MM workflow — generating
quantum-mechanical reference data before any force field optimization begins:

1. **Optimize** the transition state geometry (`opt_type="ts"`)
2. **Compute the Hessian** at the optimized geometry
3. **Extract frequencies** for validation
4. Feed the Hessian into [QFUERZA estimation](../how-it-works/theory.md)
   for initial force constant estimation

The MM backends (OpenMM, JAX, Tinker, JAX-MD) then handle iterative
force-field optimization against this reference data.

---

## See also

- [Backend comparison table](index.md#backend-overview)
- [Tutorial: Generating QM Reference Data](../tutorial.md)
- [API Reference: Psi4Backend](../reference/q2mm/backends/qm/psi4.md)
