# OpenMM Backend

The `OpenMMEngine` is Q2MM's most versatile backend, supporting both
**Harmonic** and **MM3** functional forms. It runs in-process via the
OpenMM Python API, avoiding subprocess overhead.

---

## Installation

OpenMM is available via conda-forge:

```bash
conda install -c conda-forge openmm
```

Or with pip (Linux/macOS):

```bash
pip install openmm
```

!!! tip "Verify installation"
    ```python
    import openmm
    print(openmm.version.full_version)
    print(openmm.Platform.getNumPlatforms(), "platforms available")
    ```

### Platform Detection

OpenMMEngine auto-detects the fastest available compute platform:

| Priority | Platform | Notes |
|----------|----------|-------|
| 1 | CUDA | Requires NVIDIA GPU + CUDA toolkit |
| 2 | OpenCL | AMD/Intel GPUs |
| 3 | CPU | Multi-threaded, available everywhere |
| 4 | Reference | Single-threaded, for debugging only |

Override with the `platform_name` constructor parameter if needed.

---

## Supported Energy Terms

| Term | Harmonic Mode | MM3 Mode |
|------|:---:|:---:|
| Bonds | ✅ Harmonic | ✅ Cubic/quartic |
| Angles | ✅ Harmonic | ✅ Sextic |
| Torsions | ⚠️ [#127](https://github.com/ericchansen/q2mm/issues/127) | ⚠️ [#127](https://github.com/ericchansen/q2mm/issues/127) |
| Improper torsions | ❌ | ❌ |
| vdW (LJ 12-6) | ✅ | — |
| vdW (Buckingham exp-6) | — | ✅ |
| Electrostatics | ❌ | ❌ |
| 1-4 scaling | ✅ AMBER (ε/2) | None (MM3) |

---

## Configuration

```python
from q2mm.backends.mm import OpenMMEngine

engine = OpenMMEngine(
    platform_name=None,   # auto-detect (CUDA > OpenCL > CPU > Reference)
    precision=None,       # "single", "mixed", or "double" (GPU only; default: "mixed")
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `platform_name` | `str \| None` | `None` | Force a specific OpenMM platform |
| `precision` | `str \| None` | `None` | GPU precision mode; ignored on CPU |

### Runtime Parameter Updates

OpenMMEngine supports in-place parameter updates via `update_forcefield()`,
which mutates force parameters in the existing OpenMM `Context` without
rebuilding the system. This makes iterative optimization fast.

---

## Capabilities

| Method | Supported | Notes |
|--------|:---------:|-------|
| `energy()` | ✅ | Returns kcal/mol |
| `minimize()` | ✅ | OpenMM L-BFGS minimizer |
| `hessian()` | ✅ | **Numerical** (finite-difference) |
| `frequencies()` | ✅ | From numerical Hessian |
| `energy_and_param_grad()` | ✅ | Exact for bond/angle params; torsion/vdW gradients returned as zero |
| `supports_runtime_params()` | ✅ | — |
| `supports_analytical_gradients()` | ✅ | Bond/angle only |

### Serialization

Systems can be saved to and loaded from OpenMM XML:

```python
handle = engine.create_context(molecule, forcefield)
engine.export_system_xml(handle, "system.xml")
```

---

## Limitations

- **Numerical Hessians** — `hessian()` uses finite differences, not analytical
  second derivatives. Accurate but slower than JAX's analytical Hessian.
- **Partial analytical gradients** — `energy_and_param_grad()` provides exact
  gradients for bond and angle parameters via OpenMM global-parameter
  derivatives. Torsion and vdW parameter gradients are returned as zero.
- **No improper torsions** — not yet implemented.
- **No electrostatics** — charge optimization is not supported.

---

## Example

```python
from q2mm.backends.mm import OpenMMEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule

engine = OpenMMEngine()

# Load molecule and force field
mol = Q2MMMolecule.from_xyz("molecule.xyz")
ff = ForceField.from_amber_frcmod("params.frcmod", mol)

# Single-point energy
e = engine.energy(mol, ff)
print(f"Energy: {e:.4f} kcal/mol")

# Frequencies
freqs = engine.frequencies(mol, ff)
print(f"Frequencies: {freqs}")
```

---

## See Also

- [Engine comparison table](index.md#engine-overview)
- [Parameter transferability](index.md#parameter-transferability)
- [Benchmarks](../benchmarks/index.md)
- [API Reference: OpenMMEngine](../reference/q2mm/backends/mm/openmm/)
