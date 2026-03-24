# Tinker Backend

The `TinkerEngine` wraps external Tinker executables (`analyze`, `minimize`,
`vibrate`, `testhess`) via subprocess calls. It is the only backend that
supports MM3 functional forms without OpenMM.

---

## Installation

Tinker must be installed separately. Pre-built binaries are available from
the [Tinker website](https://dasher.wustl.edu/tinker/).

The engine searches for Tinker executables in this order:

1. The `tinker_dir` constructor parameter (if provided)
2. Common installation directories (`/usr/local/bin`, `/opt/tinker/bin`, etc.)
3. Directories on `PATH`

!!! tip "Verify installation"
    ```bash
    which analyze && analyze --version
    ```

### Required Executables

| Executable | Used By |
|------------|---------|
| `analyze` | `energy()` |
| `minimize` | `minimize()` |
| `vibrate` | `frequencies()` |
| `testhess` | `hessian()` |

---

## Supported Energy Terms

| Term | Supported |
|------|:---------:|
| Bonds (MM3 cubic/quartic) | ✅ |
| Angles (MM3 sextic) | ✅ |
| Torsions | ⚠️ [#127](https://github.com/ericchansen/q2mm/issues/127) |
| Improper torsions | ❌ |
| vdW (Buckingham exp-6) | ✅ |
| Electrostatics | ✅ (Tinker default) |
| 1-4 scaling | MM3 default |

**Functional forms:** MM3 only.

---

## Configuration

```python
from q2mm.backends.mm import TinkerEngine

engine = TinkerEngine(
    tinker_dir=None,       # auto-detect Tinker installation
    params_file=None,      # auto-detect MM3 parameter file
    bond_tolerance=1.3,    # bond detection: tolerance * (r_cov_A + r_cov_B)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tinker_dir` | `str \| None` | `None` | Path to directory containing Tinker executables |
| `params_file` | `str \| None` | `None` | Path to MM3 base parameter file |
| `bond_tolerance` | `float` | `1.3` | Multiplier for covalent-radius bond detection |

---

## Capabilities

| Method | Supported | Notes |
|--------|:---------:|-------|
| `energy()` | ✅ | Via `analyze E` |
| `minimize()` | ✅ | Parses `.xyz_2` output |
| `hessian()` | ✅ | Via `testhess`; symmetrized |
| `frequencies()` | ✅ | Via `vibrate` |
| `energy_and_param_grad()` | ❌ | Not implemented |
| `supports_runtime_params()` | ❌ | Subprocess per call |
| `supports_analytical_gradients()` | ❌ | — |

### Performance Note

Each energy/frequency evaluation spawns a new Tinker subprocess, writes
temporary parameter and coordinate files, and parses text output. This
makes Tinker significantly slower per evaluation than in-process engines
(~160 ms/eval vs ~5 ms for OpenMM, ~0.1 ms for JAX).

---

## Limitations

- **MM3 only** — does not support Harmonic functional forms.
- **No runtime parameter updates** — each call writes a new parameter file
  and spawns a subprocess.
- **No analytical gradients** — `energy_and_param_grad()` is not implemented.
- **Standalone PRM limitations** — `_write_standalone_prm()` only writes
  bond, angle, and vdW terms. Template-based export is preferred for full
  `.prm` fidelity.
- **No GPU support** — runs entirely on CPU.
- **External dependency** — requires Tinker executables to be installed
  and discoverable.

---

## Example

```python
from q2mm.backends.mm import TinkerEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule

engine = TinkerEngine(tinker_dir="/opt/tinker/bin")

mol = Q2MMMolecule.from_xyz("molecule.xyz")
ff = ForceField.from_mm3_fld("mm3.fld", mol)

e = engine.energy(mol, ff)
print(f"Energy: {e:.4f} kcal/mol")

freqs = engine.frequencies(mol, ff)
print(f"Frequencies: {freqs}")
```

---

## See Also

- [Engine comparison table](index.md#engine-overview)
- [Parameter transferability](index.md#parameter-transferability)
- [Benchmarks](../benchmarks/index.md)
- [API Reference: TinkerEngine](../reference/q2mm/backends/mm/tinker/)
