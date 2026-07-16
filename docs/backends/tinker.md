# Tinker Backend

The `TinkerBackend` wraps external [Tinker](https://dasher.wustl.edu/tinker/) executables (`analyze`, `minimize`,
`vibrate`, `testhess`) via subprocess calls. It is the original subprocess-based
MM3 backend and remains useful when an external Tinker installation is available.

---

## Installation

Tinker must be installed separately. Pre-built binaries are available from
the [Tinker website](https://dasher.wustl.edu/tinker/).

The backend searches for Tinker executables in this order:

1. The `tinker_dir` constructor parameter (if provided)
2. Common installation directories (`/usr/local/bin`, `/opt/tinker/bin`, etc.)
3. Directories on `PATH`

!!! tip "Verify installation"
    ```bash
    which analyze && analyze --version
    ```

### Required executables

| Executable | Used By |
|------------|---------|
| `analyze` | `energy()` |
| `minimize` | `minimize()` |
| `vibrate` | `frequencies()` |
| `testhess` | `hessian()` |

---

## Supported energy terms

| Term | Supported |
|------|:---------:|
| Bonds (MM3 cubic/quartic) | ✅ |
| Angles (MM3 sextic) | ✅ |
| Torsions | ✅ |
| Improper torsions | ❌ |
| vdW (Buckingham exp-6) | ✅ |
| Electrostatics | ✅ (Tinker default) |
| 1-4 scaling | MM3 default |

**Functional forms:** MM3 only.

---

## Configuration

```python
from q2mm.backends.mm import TinkerBackend

backend = TinkerBackend(
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

| Prepared-session operation | Supported | Notes |
|--------|:---------:|-------|
| `energy(EnergyRequest)` | ✅ | Via `analyze E` |
| `minimize(MinimizationRequest)` | ✅ | Parses `.xyz_2` output |
| `hessian(HessianRequest)` | ✅ | Via `testhess`; symmetrized |
| `frequencies(FrequencyRequest)` | ✅ | Via `vibrate` |
| `parameter_gradient(ParameterGradientRequest)` | ❌ | Not implemented |
| `Capability.REUSABLE_STATE` | ❌ | Subprocess per call |

### Performance note

Each energy/frequency evaluation spawns a new Tinker subprocess, writes
temporary parameter and coordinate files, and parses text output. This
makes Tinker significantly slower per evaluation than in-process backends
(~160 ms/eval vs ~5 ms for OpenMM, ~0.1 ms for JAX).

---

## Limitations

- **MM3 only** — does not support Harmonic functional forms.
- **No runtime parameter updates** — each call writes a new parameter file
  and spawns a subprocess.
- **No analytical gradients** — `parameter_gradient()` is not implemented.
- **Standalone PRM limitations** — `_write_standalone_prm()` writes
  bond, angle, torsion, and vdW terms but lacks improper torsions and
  cross-terms. Template-based export is preferred for full `.prm` fidelity.
- **No GPU support** — runs entirely on CPU.
- **External dependency** — requires Tinker executables to be installed
  and discoverable.

---

## Example

```python
from q2mm.backends.contracts import EnergyRequest, FrequencyRequest, PreparationRequest
from q2mm.backends.mm.tinker import TinkerBackend
from q2mm.io.mm3 import load_mm3_fld
from q2mm.io.xyz import load_xyz

mol = load_xyz("molecule.xyz")
ff = load_mm3_fld("mm3.fld")

backend = TinkerBackend(tinker_dir="/opt/tinker/bin")
session = backend.prepare(PreparationRequest(case_id="example", molecule=mol, force_field=ff))
params = session.layout.vector(ff)

e = session.energy(EnergyRequest(parameters=params)).energy
print(f"Energy: {e:.4f} kcal/mol")

freqs = session.frequencies(FrequencyRequest(parameters=params)).frequencies
print(f"Frequencies: {freqs}")
```

---

## See also

- [Backend comparison table](index.md#backend-overview)
- [Parameter transferability](index.md#parameter-transferability)
- [Benchmarks](../benchmarks/index.md)
- [API Reference: TinkerBackend](../reference/q2mm/backends/mm/tinker.md)
