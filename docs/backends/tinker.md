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
| Electrostatics | Existing native template records only; canonical bond dipoles are rejected |
| 1-4 scaling | MM3 default |

**Functional forms:** MM3 only.

### Parameter export coverage

The public `save_tinker_prm()` serializer and the backend's standalone
writer have different supported subsets. Both reject populated canonical
stretch-bend, Urey-Bradley, bond dipole, improper torsion, CMAP, and
`nonbonded_excluded_atom_types` content rather than silently discard it.
This includes zero-valued stretch-bend/improper records and Urey-Bradley
fields supplied individually or with zero values.

| Writer path | Proper torsions | vdW reduction |
|-------------|-----------------|---------------|
| Public standalone section | Requires a template | Written |
| Public or backend template | Existing rows retain scale, phase and periodicity; scalar edits supported | Retained and editable, including an omitted source field |
| Backend standalone model | Written using its existing native convention | Nondefault values rejected |

Template export preserves unmodeled native records (such as `strbnd`,
`ureybrad`, `dipole`, `imptors`, and `opbend`) byte-for-byte alongside
supported scalar edits. This is opaque pass-through, not canonical term
support: supplying a canonical dipole or improper is still rejected even
if the template has a numerically similar native record. Native
out-of-plane and canonical Fourier improper models are not interchangeable.

Public loss gates raise `ValueError`; backend preparation and writer
loss gates raise `PreparationError`. Template row-binding and scalar
representability errors remain `ValueError`. These failures happen before
parameter output is opened or existing XYZ/key inputs are replaced.
The backend's native headers and coefficient conventions are unchanged;
sharing the loss policy does not unify distinct functional models.

CPU-only writer tests in `test/test_tinker_writers.py` cover rejection,
destination preservation, and supported file roundtrips. They do not
execute Tinker or establish native energy, geometry, or Hessian parity.

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
  bond, angle, proper torsion, and unreduced vdW terms. Unsupported
  populated content fails explicitly; see [parameter export coverage](#parameter-export-coverage)
  for the distinction between canonical terms and opaque template records.
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
