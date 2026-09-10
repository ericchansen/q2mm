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

All writer paths also reject nonempty bond-order selectors and non-generic
bond contexts, since native records have no columns for those selectors.
The canonical generic contexts `""` and `"0000 0000"` pass the loss gate.
Template binding treats only those two spellings as equivalent during its
comparison, including when scalar values are edited. It does not mutate
canonical parameters, their fingerprints, or source columns, and does not
relax any other parameter identity check.
The backend standalone model rejects `"00"` placeholder elements in bonds,
angles, proper torsions, and vdW records, even for zero coefficients,
instead of omitting those records. vdW atom types that convert to integer
zero are also rejected, even when their declared element is a real atom.
vdW records must have either a numeric type or an element available for
mapping. Native wildcard records remain supported
as template pass-through; no wildcard matching or expansion is introduced.

| Writer path | Proper torsions | vdW reduction |
|-------------|-----------------|---------------|
| Public standalone section | Requires a template | Written |
| Public or backend template | Existing rows retain scale, phase and periodicity; scalar edits supported | Retained and editable, including an omitted source field |
| Backend standalone model | Distinct folds 1-6 per native type quadruplet, including reversal-equivalent groups; existing native coefficient convention | Nondefault values rejected |

The backend standalone model uses the molecule's actual positive integer
type assignments, with one type per element and one element per type.
Zero or negative molecule types are rejected even if every force-field
label is nonnumeric. Nonnumeric molecule labels retain the existing MM3
element-to-type fallback for known elements; unknown default types and
missing standalone atomic data fail instead of inventing classes or masses.

A numeric vdW class must occur in that actual molecule map: class `5` is
not silently redirected when hydrogen uses `42`. An explicit chemical
element must agree with the element assigned to that class. Numeric-only
labels such as `VdwParam("005", ...)` remain usable when class `5` is
assigned. The canonical model infers `element="005"` in this case and
does not retain whether an element was explicit or inferred. Consequently,
an equivalent numeric element is treated as a class label, not a chemical
claim; conflicting numeric or chemical labels fail. Generic `H1` and an
empty vdW type with `element="H"` still map to hydrogen's actual type.
This boundary does not add multi-type-per-element matching.

The public standalone section has no molecule map and requires a nonempty,
single unquoted vdW type token, even when an element is supplied. It does
not invent an element fallback. Valid native wildcard `0` tokens and
public/template reduction support are unchanged.
Generated bond and angle types obey the same token rule, with exactly two
and three types respectively. Whitespace, comment delimiters (`#`/`!`),
newlines, and quotes inside emitted type tokens are rejected before output
opens. Existing environment splitting and default element-based inference
are unchanged; validation applies to the tokens actually emitted.

Both standalone writers require unique generated bond, angle, and vdW
identities, using the actual emitted types rather than descriptive labels or
row metadata. Bond reversal, angle terminal reversal, and integer class
aliases (for example `"5"`, `"005"`, and `"+5"`) share an identity.
Symbolic labels remain case-sensitive, and emitted tokens are not rewritten.
Repeated identities fail before output opens, even when their metadata,
values, or zero-valued coefficients differ. Ordered native template rows
and their source-row-bound scalar edits are not subject to this standalone
duplicate guard.
Proper torsion components sharing a native type quadruplet, including its
reverse, are grouped into one row without losing components. Their
amplitudes and phases must be finite, and the shared Tinker triplet
validator rejects malformed triples, out-of-range folds, and duplicate
folds even across differently labeled canonical environments.

Template export preserves unmodeled native records (such as `strbnd`,
`ureybrad`, `dipole`, `imptors`, and `opbend`) byte-for-byte alongside
supported scalar edits. This is opaque pass-through, not canonical term
support: supplying a canonical dipole or improper is still rejected even
if the template has a numerically similar native record. Native
out-of-plane and canonical Fourier improper models are not interchangeable.

Public loss gates raise `ValueError`; backend preparation and writer
loss gates raise `PreparationError`. Direct writer binding and scalar/record
representability errors, including template errors, remain `ValueError`;
`prepare()` maps native standalone preflight failures to `PreparationError`
before building the parameter layout.

The backend stages all native parameter records and XYZ/key text before
opening any output. Raw bond/angle/proper-torsion/vdW scalars and converted
bond/angle constants must be finite; required data must fit the native
240-byte record limit. XYZ coordinates are also checked before parameter
export, including template-backed exports. These validation failures leave
absent, existing, and source-file destinations untouched. This is preflight
preservation, not a multi-file filesystem transaction.
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

This toy H2 model illustrates the standalone writer's supported subset;
it is not a calibrated force field. Tinker must be installed and discoverable.
Full MacroModel MM3 `.fld` inputs contain canonical terms that this path
does not support and cannot be passed through the standalone writer.
For template-based use, load a supported native `.prm` with
`load_tinker_prm()`; not every stock parameter library is accepted by that
loader. Unsupported records must not be silently removed to make a
different physical model appear to work.

```python
from q2mm.backends.contracts import EnergyRequest, FrequencyRequest, PreparationRequest
from q2mm.backends.mm.tinker import TinkerBackend
from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm, VdwParam
from q2mm.models.molecule import Molecule

mol = Molecule(
    symbols=("H", "H"),
    atom_types=("5", "5"),
    geometry=((0.0, 0.0, 0.0), (0.8, 0.0, 0.0)),
)
ff = ForceField(
    functional_form=FunctionalForm.MM3,
    bonds=(BondParam(("H", "H"), equilibrium=0.74, force_constant=100.0),),
    vdws=(VdwParam("5", radius=1.2, epsilon=0.02, element="H"),),
)

backend = TinkerBackend()
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
