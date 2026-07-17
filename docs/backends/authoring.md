# Authoring a backend plugin

Q2MM backend API version 1 is the public extension contract for adding a
molecular-mechanics engine or a reference-calculation adapter without changing
Q2MM itself. A plugin publishes a lightweight manifest, implements typed
prepared sessions, and verifies the implementation with the public conformance
runner.

## Compatibility boundary

`q2mm.backends.contracts.BACKEND_API_VERSION` is `1`. An API-v1 plugin uses the
entry-point group **`q2mm.backends`** and exactly these manifest keys:

| Key | Required | API-v1 value |
|-----|:--------:|--------------|
| `backend_api_version` | yes | Integer `1` |
| `name` | yes | Registry-safe key matching the entry-point name |
| `role` | yes | `"mm"` or `"reference"` |
| `capability_ceiling` | no | List of `Capability` string values; defaults empty |
| `functional_form_ceiling` | no | List containing `"harmonic"` and/or `"mm3"` for MM; empty for reference |
| `factory` | yes | One `module:Attribute` import string |
| `probe` | no | Mapping with optional `modules` and `executables` string lists |

There are no compatibility aliases for the replaced pre-v1 names
`api_version`, `capabilities`, `forms`, role `"qm"`, `BackendRole.QM`, or the
`QM*Request` classes. They are rejected or absent intentionally.

Validation is strict. The manifest must be a JSON-safe mapping with string keys
and finite scalar, list, or nested-mapping values. Unknown keys, non-integer or
incompatible API versions, unsafe or mismatched names, invalid roles,
capabilities, forms, factory strings, and probe entries are rejected.
`factory` has exactly one colon, a dotted Python module, and one identifier.
Probe module names are dotted identifiers; executable entries contain no
whitespace or NUL bytes.

The current capability vocabulary is:

```text
energy, minimize, hessian, frequencies, parameter_gradient,
coordinate_gradient, hessian_parameter_jacobian, batched_energy,
batched_hessian, geometry_optimization, reusable_state
```

The implementation table below is the complete role matrix: MM backends cannot
declare reference-only capabilities, reference backends cannot declare MM-only
capabilities, and reference backends declare no functional forms.

## Package and manifest

Declare one entry point targeting a descriptor module, never the implementation
module:

```toml
[project.entry-points."q2mm.backends"]
my-backend = "my_backend.descriptor:MANIFEST"
```

The descriptor module contains data and may import the lightweight
`BACKEND_API_VERSION` constant:

```python
from q2mm.backends.contracts import BACKEND_API_VERSION

MANIFEST = {
    "backend_api_version": BACKEND_API_VERSION,
    "name": "my-backend",
    "role": "mm",
    "capability_ceiling": ["energy"],
    "functional_form_ceiling": ["harmonic"],
    "factory": "my_backend.backend:MyBackend",
    "probe": {"modules": ["my_engine"]},
}
```

The entry point may target the mapping directly or a zero-argument provider
returning it. Importing that descriptor module must not import the backend
implementation, enumerate devices, or initialize an engine.

## Discovery and loading guarantees

Importing `q2mm.backends.registry` does not enumerate entry points. The first
catalog, descriptor, registration, availability, or load query creates one
cached, deterministic snapshot; `registry.refresh()` discards it after a plugin
installation or test injection.

Snapshot construction imports only descriptor modules and runs cheap probes.
The implementation named by `factory` is imported only by explicit
`load_backend(name)` or `BackendDescriptor.load()`. Missing dependencies,
descriptor import failures, incompatible versions, invalid manifests, and
broken factories are isolated from healthy backends. Built-ins win conflicts
with their names. If multiple external distributions claim one name, all
claimants are rejected. `discovery_report()` provides advanced typed
diagnostics.

## Static ceilings and runtime truth

The manifest ceilings describe everything any installation or configuration of
the plugin might support. A loaded backend's immutable `BackendInfo` declares
the authoritative exact runtime subsets:

```python
BackendInfo(
    name="My engine",
    role=BackendRole.MM,
    capabilities=frozenset({Capability.ENERGY}),
    functional_forms=frozenset({"harmonic"}),
    provenance=BackendProvenance(
        backend="my-backend",
        role=BackendRole.MM,
        details={"implementation": {"name": "My engine"}},
    ),
)
```

Runtime role must equal descriptor role. Runtime capabilities and forms may be
subsets of their static ceilings but may never overclaim. Runtime provenance
must identify the descriptor's registry name and role. Provenance details are
deeply immutable, JSON-serializable structured data.

## Implementation contract

A backend exposes `info` and `prepare(PreparationRequest)`. An MM preparation
includes a `Molecule` and `ForceField`; a reference preparation includes a
`Molecule` without a force field. Prepared sessions accept the exact request
family for their role:

| Capability | MM request | Reference request | Canonical result |
|------------|------------|-------------------|------------------|
| `energy` | `EnergyRequest` | `ReferenceEnergyRequest` | `EnergyResult` |
| `minimize` | `MinimizationRequest` | — | `GeometryResult` |
| `geometry_optimization` | — | `ReferenceGeometryOptimizationRequest` | `GeometryResult` |
| `hessian` | `HessianRequest` | `ReferenceHessianRequest` | `HessianResult` |
| `frequencies` | `FrequencyRequest` | `ReferenceFrequencyRequest` | `FrequencyResult` |
| `parameter_gradient` | `ParameterGradientRequest` | — | `ParameterGradientResult` |
| `coordinate_gradient` | — | `ReferenceCoordinateGradientRequest` | `CoordinateGradientResult` |
| `hessian_parameter_jacobian` | `HessianJacobianRequest` | — | `HessianJacobianResult` |
| `batched_energy` | `BatchedEnergyRequest` | — | `BatchedEnergyResult` |
| `batched_hessian` | backend-level `prepare_hessian_batches` | — | `BatchedHessianResult` |
| `reusable_state` | Reuse the same prepared session | Reuse the same prepared session | Existing result type |

Deriving the prepared session from `AbstractPreparedBackend` provides exact
request checking, parameter-vector validation, canonical unit and shape checks,
result provenance checks, and `UnsupportedCapabilityError` gates before
undeclared implementation hooks dispatch. Backends must raise Q2MM's typed
backend errors; they must not silently substitute another engine, method, or
capability.

Canonical MM energy is kcal/mol. Reference energy is Hartree. Coordinates are
Å, Hessians Hartree/Bohr², frequencies cm⁻¹, and reference coordinate gradients
Hartree/Bohr. Input molecules, force fields, typed requests, and parameter
vectors are immutable and must remain unchanged.

## Public conformance

`q2mm.backends.conformance` imports only Q2MM core dependencies. Authors provide
an immutable typed `MMConformanceCase` or `ReferenceConformanceCase`:

```python
from q2mm.backends.conformance import MMConformanceCase, run_mm_conformance
from q2mm.backends.registry import get_descriptor, load_backend

outcome = run_mm_conformance(
    MMConformanceCase(
        descriptor=get_descriptor("my-backend"),
        backend=load_backend("my-backend"),
        molecule=small_molecule,
        force_field=small_force_field,
    )
)
```

The default bounded selection runs `ENERGY`. Pass an explicit `capabilities`
frozenset to exercise other declared operations; declared `ENERGY` must remain
selected. The runner validates descriptor/runtime ceilings, exact canonical
results and provenance, mutation safety, all undeclared public gates,
declared `REUSABLE_STATE` automatically by executing one selected operation on
the same prepared session repeatedly, and
backend-level `BATCHED_HESSIAN`. It returns deterministic
`ConformanceOutcome` or raises `ConformanceError`.

See the independently installable
[`examples/backend-plugin`](https://github.com/ericchansen/q2mm/tree/master/examples/backend-plugin)
reference implementation. It is repository guidance and is intentionally absent
from Q2MM wheels and source distributions.
