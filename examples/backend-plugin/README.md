# Q2MM backend API v1 reference plugin

This independently installable example shows the complete public contract for
an out-of-tree Q2MM molecular-mechanics backend. It is source-tree guidance for
backend authors; Q2MM's wheel and sdist intentionally exclude `examples/`.

## Descriptor-first package

The distribution declares exactly one entry point:

```toml
[project.entry-points."q2mm.backends"]
harmonic-reference = "q2mm_reference_backend.descriptor:MANIFEST"
```

`descriptor.py` contains one JSON-safe mapping:

```python
MANIFEST = {
    "backend_api_version": BACKEND_API_VERSION,
    "name": "harmonic-reference",
    "role": "mm",
    "capability_ceiling": ["energy"],
    "functional_form_ceiling": ["harmonic"],
    "factory": "q2mm_reference_backend.backend:HarmonicReferenceBackend",
    "probe": {"modules": ["numpy"]},
}
```

The entry-point name and manifest `name` must match. Importing the descriptor
must not import `backend.py`, initialize an engine, or inspect devices.
Q2MM resolves `factory` only for an explicit
`load_backend("harmonic-reference")`.

`backend.py` implements `Backend.info`, `Backend.prepare`, and a prepared
session derived from `AbstractPreparedBackend`. The base class validates typed
requests, canonical result units and shapes, provenance, parameter-vector
lengths, and undeclared-capability gates.

## Install and verify

From a Q2MM source checkout:

```bash
python -m pip install .
python -m pip install ./examples/backend-plugin
q2mm-benchmark list
```

Authors should run the public dependency-light conformance driver with a small,
deterministic molecule and force field:

```python
from q2mm.backends.conformance import MMConformanceCase, run_mm_conformance
from q2mm.backends.registry import get_descriptor, load_backend

descriptor = get_descriptor("harmonic-reference")
backend = load_backend("harmonic-reference")
outcome = run_mm_conformance(
    MMConformanceCase(
        descriptor=descriptor,
        backend=backend,
        molecule=molecule,
        force_field=force_field,
    )
)
assert outcome.backend == "harmonic-reference"
```

`BACKEND_API_VERSION == 1` is the compatibility boundary. API-v1 manifests use
only `backend_api_version`, `name`, `role`, `capability_ceiling`,
`functional_form_ceiling`, `factory`, and optional `probe`; pre-v1 field names
are rejected rather than aliased.
