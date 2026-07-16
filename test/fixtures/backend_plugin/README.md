# Out-of-tree backend plugin fixture

`q2mm-backend-plugin-fixture` is a **test-only** installable package that proves
Q2MM's lazy, descriptor-first backend plugin discovery works against an
*installed* `q2mm`. It is not part of the `q2mm` package and must never enter
the `q2mm` wheel or sdist (the release-artifact allowlist in
`scripts/check_release_artifacts.py` enforces this).

## Layout

```
backend_plugin/
├── pyproject.toml                      # declares the q2mm.backends entry point
└── q2mm_fixture_backend/
    ├── descriptor.py                   # lightweight JSON-safe manifest (no impl import)
    └── backend.py                      # HarmonicFixtureBackend (imported only on load)
```

The distribution advertises one entry point in the internal `q2mm.backends`
group, targeting the descriptor manifest:

```toml
[project.entry-points."q2mm.backends"]
harmonic-fixture = "q2mm_fixture_backend.descriptor:MANIFEST"
```

`descriptor.py` imports nothing from `backend.py`, so descriptor enumeration and
`registry.catalog()` never import the implementation. The implementation module
is imported only when the manifest's `factory` import string is resolved by an
explicit `load_backend("harmonic-fixture")` / `BackendDescriptor.load()`.

## Backend

`HarmonicFixtureBackend` declares exactly one capability — `ENERGY` — with the
`harmonic` functional form, and computes a genuine harmonic bond-stretch energy
`E = Σ k·(r − r₀)²` (kcal/mol) from the prepared molecule geometry and the force
field's matched bond parameters. Every other capability is left to
`AbstractPreparedBackend`, which raises `UnsupportedCapabilityError`.

## Usage

- **Unit tests** expose the package on an isolated `sys.path` and inject fake
  entry points (or a temporary `.dist-info`), so nothing is `pip install`ed into
  the developer environment — see `test/test_backend_discovery.py`.
- **Release artifact checker** performs the only real install: it installs the
  rebuilt `q2mm` wheel plus this fixture (`--no-deps`) in a fresh virtualenv and
  proves discovery, lazy loading, ENERGY conformance, and CLI listing
  (`external-plugin=ok`).
