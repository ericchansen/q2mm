"""Lightweight descriptor manifest for the fixture backend.

Importing this module must NOT import the backend implementation
(:mod:`q2mm_fixture_backend.backend`) or construct anything — it only declares a
JSON-safe manifest mapping.  Q2MM's discovery validates this manifest and only
resolves the ``factory`` import string on an explicit backend load.
"""

from __future__ import annotations

from q2mm.backends.contracts import BACKEND_API_VERSION

# A JSON-safe manifest mapping.  ``factory`` is an import string that points at
# the *implementation* module; it is resolved lazily, never at descriptor
# import/enumeration time.  ``backend_api_version`` is taken from the installed
# runtime contract (``BACKEND_API_VERSION``) rather than hardcoded, so the fixture
# tracks the real descriptor-API version.  Importing the contract is lightweight
# and does NOT import the backend implementation module.
MANIFEST: dict[str, object] = {
    "backend_api_version": BACKEND_API_VERSION,
    "name": "harmonic-fixture",
    "role": "mm",
    "capability_ceiling": ["energy"],
    "functional_form_ceiling": ["harmonic"],
    "factory": "q2mm_fixture_backend.backend:HarmonicFixtureBackend",
    "probe": {"modules": ["numpy"]},
}


def provider() -> dict[str, object]:
    """Return the manifest mapping (equivalent callable-provider entry point)."""
    return dict(MANIFEST)
