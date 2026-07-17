"""Lightweight, JSON-safe descriptor for the reference backend plugin."""

from q2mm.backends.contracts import BACKEND_API_VERSION

MANIFEST: dict[str, object] = {
    "backend_api_version": BACKEND_API_VERSION,
    "name": "harmonic-reference",
    "role": "mm",
    "capability_ceiling": ["energy"],
    "functional_form_ceiling": ["harmonic"],
    "factory": "q2mm_reference_backend.backend:HarmonicReferenceBackend",
    "probe": {"modules": ["numpy"]},
}
