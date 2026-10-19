"""Reference-data backend implementations."""

from __future__ import annotations

from q2mm.backends.contracts import PreparationError, PreparationRequest
from q2mm.models.molecule import Molecule


def _validate_reference_prepare(request: PreparationRequest, *, backend: str) -> Molecule:
    """Validate that a reference preparation request carries only a Molecule."""
    if request.force_field is not None:
        raise PreparationError(f"{backend} reference preparation does not accept a force_field.")
    if request.options:
        raise PreparationError(f"{backend} reference preparation does not accept per-case options.")
    if not isinstance(request.molecule, Molecule):
        raise PreparationError(f"{backend} reference preparation requires a q2mm Molecule.")
    return request.molecule
