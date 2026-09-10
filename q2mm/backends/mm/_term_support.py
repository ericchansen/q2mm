"""Private populated-term checks for built-in backend preparation."""

from __future__ import annotations

from typing import Literal

from q2mm.backends.contracts import PreparationError
from q2mm.models.forcefield import ForceField

_Term = Literal["CMAP", "bond dipoles", "stretch-bend", "Urey-Bradley", "improper torsions", "vdW reduction"]


def _validate_term_support(force_field: ForceField, *, backend: str, unsupported: frozenset[_Term]) -> None:
    """Reject populated terms in a backend's explicit unsupported subset.

    This is not a capability manifest or a claim of support for every term
    outside that subset. Only canonical force-field content is inspected;
    source metadata and molecular reference charges are not energy requests.
    """
    populated: dict[_Term, bool] = {
        "CMAP": bool(force_field.cmaps),
        "bond dipoles": any(b.dipole_moment != 0.0 for b in force_field.bonds),
        "stretch-bend": bool(force_field.stretch_bends),
        "Urey-Bradley": any(
            a.ub_force_constant is not None or a.ub_equilibrium is not None for a in force_field.angles
        ),
        "improper torsions": any(t.is_improper for t in force_field.torsions),
        "vdW reduction": any(v.reduction != 0.0 for v in force_field.vdws),
    }
    rejected = [term for term in sorted(unsupported) if populated[term]]
    if rejected:
        raise PreparationError(
            f"{backend} does not support populated {', '.join(rejected)} "
            f"for functional form {force_field.functional_form.value!r}."
        )
