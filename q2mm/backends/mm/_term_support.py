"""Private populated-term checks for built-in backend preparation."""

from __future__ import annotations

import re
from typing import Literal

from q2mm.backends.contracts import PreparationError
from q2mm.models.forcefield import ForceField, TorsionParam

_Term = Literal[
    "CMAP", "bond dipoles", "stretch-bend", "Urey-Bradley", "improper torsions", "vdW reduction", "wildcard torsions"
]


def _has_wildcard_types(torsion: TorsionParam) -> bool:
    """Recognize native X/zero tokens, not substrings or inferred unknown elements."""
    types = tuple(part.strip() for part in torsion.env_id.split("-") if part.strip())
    # An ordinary type such as X1 can have an inferred element X. A complete
    # native type quadruplet is authoritative; generic terms use elements.
    if len(types) != 4:
        types = torsion.elements
    return any(token.strip() == "X" or re.fullmatch(r"[+-]?0+", token.strip()) is not None for token in types)


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
        "wildcard torsions": any(_has_wildcard_types(t) for t in force_field.torsions),
    }
    rejected = [term for term in sorted(unsupported) if populated[term]]
    if rejected:
        raise PreparationError(
            f"{backend} does not support populated {', '.join(rejected)} "
            f"for functional form {force_field.functional_form.value!r}."
        )
