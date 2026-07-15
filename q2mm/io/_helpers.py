"""Shared I/O helper utilities used by multiple format modules.

Private module — import from ``q2mm.io`` sub-modules, not directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from q2mm.models.forcefield import (
        AngleParam,
        BondParam,
        ForceField,
        StretchBendParam,
        TorsionParam,
        VdwParam,
    )

from q2mm.models.identifiers import (
    canonicalize_angle_env_id,
    canonicalize_bond_env_id,
)

# ---------------------------------------------------------------------------
# Format compatibility
# ---------------------------------------------------------------------------

_FORMAT_COMPATIBLE_FORMS: dict[str, set[str]] = {
    "mm3_fld": {"mm3"},
    "tinker_prm": {"mm3"},
    "openmm_xml": {"mm3"},
    "amber_frcmod": {"harmonic"},
}


def _validate_form_for_format(ff: ForceField, target_format: str) -> None:
    """Raise ``ValueError`` if the force field's functional form is incompatible with *target_format*."""
    form_value = ff.functional_form.value
    allowed = _FORMAT_COMPATIBLE_FORMS.get(target_format)
    if allowed is not None and form_value not in allowed:
        raise ValueError(
            f"Cannot save a {ff.functional_form!r} force field to {target_format!r} format. "
            f"Compatible forms: {sorted(allowed)}"
        )


# ---------------------------------------------------------------------------
# Env-id splitting
# ---------------------------------------------------------------------------


def _split_env_id(env_id: str, expected_len: int) -> list[str]:
    parts = [part.strip() for part in env_id.split("-") if part.strip()]
    if len(parts) == expected_len:
        return parts
    return []


# ---------------------------------------------------------------------------
# Atom-type cleaning
# ---------------------------------------------------------------------------


def _clean_atom_types(atom_types: list[str] | tuple[str, ...] | None, expected_len: int) -> list[str]:
    if atom_types is None:
        return []
    if isinstance(atom_types, str):
        atom_types = [atom_types]
    cleaned = [
        str(atom_type).strip() for atom_type in atom_types if str(atom_type).strip() and str(atom_type).strip() != "-"
    ]
    return cleaned[:expected_len]


# ---------------------------------------------------------------------------
# Parameter map builders
# ---------------------------------------------------------------------------


def _build_param_maps(params: list, secondary_key: str) -> tuple[dict, dict]:
    """Build ff_row and secondary-key lookup dicts for a list of parameters."""
    by_row = {p.ff_row: p for p in params if p.ff_row is not None}
    by_key = {getattr(p, secondary_key): p for p in params if getattr(p, secondary_key, None)}
    return by_row, by_key


def _build_bond_maps(bonds: list[BondParam]) -> tuple[dict[int, BondParam], dict[str, BondParam]]:
    return _build_param_maps(bonds, "env_id")


def _build_angle_maps(angles: list[AngleParam]) -> tuple[dict[int, AngleParam], dict[str, AngleParam]]:
    return _build_param_maps(angles, "env_id")


def _build_sb_maps(
    stretch_bends: list[StretchBendParam],
) -> tuple[dict[int, StretchBendParam], dict[str, StretchBendParam]]:
    return _build_param_maps(stretch_bends, "env_id")


def _build_vdw_maps(vdws: list[VdwParam]) -> tuple[dict[int, VdwParam], dict[str, VdwParam]]:
    return _build_param_maps(vdws, "atom_type")


# ---------------------------------------------------------------------------
# Export matching
# ---------------------------------------------------------------------------


def _match_for_export(
    ff_row: int | None,
    atom_types: list[str] | tuple[str, ...] | None,
    by_row: dict,
    by_env: dict,
    expected_len: int,
    canonicalize_fn: Callable,
) -> Any:
    """Match a parsed file row to an internal param by ``ff_row`` or ``env_id``.

    Pure: takes the row's primitive identity data (*ff_row*, *atom_types*)
    rather than the row object itself, so it has no dependency on any
    particular format module's private staging-record type.
    """
    if ff_row is not None and ff_row in by_row:
        return by_row[ff_row]
    cleaned = _clean_atom_types(atom_types, expected_len)
    if len(cleaned) == expected_len:
        return by_env.get(canonicalize_fn(cleaned))
    return None


def _match_bond_for_export(
    ff_row: int | None,
    atom_types: list[str] | tuple[str, ...] | None,
    bond_by_row: dict[int, BondParam],
    bond_by_env: dict[str, BondParam],
) -> BondParam | None:
    return _match_for_export(ff_row, atom_types, bond_by_row, bond_by_env, 2, canonicalize_bond_env_id)


def _match_angle_for_export(
    ff_row: int | None,
    atom_types: list[str] | tuple[str, ...] | None,
    angle_by_row: dict[int, AngleParam],
    angle_by_env: dict[str, AngleParam],
) -> AngleParam | None:
    return _match_for_export(ff_row, atom_types, angle_by_row, angle_by_env, 3, canonicalize_angle_env_id)


def _match_sb_for_export(
    ff_row: int | None,
    atom_types: list[str] | tuple[str, ...] | None,
    sb_by_row: dict[int, StretchBendParam],
    sb_by_env: dict[str, StretchBendParam],
) -> StretchBendParam | None:
    return _match_for_export(ff_row, atom_types, sb_by_row, sb_by_env, 3, canonicalize_angle_env_id)


# ---------------------------------------------------------------------------
# Torsion file-value resolver (used by both MM3 and Tinker save)
# ---------------------------------------------------------------------------


def _torsion_file_value(torsions: Sequence[TorsionParam], ff_row: int | None, periodicity: int) -> float | None:
    """Return the file-convention torsion value for *ff_row*/*periodicity*, or ``None`` if unmatched.

    Matches by ``ff_row`` + ``periodicity`` (the file row's column/order).
    The ForceField stores ``V_n / 2`` (our convention), but the legacy
    file format needs ``V_n`` (the raw ``.fld``/``.prm`` value), so the
    match is multiplied by 2. Pure: returns the resolved value (or
    ``None``) instead of mutating a row in place.
    """
    for t in torsions:
        if t.ff_row == ff_row and t.periodicity == periodicity:
            return t.force_constant * 2.0
    return None


# ---------------------------------------------------------------------------
# Equilibrium-angle normalization (used by both MM3 and Tinker parse/save)
# ---------------------------------------------------------------------------


def _normalize_equilibrium_angle(value: float) -> float:
    """Fold a raw equilibrium bond-angle value (degrees) back into ``[0, 180]``.

    Force-field text files can express an equilibrium angle as a value
    above 180 degrees (or even above 360); only ``[0, 180]`` is physically
    meaningful for a bond angle, so any value read from — or about to be
    written to — a file row is folded back into that range.
    """
    if value > 180.0:
        folded = value % 360.0
        return 360.0 - folded if folded > 180.0 else folded
    return value
