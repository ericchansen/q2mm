"""Shared I/O helper utilities used by multiple format modules.

Private module — import from ``q2mm.io`` sub-modules, not directly.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

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


class _TextFileBase:
    """Shared path and lazy-line-reading base for legacy text-file parsers."""

    __slots__ = ("_lines", "path", "directory", "filename")

    def __init__(self, path: str) -> None:
        self._lines: list[str] | None = None
        self.path = os.path.abspath(path)
        self.directory = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)

    @property
    def lines(self) -> list[str]:
        """Return the lines of the file."""
        if self._lines is None:
            with open(self.path) as f:
                self._lines = f.readlines()
        return self._lines

    def write(self, path: str, lines: list[str] | None = None) -> None:
        """Write lines to file at *path* (defaults to ``self.lines``)."""
        if lines is None:
            lines = self.lines
        with open(path, "w") as f:
            f.writelines(lines)


# ---------------------------------------------------------------------------
# Format compatibility & template resolution
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


def _resolve_template_path(
    ff: ForceField,
    template_path: str | Path | None,
    source_format: str,
) -> Path | None:
    """Resolve an explicit *template_path* or fall back to ``ff.source_path`` when formats match."""
    if template_path is not None:
        return Path(template_path)
    if ff.source_format == source_format and ff.source_path is not None:
        return ff.source_path
    return None


# ---------------------------------------------------------------------------
# Env-id splitting & atom-type cleaning
# ---------------------------------------------------------------------------


def _split_env_id(env_id: str, expected_len: int) -> list[str]:
    parts = [part.strip() for part in env_id.split("-") if part.strip()]
    if len(parts) == expected_len:
        return parts
    return []


def _clean_atom_types(atom_types: Sequence[str] | None, expected_len: int) -> list[str]:
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


def _build_param_maps[P: (BondParam, AngleParam, StretchBendParam, VdwParam)](
    params: Sequence[P], secondary_key: str
) -> tuple[dict[int, P], dict[str, P]]:
    """Build ff_row and secondary-key lookup dicts for a sequence of parameters."""
    by_row = {p.ff_row: p for p in params if p.ff_row is not None}
    by_key = {getattr(p, secondary_key): p for p in params if getattr(p, secondary_key, None)}
    return by_row, by_key


def _build_bond_maps(bonds: Sequence[BondParam]) -> tuple[dict[int, BondParam], dict[str, BondParam]]:
    return _build_param_maps(bonds, "env_id")


def _build_angle_maps(angles: Sequence[AngleParam]) -> tuple[dict[int, AngleParam], dict[str, AngleParam]]:
    return _build_param_maps(angles, "env_id")


def _build_sb_maps(
    stretch_bends: Sequence[StretchBendParam],
) -> tuple[dict[int, StretchBendParam], dict[str, StretchBendParam]]:
    return _build_param_maps(stretch_bends, "env_id")


def _build_vdw_maps(vdws: Sequence[VdwParam]) -> tuple[dict[int, VdwParam], dict[str, VdwParam]]:
    return _build_param_maps(vdws, "atom_type")


# ---------------------------------------------------------------------------
# Export matching
# ---------------------------------------------------------------------------


def _match_for_export[P: (BondParam, AngleParam, StretchBendParam)](
    ff_row: int | None,
    atom_types: Sequence[str] | None,
    by_row: dict[int, P],
    by_env: dict[str, P],
    expected_len: int,
    canonicalize_fn: Callable[[Sequence[str]], str],
) -> P | None:
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
    atom_types: Sequence[str] | None,
    bond_by_row: dict[int, BondParam],
    bond_by_env: dict[str, BondParam],
) -> BondParam | None:
    return _match_for_export(ff_row, atom_types, bond_by_row, bond_by_env, 2, canonicalize_bond_env_id)


def _match_angle_for_export(
    ff_row: int | None,
    atom_types: Sequence[str] | None,
    angle_by_row: dict[int, AngleParam],
    angle_by_env: dict[str, AngleParam],
) -> AngleParam | None:
    return _match_for_export(ff_row, atom_types, angle_by_row, angle_by_env, 3, canonicalize_angle_env_id)


def _match_sb_for_export(
    ff_row: int | None,
    atom_types: Sequence[str] | None,
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
