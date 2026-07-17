"""MM3 .fld file format I/O."""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from q2mm import constants as co
from q2mm.io._helpers import (
    _build_angle_maps,
    _build_bond_maps,
    _build_sb_maps,
    _build_vdw_maps,
    _match_angle_for_export,
    _match_bond_for_export,
    _match_sb_for_export,
    _normalize_equilibrium_angle,
    _split_env_id,
    _torsion_file_value,
    _validate_form_for_format,
)
from q2mm.models.forcefield import (
    AngleParam,
    BondParam,
    ForceField,
    FunctionalForm,
    StretchBendParam,
    TorsionParam,
    VdwParam,
)
from q2mm.models.identifiers import (
    _extract_element,
    canonicalize_angle_env_id,
    canonicalize_bond_env_id,
)
from q2mm.models.units import (
    canonical_to_mm3_angle_k,
    canonical_to_mm3_bond_k,
    canonical_to_mm3_sb_k,
    mm3_angle_k_to_canonical,
    mm3_bond_k_to_canonical,
    mm3_sb_k_to_canonical,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Mm3ParameterRow:
    """One MM3 ``.fld`` file row, staged during parsing/serialization.

    Represents a single physical value read from (or to be written back
    to) an MM3 ``.fld`` file — one bond/angle/torsion/stretch-bend/vdW
    scalar — before it is converted into (or matched against) an
    immutable :class:`~q2mm.models.forcefield.BondParam` /
    :class:`~q2mm.models.forcefield.AngleParam` / etc. record. Parser-private
    to this module; never exported. Carries no optimizer-facing state
    (step sizes, allowed ranges, active/frozen partition) — that
    vocabulary lives entirely in
    :class:`q2mm.models.parameters.ParameterLayout` /
    :class:`~q2mm.models.parameters.ActiveParameterSpace`. Mutable only so
    :func:`save_mm3_fld` can overwrite ``value`` in place while staging a
    template file for re-export. ``slots=True`` enforces this exact field
    set — no arbitrary attribute can be attached later.

    Attributes:
        ptype: Parameter type (``"ae"``, ``"af"``, ``"be"``, ``"bf"``,
            ``"df"``, ``"imp1"``, ``"imp2"``, ``"sb"``, ``"q"``,
            ``"vdwr"``, ``"vdwfc"``).
        value: The row's numeric value, already in ``.fld`` (file)
            convention/units.
        ff_row: 1-based row number in the ``.fld`` file.
        ff_col: Column index within the row (1-6 depending on *ptype*;
            torsion V4/V5/V6 continuation values use 4-6).
        atom_types: Resolved atom-type strings for this row (digit
            references already resolved to concrete types).
        bond_order: Bond-order symbol from the file (``"-"`` single,
            ``"="`` double, ``"*"`` aromatic, ``"%"`` triple); only ever
            set for bond ptypes (``"be"``/``"bf"``/``"q"``).
        context: MM3 context flags (e.g. ``"O200 0000"``); only ever set
            for bond ptypes.

    """

    ptype: str
    value: float
    ff_row: int
    ff_col: int
    atom_types: list[str] = field(default_factory=list)
    bond_order: str = ""
    context: str = ""


# MM3 fixed-format column positions
COM_POS_START = 96
P_1_START = 23
P_1_END = 33
P_2_START = 34
P_2_END = 44
P_3_START = 45
P_3_END = 55
# Context flags occupy cols 56–65 (two 4-char codes separated by space)
CTX_START = 56
CTX_END = 66
# Bond-order symbol is at col 7 in standard section, col 6 in OPT
_BOND_ORDER_CHARS = frozenset({"-", "=", "*", "%"})
_GENERIC_CONTEXT = "0000 0000"


# ---------------------------------------------------------------------------
# Atom-type helpers
# ---------------------------------------------------------------------------


def _default_mm3_atom_types(elements: tuple[str, ...]) -> list[str]:
    counts: dict[str, int] = {}
    atom_types = []
    for element in elements:
        normalized = _extract_element(element)
        count = counts.get(normalized, 0) + 1
        counts[normalized] = count
        if len(normalized) == 1:
            atom_types.append(f"{normalized}{count}")
        else:
            atom_types.append(normalized[:2].upper())
    return atom_types


def _mm3_atom_types(env_id: str, elements: tuple[str, ...]) -> list[str]:
    parts = _split_env_id(env_id, len(elements))
    if parts and all(len(part) <= 2 for part in parts):
        return parts
    return _default_mm3_atom_types(elements)


# ---------------------------------------------------------------------------
# Line formatters
# ---------------------------------------------------------------------------


def _format_mm3_bond_line(atom_types: list[str], equilibrium: float, force_constant: float) -> str:
    prefix = f" 1  {atom_types[0]:>2} - {atom_types[1]:>2}{'':12}"
    return f"{prefix}{equilibrium:10.4f} {force_constant:10.4f}\n"


def _format_mm3_angle_line(atom_types: list[str], equilibrium: float, force_constant: float) -> str:
    prefix = f" 2  {atom_types[0]:>2} - {atom_types[1]:>2} - {atom_types[2]:>2}{'':7}"
    return f"{prefix}{equilibrium:10.4f} {force_constant:10.4f}\n"


def _format_mm3_torsion_line(atom_types: list[str], v1: float, v2: float, v3: float) -> str:
    prefix = f" 4  {atom_types[0]:>2} - {atom_types[1]:>2} - {atom_types[2]:>2} - {atom_types[3]:>2}  "
    return f"{prefix}{v1:10.4f} {v2:10.4f} {v3:10.4f}\n"


def _format_mm3_vdw_line(vdw: VdwParam) -> str:
    return f"  {vdw.atom_type:<3} {vdw.radius:10.4f} {vdw.epsilon:10.4f} {vdw.reduction:10.4f}                                   0000    O 1\n"


# ---------------------------------------------------------------------------
# VdW parsing / updating
# ---------------------------------------------------------------------------


def _parse_mm3_vdw_params(path: Path) -> list[VdwParam]:
    vdws: list[VdwParam] = []
    in_vdw_section = False
    for row, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped == "-6":
            in_vdw_section = True
            continue
        if not in_vdw_section:
            continue
        if stripped.startswith("-2") or "END OF NONBONDED INTERACTIONS" in stripped:
            break
        parts = raw_line.split()
        if len(parts) < 3:
            continue
        try:
            radius = float(parts[1])
            epsilon = float(parts[2])
        except ValueError:
            continue
        atom_type = parts[0]
        vdws.append(
            VdwParam(
                atom_type=atom_type,
                radius=radius,
                epsilon=epsilon,
                reduction=float(parts[3]) if len(parts) > 3 else 0.0,
                label=f"MM3 row {row}",
                ff_row=row,
            )
        )
    return vdws


def _splice_fixed(line: str, start: int, width: int, value: float) -> str:
    """Overwrite one fixed-width numeric column, preserving all other bytes.

    Returns *line* unchanged if the column falls outside the existing line
    length (the field was not present in fixed position), so trailing MM3
    columns — formal charge, context flag, opt descriptor — are never
    disturbed.
    """
    if len(line) < start + width:
        return line
    field = f"{value:{width}.4f}"
    if len(field) > width:
        # The value is too wide for the fixed column.  ``f"{v:{width}.4f}"``
        # treats *width* as a minimum, so writing it anyway would push every
        # trailing byte to the right and break the byte-stability guarantee.
        # Leave the line untouched instead.
        logger.warning(
            "Value %.4f does not fit MM3 column width %d; leaving line unchanged.",
            value,
            width,
        )
        return line
    return line[:start] + field + line[start + width :]


# Fixed MM3 vdW numeric columns (0-based start, field width). These match
# the columns the loader reads (``line[5:15]`` / ``line[16:26]``) so a
# save→load round-trip is byte-stable.
_VDW_RADIUS_COL = (5, 10)
_VDW_EPSILON_COL = (16, 10)
_VDW_REDUCTION_COL = (27, 10)


def _update_mm3_vdw_lines(path: Path, vdws: list[VdwParam]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    by_row, by_type = _build_vdw_maps(vdws)
    for index, line in enumerate(lines):
        row = index + 1
        match = by_row.get(row)
        parts = line.split()
        if match is None and parts:
            match = by_type.get(parts[0].strip())
        if match is None:
            continue
        updated = _splice_fixed(line, *_VDW_RADIUS_COL, match.radius)
        updated = _splice_fixed(updated, *_VDW_EPSILON_COL, match.epsilon)
        updated = _splice_fixed(updated, *_VDW_REDUCTION_COL, match.reduction)
        lines[index] = updated
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# MM3 label regex matchers
# ---------------------------------------------------------------------------


def match_mm3_label(mm3_label: str) -> re.Match[str] | None:
    """Check whether a line has a recognized MM3* parameter label.

    The label is the first 2 characters in the line containing the parameter
    in a Schrödinger mm3.fld file.

    Args:
        mm3_label (str): Line or string whose first 2 characters are checked.

    Returns:
        (re.Match | None): Match object if the label is recognized, else None.

    """
    return re.match(r"[\s5a-z][1-5]", mm3_label)


def match_mm3_vdw(mm3_label: str) -> re.Match[str] | None:
    """Match MM3* label for van der Waals parameters.

    Args:
        mm3_label (str): Line or string whose first 2 characters are checked.

    Returns:
        (re.Match | None): Match object if the label matches, else None.

    """
    return re.match(r"[\sa-z]6", mm3_label)


def match_mm3_bond(mm3_label: str) -> re.Match[str] | None:
    """Match MM3* label for bonds.

    Args:
        mm3_label (str): Line or string whose first 2 characters are checked.

    Returns:
        (re.Match | None): Match object if the label matches, else None.

    """
    return re.match(r"[\sa-z]1", mm3_label)


def match_mm3_angle(mm3_label: str) -> re.Match[str] | None:
    """Match MM3* label for angles.

    Args:
        mm3_label (str): Line or string whose first 2 characters are checked.

    Returns:
        (re.Match | None): Match object if the label matches, else None.

    """
    return re.match(r"[\sa-z]2", mm3_label)


def match_mm3_stretch_bend(mm3_label: str) -> re.Match[str] | None:
    """Match MM3* label for stretch-bends.

    Args:
        mm3_label (str): Line or string whose first 2 characters are checked.

    Returns:
        (re.Match | None): Match object if the label matches, else None.

    """
    return re.match(r"[\sa-z]3", mm3_label)


def match_mm3_torsion(mm3_label: str) -> re.Match[str] | None:
    """Match MM3* label for all orders of torsional parameters.

    Args:
        mm3_label (str): Line or string whose first 2 characters are checked.

    Returns:
        (re.Match | None): Match object if the label matches, else None.

    """
    return re.match(r"[\sa-z]4|54", mm3_label)


def match_mm3_lower_torsion(mm3_label: str) -> re.Match[str] | None:
    """Match MM3* label for torsions (1st through 3rd order).

    Args:
        mm3_label (str): Line or string whose first 2 characters are checked.

    Returns:
        (re.Match | None): Match object if the label matches, else None.

    """
    return re.match(r"[\sa-z]4", mm3_label)


def match_mm3_higher_torsion(mm3_label: str) -> re.Match[str] | None:
    """Match MM3* label for torsions (4th through 6th order).

    Args:
        mm3_label (str): Line or string whose first 2 characters are checked.

    Returns:
        (re.Match | None): Match object if the label matches, else None.

    """
    return re.match("54", mm3_label)


def match_mm3_improper(mm3_label: str) -> re.Match[str] | None:
    """Match MM3* label for improper torsions.

    Args:
        mm3_label (str): Line or string whose first 2 characters are checked.

    Returns:
        (re.Match | None): Match object if the label matches, else None.

    """
    return re.match(r"[\sa-z]5", mm3_label)


# ---------------------------------------------------------------------------
# SMILES helpers
# ---------------------------------------------------------------------------


def _split_smiles(smiles: str) -> list[str]:
    """Split an MM3* SMILES string into individual atom tokens."""
    split = re.split(co.RE_SPLIT_ATOMS, smiles)
    return [s for s in split if s]


def _convert_smiles_to_types(smiles: str) -> list[str]:
    """Convert an MM3* SMILES string to a list of atom types."""
    atom_types = _split_smiles(smiles)
    return _convert_to_types(atom_types, atom_types)


def _convert_to_types(atom_labels: list[str], atom_types: list[str]) -> list[str]:
    """Convert atom labels (which may be digit references) to atom types."""
    return [atom_types[int(x) - 1] if x.strip().isdigit() and x != "00" else x for x in atom_labels]


# ---------------------------------------------------------------------------
# Standalone import / export
# ---------------------------------------------------------------------------

_NONBONDED_EXCLUSION_DIRECTIVE = " C  Q2MM-NONBONDED-EXCLUDED-ATOM-TYPES "


def _parse_nonbonded_exclusions(lines: list[str]) -> tuple[str, ...]:
    """Parse Q2MM's round-trippable MM3 zero-center declaration."""
    matches = [line for line in lines if line.startswith(_NONBONDED_EXCLUSION_DIRECTIVE)]
    if len(matches) > 1:
        raise ValueError("MM3 file contains multiple Q2MM nonbonded-exclusion directives.")
    if not matches:
        return ()
    payload = matches[0][len(_NONBONDED_EXCLUSION_DIRECTIVE) :].strip()
    try:
        values = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed Q2MM nonbonded-exclusion directive in MM3 file.") from exc
    if (
        not isinstance(values, list)
        or not all(isinstance(value, str) and value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("Q2MM nonbonded-exclusion directive must contain unique non-empty atom-type strings.")
    return tuple(values)


def _write_nonbonded_exclusions(path: Path, values: tuple[str, ...]) -> None:
    """Replace the Q2MM zero-center declaration in an exported MM3 file."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines = [line for line in lines if not line.startswith(_NONBONDED_EXCLUSION_DIRECTIVE)]
    if values:
        directive = f"{_NONBONDED_EXCLUSION_DIRECTIVE}{json.dumps(list(values), separators=(',', ':'))}\n"
        lines.insert(0, directive)
    path.write_text("".join(lines), encoding="utf-8")


def _mm3_import_ff(
    path: str | Path, sub_search: str = "OPT", *, include_standard: bool = True
) -> tuple[list[_Mm3ParameterRow], list[str]]:
    """Read parameter rows from an mm3.fld file.

    Args:
        path: Path to the mm3.fld file.
        sub_search: Substructure name to look for (default ``"OPT"``).
        include_standard: When ``True`` (the default), also parse standard
            MM3 bond, angle, torsion and stretch-bend parameters from the
            main body of the file (outside the substructure section).  These
            serve as the base layer that substructure parameters override.

    Returns a ``(rows, lines)`` tuple where *rows* is the list of
    :class:`_Mm3ParameterRow` objects and *lines* is the raw file content
    (as returned by ``readlines``).

    """
    path = str(path)
    rows: list[_Mm3ParameterRow] = []
    smiles_list: list[str] = []
    sub_names: list[str] = []
    atom_types_list: list[list[str]] = []
    atom_type_equivalencies: dict[str, str] = {}

    with open(path) as f:
        all_lines = f.readlines()

    logger.log(15, f"READING: {path}")
    section_sub = False
    section_smiles = False
    section_atm_eqv = False

    for i, line in enumerate(all_lines):
        if section_atm_eqv:
            if line.startswith(" C") and len(atom_type_equivalencies) > 0:
                section_atm_eqv = False
                continue
            elif not line.startswith(" C") and not line.startswith("-5"):
                equivalency = [typ.strip() for typ in line.split()[1:]]
                for typ in equivalency[1:]:
                    atom_type_equivalencies[typ] = equivalency[0]
                continue

        # Substructure header
        if not section_sub and sub_search in line and line.startswith(" C"):
            matched = re.match(rf"\sC\s+({co.RE_SUB})\s+", line)
            assert matched is not None, f"[L{i + 1}] Can't read substructure name: {line}"
            if matched is not None:
                section_sub = True
                sub_name = matched.group(1).strip()
                sub_names.append(sub_name)
                logger.log(15, f"[L{i + 1}] Start of substructure: {sub_name}")
                section_smiles = True
                continue
        elif section_smiles is True:
            matched = re.match(rf"\s9\s+({co.RE_SMILES})\s", line)
            assert matched is not None, f"[L{i + 1}] Can't read substructure SMILES: {line}"
            smi = matched.group(1)
            smiles_list.append(smi)
            atom_types_list.append(_convert_smiles_to_types(smi))
            logger.log(15, f"  -- SMILES: {smiles_list[-1]}")
            logger.log(15, "  -- Atom types: {}".format(" ".join(atom_types_list[-1])))
            section_smiles = False
            continue
        elif section_sub and line.startswith("-3"):
            logger.log(15, f"[L{i}] End of substructure: {sub_names[-1]}")
            section_sub = False
            continue

        if sub_search in line or section_sub or include_standard:
            # Bonds
            if match_mm3_bond(line):
                logger.log(5, "[L{}] Found bond:\n{}".format(i + 1, line.strip("\n")))
                bond_order = ""
                context = ""
                if section_sub:
                    atm_lbls = [line[4:6], line[8:10]]
                    atm_typs = _convert_to_types(atm_lbls, atom_types_list[-1])
                    # OPT sections: bond-order symbol at col 6 (between atoms)
                    if len(line) > 6 and line[6] in _BOND_ORDER_CHARS:
                        bond_order = line[6]
                else:
                    atm_typs = [line[4:6], line[9:11]]
                    comment = line[COM_POS_START:].strip()
                    sub_names.append(comment)
                    # Standard section: bond-order symbol at col 7
                    if len(line) > 7 and line[7] in _BOND_ORDER_CHARS:
                        bond_order = line[7]
                    # Context flags at cols 56-65
                    if len(line) > CTX_END:
                        ctx = line[CTX_START:CTX_END].strip()
                        if ctx and ctx != "0000 0000":
                            context = ctx
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 2:
                    continue
                rows.extend(
                    (
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="be",
                            ff_col=1,
                            ff_row=i + 1,
                            value=parm_cols[0],
                            bond_order=bond_order,
                            context=context,
                        ),
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="bf",
                            ff_col=2,
                            ff_row=i + 1,
                            value=parm_cols[1],
                            bond_order=bond_order,
                            context=context,
                        ),
                    )
                )
                with contextlib.suppress(IndexError):
                    rows.append(
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="q",
                            ff_col=3,
                            ff_row=i + 1,
                            value=parm_cols[2],
                            bond_order=bond_order,
                            context=context,
                        )
                    )
                continue

            # Angles
            elif match_mm3_angle(line):
                logger.log(5, "[L{}] Found angle:\n{}".format(i + 1, line.strip("\n")))
                if section_sub:
                    atm_lbls = [line[4:6], line[8:10], line[12:14]]
                    atm_typs = _convert_to_types(atm_lbls, atom_types_list[-1])
                else:
                    atm_typs = [line[4:6], line[9:11], line[14:16]]
                    comment = line[COM_POS_START:].strip()
                    sub_names.append(comment)
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 2:
                    continue
                rows.extend(
                    (
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="ae",
                            ff_col=1,
                            ff_row=i + 1,
                            value=_normalize_equilibrium_angle(parm_cols[0]),
                        ),
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="af",
                            ff_col=2,
                            ff_row=i + 1,
                            value=parm_cols[1],
                        ),
                    )
                )
                continue

            # Stretch-bends
            elif match_mm3_stretch_bend(line):
                logger.log(5, "[L{}] Found stretch-bend:\n{}".format(i + 1, line.strip("\n")))
                if section_sub:
                    atm_lbls = [line[4:6], line[8:10], line[12:14]]
                    atm_typs = _convert_to_types(atm_lbls, atom_types_list[-1])
                else:
                    atm_typs = [line[4:6], line[9:11], line[14:16]]
                    comment = line[COM_POS_START:].strip()
                    sub_names.append(comment)
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 1:
                    continue
                rows.append(
                    _Mm3ParameterRow(
                        atom_types=atm_typs,
                        ptype="sb",
                        ff_col=1,
                        ff_row=i + 1,
                        value=parm_cols[0],
                    )
                )
                continue

            # Torsions (1st through 3rd order)
            elif match_mm3_lower_torsion(line):
                logger.log(5, "[L{}] Found torsion:\n{}".format(i + 1, line.strip("\n")))
                if section_sub:
                    atm_lbls = [line[4:6], line[8:10], line[12:14], line[16:18]]
                    atm_typs = _convert_to_types(atm_lbls, atom_types_list[-1])
                else:
                    atm_typs = [line[4:6], line[9:11], line[14:16], line[19:21]]
                    comment = line[COM_POS_START:].strip()
                    sub_names.append(comment)
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 3:
                    continue
                rows.extend(
                    (
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="df",
                            ff_col=1,
                            ff_row=i + 1,
                            value=parm_cols[0],
                        ),
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="df",
                            ff_col=2,
                            ff_row=i + 1,
                            value=parm_cols[1],
                        ),
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="df",
                            ff_col=3,
                            ff_row=i + 1,
                            value=parm_cols[2],
                        ),
                    )
                )
                continue

            # Higher order torsions (4th through 6th)
            elif match_mm3_higher_torsion(line):
                if not rows or rows[-1].ptype != "df":
                    continue
                logger.log(
                    5,
                    "[L{}] Found higher order torsion:\n{}".format(i + 1, line.strip("\n")),
                )
                atm_typs = rows[-1].atom_types
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 3:
                    continue
                rows.extend(
                    (
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="df",
                            ff_col=4,
                            ff_row=i + 1,
                            value=parm_cols[0],
                        ),
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="df",
                            ff_col=5,
                            ff_row=i + 1,
                            value=parm_cols[1],
                        ),
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="df",
                            ff_col=6,
                            ff_row=i + 1,
                            value=parm_cols[2],
                        ),
                    )
                )
                continue

            # Improper torsions
            elif match_mm3_improper(line):
                logger.log(5, "[L{}] Found torsion:\n{}".format(i + 1, line.strip("\n")))
                if section_sub:
                    atm_lbls = [line[4:6], line[8:10], line[12:14], line[16:18]]
                    atm_typs = _convert_to_types(atm_lbls, atom_types_list[-1])
                else:
                    atm_typs = [line[4:6], line[9:11], line[14:16], line[19:21]]
                    comment = line[COM_POS_START:].strip()
                    sub_names.append(comment)
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 2:
                    continue
                rows.extend(
                    (
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="imp1",
                            ff_col=1,
                            ff_row=i + 1,
                            value=parm_cols[0],
                        ),
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="imp2",
                            ff_col=2,
                            ff_row=i + 1,
                            value=parm_cols[1],
                        ),
                    )
                )
                continue

            # VdW inside substructure
            elif match_mm3_vdw(line):
                logger.log(5, "[L{}] Found vdw:\n{}".format(i + 1, line.strip("\n")))
                if not section_sub:
                    continue
                atm_lbls = [line[4:6], line[8:10]]
                atm_typs = _convert_to_types(atm_lbls, atom_types_list[-1])
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 2:
                    continue
                rows.extend(
                    (
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="vdwr",
                            ff_col=1,
                            ff_row=i + 1,
                            value=parm_cols[0],
                        ),
                        _Mm3ParameterRow(
                            atom_types=atm_typs,
                            ptype="vdwfc",
                            ff_col=2,
                            ff_row=i + 1,
                            value=parm_cols[1],
                        ),
                    )
                )
                continue

        # -6 marks start of Van der Waals section
        if line.startswith("-6"):
            continue
        if "New Atom Type Equivalencies" in line:
            section_atm_eqv = True
            continue

    logger.log(15, f"  -- Read {len(rows)} parameters.")
    return rows, all_lines


def _mm3_export_ff(path: str | Path, rows: list[_Mm3ParameterRow], lines: list[str]) -> None:
    """Write parameter rows back to an mm3.fld file at fixed column positions."""
    for row in rows:
        logger.log(1, f">>> row: {row} row.value: {row.value}")
        line = lines[row.ff_row - 1]
        if abs(row.value) > 999.0:
            logger.warning(f"Value of {row} is too high! Skipping write.")
        # Higher-order torsion amplitudes V4/V5/V6 (ff_col 4/5/6) live in the
        # same three physical parameter columns as V1/V2/V3 but on the "54"
        # continuation line, which is addressed by their own ``ff_row``.  Map
        # them onto the same columns so higher-order torsions round-trip.
        elif row.ff_col in (1, 4):
            lines[row.ff_row - 1] = line[:P_1_START] + f"{row.value:10.4f}" + line[P_1_END:]
        elif row.ff_col in (2, 5):
            lines[row.ff_row - 1] = line[:P_2_START] + f"{row.value:10.4f}" + line[P_2_END:]
        elif row.ff_col in (3, 6):
            lines[row.ff_row - 1] = line[:P_3_START] + f"{row.value:10.4f}" + line[P_3_END:]
    with open(path, "w") as f:
        f.writelines(lines)
    logger.log(10, f"WROTE: {path}")


# ---------------------------------------------------------------------------
# Public load / save
# ---------------------------------------------------------------------------


def load_mm3_fld(path: str | Path, *, include_standard: bool = True) -> ForceField:
    """Load from Schrödinger MM3 .fld file.

    Args:
        path: Path to the mm3.fld file.
        include_standard: When ``True`` (the default), load standard MM3
            parameters from the main body of the file in addition to the
            substructure section.  Standard parameters serve as the base
            layer that substructure parameters override.  Set to ``False``
            to load only substructure parameters.

    Returns:
        ForceField: A force field with bond, angle, torsion and vdW
        parameters.

    """
    parsed_rows, source_lines = _mm3_import_ff(path, include_standard=include_standard)

    bonds = []
    angles = []
    stretch_bends: list[StretchBendParam] = []
    torsions = []
    vdws = _parse_mm3_vdw_params(Path(path))

    # Pre-build lookup for equilibrium values by (ptype, ff_row)
    eq_lookup = {}
    dipole_lookup: dict[int, float] = {}  # ff_row → dipole moment (Debye)
    for row in parsed_rows:
        if row.ptype in ("be", "ae"):
            eq_lookup[(row.ptype, row.ff_row)] = row.value
        elif row.ptype == "q":
            dipole_lookup[row.ff_row] = row.value

    for row in parsed_rows:
        # Extract element letters from atom type (e.g., 'C1' -> 'C', ' F' -> 'F')
        atom_types = [t.strip() for t in row.atom_types if t.strip() and t.strip() != "-"]

        if row.ptype == "bf" and len(atom_types) >= 2:
            elems = tuple(_extract_element(t) for t in atom_types[:2])
            env_id = canonicalize_bond_env_id(atom_types[:2])
            eq_val = eq_lookup.get(("be", row.ff_row), 0.0)
            bonds.append(
                BondParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=mm3_bond_k_to_canonical(row.value),
                    label=f"MM3 row {row.ff_row}",
                    env_id=env_id,
                    ff_row=row.ff_row,
                    bond_order=row.bond_order,
                    context=row.context,
                    dipole_moment=dipole_lookup.get(row.ff_row, 0.0),
                )
            )

        elif row.ptype == "af" and len(atom_types) >= 3:
            elems = tuple(_extract_element(t) for t in atom_types[:3])
            env_id = canonicalize_angle_env_id(atom_types[:3])
            eq_val = eq_lookup.get(("ae", row.ff_row), 0.0)
            angles.append(
                AngleParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=mm3_angle_k_to_canonical(row.value),
                    label=f"MM3 row {row.ff_row}",
                    env_id=env_id,
                    ff_row=row.ff_row,
                )
            )

        elif row.ptype == "df" and len(atom_types) >= 4:
            elems = tuple(_extract_element(t) for t in atom_types[:4])
            env_id = "-".join(t.strip() for t in atom_types[:4])
            periodicity = row.ff_col
            # MM3 torsion alternates signs by order:
            #   (V1/2)(1+cos ω) + (V2/2)(1−cos 2ω) + (V3/2)(1+cos 3ω)
            #   + (V4/2)(1−cos 4ω) + (V5/2)(1+cos 5ω) + (V6/2)(1−cos 6ω)
            # The .fld stores V_n (full amplitude); our energy formula uses
            # k*(1+cos(nφ−γ)) with k = V_n/2.  Even orders need γ=180° for
            # the minus sign: (1+cos(nω−π)) = (1−cos nω).
            phase = 180.0 if periodicity % 2 == 0 else 0.0
            torsions.append(
                TorsionParam(
                    elements=elems,
                    periodicity=periodicity,
                    force_constant=row.value / 2.0,
                    phase=phase,
                    label=f"MM3 row {row.ff_row} V{periodicity}",
                    env_id=env_id,
                    ff_row=row.ff_row,
                )
            )

        elif row.ptype in ("imp1", "imp2") and len(atom_types) >= 4:
            elems = tuple(_extract_element(t) for t in atom_types[:4])
            env_id = "-".join(t.strip() for t in atom_types[:4])
            periodicity = 1 if row.ptype == "imp1" else 2
            phase = 180.0 if periodicity == 2 else 0.0
            torsions.append(
                TorsionParam(
                    elements=elems,
                    periodicity=periodicity,
                    force_constant=row.value / 2.0,
                    phase=phase,
                    label=f"MM3 row {row.ff_row} imp V{periodicity}",
                    env_id=env_id,
                    ff_row=row.ff_row,
                    is_improper=True,
                )
            )

        elif row.ptype == "sb" and len(atom_types) >= 3:
            elems = tuple(_extract_element(t) for t in atom_types[:3])
            env_id = canonicalize_angle_env_id(atom_types[:3])
            stretch_bends.append(
                StretchBendParam(
                    elements=elems,
                    force_constant=mm3_sb_k_to_canonical(row.value),
                    label=f"MM3 row {row.ff_row} SB",
                    env_id=env_id,
                    ff_row=row.ff_row,
                )
            )

    ff = ForceField(
        name=f"MM3 from {Path(path).name}",
        bonds=bonds,
        angles=angles,
        stretch_bends=stretch_bends,
        torsions=torsions,
        vdws=vdws,
        source_path=Path(path),
        source_format="mm3_fld",
        functional_form=FunctionalForm.MM3,
        nonbonded_excluded_atom_types=_parse_nonbonded_exclusions(source_lines),
    )
    return ff


def save_mm3_fld(
    ff: ForceField,
    path: str | Path,
    template_path: str | Path | None = None,
    *,
    substructure_name: str = "Generated",
    smiles: str = "AUTO",
) -> Path:
    """Write the force field to MM3 .fld format.

    If a template path is provided, or this force field came from
    :func:`load_mm3_fld`, the existing file is updated in-place via the
    legacy MM3 exporter so comments and unrelated parameters are preserved.

    Otherwise, a self-contained standard-parameter MM3 file is generated.
    """
    _validate_form_for_format(ff, "mm3_fld")
    output_path = Path(path)
    template = Path(template_path) if template_path is not None else None
    if template is None and ff.source_format == "mm3_fld" and ff.source_path is not None:
        template = ff.source_path

    if template is not None:
        template_rows, template_lines = _mm3_import_ff(template)
        updated_rows = copy.deepcopy(template_rows)
        bond_by_row, bond_by_env = _build_bond_maps(ff.bonds)
        angle_by_row, angle_by_env = _build_angle_maps(ff.angles)
        sb_by_row, sb_by_env = _build_sb_maps(ff.stretch_bends)

        for row in updated_rows:
            if row.ptype in ("bf", "be"):
                bond = _match_bond_for_export(row.ff_row, row.atom_types, bond_by_row, bond_by_env)
                if bond is not None:
                    row.value = canonical_to_mm3_bond_k(bond.force_constant) if row.ptype == "bf" else bond.equilibrium
            elif row.ptype in ("af", "ae"):
                angle = _match_angle_for_export(row.ff_row, row.atom_types, angle_by_row, angle_by_env)
                if angle is not None:
                    row.value = (
                        canonical_to_mm3_angle_k(angle.force_constant)
                        if row.ptype == "af"
                        else _normalize_equilibrium_angle(angle.equilibrium)
                    )
            elif row.ptype == "df":
                value = _torsion_file_value(ff.torsions, row.ff_row, row.ff_col)
                if value is not None:
                    row.value = value
            elif row.ptype == "sb":
                sb = _match_sb_for_export(row.ff_row, row.atom_types, sb_by_row, sb_by_env)
                if sb is not None:
                    row.value = canonical_to_mm3_sb_k(sb.force_constant)

        _mm3_export_ff(output_path, updated_rows, list(template_lines))
        if ff.vdws:
            _update_mm3_vdw_lines(output_path, ff.vdws)
        _write_nonbonded_exclusions(output_path, ff.nonbonded_excluded_atom_types)
        return output_path

    del substructure_name, smiles
    lines: list[str] = []
    if ff.nonbonded_excluded_atom_types:
        lines.insert(
            0,
            (
                f"{_NONBONDED_EXCLUSION_DIRECTIVE}"
                f"{json.dumps(list(ff.nonbonded_excluded_atom_types), separators=(',', ':'))}\n"
            ),
        )
    for bond in ff.bonds:
        lines.append(
            _format_mm3_bond_line(
                _mm3_atom_types(bond.env_id, bond.elements),
                bond.equilibrium,
                canonical_to_mm3_bond_k(bond.force_constant),
            )
        )
    for angle in ff.angles:
        lines.append(
            _format_mm3_angle_line(
                _mm3_atom_types(angle.env_id, angle.elements),
                angle.equilibrium,
                canonical_to_mm3_angle_k(angle.force_constant),
            )
        )
    if ff.torsions:
        # Group torsions by env_id to combine V1/V2/V3 on one line
        torsion_groups: dict[str, dict[int, float]] = {}
        torsion_elements: dict[str, tuple[str, ...]] = {}
        for tor in ff.torsions:
            key = tor.env_id or "-".join(tor.elements)
            if key not in torsion_groups:
                torsion_groups[key] = {}
                torsion_elements[key] = tor.elements
            torsion_groups[key][tor.periodicity] = (
                tor.force_constant * 2.0
            )  # V_n = 2*k (MM3 .fld stores V, we store V/2)
        for key, vs in torsion_groups.items():
            atom_types = _mm3_atom_types(key, torsion_elements[key])
            lines.append(_format_mm3_torsion_line(atom_types, vs.get(1, 0.0), vs.get(2, 0.0), vs.get(3, 0.0)))
    if ff.vdws:
        lines.extend(["-6\n"])
        for vdw in ff.vdws:
            lines.append(_format_mm3_vdw_line(vdw))
        lines.extend([" END OF NONBONDED INTERACTIONS\n", "-2\n"])
    else:
        lines.append("-2\n")
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
