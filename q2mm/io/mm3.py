"""MM3 .fld file format I/O."""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
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
        atom_types: Source atom-type or pattern tokens for this row
            (digit references resolved; literal tokens are not expanded).
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
_MM3_FIELD_WIDTH = P_1_END - P_1_START
# Context flags occupy cols 56–65 (two 4-char codes separated by space)
CTX_START = 56
CTX_END = 66
# Bond-order symbol is at col 7 in standard rows, col 6 in substructures.
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


def _require_finite_mm3_values(*values: float, context: str) -> None:
    """Reject nonfinite canonical values before conversion or normalization."""
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Cannot save MM3 {context}: numeric values must be finite.")


def _format_mm3_field(value: float, width: int = _MM3_FIELD_WIDTH, *, context: str = "numeric field") -> str:
    """Format one finite file value without allowing rounding to widen its column."""
    _require_finite_mm3_values(value, context=context)
    text = f"{value:{width}.4f}"
    if len(text) > width:
        raise ValueError(
            f"Cannot save MM3 {context}: value {value!r} does not fit a {width}-character, four-decimal field."
        )
    return text


def _format_mm3_bond_line(atom_types: list[str], equilibrium: float, force_constant: float) -> str:
    prefix = f" 1  {atom_types[0]:>2} - {atom_types[1]:>2}{'':12}"
    fields = " ".join(_format_mm3_field(value, context="bond") for value in (equilibrium, force_constant))
    return f"{prefix}{fields}\n"


def _format_mm3_angle_line(atom_types: list[str], equilibrium: float, force_constant: float) -> str:
    prefix = f" 2  {atom_types[0]:>2} - {atom_types[1]:>2} - {atom_types[2]:>2}{'':7}"
    fields = " ".join(_format_mm3_field(value, context="angle") for value in (equilibrium, force_constant))
    return f"{prefix}{fields}\n"


def _format_mm3_torsion_line(atom_types: list[str], v1: float, v2: float, v3: float) -> str:
    prefix = f" 4  {atom_types[0]:>2} - {atom_types[1]:>2} - {atom_types[2]:>2} - {atom_types[3]:>2}  "
    fields = " ".join(_format_mm3_field(value, context="proper torsion") for value in (v1, v2, v3))
    return f"{prefix}{fields}\n"


def _format_mm3_vdw_line(vdw: VdwParam) -> str:
    fields = " ".join(
        _format_mm3_field(value, context=f"vdW {name}")
        for name, value in (("radius", vdw.radius), ("epsilon", vdw.epsilon), ("reduction", vdw.reduction))
    )
    return f"  {vdw.atom_type:<3} {fields}                                   0000    O 1\n"


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


def _splice_fixed(line: str, start: int, width: int, value: float, *, context: str = "numeric field") -> str:
    """Replace a representable fixed field, rejecting missing columns or lossy edits."""
    if start < 0 or len(line.rstrip("\r\n")) < start + width:
        raise ValueError(f"Cannot save MM3 {context}: no existing {width}-character field.")
    field = _format_mm3_field(value, width, context=context)
    return line[:start] + field + line[start + width :]


def _update_mm3_vdw_lines(lines: list[str], vdws: tuple[VdwParam, ...]) -> list[str]:
    """Stage representable vdW edits without opening the destination."""
    lines = list(lines)
    by_row, by_type = _build_vdw_maps(vdws)
    for index, line in enumerate(lines):
        row = index + 1
        match = by_row.get(row)
        parts = list(re.finditer(r"\S+", line))
        if match is None and parts:
            match = by_type.get(parts[0].group())
        if match is None:
            continue
        updated = line
        for column, (name, value) in enumerate(
            (("radius", match.radius), ("epsilon", match.epsilon), ("reduction", match.reduction)), start=1
        ):
            if len(parts) <= column:
                raise ValueError(f"Cannot save MM3 vdW row {row} {name}: no existing 10-character numeric field.")
            start = parts[column].end() - _MM3_FIELD_WIDTH
            if start <= parts[column - 1].end() or parts[column].start() < start:
                raise ValueError(f"Cannot save MM3 vdW row {row} {name}: no existing 10-character numeric field.")
            try:
                float(parts[column].group())
            except ValueError as exc:
                raise ValueError(f"Cannot save MM3 vdW row {row} {name}: template field is not numeric.") from exc
            # Source and generated atom-type prefixes differ in width. Locate
            # each right-aligned field from its original token, not a guessed offset.
            updated = _splice_fixed(updated, start, _MM3_FIELD_WIDTH, value, context=f"vdW row {row} {name}")
        lines[index] = updated
    return lines


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
    for label in atom_labels:
        token = label.strip()
        if not token or token == "-":
            raise ValueError("Empty atom label.")
        if token.isdigit() and token != "00" and not 1 <= int(token) <= len(atom_types):
            raise ValueError(f"Atom reference {token!r} is outside 1..{len(atom_types)}.")
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
        sub_search: Case-sensitive substring selecting substructure names
            when ``include_standard=False`` (default ``"OPT"``).
        include_standard: When ``True`` (the default), parse supported
            standard rows and all physical substructure blocks, including
            non-OPT blocks. When ``False``, import only matching blocks.
            Selection never changes a block's column layout or source rows.

    Returns a ``(rows, lines)`` tuple where *rows* is the list of
    :class:`_Mm3ParameterRow` objects and *lines* is the raw file content
    (as returned by ``readlines``).

    Native ``-3`` format declares a substructure title followed by a ``9``
    pattern; a selected title without that pattern is invalid. Outside that
    format, ordinary ``C`` comments remain comments. Existing abbreviated
    templates with adjacent ``C``/``9`` records are also accepted.
    The format applies to subsequent records until another format directive,
    not just to one following record (MacroModel 9.7 Reference Manual, D.4.4).

    """
    path = str(path)
    rows: list[_Mm3ParameterRow] = []
    atom_type_equivalencies: dict[str, str] = {}

    with open(path) as f:
        all_lines = f.readlines()

    logger.log(15, f"READING: {path}")
    section_sub = False
    section_sub_format = False
    section_smiles = False
    section_atm_eqv = False
    sub_name = ""
    include_sub = False
    atom_types: list[str] = []
    last_torsion_types: list[str] | None = None

    def substructure_types(labels: list[str], row_number: int) -> list[str]:
        try:
            return _convert_to_types(labels, atom_types)
        except ValueError as exc:
            raise ValueError(f"{path}: row {row_number}, substructure {sub_name!r}: {exc}") from exc

    for i, line in enumerate(all_lines):
        if section_atm_eqv:
            if line.startswith(" C") and len(atom_type_equivalencies) > 0:
                section_atm_eqv = False
            elif not line.startswith(" C") and not line.startswith("-5"):
                equivalency = [typ.strip() for typ in line.split()[1:]]
                for typ in equivalency[1:]:
                    atom_type_equivalencies[typ] = equivalency[0]
                continue

        if line.startswith("-"):
            section_sub_format = line.startswith("-3")

        # Native -3 format identifies titles even when the pattern is missing.
        # Keep C/9 lookahead for abbreviated templates, not for ordinary comments.
        if line.startswith(" C") and (
            (section_sub_format and not section_sub)
            or (i + 1 < len(all_lines) and re.match(r"\s9", all_lines[i + 1]) is not None)
        ):
            if section_smiles and include_sub:
                raise ValueError(f"{path}: row {i + 1}, substructure {sub_name!r}: missing pattern before next title.")
            if section_sub:
                raise ValueError(f"{path}: row {i + 1}, substructure {sub_name!r}: missing -3 before next block.")
            sub_name = line[2:].strip()
            section_sub = True
            include_sub = include_standard or sub_search in sub_name
            atom_types = []
            last_torsion_types = None
            section_smiles = True
            logger.log(15, f"[L{i + 1}] Start of substructure: {sub_name}")
            continue
        elif section_sub and line.startswith("-3") and not (section_smiles and include_sub):
            logger.log(15, f"[L{i + 1}] End of substructure: {sub_name}")
            section_sub = False
            section_smiles = False
            include_sub = False
            atom_types = []
            last_torsion_types = None
            continue
        elif section_smiles:
            if include_sub:
                matched = re.fullmatch(rf"\s9\s+({co.RE_SMILES})\s*", line)
                if matched is None:
                    raise ValueError(f"{path}: row {i + 1}, substructure {sub_name!r}: unsupported or missing pattern.")
                try:
                    atom_types = _convert_smiles_to_types(matched.group(1))
                    if not atom_types:
                        raise ValueError("Pattern contains no atom labels.")
                except ValueError as exc:
                    raise ValueError(f"{path}: row {i + 1}, substructure {sub_name!r}: {exc}") from exc
                logger.log(15, "  -- Atom types: %s", " ".join(atom_types))
            section_smiles = False
            continue
        if (
            line.startswith("-")
            or match_mm3_vdw(line)
            or (match_mm3_label(line) and not match_mm3_higher_torsion(line))
        ):
            last_torsion_types = None

        if include_sub or (include_standard and not section_sub):
            # Bonds
            if match_mm3_bond(line):
                logger.log(5, "[L{}] Found bond:\n{}".format(i + 1, line.strip("\n")))
                bond_order = ""
                context = ""
                if section_sub:
                    atm_lbls = [line[4:6], line[8:10]]
                    atm_typs = substructure_types(atm_lbls, i + 1)
                    # Substructure sections: bond-order symbol between labels.
                    if len(line) > 6 and line[6] in _BOND_ORDER_CHARS:
                        bond_order = line[6]
                else:
                    atm_typs = [line[4:6], line[9:11]]
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
                    atm_typs = substructure_types(atm_lbls, i + 1)
                else:
                    atm_typs = [line[4:6], line[9:11], line[14:16]]
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
                    atm_typs = substructure_types(atm_lbls, i + 1)
                else:
                    atm_typs = [line[4:6], line[9:11], line[14:16]]
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
                    atm_typs = substructure_types(atm_lbls, i + 1)
                else:
                    atm_typs = [line[4:6], line[9:11], line[14:16], line[19:21]]
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 3:
                    continue
                last_torsion_types = atm_typs
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
                if last_torsion_types is None:
                    scope = f"substructure {sub_name!r}" if section_sub else "standard section"
                    raise ValueError(
                        f"{path}: row {i + 1}, {scope}: torsion continuation has no lower torsion in this scope."
                    )
                logger.log(
                    5,
                    "[L{}] Found higher order torsion:\n{}".format(i + 1, line.strip("\n")),
                )
                atm_typs = last_torsion_types
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
                    atm_typs = substructure_types(atm_lbls, i + 1)
                else:
                    atm_typs = [line[4:6], line[9:11], line[14:16], line[19:21]]
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
                atm_typs = substructure_types(atm_lbls, i + 1)
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

    if section_smiles and include_sub:
        raise ValueError(f"{path}: row {len(all_lines)}, substructure {sub_name!r}: missing pattern after title.")
    if section_sub:
        raise ValueError(f"{path}: row {len(all_lines)}, substructure {sub_name!r}: unterminated block (missing -3).")
    logger.log(15, f"  -- Read {len(rows)} parameters.")
    return rows, all_lines


def _mm3_export_ff(path: str | Path, rows: list[_Mm3ParameterRow], lines: list[str]) -> None:
    """Write parameter rows back to an mm3.fld file at fixed column positions."""
    lines = list(lines)
    for row in rows:
        logger.log(1, f">>> row: {row} row.value: {row.value}")
        line = lines[row.ff_row - 1]
        kind = "improper" if row.ptype in ("imp1", "imp2") else row.ptype
        context = f"{kind} row {row.ff_row}"
        # Higher-order torsion amplitudes V4/V5/V6 (ff_col 4/5/6) live in the
        # same three physical parameter columns as V1/V2/V3 but on the "54"
        # continuation line, which is addressed by their own ``ff_row``.  Map
        # them onto the same columns so higher-order torsions round-trip.
        if row.ff_col in (1, 4):
            lines[row.ff_row - 1] = _splice_fixed(line, P_1_START, _MM3_FIELD_WIDTH, row.value, context=context)
        elif row.ff_col in (2, 5):
            lines[row.ff_row - 1] = _splice_fixed(line, P_2_START, _MM3_FIELD_WIDTH, row.value, context=context)
        elif row.ff_col in (3, 6):
            lines[row.ff_row - 1] = _splice_fixed(line, P_3_START, _MM3_FIELD_WIDTH, row.value, context=context)
    with open(path, "w") as f:
        f.writelines(lines)
    logger.log(10, f"WROTE: {path}")


def _mm3_phase_is_representable(torsion: TorsionParam) -> bool:
    """Check the fixed odd/even phase convention; a zero amplitude has no phase dependence."""
    phase = 180.0 if torsion.periodicity % 2 == 0 else 0.0
    return (
        math.isfinite(torsion.force_constant)
        and math.isfinite(torsion.phase)
        and (torsion.force_constant == 0.0 or torsion.phase % 360.0 == phase)
    )


def _validate_mm3_torsion_phase(torsion: TorsionParam, *, context: str) -> None:
    """Require finite amplitude and phase under the existing MM3 phase convention."""
    if not _mm3_phase_is_representable(torsion):
        phase = 180.0 if torsion.periodicity % 2 == 0 else 0.0
        raise ValueError(
            f"Cannot save MM3 {context}: amplitude and phase must be finite; periodicity {torsion.periodicity} "
            f"requires phase {phase} modulo 360 degrees for a nonzero amplitude."
        )


def _validate_mm3_template_impropers(torsions: tuple[TorsionParam, ...], rows: list[_Mm3ParameterRow]) -> None:
    """Reject improper edits that cannot be applied to an existing template column."""
    improper_rows = {(row.ff_row, row.ff_col): row for row in rows if row.ptype in ("imp1", "imp2")}
    row_numbers = {row_number for row_number, _ in improper_rows}
    seen: set[tuple[int, int]] = set()
    for torsion in torsions:
        if not torsion.is_improper and torsion.ff_row not in row_numbers:
            continue
        row = improper_rows.get((torsion.ff_row, torsion.periodicity)) if torsion.ff_row is not None else None
        if row is None:
            raise ValueError(
                f"Cannot save MM3 improper torsion at row {torsion.ff_row}, periodicity {torsion.periodicity}: "
                "no matching imp1/imp2 template column."
            )
        key = (row.ff_row, row.ff_col)
        if key in seen:
            raise ValueError(f"Cannot save MM3 improper row {row.ff_row}: duplicate periodicity {row.ff_col}.")
        seen.add(key)
        atom_types = [t.strip() for t in row.atom_types if t.strip() and t.strip() != "-"]
        if (
            not torsion.is_improper
            or torsion.env_id != "-".join(atom_types)
            or torsion.elements != tuple(_extract_element(t) for t in atom_types)
        ):
            raise ValueError(f"Cannot save MM3 improper row {row.ff_row}: interaction kind or atom identity changed.")
        _validate_mm3_torsion_phase(torsion, context=f"improper row {row.ff_row}")


def _validate_mm3_standalone(ff: ForceField) -> None:
    """Reject populated features that the standalone MM3 writer cannot represent."""
    if ff.cmaps:
        raise ValueError("Cannot save standalone MM3 CMAP grids; no CMAP representation is emitted.")
    if ff.stretch_bends:
        raise ValueError("Cannot save standalone MM3 stretch-bend terms; use a source template.")
    for index, angle in enumerate(ff.angles):
        if angle.ub_force_constant is not None or angle.ub_equilibrium is not None:
            raise ValueError(f"Cannot save standalone MM3 angle {index}: Urey-Bradley fields are unsupported.")
        _require_finite_mm3_values(angle.equilibrium, angle.force_constant, context=f"standalone angle {index}")
    for index, torsion in enumerate(ff.torsions):
        if torsion.is_improper:
            raise ValueError(f"Cannot save standalone MM3 improper torsion {index}; use a source template.")
        if torsion.periodicity not in (1, 2, 3):
            raise ValueError(
                f"Cannot save standalone MM3 torsion {index}: periodicity {torsion.periodicity} "
                "is unsupported; only V1/V2/V3 are emitted."
            )
        _validate_mm3_torsion_phase(torsion, context=f"standalone proper torsion {index}")
    for index, bond in enumerate(ff.bonds):
        _require_finite_mm3_values(bond.equilibrium, bond.force_constant, context=f"standalone bond {index}")
        if bond.bond_order not in ("", "-"):
            raise ValueError(f"Cannot save standalone MM3 bond {index}: bond order {bond.bond_order!r} would be lost.")
        if bond.context not in ("", _GENERIC_CONTEXT):
            raise ValueError(f"Cannot save standalone MM3 bond {index}: bond context {bond.context!r} would be lost.")
        if bond.dipole_moment != 0.0:
            raise ValueError(
                f"Cannot save standalone MM3 bond {index}: bond dipole {bond.dipole_moment!r} would be lost."
            )


# ---------------------------------------------------------------------------
# Public load / save
# ---------------------------------------------------------------------------


def load_mm3_fld(path: str | Path, *, include_standard: bool = True) -> ForceField:
    """Load from Schrödinger MM3 .fld file.

    Args:
        path: Path to the mm3.fld file.
        include_standard: When ``True`` (the default), load supported
            standard parameters and all physical substructure blocks,
            including non-OPT blocks. Set to ``False`` to load only
            ``OPT``-named blocks' bonded parameters. The global vdW table
            is loaded in either mode; this flag does not define its
            active/frozen partition.

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
    Source-matched ``imp1``/``imp2`` amplitudes are updated in their original
    columns (four decimal places in file units). Unrepresentable improper
    edits, including changed interaction identity or nonzero-amplitude
    phase, raise ``ValueError`` before the destination is opened for writing.
    Proper template torsions, including V4/V5/V6 continuation rows, follow
    the same phase rule. All emitted scalars use finite 10-character fields
    at four-decimal precision, including vdW reduction and converted force
    constants. Overflow after conversion or rounding, missing fixed fields,
    and oversized source vdW tokens are rejected before any write; fitting
    large values are serialized rather than silently skipped.

    Source-backed bonds, angles, and stretch-bends update only their exact
    ``ff_row``; a missing source row is rejected before writing. Parameters
    without ``ff_row`` retain environment-based template updates. Unchanged
    bonded numeric fields retain their original spelling.

    Otherwise, a self-contained standard-parameter MM3 file is generated.
    This limited writer rejects populated CMAP, stretch-bend and Urey-Bradley
    fields, improper torsions, periodicities outside 1-3 (including zero
    amplitudes), non-single declared bond orders, non-generic bond contexts,
    and nonzero bond dipoles before modifying the destination. Nonzero proper
    torsions require phases equivalent to 0 degrees for odd orders or 180
    degrees for even orders; zero amplitudes may retain arbitrary finite
    phases in memory (output reloads the canonical phase). Empty bond
    order and generic context (empty or ``"0000 0000"``) remain accepted.
    These checks do not add support for any new physical terms.
    """
    _validate_form_for_format(ff, "mm3_fld")
    output_path = Path(path)
    template = Path(template_path) if template_path is not None else None
    if template is None and ff.source_format == "mm3_fld" and ff.source_path is not None:
        template = ff.source_path

    if template is not None:
        template_rows, template_lines = _mm3_import_ff(template)
        _validate_mm3_template_impropers(ff.torsions, template_rows)
        for torsion in ff.proper_torsions:
            _validate_mm3_torsion_phase(torsion, context=f"proper torsion row {torsion.ff_row}")
        updated_rows = copy.deepcopy(template_rows)
        bond_by_row, _ = _build_bond_maps(ff.bonds)
        angle_by_row, _ = _build_angle_maps(ff.angles)
        sb_by_row, _ = _build_sb_maps(ff.stretch_bends)
        _, bond_by_env = _build_bond_maps([bond for bond in ff.bonds if bond.ff_row is None])
        _, angle_by_env = _build_angle_maps([angle for angle in ff.angles if angle.ff_row is None])
        _, sb_by_env = _build_sb_maps([sb for sb in ff.stretch_bends if sb.ff_row is None])

        for ptype, by_row in (("bf", bond_by_row), ("af", angle_by_row), ("sb", sb_by_row)):
            template_source_rows = {row.ff_row for row in template_rows if row.ptype == ptype}
            missing = by_row.keys() - template_source_rows
            if missing:
                raise ValueError(f"{template}: source rows {sorted(missing)} for {ptype!r} are not in the template.")

        for row in updated_rows:
            if row.ptype in ("bf", "be"):
                bond = _match_bond_for_export(row.ff_row, row.atom_types, bond_by_row, bond_by_env)
                if bond is not None:
                    _require_finite_mm3_values(bond.equilibrium, bond.force_constant, context=f"bond row {row.ff_row}")
                    row.value = canonical_to_mm3_bond_k(bond.force_constant) if row.ptype == "bf" else bond.equilibrium
            elif row.ptype in ("af", "ae"):
                angle = _match_angle_for_export(row.ff_row, row.atom_types, angle_by_row, angle_by_env)
                if angle is not None:
                    _require_finite_mm3_values(
                        angle.equilibrium, angle.force_constant, context=f"angle row {row.ff_row}"
                    )
                    row.value = (
                        canonical_to_mm3_angle_k(angle.force_constant)
                        if row.ptype == "af"
                        else _normalize_equilibrium_angle(angle.equilibrium)
                    )
            elif row.ptype in ("df", "imp1", "imp2"):
                value = _torsion_file_value(ff.torsions, row.ff_row, row.ff_col)
                if value is not None:
                    row.value = value
            elif row.ptype == "sb":
                sb = _match_sb_for_export(row.ff_row, row.atom_types, sb_by_row, sb_by_env)
                if sb is not None:
                    _require_finite_mm3_values(sb.force_constant, context=f"stretch-bend row {row.ff_row}")
                    row.value = canonical_to_mm3_sb_k(sb.force_constant)

        changed_rows = [
            updated
            for original, updated in zip(template_rows, updated_rows, strict=True)
            if updated.value != original.value
        ]
        staged_lines = _update_mm3_vdw_lines(template_lines, ff.vdws)
        _mm3_export_ff(output_path, changed_rows, staged_lines)
        _write_nonbonded_exclusions(output_path, ff.nonbonded_excluded_atom_types)
        return output_path

    _validate_mm3_standalone(ff)
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
