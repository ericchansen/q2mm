"""MM3 .fld file format I/O."""

from __future__ import annotations

import contextlib
import copy
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from q2mm import constants as co
from q2mm.io._helpers import (
    Param,
    _build_angle_maps,
    _build_bond_maps,
    _build_sb_maps,
    _build_vdw_maps,
    _match_angle_for_export,
    _match_bond_for_export,
    _match_sb_for_export,
    _split_env_id,
    _update_torsion_param,
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

# MM3 fixed-format column positions
COM_POS_START = 96
P_1_START = 23
P_1_END = 33
P_2_START = 34
P_2_END = 44
P_3_START = 45
P_3_END = 55


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
    prefix = f" 1{atom_types[0]:>4}{atom_types[1]:>4}{'':13}"
    return f"{prefix}{equilibrium:10.4f} {force_constant:10.4f}\n"


def _format_mm3_angle_line(atom_types: list[str], equilibrium: float, force_constant: float) -> str:
    prefix = f" 2{atom_types[0]:>4}{atom_types[1]:>4}{atom_types[2]:>4}{'':9}"
    return f"{prefix}{equilibrium:10.4f} {force_constant:10.4f}\n"


def _format_mm3_torsion_line(atom_types: list[str], v1: float, v2: float, v3: float) -> str:
    prefix = f" 4{atom_types[0]:>4}{atom_types[1]:>4}{atom_types[2]:>4}{atom_types[3]:>4}{'':5}"
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
        tail = " ".join(parts[4:]) if len(parts) > 4 else ""
        lines[index] = f"  {match.atom_type:<3} {match.radius:10.4f} {match.epsilon:10.4f} {match.reduction:10.4f}" + (
            f" {tail}" if tail else ""
        )
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


def _mm3_import_ff(
    path: str | Path, sub_search: str = "OPT", *, include_standard: bool = True
) -> tuple[list[Param], list[str]]:
    """Read parameters from an mm3.fld file.

    Args:
        path: Path to the mm3.fld file.
        sub_search: Substructure name to look for (default ``"OPT"``).
        include_standard: When ``True`` (the default), also parse standard
            MM3 bond, angle, torsion and stretch-bend parameters from the
            main body of the file (outside the substructure section).  These
            serve as the base layer that substructure parameters override.

    Returns a ``(params, lines)`` tuple where *params* is the list of
    :class:`Param` objects and *lines* is the raw file content (as
    returned by ``readlines``).

    """
    path = str(path)
    params: list[Param] = []
    smiles_list: list[str] = []
    sub_names: list[str] = []
    atom_types_list: list[list[str]] = []
    atom_type_equivalencies: dict[str, str] = {}

    with open(path) as f:
        all_lines = f.readlines()

    logger.log(15, f"READING: {path}")
    section_sub = False
    section_smiles = False
    section_vdw = False
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

        # Van der Waals in the -6 section
        if "OPT" in line and section_vdw:
            logger.log(5, "[L{}] Found Van der Waals:\n{}".format(i + 1, line.strip("\n")))
            atm = line[2:5]
            rad = line[5:15]
            eps = line[16:26]
            params.extend(
                (
                    Param(atom_types=atm, ptype="vdwr", ff_col=1, ff_row=i + 1, value=float(rad)),
                    Param(atom_types=atm, ptype="vdwe", ff_col=2, ff_row=i + 1, value=float(eps)),
                )
            )
            continue

        if sub_search in line or section_sub or include_standard:
            # Bonds
            if match_mm3_bond(line):
                logger.log(5, "[L{}] Found bond:\n{}".format(i + 1, line.strip("\n")))
                if section_sub:
                    atm_lbls = [line[4:6], line[8:10]]
                    atm_typs = _convert_to_types(atm_lbls, atom_types_list[-1])
                else:
                    atm_typs = [line[4:6], line[9:11]]
                    atm_lbls = atm_typs
                    comment = line[COM_POS_START:].strip()
                    sub_names.append(comment)
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 2:
                    continue
                params.extend(
                    (
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="be",
                            ff_col=1,
                            ff_row=i + 1,
                            label=line[:2],
                            value=parm_cols[0],
                        ),
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="bf",
                            ff_col=2,
                            ff_row=i + 1,
                            label=line[:2],
                            value=parm_cols[1],
                        ),
                    )
                )
                with contextlib.suppress(IndexError):
                    params.append(
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="q",
                            ff_col=3,
                            ff_row=i + 1,
                            label=line[:2],
                            value=parm_cols[2],
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
                    atm_lbls = atm_typs
                    comment = line[COM_POS_START:].strip()
                    sub_names.append(comment)
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 2:
                    continue
                params.extend(
                    (
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="ae",
                            ff_col=1,
                            ff_row=i + 1,
                            label=line[:2],
                            value=parm_cols[0],
                        ),
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="af",
                            ff_col=2,
                            ff_row=i + 1,
                            label=line[:2],
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
                    atm_lbls = atm_typs
                    comment = line[COM_POS_START:].strip()
                    sub_names.append(comment)
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 1:
                    continue
                params.append(
                    Param(
                        atom_labels=atm_lbls,
                        atom_types=atm_typs,
                        ptype="sb",
                        ff_col=1,
                        ff_row=i + 1,
                        label=line[:2],
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
                    atm_lbls = atm_typs
                    comment = line[COM_POS_START:].strip()
                    sub_names.append(comment)
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 3:
                    continue
                params.extend(
                    (
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="df",
                            ff_col=1,
                            ff_row=i + 1,
                            label=line[:2],
                            value=parm_cols[0],
                        ),
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="df",
                            ff_col=2,
                            ff_row=i + 1,
                            label=line[:2],
                            value=parm_cols[1],
                        ),
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="df",
                            ff_col=3,
                            ff_row=i + 1,
                            label=line[:2],
                            value=parm_cols[2],
                        ),
                    )
                )
                continue

            # Higher order torsions (4th through 6th)
            elif match_mm3_higher_torsion(line):
                if not params or params[-1].ptype != "df":
                    continue
                logger.log(
                    5,
                    "[L{}] Found higher order torsion:\n{}".format(i + 1, line.strip("\n")),
                )
                atm_lbls = params[-1].atom_labels
                atm_typs = params[-1].atom_types
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 3:
                    continue
                params.extend(
                    (
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="df",
                            ff_col=1,
                            ff_row=i + 1,
                            label=line[:2],
                            value=parm_cols[0],
                        ),
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="df",
                            ff_col=2,
                            ff_row=i + 1,
                            label=line[:2],
                            value=parm_cols[1],
                        ),
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="df",
                            ff_col=3,
                            ff_row=i + 1,
                            label=line[:2],
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
                    atm_lbls = atm_typs
                    comment = line[COM_POS_START:].strip()
                    sub_names.append(comment)
                try:
                    parm_cols = [float(x) for x in line[P_1_START:P_3_END].split()]
                except ValueError:
                    continue
                if len(parm_cols) < 2:
                    continue
                params.extend(
                    (
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="imp1",
                            ff_col=1,
                            ff_row=i + 1,
                            label=line[:2],
                            value=parm_cols[0],
                        ),
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="imp2",
                            ff_col=2,
                            ff_row=i + 1,
                            label=line[:2],
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
                params.extend(
                    (
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="vdwr",
                            ff_col=1,
                            ff_row=i + 1,
                            label=line[:2],
                            value=parm_cols[0],
                        ),
                        Param(
                            atom_labels=atm_lbls,
                            atom_types=atm_typs,
                            ptype="vdwfc",
                            ff_col=2,
                            ff_row=i + 1,
                            label=line[:2],
                            value=parm_cols[1],
                        ),
                    )
                )
                continue

        # -6 marks start of Van der Waals section
        if line.startswith("-6"):
            section_vdw = True
            continue
        if "New Atom Type Equivalencies" in line:
            section_atm_eqv = True
            continue

    logger.log(15, f"  -- Read {len(params)} parameters.")
    return params, all_lines


def _mm3_export_ff(path: str | Path, params: list[Param], lines: list[str]) -> None:
    """Write parameters back to an mm3.fld file at fixed column positions."""
    for param in params:
        logger.log(1, f">>> param: {param} param.value: {param.value}")
        line = lines[param.ff_row - 1]
        if abs(param.value) > 999.0:
            logger.warning(f"Value of {param} is too high! Skipping write.")
        elif param.ff_col == 1:
            lines[param.ff_row - 1] = line[:P_1_START] + f"{param.value:10.4f}" + line[P_1_END:]
        elif param.ff_col == 2:
            lines[param.ff_row - 1] = line[:P_2_START] + f"{param.value:10.4f}" + line[P_2_END:]
        elif param.ff_col == 3:
            lines[param.ff_row - 1] = line[:P_3_START] + f"{param.value:10.4f}" + line[P_3_END:]
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
    parsed_params, _ = _mm3_import_ff(path, include_standard=include_standard)

    bonds = []
    angles = []
    stretch_bends: list[StretchBendParam] = []
    torsions = []
    vdws = _parse_mm3_vdw_params(Path(path))

    # Pre-build lookup for equilibrium values by (ptype, ff_row)
    eq_lookup = {}
    for p in parsed_params:
        if p.ptype in ("be", "ae"):
            eq_lookup[(p.ptype, p.ff_row)] = p.value

    for param in parsed_params:
        # Extract element letters from atom type (e.g., 'C1' -> 'C', ' F' -> 'F')
        atom_types = [t.strip() for t in param.atom_types if t.strip() and t.strip() != "-"]

        if param.ptype == "bf" and len(atom_types) >= 2:
            elems = tuple(_extract_element(t) for t in atom_types[:2])
            env_id = canonicalize_bond_env_id(atom_types[:2])
            eq_val = eq_lookup.get(("be", param.ff_row), 0.0)
            bonds.append(
                BondParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=mm3_bond_k_to_canonical(param.value),
                    label=f"MM3 row {param.ff_row}",
                    env_id=env_id,
                    ff_row=param.ff_row,
                )
            )

        elif param.ptype == "af" and len(atom_types) >= 3:
            elems = tuple(_extract_element(t) for t in atom_types[:3])
            env_id = canonicalize_angle_env_id(atom_types[:3])
            eq_val = eq_lookup.get(("ae", param.ff_row), 0.0)
            angles.append(
                AngleParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=mm3_angle_k_to_canonical(param.value),
                    label=f"MM3 row {param.ff_row}",
                    env_id=env_id,
                    ff_row=param.ff_row,
                )
            )

        elif param.ptype == "df" and len(atom_types) >= 4:
            elems = tuple(_extract_element(t) for t in atom_types[:4])
            env_id = "-".join(t.strip() for t in atom_types[:4])
            periodicity = getattr(param, "ff_col", 1)
            torsions.append(
                TorsionParam(
                    elements=elems,
                    periodicity=periodicity,
                    force_constant=param.value,
                    label=f"MM3 row {param.ff_row} V{periodicity}",
                    env_id=env_id,
                    ff_row=param.ff_row,
                )
            )

        elif param.ptype in ("imp1", "imp2") and len(atom_types) >= 4:
            elems = tuple(_extract_element(t) for t in atom_types[:4])
            env_id = "-".join(t.strip() for t in atom_types[:4])
            periodicity = 1 if param.ptype == "imp1" else 2
            torsions.append(
                TorsionParam(
                    elements=elems,
                    periodicity=periodicity,
                    force_constant=param.value,
                    label=f"MM3 row {param.ff_row} imp V{periodicity}",
                    env_id=env_id,
                    ff_row=param.ff_row,
                    is_improper=True,
                )
            )

        elif param.ptype == "sb" and len(atom_types) >= 3:
            elems = tuple(_extract_element(t) for t in atom_types[:3])
            env_id = canonicalize_angle_env_id(atom_types[:3])
            stretch_bends.append(
                StretchBendParam(
                    elements=elems,
                    force_constant=mm3_sb_k_to_canonical(param.value),
                    label=f"MM3 row {param.ff_row} SB",
                    env_id=env_id,
                    ff_row=param.ff_row,
                )
            )

    return ForceField(
        name=f"MM3 from {Path(path).name}",
        bonds=bonds,
        angles=angles,
        stretch_bends=stretch_bends,
        torsions=torsions,
        vdws=vdws,
        source_path=Path(path),
        source_format="mm3_fld",
        functional_form=FunctionalForm.MM3,
    )


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

    Otherwise, a minimal bond/angle-only MM3 substructure is generated.
    """
    _validate_form_for_format(ff, "mm3_fld")
    output_path = Path(path)
    template = Path(template_path) if template_path is not None else None
    if template is None and ff.source_format == "mm3_fld" and ff.source_path is not None:
        template = ff.source_path

    if template is not None:
        template_params, template_lines = _mm3_import_ff(template)
        updated_params = copy.deepcopy(template_params)
        bond_by_row, bond_by_env = _build_bond_maps(ff.bonds)
        angle_by_row, angle_by_env = _build_angle_maps(ff.angles)
        sb_by_row, sb_by_env = _build_sb_maps(ff.stretch_bends)

        for param in updated_params:
            if param.ptype in ("bf", "be"):
                bond = _match_bond_for_export(param, bond_by_row, bond_by_env)
                if bond is not None:
                    param.value = (
                        canonical_to_mm3_bond_k(bond.force_constant) if param.ptype == "bf" else bond.equilibrium
                    )
            elif param.ptype in ("af", "ae"):
                angle = _match_angle_for_export(param, angle_by_row, angle_by_env)
                if angle is not None:
                    param.value = (
                        canonical_to_mm3_angle_k(angle.force_constant) if param.ptype == "af" else angle.equilibrium
                    )
            elif param.ptype == "df":
                _update_torsion_param(param, ff.torsions)
            elif param.ptype == "sb":
                sb = _match_sb_for_export(param, sb_by_row, sb_by_env)
                if sb is not None:
                    param.value = canonical_to_mm3_sb_k(sb.force_constant)

        _mm3_export_ff(output_path, updated_params, list(template_lines))
        if ff.vdws:
            _update_mm3_vdw_lines(output_path, ff.vdws)
        return output_path

    lines = [f" C  OPT {substructure_name}\n", f" 9  {smiles}\n"]
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
            torsion_groups[key][tor.periodicity] = tor.force_constant
        for key, vs in torsion_groups.items():
            atom_types = _mm3_atom_types(key, torsion_elements[key])
            lines.append(_format_mm3_torsion_line(atom_types, vs.get(1, 0.0), vs.get(2, 0.0), vs.get(3, 0.0)))
    lines.append("-3\n")
    if ff.vdws:
        lines.extend(["-6\n"])
        for vdw in ff.vdws:
            lines.append(_format_mm3_vdw_line(vdw))
        lines.extend([" END OF NONBONDED INTERACTIONS\n", "-2\n"])
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
