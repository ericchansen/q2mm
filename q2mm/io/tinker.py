"""Tinker .prm file format I/O."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from q2mm.io._helpers import (
    Param,
    _build_angle_maps,
    _build_bond_maps,
    _build_vdw_maps,
    _clean_atom_types,
    _match_angle_for_export,
    _match_bond_for_export,
    _split_env_id,
    _update_torsion_param,
    _validate_form_for_format,
)
from q2mm.models.forcefield import (
    AngleParam,
    BondParam,
    ForceField,
    FunctionalForm,
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
    mm3_angle_k_to_canonical,
    mm3_bond_k_to_canonical,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Atom-type helpers
# ---------------------------------------------------------------------------


def _default_tinker_atom_types(elements: tuple[str, ...]) -> list[str]:
    counts: dict[str, int] = {}
    atom_types = []
    for element in elements:
        count = counts.get(element, 0) + 1
        counts[element] = count
        atom_types.append(f"{element}{count}")
    return atom_types


def _tinker_atom_types(env_id: str, elements: tuple[str, ...]) -> list[str]:
    return _split_env_id(env_id, len(elements)) or _default_tinker_atom_types(elements)


# ---------------------------------------------------------------------------
# Line formatters
# ---------------------------------------------------------------------------


def _format_tinker_bond_line(atom_types: list[str], force_constant: float, equilibrium: float) -> str:
    return f"bond   {atom_types[0]:>4} {atom_types[1]:>4} {force_constant:10.4f} {equilibrium:10.4f}\n"


def _format_tinker_angle_line(atom_types: list[str], force_constant: float, equilibrium: float) -> str:
    return (
        f"angle  {atom_types[0]:>4} {atom_types[1]:>4} {atom_types[2]:>4} {force_constant:10.4f} {equilibrium:10.4f}\n"
    )


def _format_tinker_vdw_line(atom_type: str, radius: float, epsilon: float, reduction: float = 0.0) -> str:
    return f"vdw    {atom_type:>4} {radius:10.4f} {epsilon:10.4f} {reduction:10.4f}\n"


# ---------------------------------------------------------------------------
# VdW parsing / updating
# ---------------------------------------------------------------------------


def _parse_tinker_vdw_params(path: Path) -> list[VdwParam]:
    vdws: list[VdwParam] = []
    q2mm_sec = False
    gather_data = False
    for row, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if not q2mm_sec and "# Q2MM" in raw_line:
            q2mm_sec = True
            continue
        if q2mm_sec and raw_line.startswith("#"):
            gather_data = "OPT" in raw_line
            continue
        if not gather_data:
            continue
        parts = raw_line.split()
        if not parts or parts[0] != "vdw" or len(parts) < 4:
            continue
        vdws.append(
            VdwParam(
                atom_type=parts[1],
                radius=float(parts[2]),
                epsilon=float(parts[3]),
                reduction=float(parts[4]) if len(parts) > 4 else 0.0,
                label=f"Tinker row {row}",
                ff_row=row,
            )
        )
    return vdws


def _parse_generic_tinker_prm(path: Path) -> tuple[list[BondParam], list[AngleParam], list[VdwParam]]:
    bonds: list[BondParam] = []
    angles: list[AngleParam] = []
    vdws: list[VdwParam] = []
    atom_elements: dict[str, str] = {}

    for row, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = raw_line.split()
        record = parts[0].lower()

        if record == "atom" and len(parts) >= 3:
            # Standard Tinker: atom <type> <symbol> "desc" <anum> <mass> <val>
            # AMOEBA-style:    atom <type> <class> <symbol> "desc" ...
            # Distinguish: if parts[2] is purely numeric, it's a class field.
            symbol_col = 2
            if parts[2].isdigit() and len(parts) >= 4:
                symbol_col = 3
            atom_elements[parts[1]] = _extract_element(parts[symbol_col])
            continue

        if record.startswith("bond") and len(parts) >= 5:
            atom_types = parts[1:3]
            elements = tuple(atom_elements.get(atom_type, _extract_element(atom_type)) for atom_type in atom_types)
            bonds.append(
                BondParam(
                    elements=elements,
                    equilibrium=float(parts[4]),
                    force_constant=float(parts[3]),
                    label=f"Tinker row {row}",
                    env_id=canonicalize_bond_env_id(atom_types),
                    ff_row=row,
                )
            )
            continue

        if record.startswith("angle") and len(parts) >= 6:
            atom_types = parts[1:4]
            elements = tuple(atom_elements.get(atom_type, _extract_element(atom_type)) for atom_type in atom_types)
            angles.append(
                AngleParam(
                    elements=elements,
                    equilibrium=float(parts[5]),
                    force_constant=float(parts[4]),
                    label=f"Tinker row {row}",
                    env_id=canonicalize_angle_env_id(atom_types),
                    ff_row=row,
                )
            )
            continue

        if record == "vdw" and len(parts) >= 4:
            atom_type = parts[1]
            vdws.append(
                VdwParam(
                    atom_type=atom_type,
                    radius=float(parts[2]),
                    epsilon=float(parts[3]),
                    reduction=float(parts[4]) if len(parts) > 4 else 0.0,
                    label=f"Tinker row {row}",
                    ff_row=row,
                    element=atom_elements.get(atom_type, _extract_element(atom_type)),
                )
            )

    return bonds, angles, vdws


def _update_tinker_vdw_lines(path: Path, vdws: list[VdwParam]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    by_row, by_type = _build_vdw_maps(vdws)
    for index, line in enumerate(lines):
        row = index + 1
        parts = line.split()
        if not parts or parts[0] != "vdw":
            continue
        match = by_row.get(row)
        if match is None and len(parts) > 1:
            match = by_type.get(parts[1].strip())
        if match is None:
            continue
        base = f"vdw    {match.atom_type:>4} {match.radius:10.4f} {match.epsilon:10.4f}"
        if match.reduction != 0.0:
            base += f" {match.reduction:10.4f}"
        lines[index] = base
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Legacy Tinker FF import / export (replaces q2mm.parsers.tinker_ff.TinkerFF)
# ---------------------------------------------------------------------------

_BONDS = ["bond", "bond3", "bond4", "bond5"]
_PIBONDS = ["pibond", "pibond3", "pibond4", "pibond5"]
_ANGLES = ["angle", "angle3", "angle4", "angle5"]
_TORSIONS = ["torsion", "torsion4", "torsion5"]
_DIPOLES = ["dipole", "dipole3", "dipole4", "dipole5"]

logger = logging.getLogger(__name__)


def _tinker_import_ff(path: str | Path) -> tuple[list[Param], list[str]]:
    """Read Q2MM-marked parameters from a Tinker .prm file.

    Returns a ``(params, lines)`` tuple where *params* is a list of
    :class:`Param` objects and *lines* is the full file content as a list
    of strings (needed later by :func:`_tinker_export_ff`).
    """
    path = str(path)
    params: list[Param] = []
    q2mm_sec = False
    gather_data = False
    with open(path) as f:
        logger.log(15, f"READING: {path}")
        for i, line in enumerate(f):
            split = line.split()
            if not q2mm_sec and "# Q2MM" in line:
                q2mm_sec = True
            elif q2mm_sec and line and line[0] == "#":
                if "OPT" in line:
                    gather_data = True
                else:
                    gather_data = False
            if gather_data and split:
                if split[0] in _BONDS:
                    at = [split[1], split[2]]
                    params.extend(
                        (
                            Param(atom_types=at, ptype="bf", ff_col=1, ff_row=i + 1, value=float(split[3])),
                            Param(atom_types=at, ptype="be", ff_col=2, ff_row=i + 1, value=float(split[4])),
                        )
                    )
                if split[0] in _DIPOLES:
                    at = [split[1], split[2]]
                    params.extend(
                        (
                            Param(atom_types=at, ptype="q", ff_col=1, ff_row=i + 1, value=float(split[3])),
                            Param(atom_types=at, ptype="q_p", ff_col=2, ff_row=i + 1, value=float(split[4])),
                        )
                    )
                if split[0] in _PIBONDS:
                    at = [split[1], split[2]]
                    params.extend(
                        (
                            Param(atom_types=at, ptype="pi_b", ff_col=1, ff_row=i + 1, value=float(split[3])),
                            Param(atom_types=at, ptype="pi_t", ff_col=2, ff_row=i + 1, value=float(split[4])),
                        )
                    )
                if split[0] in _ANGLES:
                    at = [split[1], split[2], split[3]]
                    params.extend(
                        (
                            Param(atom_types=at, ptype="af", ff_col=1, ff_row=i + 1, value=float(split[4])),
                            Param(atom_types=at, ptype="ae", ff_col=2, ff_row=i + 1, value=float(split[5])),
                        )
                    )
                    if len(split) == 8:
                        params.extend(
                            (
                                Param(atom_types=at, ptype="ae", ff_col=3, ff_row=i + 1, value=float(split[6])),
                                Param(atom_types=at, ptype="ae", ff_col=4, ff_row=i + 1, value=float(split[7])),
                            )
                        )
                    elif len(split) == 7:
                        params.append(Param(atom_types=at, ptype="ae", ff_col=3, ff_row=i + 1, value=float(split[6])))
                if split[0] in _TORSIONS:
                    at = [split[1], split[2], split[3], split[4]]
                    params.extend(
                        (
                            Param(atom_types=at, ptype="df", ff_col=1, ff_row=i + 1, value=float(split[5])),
                            Param(atom_types=at, ptype="df", ff_col=2, ff_row=i + 1, value=float(split[8])),
                            Param(atom_types=at, ptype="df", ff_col=3, ff_row=i + 1, value=float(split[11])),
                        )
                    )
                if split[0] == "opbend":
                    at = [split[1], split[2], split[3], split[4]]
                    params.append(Param(atom_types=at, ptype="op_b", ff_col=1, ff_row=i + 1, value=float(split[5])))
                if split[0] == "vdw":
                    at = [split[1]]
                    params.append(Param(atom_types=at, ptype="vdw", ff_col=1, ff_row=i + 1, value=float(split[2])))
    logger.log(15, f"  -- Read {len(params)} parameters.")

    with open(path) as f:
        lines = f.readlines()
    return params, lines


def _tinker_export_ff(path: str | Path, params: list[Param], lines: list[str]) -> None:
    """Write parameters back to a Tinker .prm file.

    Uses keyword-based column detection (bond, angle, torsion, opbend, vdw)
    to reconstruct each line with updated values.
    """
    for param in params:
        logger.log(1, f">>> param: {param} param.value: {param.value}")
        line = lines[param.ff_row - 1]
        if abs(param.value) > 999.0:
            logger.warning(f"Value of {param} is too high! Skipping write.")
        else:
            col = int(param.ff_col - 1)
            linesplit = line.split()
            value = f"{param.value:7.3f}"
            par = format(linesplit[0], "<10")
            space5 = " " * 5

            if "bond" in line:
                atoms = "".join([format(el, ">5") for el in linesplit[1:3]]) + space5 * 2
                linesplit[3 + col] = value
                const = "".join([format(el, ">12") for el in linesplit[3:]])
            elif "angle" in line:
                atoms = "".join([format(el, ">5") for el in linesplit[1:4]]) + space5
                linesplit[4 + col] = value
                const = "".join([format(el, ">12") for el in linesplit[4:]])
            elif "torsion" in line:
                atoms = "".join([format(el, ">5") for el in linesplit[1:5]]) + space5
                linesplit[5 + 3 * col] = value
                const = "".join([format(el, ">8") for el in linesplit[5:]])
            elif "opbend" in line:
                atoms = "".join([format(el, ">5") for el in linesplit[1:5]]) + space5
                linesplit[5 + col] = value
                const = "".join([format(el, ">12") for el in linesplit[5:]])
            elif "vdw" in line:
                atoms = format(linesplit[1], ">5") + space5 * 3
                linesplit[2 + col] = value
                const = "".join([format(el, ">12") for el in linesplit[2:]])
            lines[param.ff_row - 1] = par + atoms + const + "\n"
    with open(path, "w") as f:
        f.writelines(lines)
    logger.log(10, f"WROTE: {path}")


# ---------------------------------------------------------------------------
# Public load / save
# ---------------------------------------------------------------------------


def load_tinker_prm(path: str | Path) -> ForceField:
    """Load bond and angle parameters from a Tinker .prm file."""
    params, _lines = _tinker_import_ff(path)

    if not params:
        bonds, angles, vdws = _parse_generic_tinker_prm(Path(path))
        for b in bonds:
            b.force_constant = mm3_bond_k_to_canonical(b.force_constant)
        for a in angles:
            a.force_constant = mm3_angle_k_to_canonical(a.force_constant)
        return ForceField(
            name=f"Tinker from {Path(path).name}",
            bonds=bonds,
            angles=angles,
            vdws=vdws,
            source_path=Path(path),
            source_format="tinker_prm",
            functional_form=FunctionalForm.MM3,
        )

    bonds = []
    angles = []
    torsions = []
    vdws = _parse_tinker_vdw_params(Path(path))

    eq_lookup: dict[tuple[str, int], float] = {}
    for param in params:
        if param.ptype == "be" or (param.ptype == "ae" and getattr(param, "ff_col", None) == 2):
            eq_lookup[(param.ptype, param.ff_row)] = param.value

    for param in params:
        atom_types = _clean_atom_types(getattr(param, "atom_types", None), 4)

        if param.ptype == "bf" and len(atom_types) >= 2:
            elems = tuple(_extract_element(t) for t in atom_types[:2])
            env_id = canonicalize_bond_env_id(atom_types[:2])
            eq_val = eq_lookup.get(("be", param.ff_row), 0.0)
            bonds.append(
                BondParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=mm3_bond_k_to_canonical(param.value),
                    label=f"Tinker row {param.ff_row}",
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
                    label=f"Tinker row {param.ff_row}",
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
                    label=f"Tinker row {param.ff_row} V{periodicity}",
                    env_id=env_id,
                    ff_row=param.ff_row,
                )
            )

    return ForceField(
        name=f"Tinker from {Path(path).name}",
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        vdws=vdws,
        source_path=Path(path),
        source_format="tinker_prm",
        functional_form=FunctionalForm.MM3,
    )


def save_tinker_prm(
    ff: ForceField,
    path: str | Path,
    template_path: str | Path | None = None,
    *,
    section_name: str = "OPT Generated",
) -> Path:
    """Write the force field to Tinker .prm format.

    If a template path is provided, or this force field came from
    :func:`load_tinker_prm`, the existing file is updated via the legacy
    exporter. Otherwise, a minimal Q2MM bond/angle section is written.
    """
    _validate_form_for_format(ff, "tinker_prm")
    output_path = Path(path)
    template = Path(template_path) if template_path is not None else None
    if template is None and ff.source_format == "tinker_prm" and ff.source_path is not None:
        template = ff.source_path

    if template is not None:
        template_params, template_lines = _tinker_import_ff(template)
        updated_params = copy.deepcopy(template_params)
        bond_by_row, bond_by_env = _build_bond_maps(ff.bonds)
        angle_by_row, angle_by_env = _build_angle_maps(ff.angles)

        for param in updated_params:
            if param.ptype in ("bf", "be"):
                bond = _match_bond_for_export(param, bond_by_row, bond_by_env)
                if bond is not None:
                    param.value = (
                        canonical_to_mm3_bond_k(bond.force_constant) if param.ptype == "bf" else bond.equilibrium
                    )
            elif param.ptype == "af":
                angle = _match_angle_for_export(param, angle_by_row, angle_by_env)
                if angle is not None:
                    param.value = canonical_to_mm3_angle_k(angle.force_constant)
            elif param.ptype == "ae" and getattr(param, "ff_col", None) == 2:
                angle = _match_angle_for_export(param, angle_by_row, angle_by_env)
                if angle is not None:
                    param.value = angle.equilibrium
            elif param.ptype == "df":
                _update_torsion_param(param, ff.torsions)

        updated_lines = list(template_lines)
        _tinker_export_ff(str(output_path), updated_params, updated_lines)
        if ff.vdws:
            _update_tinker_vdw_lines(output_path, ff.vdws)
        return output_path

    lines = ["# Q2MM\n", f"# {section_name}\n"]
    for bond in ff.bonds:
        lines.append(
            _format_tinker_bond_line(
                _tinker_atom_types(bond.env_id, bond.elements),
                canonical_to_mm3_bond_k(bond.force_constant),
                bond.equilibrium,
            )
        )
    for angle in ff.angles:
        lines.append(
            _format_tinker_angle_line(
                _tinker_atom_types(angle.env_id, angle.elements),
                canonical_to_mm3_angle_k(angle.force_constant),
                angle.equilibrium,
            )
        )
    for vdw in ff.vdws:
        lines.append(_format_tinker_vdw_line(vdw.atom_type, vdw.radius, vdw.epsilon, vdw.reduction))
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
