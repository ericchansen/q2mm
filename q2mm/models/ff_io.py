"""Force field format I/O — load and save ForceField objects.

Standalone functions that handle format-specific conversion, keeping
the ForceField dataclass itself format-agnostic.
"""

from __future__ import annotations

import copy
from pathlib import Path

from q2mm.models.forcefield import (
    AngleParam,
    BondParam,
    ForceField,
    VdwParam,
    _build_angle_maps,
    _build_bond_maps,
    _clean_atom_types,
    _format_mm3_angle_line,
    _format_mm3_bond_line,
    _format_mm3_vdw_line,
    _format_tinker_angle_line,
    _format_tinker_bond_line,
    _format_tinker_vdw_line,
    _match_angle_for_export,
    _match_bond_for_export,
    _mm3_atom_types,
    _parse_generic_tinker_prm,
    _parse_mm3_vdw_params,
    _parse_tinker_vdw_params,
    _tinker_atom_types,
    _update_mm3_vdw_lines,
    _update_tinker_vdw_lines,
)
from q2mm.models.identifiers import (
    _extract_element,
    canonicalize_angle_env_id,
    canonicalize_bond_env_id,
)


def load_mm3_fld(path: str | Path) -> ForceField:
    """Load from Schrödinger MM3 .fld file.

    Parses bond and angle parameters from the substructure sections,
    extracting element letters from MM3 atom type codes.
    """
    from q2mm.parsers.mm3 import MM3 as MM3Parser

    parser = MM3Parser(str(path))
    parser.import_ff()

    bonds = []
    angles = []
    vdws = _parse_mm3_vdw_params(Path(path))

    # Pre-build lookup for equilibrium values by (ptype, ff_row)
    eq_lookup = {}
    for p in parser.params:
        if p.ptype in ("be", "ae"):
            eq_lookup[(p.ptype, p.ff_row)] = p.value

    for param in parser.params:
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
                    force_constant=param.value,
                    label=f"MM3 row {param.ff_row}",
                    env_id=env_id,
                    ff_row=param.ff_row,
                )
            )

        elif param.ptype == "af" and len(atom_types) >= 2:
            # Angle: extract center and outer elements
            if len(atom_types) >= 3:
                elems = tuple(_extract_element(t) for t in atom_types[:3])
                env_id = canonicalize_angle_env_id(atom_types[:3])
            else:
                elems = (_extract_element(atom_types[0]), _extract_element(atom_types[1]), "?")
                env_id = canonicalize_angle_env_id(atom_types[:2])
            eq_val = eq_lookup.get(("ae", param.ff_row), 0.0)
            angles.append(
                AngleParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=param.value,
                    label=f"MM3 row {param.ff_row}",
                    env_id=env_id,
                    ff_row=param.ff_row,
                )
            )

    return ForceField(
        name=f"MM3 from {Path(path).name}",
        bonds=bonds,
        angles=angles,
        vdws=vdws,
        source_path=Path(path),
        source_format="mm3_fld",
    )


def save_mm3_fld(
    ff: ForceField,
    path: str | Path,
    template_path: str | Path | None = None,
    *,
    substructure_name: str = "OPT Generated",
    smiles: str = "AUTO",
) -> Path:
    """Write the force field to MM3 .fld format.

    If a template path is provided, or this force field came from
    :func:`load_mm3_fld`, the existing file is updated in-place via the
    legacy MM3 exporter so comments and unrelated parameters are preserved.

    Otherwise, a minimal bond/angle-only MM3 substructure is generated.
    """
    output_path = Path(path)
    template = Path(template_path) if template_path is not None else None
    if template is None and ff.source_format == "mm3_fld" and ff.source_path is not None:
        template = ff.source_path

    if template is not None:
        from q2mm.parsers.mm3 import MM3

        parser = MM3(str(template))
        parser.import_ff()
        updated_params = copy.deepcopy(parser.params)
        bond_by_row, bond_by_env = _build_bond_maps(ff.bonds)
        angle_by_row, angle_by_env = _build_angle_maps(ff.angles)

        for param in updated_params:
            if param.ptype in ("bf", "be"):
                bond = _match_bond_for_export(param, bond_by_row, bond_by_env)
                if bond is not None:
                    param.value = bond.force_constant if param.ptype == "bf" else bond.equilibrium
            elif param.ptype in ("af", "ae"):
                angle = _match_angle_for_export(param, angle_by_row, angle_by_env)
                if angle is not None:
                    param.value = angle.force_constant if param.ptype == "af" else angle.equilibrium

        updated_lines = list(parser.lines)
        parser.export_ff(path=str(output_path), params=updated_params, lines=updated_lines)
        if ff.vdws:
            _update_mm3_vdw_lines(output_path, ff.vdws)
        return output_path

    lines = [f" C  {substructure_name}\n", f" 9  {smiles}\n"]
    for bond in ff.bonds:
        lines.append(
            _format_mm3_bond_line(
                _mm3_atom_types(bond.env_id, bond.elements), bond.equilibrium, bond.force_constant
            )
        )
    for angle in ff.angles:
        lines.append(
            _format_mm3_angle_line(
                _mm3_atom_types(angle.env_id, angle.elements), angle.equilibrium, angle.force_constant
            )
        )
    lines.append("-3\n")
    if ff.vdws:
        lines.extend(["-6\n"])
        for vdw in ff.vdws:
            lines.append(_format_mm3_vdw_line(vdw))
        lines.extend([" END OF NONBONDED INTERACTIONS\n", "-2\n"])
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


def load_tinker_prm(path: str | Path) -> ForceField:
    """Load bond and angle parameters from a Tinker .prm file."""
    from q2mm.parsers.tinker_ff import TinkerFF

    parser = TinkerFF(str(path))
    parser.import_ff()

    if not parser.params:
        bonds, angles, vdws = _parse_generic_tinker_prm(Path(path))
        return ForceField(
            name=f"Tinker from {Path(path).name}",
            bonds=bonds,
            angles=angles,
            vdws=vdws,
            source_path=Path(path),
            source_format="tinker_prm",
        )

    bonds = []
    angles = []
    vdws = _parse_tinker_vdw_params(Path(path))

    eq_lookup: dict[tuple[str, int], float] = {}
    for param in parser.params:
        if param.ptype == "be" or (param.ptype == "ae" and getattr(param, "ff_col", None) == 2):
            eq_lookup[(param.ptype, param.ff_row)] = param.value

    for param in parser.params:
        atom_types = _clean_atom_types(getattr(param, "atom_types", None), 4)

        if param.ptype == "bf" and len(atom_types) >= 2:
            elems = tuple(_extract_element(t) for t in atom_types[:2])
            env_id = canonicalize_bond_env_id(atom_types[:2])
            eq_val = eq_lookup.get(("be", param.ff_row), 0.0)
            bonds.append(
                BondParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=param.value,
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
                    force_constant=param.value,
                    label=f"Tinker row {param.ff_row}",
                    env_id=env_id,
                    ff_row=param.ff_row,
                )
            )

    return ForceField(
        name=f"Tinker from {Path(path).name}",
        bonds=bonds,
        angles=angles,
        vdws=vdws,
        source_path=Path(path),
        source_format="tinker_prm",
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
    output_path = Path(path)
    template = Path(template_path) if template_path is not None else None
    if template is None and ff.source_format == "tinker_prm" and ff.source_path is not None:
        template = ff.source_path

    if template is not None:
        from q2mm.parsers.tinker_ff import TinkerFF

        parser = TinkerFF(str(template))
        parser.import_ff()
        updated_params = copy.deepcopy(parser.params)
        bond_by_row, bond_by_env = _build_bond_maps(ff.bonds)
        angle_by_row, angle_by_env = _build_angle_maps(ff.angles)

        for param in updated_params:
            if param.ptype in ("bf", "be"):
                bond = _match_bond_for_export(param, bond_by_row, bond_by_env)
                if bond is not None:
                    param.value = bond.force_constant if param.ptype == "bf" else bond.equilibrium
            elif param.ptype == "af":
                angle = _match_angle_for_export(param, angle_by_row, angle_by_env)
                if angle is not None:
                    param.value = angle.force_constant
            elif param.ptype == "ae" and getattr(param, "ff_col", None) == 2:
                angle = _match_angle_for_export(param, angle_by_row, angle_by_env)
                if angle is not None:
                    param.value = angle.equilibrium

        updated_lines = list(parser.lines)
        parser.export_ff(path=str(output_path), params=updated_params, lines=updated_lines)
        if ff.vdws:
            _update_tinker_vdw_lines(output_path, ff.vdws)
        return output_path

    lines = ["# Q2MM\n", f"# {section_name}\n"]
    for bond in ff.bonds:
        lines.append(
            _format_tinker_bond_line(
                _tinker_atom_types(bond.env_id, bond.elements), bond.force_constant, bond.equilibrium
            )
        )
    for angle in ff.angles:
        lines.append(
            _format_tinker_angle_line(
                _tinker_atom_types(angle.env_id, angle.elements), angle.force_constant, angle.equilibrium
            )
        )
    for vdw in ff.vdws:
        lines.append(_format_tinker_vdw_line(vdw.atom_type, vdw.radius, vdw.epsilon, vdw.reduction))
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
