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
    TorsionParam,
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
    torsions = []
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

    return ForceField(
        name=f"MM3 from {Path(path).name}",
        bonds=bonds,
        angles=angles,
        torsions=torsions,
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
            elif param.ptype == "df":
                _update_torsion_param(param, ff.torsions)

        updated_lines = list(parser.lines)
        parser.export_ff(path=str(output_path), params=updated_params, lines=updated_lines)
        if ff.vdws:
            _update_mm3_vdw_lines(output_path, ff.vdws)
        return output_path

    lines = [f" C  {substructure_name}\n", f" 9  {smiles}\n"]
    for bond in ff.bonds:
        lines.append(
            _format_mm3_bond_line(_mm3_atom_types(bond.env_id, bond.elements), bond.equilibrium, bond.force_constant)
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
    torsions = []
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
            elif param.ptype == "df":
                _update_torsion_param(param, ff.torsions)

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


def save_openmm_xml(
    ff: ForceField,
    path: str | Path,
    molecule=None,
) -> Path:
    """Write the force field to a standalone OpenMM ForceField XML file.

    Produces a ``<ForceField>`` XML document loadable by
    ``openmm.app.ForceField(path)``.  Custom force definitions use MM3
    functional forms (cubic bond stretch, sextic angle bend, buffered
    14-7 vdW) so the resulting system is physically equivalent to what
    :class:`~q2mm.backends.mm.openmm.OpenMMEngine` builds
    programmatically.

    A *molecule* (or iterable of molecules) can be provided to generate
    ``<Residues>`` and ``<AtomTypes>`` sections.  If omitted, only the
    ``<CustomBondForce>``, ``<CustomAngleForce>``, and
    ``<CustomNonbondedForce>`` definitions are written — the user must
    supply their own topology when loading.

    Args:
        ff: Force field to export.
        path: Output file path.
        molecule: Optional :class:`~q2mm.models.molecule.Q2MMMolecule`
            or list thereof.  Used to generate ``<AtomTypes>`` and
            ``<Residues>`` sections.

    Returns:
        The resolved output path.
    """
    import xml.etree.ElementTree as ET

    from q2mm.constants import (
        KCAL_TO_KJ,
        MDYNA_TO_KJMOLA2,
        MM3_ANGLE_C3,
        MM3_ANGLE_C4,
        MM3_ANGLE_C5,
        MM3_ANGLE_C6,
        MM3_BOND_C3,
        MM3_BOND_C4,
        MM3_STR,
        MASSES,
        RAD_TO_DEG,
    )

    root = ET.Element("ForceField")

    # ---- Atom types & residues (if molecule provided) ----
    molecules = []
    if molecule is not None:
        from q2mm.models.molecule import Q2MMMolecule

        if isinstance(molecule, Q2MMMolecule):
            molecules = [molecule]
        else:
            molecules = list(molecule)

    if molecules:
        atom_types_el = ET.SubElement(root, "AtomTypes")
        # Collect unique (element, atom_type) pairs across all molecules
        seen_types: dict[str, tuple[str, float]] = {}
        for mol in molecules:
            for symbol, atype in zip(mol.symbols, mol.atom_types):
                type_name = atype if atype else symbol
                if type_name not in seen_types:
                    mass = MASSES.get(symbol, 0.0)
                    seen_types[type_name] = (symbol, mass)

        for type_name, (element, mass) in sorted(seen_types.items()):
            ET.SubElement(
                atom_types_el,
                "Type",
                name=f"q2mm-{type_name}",
                element=element,
                mass=f"{mass:.4f}",
            ).set("class", type_name)

        residues_el = ET.SubElement(root, "Residues")
        for mol_idx, mol in enumerate(molecules):
            res_name = f"Q2MM{mol_idx}" if len(molecules) > 1 else "Q2MM"
            res_el = ET.SubElement(residues_el, "Residue", name=res_name)
            for i, (symbol, atype) in enumerate(zip(mol.symbols, mol.atom_types)):
                type_name = atype if atype else symbol
                ET.SubElement(
                    res_el,
                    "Atom",
                    name=f"{symbol}{i + 1}",
                    type=f"q2mm-{type_name}",
                )
            for bond in mol.bonds:
                ET.SubElement(
                    res_el,
                    "Bond",
                    atomName1=f"{mol.symbols[bond.atom_i]}{bond.atom_i + 1}",
                    atomName2=f"{mol.symbols[bond.atom_j]}{bond.atom_j + 1}",
                )

    # ---- Custom bond force (MM3 cubic stretch) ----
    if ff.bonds:
        bond_expr = f"k*(10*(r-r0))^2*(1-{MM3_BOND_C3}*(10*(r-r0))+{MM3_BOND_C4}*(10*(r-r0))^2)"
        bond_force_el = ET.SubElement(
            root,
            "CustomBondForce",
            energy=bond_expr,
        )
        ET.SubElement(bond_force_el, "PerBondParameter", name="k")
        ET.SubElement(bond_force_el, "PerBondParameter", name="r0")

        for bond in ff.bonds:
            k_openmm = 0.5 * float(bond.force_constant) * MDYNA_TO_KJMOLA2
            r0_nm = float(bond.equilibrium) * 0.1

            if molecules:
                # Add typed bonds (class1/class2 matching)
                env_parts = bond.env_id.split("-") if bond.env_id else list(bond.elements)
                class1 = env_parts[0] if len(env_parts) >= 2 else bond.elements[0]
                class2 = env_parts[1] if len(env_parts) >= 2 else bond.elements[1]
                bond_el = ET.SubElement(bond_force_el, "Bond")
                bond_el.set("class1", class1)
                bond_el.set("class2", class2)
                bond_el.set("k", f"{k_openmm:.6f}")
                bond_el.set("r0", f"{r0_nm:.6f}")
            else:
                bond_el = ET.SubElement(bond_force_el, "Bond")
                env_parts = bond.env_id.split("-") if bond.env_id else list(bond.elements)
                bond_el.set("class1", env_parts[0] if len(env_parts) >= 2 else bond.elements[0])
                bond_el.set("class2", env_parts[1] if len(env_parts) >= 2 else bond.elements[1])
                bond_el.set("k", f"{k_openmm:.6f}")
                bond_el.set("r0", f"{r0_nm:.6f}")

    # ---- Custom angle force (MM3 sextic bend) ----
    if ff.angles:
        angle_expr = (
            f"k*(theta-theta0)^2*("
            f"1+{MM3_ANGLE_C3}*((theta-theta0)*{RAD_TO_DEG})"
            f"+{MM3_ANGLE_C4}*((theta-theta0)*{RAD_TO_DEG})^2"
            f"+{MM3_ANGLE_C5}*((theta-theta0)*{RAD_TO_DEG})^3"
            f"+{MM3_ANGLE_C6}*((theta-theta0)*{RAD_TO_DEG})^4"
            f")"
        )
        angle_force_el = ET.SubElement(
            root,
            "CustomAngleForce",
            energy=angle_expr,
        )
        ET.SubElement(angle_force_el, "PerAngleParameter", name="k")
        ET.SubElement(angle_force_el, "PerAngleParameter", name="theta0")

        for angle in ff.angles:
            import math

            k_openmm = 0.5 * float(angle.force_constant) * MM3_STR
            theta0_rad = math.radians(float(angle.equilibrium))

            env_parts = angle.env_id.split("-") if angle.env_id else list(angle.elements)
            class1 = env_parts[0] if len(env_parts) >= 3 else angle.elements[0]
            class2 = env_parts[1] if len(env_parts) >= 3 else angle.elements[1]
            class3 = env_parts[2] if len(env_parts) >= 3 else angle.elements[2]
            angle_el = ET.SubElement(angle_force_el, "Angle")
            angle_el.set("class1", class1)
            angle_el.set("class2", class2)
            angle_el.set("class3", class3)
            angle_el.set("k", f"{k_openmm:.6f}")
            angle_el.set("theta0", f"{theta0_rad:.6f}")

    # ---- Custom torsion force ----
    if ff.torsions:
        import math

        torsion_expr = "k*(1+cos(n*theta-phase))"
        torsion_force_el = ET.SubElement(
            root,
            "CustomTorsionForce",
            energy=torsion_expr,
        )
        ET.SubElement(torsion_force_el, "PerTorsionParameter", name="k")
        ET.SubElement(torsion_force_el, "PerTorsionParameter", name="n")
        ET.SubElement(torsion_force_el, "PerTorsionParameter", name="phase")

        for torsion in ff.torsions:
            k_kj = float(torsion.force_constant) * KCAL_TO_KJ
            env_parts = torsion.env_id.split("-") if torsion.env_id else list(torsion.elements)
            class1 = env_parts[0] if len(env_parts) >= 4 else torsion.elements[0]
            class2 = env_parts[1] if len(env_parts) >= 4 else torsion.elements[1]
            class3 = env_parts[2] if len(env_parts) >= 4 else torsion.elements[2]
            class4 = env_parts[3] if len(env_parts) >= 4 else torsion.elements[3]
            tor_el = ET.SubElement(torsion_force_el, "Torsion")
            tor_el.set("class1", class1)
            tor_el.set("class2", class2)
            tor_el.set("class3", class3)
            tor_el.set("class4", class4)
            tor_el.set("k", f"{k_kj:.6f}")
            tor_el.set("n", str(torsion.periodicity))
            tor_el.set("phase", f"{math.radians(float(torsion.phase)):.6f}")

    # ---- Custom nonbonded vdW force (MM3 buffered 14-7) ----
    if ff.vdws:
        vdw_expr = "epsilon*(-2.25*(rv/r)^6+184000*exp(-12*r/rv));rv=radius1+radius2;epsilon=sqrt(epsilon1*epsilon2)"
        vdw_force_el = ET.SubElement(
            root,
            "CustomNonbondedForce",
            energy=vdw_expr,
            bondCutoff="2",
        )
        ET.SubElement(vdw_force_el, "PerParticleParameter", name="radius")
        ET.SubElement(vdw_force_el, "PerParticleParameter", name="epsilon")

        for vdw in ff.vdws:
            r_nm = float(vdw.radius) * 0.1
            eps_kj = float(vdw.epsilon) * KCAL_TO_KJ
            type_name = vdw.atom_type if vdw.atom_type else vdw.element
            atom_el = ET.SubElement(vdw_force_el, "Atom")
            atom_el.set("class", type_name)
            atom_el.set("radius", f"{r_nm:.6f}")
            atom_el.set("epsilon", f"{eps_kj:.6f}")

    # ---- Write XML ----
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    output_path = Path(path)
    tree.write(output_path, encoding="unicode", xml_declaration=True)
    return output_path


def _update_torsion_param(param, torsions: list[TorsionParam]) -> None:
    """Update a legacy ``df`` param from the ForceField's torsion list.

    Matches by ``ff_row`` + ``ff_col`` (periodicity).
    """
    periodicity = getattr(param, "ff_col", 1)
    ff_row = getattr(param, "ff_row", None)
    for t in torsions:
        if t.ff_row == ff_row and t.periodicity == periodicity:
            param.value = t.force_constant
            return
