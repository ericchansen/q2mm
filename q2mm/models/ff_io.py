"""Force field format I/O — load and save ForceField objects.

Standalone functions that handle format-specific conversion, keeping
the ForceField dataclass itself format-agnostic.
"""

from __future__ import annotations

import contextlib
import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from q2mm.models.molecule import Q2MMMolecule

from q2mm.models.forcefield import (
    AngleParam,
    BondParam,
    ForceField,
    FunctionalForm,
    TorsionParam,
    VdwParam,
    _build_angle_maps,
    _build_bond_maps,
    _clean_atom_types,
    _format_mm3_angle_line,
    _format_mm3_bond_line,
    _format_mm3_torsion_line,
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
from q2mm.models.units import (
    canonical_to_mm3_angle_k,
    canonical_to_mm3_bond_k,
    mm3_angle_k_to_canonical,
    mm3_bond_k_to_canonical,
)


_FORMAT_COMPATIBLE_FORMS: dict[str, set[str]] = {
    "mm3_fld": {"mm3"},
    "tinker_prm": {"mm3"},
    "openmm_xml": {"mm3"},
    "amber_frcmod": {"harmonic"},
}


def _validate_form_for_format(ff: ForceField, target_format: str) -> None:
    """Raise ``ValueError`` if the force field's functional form is incompatible with *target_format*."""
    form = getattr(ff, "functional_form", None)
    if form is None:
        return
    form_value = form.value if hasattr(form, "value") else str(form)
    allowed = _FORMAT_COMPATIBLE_FORMS.get(target_format)
    if allowed is not None and form_value not in allowed:
        raise ValueError(
            f"Cannot save a {form!r} force field to {target_format!r} format. Compatible forms: {sorted(allowed)}"
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
                    force_constant=mm3_bond_k_to_canonical(param.value),
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

    return ForceField(
        name=f"MM3 from {Path(path).name}",
        bonds=bonds,
        angles=angles,
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
    substructure_name: str = "OPT Generated",
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

        updated_lines = list(parser.lines)
        parser.export_ff(path=str(output_path), params=updated_params, lines=updated_lines)
        if ff.vdws:
            _update_mm3_vdw_lines(output_path, ff.vdws)
        return output_path

    lines = [f" C  {substructure_name}\n", f" 9  {smiles}\n"]
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


def load_tinker_prm(path: str | Path) -> ForceField:
    """Load bond and angle parameters from a Tinker .prm file."""
    from q2mm.parsers.tinker_ff import TinkerFF

    parser = TinkerFF(str(path))
    parser.import_ff()

    if not parser.params:
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

        updated_lines = list(parser.lines)
        parser.export_ff(path=str(output_path), params=updated_params, lines=updated_lines)
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


def save_openmm_xml(
    ff: ForceField,
    path: str | Path,
    molecule: Q2MMMolecule | list[Q2MMMolecule] | None = None,
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
    ``<CustomBondForce>``, ``<CustomAngleForce>``, ``<CustomTorsionForce>``,
    and ``<CustomNonbondedForce>`` definitions are written — the user must
    supply their own topology when loading.

    Args:
        ff (ForceField): Force field to export.
        path (str | Path): Output file path.
        molecule (Q2MMMolecule | list[Q2MMMolecule] | None): Optional molecule
            or list thereof.  Used to generate ``<AtomTypes>`` and
            ``<Residues>`` sections.

    Returns:
        The resolved output path.

    """
    _validate_form_for_format(ff, "openmm_xml")
    import xml.etree.ElementTree as ET

    from q2mm.constants import (
        MM3_ANGLE_C3,
        MM3_ANGLE_C4,
        MM3_ANGLE_C5,
        MM3_ANGLE_C6,
        MM3_BOND_C3,
        MM3_BOND_C4,
        MASSES,
    )
    from q2mm.models.units import (
        RAD_TO_DEG,
        ang_to_nm,
        canonical_to_openmm_bond_k,
        canonical_to_openmm_angle_k,
        canonical_to_openmm_torsion_k,
        canonical_to_openmm_epsilon,
        deg_to_rad,
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
                    if symbol not in MASSES:
                        raise ValueError(f"No atomic mass for element '{symbol}' when building <AtomTypes>.")
                    seen_types[type_name] = (symbol, MASSES[symbol])
                else:
                    prev_symbol, _ = seen_types[type_name]
                    if prev_symbol != symbol:
                        raise ValueError(
                            f"Inconsistent element for atom type '{type_name}': "
                            f"seen both '{prev_symbol}' and '{symbol}' across exported molecules."
                        )

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
            k_openmm = canonical_to_openmm_bond_k(bond.force_constant)
            r0_nm = ang_to_nm(bond.equilibrium)

            env_parts = bond.env_id.split("-") if bond.env_id else list(bond.elements)
            class1 = env_parts[0] if len(env_parts) >= 2 else bond.elements[0]
            class2 = env_parts[1] if len(env_parts) >= 2 else bond.elements[1]
            bond_el = ET.SubElement(bond_force_el, "Bond")
            bond_el.set("class1", class1)
            bond_el.set("class2", class2)
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
            k_openmm = canonical_to_openmm_angle_k(angle.force_constant)
            theta0_rad = deg_to_rad(angle.equilibrium)

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
            k_kj = canonical_to_openmm_torsion_k(torsion.force_constant)
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
            tor_el.set("phase", f"{deg_to_rad(torsion.phase):.6f}")

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
            r_nm = ang_to_nm(vdw.radius)
            eps_kj = canonical_to_openmm_epsilon(vdw.epsilon)
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


# ---- AMBER .frcmod I/O ----

_FRCMOD_SECTIONS = frozenset({"MASS", "BOND", "ANGLE", "ANGL", "DIHE", "IMPROPER", "NONBON", "NONB"})

# Average atomic masses for nearest-match element inference.  AMBER force
# fields report *average* masses in the MASS section, which differ from the
# monoisotopic values in q2mm.elements.  This table covers elements commonly
# encountered in molecular-mechanics force fields.
_AVG_MASS_ELEMENT: list[tuple[float, str]] = sorted(
    [
        (1.008, "H"),
        (4.003, "He"),
        (6.941, "Li"),
        (9.012, "Be"),
        (10.81, "B"),
        (12.011, "C"),
        (14.007, "N"),
        (15.999, "O"),
        (18.998, "F"),
        (22.990, "Na"),
        (24.305, "Mg"),
        (26.982, "Al"),
        (28.086, "Si"),
        (30.974, "P"),
        (32.065, "S"),
        (35.453, "Cl"),
        (39.098, "K"),
        (40.078, "Ca"),
        (47.867, "Ti"),
        (51.996, "Cr"),
        (54.938, "Mn"),
        (55.845, "Fe"),
        (58.933, "Co"),
        (58.693, "Ni"),
        (63.546, "Cu"),
        (65.38, "Zn"),
        (79.904, "Br"),
        (95.94, "Mo"),
        (101.07, "Ru"),
        (102.91, "Rh"),
        (106.42, "Pd"),
        (107.87, "Ag"),
        (112.41, "Cd"),
        (118.71, "Sn"),
        (126.90, "I"),
        (183.84, "W"),
        (190.23, "Os"),
        (192.22, "Ir"),
        (195.08, "Pt"),
        (196.97, "Au"),
    ]
)


def _element_from_mass(mass: float, tolerance: float = 1.5) -> str | None:
    """Find the element whose average atomic mass is closest to *mass*.

    Returns ``None`` if no element is within *tolerance* amu.
    """
    best_sym: str | None = None
    best_diff = tolerance
    for m, sym in _AVG_MASS_ELEMENT:
        diff = abs(m - mass)
        if diff < best_diff:
            best_diff = diff
            best_sym = sym
        elif m > mass + tolerance:
            break
    return best_sym


# Lowercase GAFF/AMBER two-character type names that genuinely represent
# two-letter elements.  Everything else follows the GAFF convention of
# element = first character.  This prevents _extract_element() from
# misidentifying types like ``ca`` (aromatic C) as Ca (calcium).
_GAFF_TWO_LETTER_ELEMENTS: frozenset[str] = frozenset(
    {"cl", "br", "zn", "cu", "fe", "mn", "co", "ni", "pd", "pt", "au", "ag", "ru", "rh", "ir"}
)


def _amber_type_to_element(atom_type: str, mass_map: dict[str, float] | None = None) -> str:
    """Infer element from AMBER/GAFF atom type.

    Uses *mass_map* (from the MASS section) when available — this gives
    definitive results.  Falls back to the GAFF convention: if the
    lowercase type is a known two-letter element (``cl``, ``br``, ``zn``
    etc.) return that element, otherwise the element is the first
    character uppercased.
    """
    t = atom_type.strip()
    if not t:
        return "X"
    if mass_map and t in mass_map:
        elem = _element_from_mass(mass_map[t])
        if elem is not None:
            return elem
    # GAFF fallback: check known two-letter element types, then first-char.
    lower = t.lower()
    if lower in _GAFF_TWO_LETTER_ELEMENTS:
        return lower.title()
    return t[0].upper()


def _parse_amber_types(line: str, n_types: int) -> tuple[list[str], str]:
    """Extract *n_types* AMBER atom types from the start of *line*.

    Each type occupies 2 characters, separated by ``-``.  Returns the
    list of stripped type strings and the remainder of the line.
    """
    end = n_types * 3 - 1  # 2 chars per type + 1 dash between each pair
    types = [line[i * 3 : i * 3 + 2].strip() for i in range(n_types)]
    return types, line[end:]


def _parse_floats(text: str) -> list[float]:
    """Parse leading numeric tokens from *text*, stopping at comments."""
    vals: list[float] = []
    for token in text.split():
        try:
            vals.append(float(token))
        except ValueError:
            break
    return vals


def load_amber_frcmod(path: str | Path) -> ForceField:
    """Load from standard AMBER .frcmod file.

    Parses MASS, BOND, ANGLE/ANGL, DIHE, IMPROPER, and NONBON sections.
    Atom type → element mapping uses the MASS section when present,
    falling back to the GAFF convention (first character).
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    bonds: list[BondParam] = []
    angles: list[AngleParam] = []
    torsions: list[TorsionParam] = []
    vdws: list[VdwParam] = []
    mass_map: dict[str, float] = {}

    section: str | None = None
    for row, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Section headers
        if stripped in _FRCMOD_SECTIONS:
            section = stripped
            if section in ("ANGL",):
                section = "ANGLE"
            if section == "NONB":
                section = "NONBON"
            continue

        # Blank line ends section
        if not stripped:
            section = None
            continue

        # Skip comments and the remark line (row 1 before any section)
        if stripped.startswith("#") or section is None:
            continue

        if section == "MASS":
            parts = stripped.split()
            if len(parts) >= 2:
                with contextlib.suppress(ValueError):
                    mass_map[parts[0]] = float(parts[1])

        elif section == "BOND":
            types, rest = _parse_amber_types(line, 2)
            vals = _parse_floats(rest)
            if len(types) == 2 and all(types) and len(vals) >= 2:
                elems = tuple(_amber_type_to_element(t, mass_map) for t in types)
                bonds.append(
                    BondParam(
                        elements=elems,
                        equilibrium=vals[1],
                        force_constant=vals[0],
                        env_id="-".join(types),
                        ff_row=row,
                        label=f"frcmod row {row}",
                    )
                )

        elif section == "ANGLE":
            types, rest = _parse_amber_types(line, 3)
            vals = _parse_floats(rest)
            if len(types) == 3 and all(types) and len(vals) >= 2:
                elems = tuple(_amber_type_to_element(t, mass_map) for t in types)
                angles.append(
                    AngleParam(
                        elements=elems,
                        equilibrium=vals[1],
                        force_constant=vals[0],
                        env_id="-".join(types),
                        ff_row=row,
                        label=f"frcmod row {row}",
                    )
                )

        elif section == "DIHE":
            types, rest = _parse_amber_types(line, 4)
            vals = _parse_floats(rest)
            # vals: IDIVF, barrier, phase, periodicity
            if len(types) == 4 and all(types) and len(vals) >= 4:
                idivf = int(vals[0]) if vals[0] != 0 else 1
                barrier = vals[1]
                phase = vals[2]
                periodicity = abs(int(vals[3]))
                k = barrier / idivf
                elems = tuple(_amber_type_to_element(t, mass_map) for t in types)
                torsions.append(
                    TorsionParam(
                        elements=elems,
                        periodicity=periodicity or 1,
                        force_constant=k,
                        phase=phase,
                        env_id="-".join(types),
                        ff_row=row,
                        label=f"frcmod row {row}",
                    )
                )

        elif section == "IMPROPER":
            types, rest = _parse_amber_types(line, 4)
            vals = _parse_floats(rest)
            # vals: barrier, phase, periodicity (no IDIVF)
            if len(types) == 4 and all(types) and len(vals) >= 3:
                elems = tuple(_amber_type_to_element(t, mass_map) for t in types)
                periodicity = abs(int(vals[2]))
                torsions.append(
                    TorsionParam(
                        elements=elems,
                        periodicity=periodicity or 1,
                        force_constant=vals[0],
                        phase=vals[1],
                        env_id="-".join(types),
                        ff_row=row,
                        label=f"frcmod row {row} (improper)",
                        is_improper=True,
                    )
                )

        elif section == "NONBON":
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    atype = parts[0]
                    radius, epsilon = float(parts[1]), float(parts[2])
                    elem = _amber_type_to_element(atype, mass_map)
                    vdws.append(
                        VdwParam(
                            atom_type=atype,
                            radius=radius,
                            epsilon=epsilon,
                            element=elem,
                            ff_row=row,
                            label=f"frcmod row {row}",
                        )
                    )
                except ValueError:
                    pass

    return ForceField(
        name=f"AMBER from {path.name}",
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        vdws=vdws,
        source_path=path,
        source_format="amber_frcmod",
        functional_form=FunctionalForm.HARMONIC,
    )


def save_amber_frcmod(
    ff: ForceField,
    path: str | Path,
    template_path: str | Path | None = None,
    *,
    remark: str = "Q2MM generated frcmod",
) -> Path:
    """Write the force field to AMBER .frcmod format.

    If *template_path* is provided (or the ForceField was loaded from a
    .frcmod file), the template is updated in-place, preserving comments
    and unrelated sections.  Otherwise a standalone file is generated.
    """
    _validate_form_for_format(ff, "amber_frcmod")
    output_path = Path(path)
    template = Path(template_path) if template_path is not None else None
    if template is None and ff.source_format == "amber_frcmod" and ff.source_path is not None:
        template = ff.source_path

    if template is not None:
        return _save_amber_frcmod_template(ff, output_path, template)

    return _save_amber_frcmod_standalone(ff, output_path, remark)


def _extract_amber_trailing(line: str, n_types: int, n_values: int) -> str:
    """Extract trailing comment/text after the numeric fields in a frcmod line.

    *n_types* is the number of atom types (2 for BOND, 3 for ANGLE, etc.)
    and *n_values* is the expected count of numeric columns.  Returns the
    trailing text (including any leading whitespace) or empty string.
    """
    _, rest = _parse_amber_types(line, n_types)
    # Walk through *rest* consuming numeric tokens
    pos = 0
    consumed = 0
    while consumed < n_values and pos < len(rest):
        # Skip whitespace
        while pos < len(rest) and rest[pos] in " \t":
            pos += 1
        if pos >= len(rest):
            break
        # Try to consume a numeric token
        tok_start = pos
        while pos < len(rest) and rest[pos] not in " \t\n":
            pos += 1
        tok = rest[tok_start:pos]
        try:
            float(tok)
            consumed += 1
        except ValueError:
            break
    return rest[pos:].rstrip("\n")


def _format_amber_bond_line(types: list[str], k: float, r0: float, suffix: str = "") -> str:
    return f"{types[0]:<2}-{types[1]:<2} {k:12.4f} {r0:10.4f}{suffix}\n"


def _format_amber_angle_line(types: list[str], k: float, theta0: float, suffix: str = "") -> str:
    return f"{types[0]:<2}-{types[1]:<2}-{types[2]:<2} {k:12.4f} {theta0:10.4f}{suffix}\n"


def _format_amber_dihe_line(
    types: list[str], k: float, phase: float, periodicity: int, suffix: str = "", *, idivf: int = 1
) -> str:
    return f"{types[0]:<2}-{types[1]:<2}-{types[2]:<2}-{types[3]:<2}   {idivf} {k:10.4f} {phase:8.3f} {float(periodicity):8.3f}{suffix}\n"


def _format_amber_improper_line(types: list[str], k: float, phase: float, periodicity: int, suffix: str = "") -> str:
    return f"{types[0]:<2}-{types[1]:<2}-{types[2]:<2}-{types[3]:<2} {k:10.4f} {phase:8.1f} {float(periodicity):8.1f}{suffix}\n"


def _format_amber_nonbon_line(atom_type: str, radius: float, epsilon: float, suffix: str = "") -> str:
    return f"{atom_type:<2} {radius:10.4f} {epsilon:10.4f}{suffix}\n"


def _amber_env_types(env_id: str, elements: tuple[str, ...]) -> list[str]:
    """Get AMBER-style atom types from env_id, falling back to element symbols."""
    parts = [p.strip() for p in env_id.split("-") if p.strip()] if env_id else []
    if len(parts) == len(elements):
        return parts
    return [e.lower() for e in elements]


def _save_amber_frcmod_standalone(ff: ForceField, output_path: Path, remark: str) -> Path:
    """Generate a standalone .frcmod file from scratch."""
    lines = [f"{remark}\n"]

    if ff.bonds:
        lines.append("BOND\n")
        for bond in ff.bonds:
            types = _amber_env_types(bond.env_id, bond.elements)
            lines.append(_format_amber_bond_line(types, bond.force_constant, bond.equilibrium))
        lines.append("\n")

    if ff.angles:
        lines.append("ANGLE\n")
        for angle in ff.angles:
            types = _amber_env_types(angle.env_id, angle.elements)
            lines.append(_format_amber_angle_line(types, angle.force_constant, angle.equilibrium))
        lines.append("\n")

    if ff.torsions:
        proper = [t for t in ff.torsions if not t.is_improper]
        improper = [t for t in ff.torsions if t.is_improper]
        if proper:
            lines.append("DIHE\n")
            for tor in proper:
                types = _amber_env_types(tor.env_id, tor.elements)
                lines.append(_format_amber_dihe_line(types, tor.force_constant, tor.phase, tor.periodicity))
            lines.append("\n")
        if improper:
            lines.append("IMPROPER\n")
            for tor in improper:
                types = _amber_env_types(tor.env_id, tor.elements)
                lines.append(_format_amber_improper_line(types, tor.force_constant, tor.phase, tor.periodicity))
            lines.append("\n")

    if ff.vdws:
        lines.append("NONBON\n")
        for vdw in ff.vdws:
            lines.append(_format_amber_nonbon_line(vdw.atom_type, vdw.radius, vdw.epsilon))
        lines.append("\n")

    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


def _save_amber_frcmod_template(ff: ForceField, output_path: Path, template: Path) -> Path:
    """Update parameter values in an existing .frcmod template."""
    src_lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
    bond_by_row = {b.ff_row: b for b in ff.bonds if b.ff_row is not None}
    angle_by_row = {a.ff_row: a for a in ff.angles if a.ff_row is not None}
    torsion_by_row = {t.ff_row: t for t in ff.torsions if t.ff_row is not None}
    vdw_by_row = {v.ff_row: v for v in ff.vdws if v.ff_row is not None}

    section: str | None = None
    out_lines: list[str] = []
    for row, line in enumerate(src_lines, start=1):
        stripped = line.strip()

        if stripped in _FRCMOD_SECTIONS:
            section = stripped
            if section in ("ANGL",):
                section = "ANGLE"
            if section == "NONB":
                section = "NONBON"
            out_lines.append(line)
            continue

        if not stripped:
            section = None
            out_lines.append(line)
            continue

        updated = False
        if section == "BOND" and row in bond_by_row:
            b = bond_by_row[row]
            types, _ = _parse_amber_types(line, 2)
            suffix = _extract_amber_trailing(line, 2, 2)
            out_lines.append(_format_amber_bond_line(types, b.force_constant, b.equilibrium, suffix))
            updated = True
        elif section == "ANGLE" and row in angle_by_row:
            a = angle_by_row[row]
            types, _ = _parse_amber_types(line, 3)
            suffix = _extract_amber_trailing(line, 3, 2)
            out_lines.append(_format_amber_angle_line(types, a.force_constant, a.equilibrium, suffix))
            updated = True
        elif section in ("DIHE", "IMPROPER") and row in torsion_by_row:
            t = torsion_by_row[row]
            types, rest = _parse_amber_types(line, 4)
            n_vals = 3 if section == "IMPROPER" else 4
            suffix = _extract_amber_trailing(line, 4, n_vals)
            if section == "IMPROPER":
                out_lines.append(_format_amber_improper_line(types, t.force_constant, t.phase, t.periodicity, suffix))
            else:
                # Preserve the template's IDIVF and reconstruct the barrier
                # so the written line matches the original IDIVF column.
                orig_vals = _parse_floats(rest)
                idivf = int(orig_vals[0]) if orig_vals and orig_vals[0] != 0 else 1
                barrier = t.force_constant * idivf
                out_lines.append(_format_amber_dihe_line(types, barrier, t.phase, t.periodicity, suffix, idivf=idivf))
            updated = True
        elif section == "NONBON" and row in vdw_by_row:
            v = vdw_by_row[row]
            # NONBON uses whitespace-delimited fields (atom_type, radius, epsilon).
            # Preserve any trailing text after those 3 tokens.
            tokens = stripped.split()
            tail = ""
            if len(tokens) > 3:
                third_end = stripped.index(tokens[2]) + len(tokens[2])
                tail = stripped[third_end:]
            out_lines.append(_format_amber_nonbon_line(v.atom_type, v.radius, v.epsilon, tail))
            updated = True

        if not updated:
            out_lines.append(line)

    output_path.write_text("".join(out_lines), encoding="utf-8")
    return output_path


def _update_torsion_param(param: Any, torsions: list[TorsionParam]) -> None:
    """Update a legacy ``df`` param from the ForceField's torsion list.

    Matches by ``ff_row`` + ``ff_col`` (periodicity).
    """
    periodicity = getattr(param, "ff_col", 1)
    ff_row = getattr(param, "ff_row", None)
    for t in torsions:
        if t.ff_row == ff_row and t.periodicity == periodicity:
            param.value = t.force_constant
            return
