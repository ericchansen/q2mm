"""OpenMM XML file format I/O."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from q2mm.io._helpers import _validate_form_for_format
from q2mm.models.forcefield import ForceField

logger = logging.getLogger(__name__)

# MM3/Tinker wildcard atom type — means "any atom".  OpenMM XML has no
# wildcard concept, so terms containing this type must be skipped.
_WILDCARD_TYPES = frozenset({"00"})

if TYPE_CHECKING:
    from q2mm.models.molecule import Molecule


def save_openmm_xml(
    ff: ForceField,
    path: str | Path,
    molecule: Molecule | list[Molecule] | None = None,
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
        molecule (Molecule | list[Molecule] | None): Optional molecule
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
        canonical_to_openmm_angle_k,
        canonical_to_openmm_bond_k,
        canonical_to_openmm_epsilon,
        canonical_to_openmm_torsion_k,
        deg_to_rad,
    )

    root = ET.Element("ForceField")

    # ---- Atom types & residues (if molecule provided) ----
    molecules = []
    if molecule is not None:
        from q2mm.models.molecule import Molecule

        if isinstance(molecule, Molecule):
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
            if _WILDCARD_TYPES & {class1, class2}:
                logger.debug("Skipping bond %s — wildcard type '00' has no OpenMM equivalent", bond.env_id)
                continue
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
            if _WILDCARD_TYPES & {class1, class2, class3}:
                logger.debug("Skipping angle %s — wildcard type '00' has no OpenMM equivalent", angle.env_id)
                continue
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
            if _WILDCARD_TYPES & {class1, class2, class3, class4}:
                logger.debug("Skipping torsion %s — wildcard type '00' has no OpenMM equivalent", torsion.env_id)
                continue
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
