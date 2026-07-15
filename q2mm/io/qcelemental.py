"""QCElemental interop for Q2MM molecules.

Converts between :class:`~q2mm.models.molecule.Molecule` and
``qcelemental.models.Molecule`` (geometry in Bohr, per QCElemental
convention, vs. Q2MM's canonical Ångström).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from q2mm.models.identifiers import canonicalize_bond_env_id
from q2mm.models.molecule import Bond, Molecule

try:
    import qcelemental as qcel

    _HAS_QCEL = True
except ImportError:  # pragma: no cover - exercised only without qcelemental installed
    qcel = None
    _HAS_QCEL = False


_QCEL_TO_CANONICAL_ORDER = {
    1.0: "-",
    1.5: "*",
    2.0: "=",
    3.0: "%",
}
_CANONICAL_TO_QCEL_ORDER = {value: key for key, value in _QCEL_TO_CANONICAL_ORDER.items()}
_EXPLICIT_EMPTY_CONNECTIVITY_KEY = "q2mm_explicit_empty_connectivity"


def molecule_from_qcel(mol: Any, name: str = "") -> Molecule:
    """Create a :class:`Molecule` from a QCElemental Molecule object.

    Args:
        mol: QCElemental ``Molecule`` with geometry in Bohr.
        name: Display name for the molecule.

    Returns:
        A new :class:`Molecule` with geometry converted to Ångströms.

    Raises:
        ImportError: If ``qcelemental`` is not installed.

    """
    if not _HAS_QCEL:
        raise ImportError("qcelemental required: pip install qcelemental")
    coords_bohr = np.array(mol.geometry).reshape(-1, 3)
    coords_ang = coords_bohr * qcel.constants.bohr2angstroms
    symbols = tuple(mol.symbols)
    atom_types = symbols

    connectivity = getattr(mol, "connectivity", None)
    extras = dict(getattr(mol, "extras", {}) or {})
    bonds: tuple[Bond, ...] | None
    if extras.get(_EXPLICIT_EMPTY_CONNECTIVITY_KEY):
        bonds = ()
    elif connectivity is None:
        bonds = None
    else:
        parsed_bonds = []
        for atom_i, atom_j, source_order in connectivity:
            numeric_order = float(source_order)
            parsed_bonds.append(
                Bond(
                    atom_i=int(atom_i),
                    atom_j=int(atom_j),
                    elements=(symbols[int(atom_i)], symbols[int(atom_j)]),
                    length=float(np.linalg.norm(coords_ang[int(atom_i)] - coords_ang[int(atom_j)])),
                    env_id=canonicalize_bond_env_id([atom_types[int(atom_i)], atom_types[int(atom_j)]]),
                    bond_order=_QCEL_TO_CANONICAL_ORDER.get(numeric_order, ""),
                    source_bond_order=str(source_order),
                )
            )
        bonds = tuple(parsed_bonds)

    return Molecule(
        symbols=symbols,
        atom_types=atom_types,
        geometry=coords_ang,
        charge=mol.molecular_charge,
        multiplicity=mol.molecular_multiplicity,
        name=name,
        bonds=bonds,
    )


def molecule_to_qcel(molecule: Molecule) -> Any:
    """Convert a :class:`Molecule` to a QCElemental Molecule.

    Args:
        molecule: Source molecule.

    Returns:
        A ``qcelemental.models.Molecule`` with geometry in Bohr and
        connectivity derived from detected/supplied bonds.

    Raises:
        ImportError: If ``qcelemental`` is not installed.

    """
    if not _HAS_QCEL:
        raise ImportError("qcelemental required: pip install qcelemental")
    coords_bohr = molecule.geometry / qcel.constants.bohr2angstroms
    kwargs: dict[str, Any] = {
        "symbols": list(molecule.symbols),
        "geometry": coords_bohr.flatten().tolist(),
        "molecular_charge": molecule.charge,
        "molecular_multiplicity": molecule.multiplicity,
    }
    if molecule.bonds_explicit and molecule.bonds:
        connectivity = []
        for bond in molecule.bonds:
            order = None
            if bond.source_bond_order is not None:
                try:
                    order = float(bond.source_bond_order)
                except ValueError:
                    order = None
            if order is None:
                order = _CANONICAL_TO_QCEL_ORDER.get(bond.bond_order, 1.0)
            connectivity.append((bond.atom_i, bond.atom_j, order))
        kwargs["connectivity"] = connectivity
    elif molecule.bonds_explicit:
        kwargs["extras"] = {_EXPLICIT_EMPTY_CONNECTIVITY_KEY: True}
    return qcel.models.Molecule(**kwargs)
