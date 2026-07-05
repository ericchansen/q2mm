"""Internal OpenMM term records.

Private dataclasses mapping molecule bonds, angles, torsions, van der Waals
particles, Urey-Bradley pairs, and CMAP corrections to their OpenMM force
indices.  These are implementation details of
:mod:`q2mm.backends.mm.openmm` and are not part of the public API.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _BondTerm:
    """Internal record mapping a molecule bond to its OpenMM force index.

    Attributes:
        force_index: Index of this bond in the OpenMM bond force object.
        atom_i: First atom index.
        atom_j: Second atom index.
        elements: Element symbols for the two atoms.
        env_id: Chemical environment identifier for parameter matching.
        ff_row: Row index in the source force field file, if applicable.

    """

    force_index: int
    atom_i: int
    atom_j: int
    elements: tuple[str, str]
    env_id: str = ""
    ff_row: int | None = None


@dataclass
class _AngleTerm:
    """Internal record mapping a molecule angle to its OpenMM force index.

    Attributes:
        force_index: Index of this angle in the OpenMM angle force object.
        atom_i: First atom index.
        atom_j: Central atom index.
        atom_k: Third atom index.
        elements: Element symbols for the three atoms.
        env_id: Chemical environment identifier for parameter matching.
        ff_row: Row index in the source force field file, if applicable.

    """

    force_index: int
    atom_i: int
    atom_j: int
    atom_k: int
    elements: tuple[str, str, str]
    env_id: str = ""
    ff_row: int | None = None


@dataclass
class _VdwTerm:
    """Internal record mapping a molecule atom to its OpenMM vdW particle.

    Attributes:
        particle_index: Index of this particle in the OpenMM vdW force.
        atom_type: Atom type label for parameter matching.
        element: Element symbol.
        ff_row: Row index in the source force field file, if applicable.

    """

    particle_index: int
    atom_type: str = ""
    element: str = ""
    ff_row: int | None = None


@dataclass
class _Exception14:
    """A 1-4 nonbonded exception whose parameters must track particle updates.

    Attributes:
        exception_index: Index of this exception in the OpenMM NonbondedForce.
        particle_i: First particle index.
        particle_j: Second particle index.

    """

    exception_index: int
    particle_i: int
    particle_j: int


@dataclass
class _TorsionTerm:
    """Internal record mapping a molecule torsion to its OpenMM force index.

    Attributes:
        force_index: Index of this torsion in the OpenMM torsion force object.
        atom_i: First atom index.
        atom_j: Second atom index.
        atom_k: Third atom index.
        atom_l: Fourth atom index.
        elements: Element symbols for the four atoms.
        periodicity: Fourier component periodicity (1, 2, or 3).
        env_id: Chemical environment identifier for parameter matching.
        ff_row: Row index in the source force field file, if applicable.
        is_improper: Whether this term is an improper torsion.

    """

    force_index: int
    atom_i: int
    atom_j: int
    atom_k: int
    atom_l: int
    elements: tuple[str, str, str, str]
    periodicity: int = 1
    env_id: str = ""
    ff_row: int | None = None
    is_improper: bool = False


@dataclass
class _UBTerm:
    """Internal record mapping a Urey-Bradley 1-3 pair to its OpenMM force index.

    Attributes:
        force_index: Index of this UB bond in the OpenMM bond force object.
        atom_i: First atom of the angle (outer).
        atom_k: Third atom of the angle (outer).
        elements: Element symbols for the three angle atoms (for matching).
        env_id: Chemical environment identifier for parameter matching.
        ff_row: Row index in the source force field file, if applicable.

    """

    force_index: int
    atom_i: int
    atom_k: int
    elements: tuple[str, str, str]
    env_id: str = ""
    ff_row: int | None = None


@dataclass
class _CmapTerm:
    """Internal record mapping a CMAP correction to its OpenMM force index.

    Attributes:
        torsion_index: Index of this CMAP torsion in the CMAPTorsionForce.
        map_index: Index of the CMAP grid in the CMAPTorsionForce.
        phi_atoms: Atom indices for the φ dihedral (4 atoms).
        psi_atoms: Atom indices for the ψ dihedral (4 atoms).
        phi_types: Atom types for the φ dihedral.
        psi_types: Atom types for the ψ dihedral.

    """

    torsion_index: int
    map_index: int
    phi_atoms: tuple[int, int, int, int]
    psi_atoms: tuple[int, int, int, int]
    phi_types: tuple[str, str, str, str]
    psi_types: tuple[str, str, str, str]
