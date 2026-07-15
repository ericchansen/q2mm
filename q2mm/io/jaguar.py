"""Parsers for Schrödinger Jaguar input and output files.

Provides ``JaguarIn`` for reading Jaguar ``.in`` files (including
Hessian data) and ``JaguarOut`` for reading Jaguar ``.out`` files
(structures, eigenvalues, eigenvectors, and frequencies).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from importlib.util import find_spec
from string import digits
from typing import Any

import numpy as np

from q2mm import constants as co
from q2mm.models.hessian import HessianProvenance, HessianUnits
from q2mm.models.identifiers import _extract_element
from q2mm.models.molecule import Molecule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Pint integration — cold-path unit tagging (once per file load)
# ---------------------------------------------------------------------------
# When pint is installed (``pip install q2mm[qm]``) and the caller passes
# ``tag_units=True``, ``JaguarIn.get_hessian`` returns a ``pint.Quantity``
# tagged as ``hartree/bohr**2``.  Callers that pass the result to
# ``Molecule.with_hessian`` get automatic unit validation: an
# incompatible tag (e.g. ``kJ/(mol·Å²)``) raises
# ``pint.errors.DimensionalityError`` instead of silently producing wrong
# force constants.  The default (``tag_units=False``) always returns a bare
# ``np.ndarray``.  See ``docs/how-it-works/architecture.md`` §"Unit type
# system: NewType vs Pint" for details.
_HAS_PINT: bool = find_spec("pint") is not None
_pint_ureg: Any = None  # lazy — created on first get_hessian() call


def _get_pint_ureg() -> Any:  # returns pint.UnitRegistry or None
    """Return the module-level pint UnitRegistry, creating it on first call."""
    global _pint_ureg
    if _HAS_PINT and _pint_ureg is None:
        import pint  # noqa: PLC0415

        _pint_ureg = pint.UnitRegistry()
    return _pint_ureg


@dataclass
class _JaguarAtom:
    """One atom parsed from a Jaguar ``.out`` geometry block.

    Jaguar labels carry the element with trailing digits (e.g. ``"Rh1"``);
    it has no separate force-field type label and no atomic number, so
    there is nothing else to resolve an element from.
    """

    element: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @property
    def coords(self) -> np.ndarray:
        """Cartesian coordinates of this atom, as ``[x, y, z]``."""
        return np.array([self.x, self.y, self.z])

    @property
    def symbol(self) -> str:
        """Resolve this atom's element symbol."""
        return _extract_element(self.element) if self.element else ""

    @property
    def is_dummy(self) -> bool:
        """Whether this is a dummy atom (Jaguar's ``"X"`` placeholder)."""
        return self.symbol == "X"


@dataclass
class _JaguarRecord:
    """One structure/conformer staging record parsed from a Jaguar ``.out`` file."""

    origin_name: str
    atoms: list[_JaguarAtom] = field(default_factory=list)


def _molecule_from_jaguar_record(record: _JaguarRecord) -> Molecule:
    """Convert a :class:`_JaguarRecord` into a :class:`~q2mm.models.molecule.Molecule`.

    Bonds/angles/torsions are always inferred from geometry — Jaguar
    ``.out`` files carry no connectivity data. Charge, multiplicity, and
    Hessian are not available from this file format (the Hessian, when
    needed, comes from a separate :class:`JaguarIn` file via
    :meth:`JaguarIn.get_hessian` and is attached by the caller).
    """
    symbols = [atom.symbol for atom in record.atoms]
    coords = [atom.coords for atom in record.atoms]
    return Molecule(
        symbols=tuple(symbols),
        geometry=np.array(coords, dtype=float),
        name=record.origin_name,
    )


class JaguarIn:
    """Retrieve data from Jaguar ``.in`` files.

    The Hessian is **not** mass-weighted and is returned in atomic units
    (Hartree/Bohr²), matching the convention used by Gaussian .fchk and
    Psi4 outputs.  Jaguar stores the Hessian in the ``&hess`` section of
    its ``.in`` file in Hartree/Bohr² (confirmed empirically: raw diagonal
    elements give frequencies that match the Jaguar ``.out`` exactly when
    treated as atomic units).
    """

    def __init__(self, path: str) -> None:
        """Initialize a JaguarIn instance.

        Args:
            path (str): Path to the Jaguar ``.in`` file.

        """
        self._lines = None
        self.path = os.path.abspath(path)
        self.directory = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)
        self._hessian = None
        self._empty_atoms = None

    @property
    def lines(self) -> list[str]:
        """Return the lines of the file.

        Returns:
            (list[str]): Lines of the file.

        """
        if self._lines is None:
            with open(self.path) as f:
                self._lines = f.readlines()
        return self._lines

    def write(self, path: str, lines: list[str] | None = None) -> None:
        """Write lines to file at path.

        Args:
            path (str): Location of file to write.
            lines (list[str] | None): Lines to write. Defaults to
                ``self.lines``.

        """
        if lines is None:
            lines = self.lines
        with open(path, "w") as f:
            for line in lines:
                f.write(line)

    def get_hessian(self, num_atoms: int, *, tag_units: bool = False) -> np.ndarray:
        """Read the Hessian matrix from a Jaguar ``.in`` file.

        Automatically removes Hessian elements corresponding to dummy
        atoms.  That removal is currently disabled to minimize Schrödinger
        dependence because current use cases have no dummy or empty atoms,
        but it should be restored if dummy atoms are used in the future.

        Args:
            num_atoms (int): Number of atoms in the system.
            tag_units: When *True* and ``pint`` is installed, wrap the
                returned array in ``pint.Quantity(array, 'hartree/bohr**2')``
                so callers can validate units with ``.to(...)``.
                Default is *False* (always returns a bare ``ndarray``).

        Returns:
            (numpy.ndarray): 2-D Hessian matrix of shape
                ``(num_atoms * 3, num_atoms * 3)`` in Hartree/Bohr²
                (atomic units, no additional conversion applied).

        """
        if self._hessian is None:
            num = num_atoms

            assert num != 0, f"Zero atoms found when loading Hessian from {self.path}!"
            hessian = np.zeros([num * 3, num * 3], dtype=float)
            logger.log(5, f"  -- Created {hessian.shape} Hessian matrix (including dummy atoms).")
            with open(self.path) as f:
                section_hess = False
                for line in f:
                    if section_hess and line.startswith("&"):
                        section_hess = False
                        hessian += np.tril(hessian, -1).T
                    if section_hess:
                        cols = line.split()
                        if len(cols) == 1:
                            hess_col = int(cols[0])
                        elif len(cols) > 1:
                            hess_row = int(cols[0])
                            for i, hess_ele in enumerate(cols[1:]):
                                hessian[hess_row - 1, i + hess_col - 1] = float(hess_ele)
                    if "&hess" in line:
                        section_hess = True

            logger.log(1, f">>> hessian:\n{hessian}")
            logger.log(5, f"  -- Created {hessian.shape} Hessian matrix (w/o dummy atoms).")
            # Jaguar stores the Hessian in atomic units (Hartree/Bohr²).
            # We return it as-is to match the convention of other QM parsers
            # (Gaussian .fchk, Psi4).  Downstream code (Seminario, frequency
            # computation) expects AU when au_hessian=True / au_units=True.
            self._hessian = hessian
            logger.log(1, f">>> hessian.shape: {hessian.shape}")
        # Cold-path unit tagging: wrap in pint.Quantity when requested.
        if tag_units:
            ureg = _get_pint_ureg()
            if ureg is not None:
                return ureg.Quantity(self._hessian, "hartree/bohr**2")
        return self._hessian

    def attach_hessian(self, molecule: Molecule) -> Molecule:
        """Return *molecule* with this Jaguar input's Hessian and provenance."""
        return molecule.with_hessian(
            self.get_hessian(molecule.n_atoms),
            HessianProvenance(
                units=HessianUnits.ATOMIC,
                source="jaguar",
                path=self.path,
            ),
        )


class JaguarOut:
    """Retrieve data from Schrödinger Jaguar ``.out`` files.

    Eigenvalues and eigenvectors are **not** mass-weighted.
    """

    def __init__(self, path: str) -> None:
        """Initialize a JaguarOut instance.

        Args:
            path (str): Path to the Jaguar ``.out`` file.

        """
        self._lines = None
        self.path = os.path.abspath(path)
        self.directory = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)
        self._records = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._frequencies = None
        self._dummy_atom_eigenvector_indices = None

    @property
    def lines(self) -> list[str]:
        """Return the lines of the file.

        Returns:
            (list[str]): Lines of the file.

        """
        if self._lines is None:
            with open(self.path) as f:
                self._lines = f.readlines()
        return self._lines

    def write(self, path: str, lines: list[str] | None = None) -> None:
        """Write lines to file at path.

        Args:
            path (str): Location of file to write.
            lines (list[str] | None): Lines to write. Defaults to
                ``self.lines``.

        """
        if lines is None:
            lines = self.lines
        with open(path, "w") as f:
            for line in lines:
                f.write(line)

    @property
    def _records_parsed(self) -> list[_JaguarRecord]:
        """Records parsed from the output file (private staging).

        Not part of the public API — see :attr:`molecules`.
        """
        if self._records is None:
            self.import_file()
        return self._records

    @property
    def molecules(self) -> list[Molecule]:
        """Parsed structures as :class:`~q2mm.models.molecule.Molecule` objects."""
        return [_molecule_from_jaguar_record(record) for record in self._records_parsed]

    @property
    def eigenvalues(self) -> np.ndarray:
        """numpy.ndarray: Eigenvalues derived from force constants and frequencies."""
        if self._eigenvalues is None:
            self.import_file()
        return self._eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        """numpy.ndarray: Cartesian eigenvectors with dummy-atom rows removed."""
        if self._eigenvectors is None:
            self.import_file()
        return self._eigenvectors

    @property
    def frequencies(self) -> np.ndarray:
        """numpy.ndarray: Vibrational frequencies in cm⁻¹."""
        if self._frequencies is None:
            self.import_file()
        return self._frequencies

    @property
    def dummy_atom_eigenvector_indices(self) -> list[int]:
        """list[int]: Row indices in the eigenvector matrix that correspond to dummy atoms."""
        if self._dummy_atom_eigenvector_indices is None:
            self.import_file()
        return self._dummy_atom_eigenvector_indices

    def import_file(self) -> None:
        """Parse the Jaguar ``.out`` file and populate all cached properties.

        Reads structures, frequencies, force constants, and eigenvectors
        from the file. Dummy-atom contributions are removed from the
        eigenvectors.
        """
        logger.log(10, f"READING: {self.filename}")
        frequencies = []
        force_constants = []
        eigenvectors = []
        records: list[_JaguarRecord] = []
        with open(self.path) as f:
            section_geometry = False
            section_eigenvalues = False
            section_eigenvectors = False
            current_record: _JaguarRecord | None = None
            temp_eigenvectors = []
            for i, line in enumerate(f):
                if section_geometry:
                    cols = line.split()
                    if len(cols) == 0:
                        section_geometry = False
                        records.append(current_record)
                        continue
                    elif len(cols) == 1:
                        pass
                    else:
                        match = re.match(rf"\s+([\d\w]+)\s+({co.RE_FLOAT})\s+({co.RE_FLOAT})\s+({co.RE_FLOAT})", line)
                        if match is not None:
                            current_atom = _JaguarAtom()
                            current_atom.element = match.group(1).translate(str.maketrans("", "", digits))
                            current_atom.x = float(match.group(2))
                            current_atom.y = float(match.group(3))
                            current_atom.z = float(match.group(4))
                            current_record.atoms.append(current_atom)
                            logger.log(
                                0,
                                f"{current_atom.element:<3}{current_atom.x:>12.6f}{current_atom.y:>12.6f}"
                                f"{current_atom.z:>12.6f}",
                            )
                if "geometry:" in line:
                    section_geometry = True
                    current_record = _JaguarRecord(self.filename)
                    logger.log(5, f"[L{i + 1}] Located geometry.")
                if (
                    "Number of imaginary frequencies" in line
                    or "Writing vibrational" in line
                    or "Thermochemical properties at" in line
                ):
                    section_eigenvalues = False
                if section_eigenvectors is True:
                    cols = line.split()
                    if len(cols) == 0:
                        section_eigenvectors = False
                        eigenvectors.extend(temp_eigenvectors)
                        continue
                    else:
                        for i, x in enumerate(cols[2:]):
                            if not len(temp_eigenvectors) > i:
                                temp_eigenvectors.append([])
                            temp_eigenvectors[i].append(float(x))
                if section_eigenvalues is True and section_eigenvectors is False:
                    if "frequencies" in line:
                        cols = line.split()
                        frequencies.extend(map(float, cols[1:]))
                    if "force const" in line:
                        cols = line.split()
                        force_constants.extend(map(float, cols[2:]))
                        section_eigenvectors = True
                        temp_eigenvectors = [[]]
                if "normal modes in" in line:
                    section_eigenvalues = True
        logger.log(1, f">>> len(frequencies): {len(frequencies)}")
        logger.log(1, f">>> frequencies:\n{frequencies}")
        # logger.log(1, '>>> frequencies:\n{}'.format(
        #         [x / co.FORCE_CONVERSION for x in frequencies]))
        # logger.log(1, '>>> frequencies:\n{}'.format(
        #         [x * 4.55633e-6 for x in frequencies]))
        # logger.log(1, '>>> frequencies:\n{}'.format(
        #         [x * 1.23981e-4 for x in frequencies]))
        # logger.log(1, '>>> frequencies:\n{}'.format(
        #         [x / 219474.6305 for x in frequencies]))
        eigenvalues = [
            -fc / co.FORCE_CONVERSION if f < 0 else fc / co.FORCE_CONVERSION
            for fc, f in zip(force_constants, frequencies)
        ]
        logger.log(1, f">>> eigenvalues:\n{eigenvalues}")
        # Remove eigenvector components related to dummy atoms.
        # Find the index of the atoms that are dummies.
        dummy_atom_indices = []
        for i, atom in enumerate(records[-1].atoms):
            if atom.is_dummy:
                dummy_atom_indices.append(i)
        logger.log(10, f"  -- Located {len(dummy_atom_indices)} dummy atoms.")
        # Correlate those indices to the rows in the cartesian eigenvector.
        dummy_atom_eigenvector_indices = []
        for dummy_atom_index in dummy_atom_indices:
            start = dummy_atom_index * 3
            dummy_atom_eigenvector_indices.append(start)
            dummy_atom_eigenvector_indices.append(start + 1)
            dummy_atom_eigenvector_indices.append(start + 2)
        new_eigenvectors = []
        # Create new eigenvectors without the rows corresponding to the
        # dummy atoms.
        for eigenvector in eigenvectors:
            new_eigenvectors.append([])
            for i, eigenvector_row in enumerate(eigenvector):
                if i not in dummy_atom_eigenvector_indices:
                    new_eigenvectors[-1].append(eigenvector_row)
        # Replace old eigenvectors with new where dummy atoms aren't included.
        eigenvectors = np.array(new_eigenvectors)
        self._dummy_atom_eigenvector_indices = dummy_atom_eigenvector_indices
        self._records = records
        self._eigenvalues = np.array(eigenvalues)
        self._eigenvectors = np.array(eigenvectors)
        self._frequencies = np.array(frequencies)
        logger.log(5, f"  -- Read {len(self._records)} structures")
        logger.log(5, f"  -- Read {len(self.frequencies)} frequencies.")
        logger.log(5, f"  -- Read {len(self.eigenvalues)} eigenvalues.")
        logger.log(5, f"  -- Read {self.eigenvectors.shape} eigenvectors.")
