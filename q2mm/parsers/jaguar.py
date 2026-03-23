"""Parsers for Schrödinger Jaguar input and output files.

Provides ``JaguarIn`` for reading Jaguar ``.in`` files (including
Hessian data) and ``JaguarOut`` for reading Jaguar ``.out`` files
(structures, eigenvalues, eigenvectors, and frequencies).
"""

from __future__ import annotations
import logging
import numpy as np
import os
import re
from string import digits
from q2mm import constants as co
from q2mm.parsers.base import File
from q2mm.parsers.structures import Atom, Structure

logger = logging.getLogger(__name__)


class JaguarIn(File):
    """Retrieve data from Jaguar ``.in`` files.

    The Hessian is not mass-weighted. Hessian units are assumed to be
    kJ/(mol·Å²).
    """

    def __init__(self, path):
        """Initialize a JaguarIn instance.

        Args:
            path (str): Path to the Jaguar ``.in`` file.
        """
        super().__init__(path)
        self._structures = None
        self._hessian = None
        self._empty_atoms = None
        self._lines = None

    def get_hessian(self, num_atoms: int):
        """Read the Hessian matrix from a Jaguar ``.in`` file.

        Automatically removes Hessian elements corresponding to dummy
        atoms.  That removal is currently disabled to minimize Schrödinger
        dependence because current use cases have no dummy or empty atoms,
        but it should be restored if dummy atoms are used in the future.

        Args:
            num_atoms (int): Number of atoms in the system.

        Returns:
            (numpy.ndarray): 2-D Hessian matrix of shape
                ``(num_atoms * 3, num_atoms * 3)`` after unit conversion.
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
            self._hessian = (
                hessian * co.HESSIAN_CONVERSION
            )  # TODO find a more universal way to manage units, JAGUAR IGNORED UNITS SETTINGS????!
            logger.log(1, f">>> hessian.shape: {hessian.shape}")
        return self._hessian

    def gen_lines(self):
        """Generate output lines for the Jaguar ``.in`` file.

        Since it would be difficult to reproduce all original data, the
        written version will be missing much of the data in the original.
        The Schrödinger API may provide a better mechanism for that.

        The intent is to include the ability to write out an atomic
        section with the ESP data that we would want.

        Returns:
            (list[str]): Generated lines for the ``.in`` file.
        """
        lines = []
        mae_name = None
        lines.append(f"MAEFILE: {mae_name}")
        lines.append("&gen")
        lines.append("&")
        lines.append("&zmat")
        # Just use the 1st structure. I don't imagine a Jaguar input file
        # ever containing more than one structure.
        struct = self.structures[0]
        lines.extend(struct.format_coords(format="gauss"))
        lines.append("&")
        return lines


class JaguarOut(File):
    """Retrieve data from Schrödinger Jaguar ``.out`` files.

    Eigenvalues and eigenvectors are **not** mass-weighted.
    """

    def __init__(self, path):
        """Initialize a JaguarOut instance.

        Args:
            path (str): Path to the Jaguar ``.out`` file.
        """
        super().__init__(path)
        self._structures = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._frequencies = None
        self._dummy_atom_eigenvector_indices = None
        # self._force_constants = None

    @property
    def structures(self):
        """list[Structure]: Parsed molecular structures from the output file."""
        if self._structures is None:
            self.import_file()
        return self._structures

    @property
    def eigenvalues(self):
        """numpy.ndarray: Eigenvalues derived from force constants and frequencies."""
        if self._eigenvalues is None:
            self.import_file()
        return self._eigenvalues

    @property
    def eigenvectors(self):
        """numpy.ndarray: Cartesian eigenvectors with dummy-atom rows removed."""
        if self._eigenvectors is None:
            self.import_file()
        return self._eigenvectors

    @property
    def frequencies(self):
        """numpy.ndarray: Vibrational frequencies in cm⁻¹."""
        if self._frequencies is None:
            self.import_file()
        return self._frequencies

    @property
    def dummy_atom_eigenvector_indices(self):
        """list[int]: Row indices in the eigenvector matrix that correspond to dummy atoms."""
        if self._dummy_atom_eigenvector_indices is None:
            self.import_file()
        return self._dummy_atom_eigenvector_indices

    def import_file(self):
        """Parse the Jaguar ``.out`` file and populate all cached properties.

        Reads structures, frequencies, force constants, and eigenvectors
        from the file. Dummy-atom contributions are removed from the
        eigenvectors.
        """
        logger.log(10, f"READING: {self.filename}")
        frequencies = []
        force_constants = []
        eigenvectors = []
        structures = []
        with open(self.path) as f:
            section_geometry = False
            section_eigenvalues = False
            section_eigenvectors = False
            for i, line in enumerate(f):
                if section_geometry:
                    cols = line.split()
                    if len(cols) == 0:
                        section_geometry = False
                        structures.append(current_structure)
                        continue
                    elif len(cols) == 1:
                        pass
                    else:
                        match = re.match(rf"\s+([\d\w]+)\s+({co.RE_FLOAT})\s+({co.RE_FLOAT})\s+({co.RE_FLOAT})", line)
                        if match is not None:
                            current_atom = Atom()
                            current_atom.element = match.group(1).translate(str.maketrans("", "", digits))
                            current_atom.x = float(match.group(2))
                            current_atom.y = float(match.group(3))
                            current_atom.z = float(match.group(4))
                            current_structure.atoms.append(current_atom)
                            logger.log(
                                0,
                                f"{current_atom.element:<3}{current_atom.x:>12.6f}{current_atom.y:>12.6f}"
                                f"{current_atom.z:>12.6f}",
                            )
                if "geometry:" in line:
                    section_geometry = True
                    current_structure = Structure(self.filename)
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
        for i, atom in enumerate(structures[-1].atoms):
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
        self._structures = structures
        self._eigenvalues = np.array(eigenvalues)
        self._eigenvectors = np.array(eigenvectors)
        self._frequencies = np.array(frequencies)
        # self._force_constants = np.array(force_constants)
        logger.log(5, f"  -- Read {len(self.structures)} structures")
        logger.log(5, f"  -- Read {len(self.frequencies)} frequencies.")
        logger.log(5, f"  -- Read {len(self.eigenvalues)} eigenvalues.")
        logger.log(5, f"  -- Read {self.eigenvectors.shape} eigenvectors.")
        # num_atoms = len(structures[-1].atoms)
        # logger.log(5,
        #            '  -- ({}, {}) eigenvectors expected for linear '
        #            'molecule.'.format(
        #         num_atoms * 3 - 5, num_atoms * 3))
        # logger.log(5, '  -- ({}, {}) eigenvectors expected for nonlinear '
        #            'molecule.'.format(
        #         num_atoms * 3 - 6, num_atoms * 3))
