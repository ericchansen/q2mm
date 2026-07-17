"""Parser for Gaussian log files.

Extracts structures, Hessians, eigenvectors, eigenvalues, frequencies,
and ESP data from Gaussian output files.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from q2mm import constants as co
from q2mm.constants import DEFAULT_BOND_TOLERANCE
from q2mm.models.hessian import HessianProvenance, HessianUnits, hessian_to_atomic_units
from q2mm.models.identifiers import _extract_element
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet

logger = logging.getLogger(__name__)


@dataclass
class _GaussianAtom:
    """One atom parsed from a Gaussian log file.

    Gaussian atoms carry either an atomic number (``"Standard orientation"``
    sections in the log body) or an explicit element symbol (the terse
    archive block) — never both, and never a separate force-field type
    label, so there is no Gaussian-specific type-vs-element ambiguity to
    resolve.
    """

    index: int | None = None
    atomic_num: int | None = None
    element: str | None = None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @property
    def coords(self) -> np.ndarray:
        """Cartesian coordinates of this atom, as ``[x, y, z]``."""
        return np.array([self.x, self.y, self.z])

    @property
    def symbol(self) -> str:
        """Resolve this atom's element symbol.

        Prefers an explicit ``element`` (archive block); falls back to a
        periodic-table lookup from ``atomic_num`` (log body).
        """
        if self.element:
            return _extract_element(self.element)
        if self.atomic_num is not None and self.atomic_num >= 1:
            return list(co.MASSES.keys())[self.atomic_num - 1]
        return ""


@dataclass
class _GaussianRecord:
    """One structure/conformer staging record parsed from a Gaussian log file."""

    origin_name: str
    source_path: str
    atoms: list[_GaussianAtom] = field(default_factory=list)
    props: dict[str, str] = field(default_factory=dict)
    hess: np.ndarray | None = None
    hessian_units: HessianUnits | None = None


def _molecule_from_gaussian_record(record: _GaussianRecord) -> Molecule:
    """Convert a :class:`_GaussianRecord` into a :class:`~q2mm.models.molecule.Molecule`.

    Bonds/angles/torsions are always inferred from geometry — Gaussian logs
    carry no connectivity data. Charge/multiplicity come from the archive's
    ``props`` (defaulting to neutral singlet); the Hessian (if present) is
    converted once from its recorded unit provenance to canonical
    Hartree/Bohr².
    """
    symbols = [atom.symbol for atom in record.atoms]
    coords = [atom.coords for atom in record.atoms]
    charge = int(record.props.get("charge", 0))
    multiplicity = int(record.props.get("multiplicity", 1))

    resolved_hessian = hessian_to_atomic_units(record.hess, record.hessian_units)
    resolved_provenance = None
    if resolved_hessian is not None:
        assert record.hessian_units is not None
        resolved_provenance = HessianProvenance(
            units=record.hessian_units,
            source="gaussian",
            path=record.source_path,
        )

    return Molecule(
        symbols=tuple(symbols),
        geometry=np.array(coords, dtype=float),
        charge=charge,
        multiplicity=multiplicity,
        name=record.origin_name,
        hessian=resolved_hessian,
        hessian_provenance=resolved_provenance,
    )


def _gaussian_archive_records(path: str, filename: str, *, au_hessian: bool) -> list[_GaussianRecord]:
    """Parse every complete Gaussian archive record in file order."""
    with open(path) as handle:
        archives = re.findall("(?s)(\\s1\\\\1\\\\.*?[\\\\\n\\s]+@)", handle.read())
    if not archives:
        raise IndexError(f"No Gaussian archive found in {filename}.")

    records: list[_GaussianRecord] = []
    for archive in archives:
        record = _GaussianRecord(filename, path)
        sections = archive.replace("\n ", "").split("\\\\")
        section_index = 0

        general = sections[section_index]
        section_index += 1
        general_match = re.search(
            "\\s1\\\\1\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\(?P<user>.*?)\\\\(?P<date>.*?)\\\\.*?",
            general,
        )
        if general_match is None:
            raise ValueError(f"Malformed Gaussian archive general section in {filename}.")
        record.props["user"] = general_match.group("user")
        record.props["date"] = general_match.group("date")

        section_index += 2  # route/commands and title/comment
        coordinate_match = re.search(
            "(?P<charge>.*?),(?P<multiplicity>.*?)\\\\(?P<atoms>.*)",
            sections[section_index],
        )
        section_index += 1
        if coordinate_match is None:
            raise ValueError(f"Malformed Gaussian archive coordinate section in {filename}.")
        record.props["charge"] = coordinate_match.group("charge")
        record.props["multiplicity"] = coordinate_match.group("multiplicity")
        atom_rows = coordinate_match.group("atoms").split("\\")
        for atom_row in atom_rows:
            columns = atom_row.split(",")
            if len(columns) == 4:
                element, x, y, z = columns
            elif len(columns) == 5:
                element, x, y, z = columns[0], columns[2], columns[3], columns[4]
            else:
                raise ValueError(f"Unsupported Gaussian archive coordinate row {atom_row!r} in {filename}.")
            record.atoms.append(_GaussianAtom(element=element, x=float(x), y=float(y), z=float(z)))

        for item in sections[section_index].split("\\"):
            property_name, property_value = item.split("=", maxsplit=1)
            record.props[property_name] = property_value
        section_index += 1

        if sections[section_index] != "@":
            triangle = sections[section_index].split(",")
            dimension = len(record.atoms) * 3
            expected = dimension * (dimension + 1) // 2
            if len(triangle) != expected:
                raise ValueError(
                    f"Gaussian archive Hessian in {filename} has {len(triangle)} elements; expected {expected}."
                )
            hessian = np.zeros((dimension, dimension), dtype=float)
            hessian[np.tril_indices_from(hessian)] = triangle
            hessian += np.tril(hessian, -1).T
            if not au_hessian:
                hessian *= co.HESSIAN_AU_TO_KJMOLA2
            record.hess = hessian
            record.hessian_units = HessianUnits.ATOMIC if au_hessian else HessianUnits.KJ_MOL_ANGSTROM2
        records.append(record)
    return records


class GaussLog:
    """Retrieves data from Gaussian log files.

    If you are extracting frequencies/Hessian data from this file, use
    the keyword NoSymmetry when running the Gaussian calculation.
    """

    __slots__ = [
        "_lines",
        "path",
        "directory",
        "filename",
        "_evals",
        "_evecs",
        "_frequencies_cm",
        "_records",
        "_esp_rms",
        "_au_hessian",
    ]

    def __init__(self, path: str, au_hessian: bool = False) -> None:
        """Instantiate a file object for the file at the location path passed.

        Populates the directory and filename properties as well.

        Args:
            path (str): Location of the Gaussian log file.
            au_hessian (bool, optional): If True, the Hessian will not be
                converted to kJ/(mol·Å²) but rather left in atomic units
                (Hartree/Bohr²). Defaults to False.

        """
        self._lines = None
        self.path = os.path.abspath(path)
        self.directory = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)
        self._evals = None
        self._evecs = None
        self._frequencies_cm = None
        self._records = None
        self._esp_rms = None
        self._au_hessian = au_hessian

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
    def evecs(self) -> np.ndarray | None:
        """Returns eigenvectors of frequency analysis if applicable.

        If not yet parsed, parses them from the log body, not the archive.

        Returns:
            (np.ndarray | None): Mass-weighted, normalized eigenvectors of the
                Gaussian frequency analysis, or None if not a frequency job.

        """
        if self._evecs is None:
            self.read_out()
        return self._evecs

    @property
    def evals(self) -> np.ndarray | None:
        """Returns eigenvalues of frequency analysis if applicable.

        If not yet parsed, parses them from the log body, not the archive.

        These are mass-weighted force constants in atomic units
        (sign × force_constant / (reduced_mass × AU_TO_MDYNA)), NOT vibrational
        frequencies in cm⁻¹.  Use :attr:`frequencies` for cm⁻¹ values.

        Returns:
            (np.ndarray | None): Mass-weighted force constants in atomic units,
                or None if not a frequency job.

        """
        if self._evals is None:
            self.read_out()
        return self._evals

    @property
    def frequencies(self) -> np.ndarray | None:
        """Vibrational frequencies in cm⁻¹ from the Gaussian frequency analysis.

        Negative values indicate imaginary modes (transition states).

        Returns:
            (np.ndarray | None): Frequencies in cm⁻¹, or None if not a frequency job.

        """
        if self._frequencies_cm is None:
            self.read_out()
        return self._frequencies_cm

    @property
    def _records_parsed(self) -> list[_GaussianRecord]:
        """Records parsed from the Gaussian log file archive (private staging).

        Not part of the public API — see :attr:`molecules`.
        """
        if self._records is None:
            self.read_archive()
        return self._records

    @property
    def molecules(self) -> list[Molecule]:
        """Parsed structures as :class:`~q2mm.models.molecule.Molecule` objects.

        Each record is converted via :func:`_molecule_from_gaussian_record`,
        preserving any Hessian data found in the archive and normalizing it
        to Hartree/Bohr² from the parser-recorded unit provenance.

        """
        return [_molecule_from_gaussian_record(record) for record in self._records_parsed]

    def read_out(self) -> None:
        """Read force constant and eigenvector data from a frequency calculation.

        Populates ``_evals``, ``_evecs``, ``_frequencies_cm``, ``_records``,
        and ``_esp_rms`` from the Gaussian log file body.
        """
        logger.log(5, f"READING: {self.filename}")
        self._evals = []
        self._evecs = []
        self._frequencies_cm = []
        self._records = []
        force_constants = []
        evecs = []
        with open(self.path) as f:
            # The keyword "harmonic" shows up before the section we're
            # interested in. It can show up multiple times depending on the
            # options in the Gaussian .com file.
            past_first_harm = False
            # High precision mode, turned on by including "freq=hpmodes" in the
            # Gaussian .com file.
            hpmodes = False
            file_iterator = iter(f)
            # This while loop breaks when the end of the file is reached, or
            # if the high quality modes have been read already.
            while True:
                try:
                    line = next(file_iterator)
                except StopIteration:
                    break
                if "Charges from ESP fit" in line:
                    pattern = re.compile(rf"RMS=\s+({co.RE_FLOAT})")
                    match = pattern.search(line)
                    self._esp_rms = float(match.group(1))
                # Gathering some geometric information.
                elif "Standard orientation:" in line:
                    self._records.append(_GaussianRecord(self.filename, self.path))
                    next(file_iterator)
                    next(file_iterator)
                    next(file_iterator)
                    next(file_iterator)
                    line = next(file_iterator)
                    while "---" not in line:
                        cols = line.split()
                        self._records[-1].atoms.append(
                            _GaussianAtom(
                                index=int(cols[0]),
                                atomic_num=int(cols[1]),
                                x=float(cols[3]),
                                y=float(cols[4]),
                                z=float(cols[5]),
                            )
                        )
                        line = next(file_iterator)
                    logger.log(
                        5,
                        f"  -- Found {len(self._records[-1].atoms)} atoms.",
                    )
                elif "Harmonic" in line:
                    # The high quality eigenvectors come before the low quality
                    # ones. If you see "Harmonic" again, it means you're at the
                    # low quality ones now, so break.
                    if past_first_harm:
                        break
                    else:
                        past_first_harm = True
                elif "Frequencies" in line:
                    # We're going to keep reusing these.
                    # We accumulate sets of eigevectors and eigenvalues, add
                    # them to self._evecs and self._evals, and then reuse this
                    # for the next set.
                    del force_constants[:]
                    del evecs[:]
                    # Values inside line look like:
                    #     "Frequencies --- xxxx.xxxx xxxx.xxxx"
                    # That's why we remove the 1st two columns. This is
                    # consistent with and without "hpmodes".
                    # For "hpmodes" option, there are 5 of these frequencies.
                    # Without "hpmodes", there are 3.
                    # Thus the eigenvectors and eigenvalues will come in sets of
                    # either 5 or 3.
                    cols = line.split()
                    for frequency in map(float, cols[2:]):
                        # Store actual frequency in cm⁻¹
                        self._frequencies_cm.append(frequency)
                        # Has 1. or -1. depending on the sign of the frequency.
                        if frequency < 0.0:
                            force_constants.append(-1.0)
                        else:
                            force_constants.append(1.0)
                        # For now this is empty, but we will add to it soon.
                        evecs.append([])

                    # Moving on to the reduced masses.
                    line = next(file_iterator)
                    cols = line.split()
                    # Again, trim the "Reduced masses ---".
                    # It's "Red. masses --" for without "hpmodes".
                    for i, mass in enumerate(map(float, cols[3:])):
                        # +/- 1 / reduced mass
                        force_constants[i] = force_constants[i] / mass

                    # Now we are on the line with the force constants.
                    line = next(file_iterator)
                    cols = line.split()
                    # Trim "Force constants ---". It's "Frc consts --" without
                    # "hpmodes".
                    for i, force_constant in enumerate(map(float, cols[3:])):
                        # AU_TO_MDYNA conversion (15.569141)
                        force_constants[i] *= force_constant / co.AU_TO_MDYNA

                    # Force constants were calculated above as follows:
                    #    a = +/- 1 depending on the sign of the frequency
                    #    b = a / reduced mass (obtained from the Gaussian log)
                    #    c = b * force constant / conversion factor (force
                    #         (constant obtained from Gaussian log) (conversion
                    #         factor is inside constants module)

                    # Skip the IR intensities.
                    next(file_iterator)
                    # This is different depending on whether you use "hpmodes".
                    line = next(file_iterator)
                    # "Coord" seems to only appear when the "hpmodes" is used.
                    if "Coord" in line:
                        hpmodes = True
                    # This is different depending on whether you use
                    # "freq=projected".
                    line = next(file_iterator)
                    # The "projected" keyword seems to add "IRC Coupling".
                    if "IRC Coupling" in line:
                        line = next(file_iterator)
                    # We're on to the eigenvectors.
                    # Until the end of this section containing the eigenvectors,
                    # the number of columns remains constant. When that changes,
                    # we know we're to the next set of frequencies, force
                    # constants and eigenvectors.
                    # Actually check that we've moved on, sometimes a "Depolar" entry is
                    if "Depolar" in line:
                        line = next(file_iterator)
                    if "Atom" in line:
                        line = next(file_iterator)
                    cols = line.split()
                    cols_len = len(cols)

                    while len(cols) == cols_len:
                        # This will come after all the eigenvectors have been
                        # read. We can break out then.
                        if "Harmonic" in line:
                            break
                        # If "hpmodes" is used, you have an extra column here
                        # that is simply an index.
                        if hpmodes:
                            cols = cols[1:]
                        # cols corresponds to line(s) (maybe only 1st line)
                        # under section "Coord Atom Element:" (at least for
                        # "hpmodes").

                        # Just the square root of the mass from co.MASSES.
                        # co.MASSES currently has the average mass.
                        # Gaussian may use the mass of the most abundant
                        # isotope. This may be a problem.
                        mass_sqrt = np.sqrt(list(co.MASSES.items())[int(cols[1]) - 1][1])

                        cols = cols[2:]
                        # This corresponds to the same line still, but without
                        # the atom elements.

                        # This loop expands the LoL, evecs, as so.
                        # Iteration 1: each sub-list has 1 element
                        # Iteration 2: each sub-list has 2 elements
                        # ... etc. until the length of the sublist is equal to
                        # the number of atoms. Remember, for low precision
                        # eigenvectors it only adds in sets of 3, not 5.

                        # Elements of evecs are simply the data under
                        # "Coord Atom Element" multiplied by the square root
                        # of the weight.
                        for i in range(len(evecs)):
                            if hpmodes:
                                # evecs is a LoL. Length of sublist is
                                # equal to # of columns in section "Coord Atom
                                # Element" minus 3, for the 1st 3 columns
                                # (index, atom index, atomic number).
                                evecs[i].append(float(cols[i]) * mass_sqrt)
                            else:
                                # This is fow low precision eigenvectors. It's a
                                # funny way to go in sets of 3. Take a look at
                                # your low precision Gaussian log and it will
                                # make more sense.
                                for _ in range(3):
                                    x = float(cols.pop(0))
                                    evecs[i].append(x * mass_sqrt)
                        line = next(file_iterator)
                        cols = line.split()

                    # Here the overall number of eigenvalues and eigenvectors is
                    # increased by 5 (high precision) or 3 (low precision). The
                    # total number goes to 3N - 6 for non-linear and 3N - 5 for
                    # linear. Same goes for self._evecs.
                    for i in range(len(evecs)):
                        self._evals.append(force_constants[i])
                        self._evecs.append(evecs[i])
                    # We know we're done if this is in the line.
                    if "Harmonic" in line:
                        break
        if self._evals and self._evecs:
            for evec in self._evecs:
                # Each evec is a single eigenvector.
                # Add up the sum of squares over an eigenvector.
                sum_of_squares = 0.0
                # Appropriately named, element is an element of that single
                # eigenvector.
                for element in evec:
                    sum_of_squares += element * element
                # Now x is the inverse of the square root of the sum of squares
                # for an individual eigenvector.
                element = 1 / np.sqrt(sum_of_squares)
                for i in range(len(evec)):
                    evec[i] *= element
            self._evals = np.array(self._evals)
            self._evecs = np.array(self._evecs)
            self._frequencies_cm = np.array(self._frequencies_cm)
            logger.log(1, f">>> self._evals: {self._evals}")
            logger.log(1, f">>> self._evecs: {self._evecs}")
            logger.log(5, f"  -- {len(self._records)} structures found.")

    # May want to move some attributes assigned to the structure class onto
    # this filetype class.
    def read_archive(self) -> None:
        """Read the last complete archive section, preserving legacy behavior."""
        logger.log(5, f"READING: {self.filename}")
        try:
            self._records = [_gaussian_archive_records(self.path, self.filename, au_hessian=self._au_hessian)[-1]]
        except IndexError:
            logger.warning("  -- Couldn't locate archive.")
            raise
        logger.log(5, "  -- Located last archive.")

    def read_archives(self) -> None:
        """Read every complete archive section in file order."""
        logger.log(5, f"READING ALL ARCHIVES: {self.filename}")
        try:
            self._records = _gaussian_archive_records(self.path, self.filename, au_hessian=self._au_hessian)
        except IndexError:
            logger.warning("  -- Couldn't locate an archive.")
            raise
        logger.log(5, "  -- Located %d archive(s).", len(self._records))


def load_gaussian_reference(
    path: str | Path,
    *,
    weights: dict[str, float] | None = None,
    bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
    charge: int | None = None,
    multiplicity: int | None = None,
    include_frequencies: bool = False,
    skip_imaginary: bool = False,
    au_hessian: bool = True,
    include_eigenmatrix: bool = True,
) -> tuple[ObservationSet, Molecule]:
    """Build reference data from a Gaussian log file.

    Parses the log file for the optimised geometry and vibrational
    frequencies, then auto-populates bond lengths, angles, and (when a
    Hessian is available) eigenmatrix data via
    :meth:`~q2mm.models.observations.ObservationSet.from_molecule`.

    By default, **frequencies are not included** — eigenmatrix
    training from the Hessian is the standard Q2MM approach per
    Norrby & Liljefors (1998). Set ``include_frequencies=True``
    to add them.

    Args:
        path (str | Path): Path to the Gaussian ``.log`` file
            (from an ``opt freq`` job).
        weights (dict[str, float] | None): Weight overrides (same
            keys as :meth:`~q2mm.models.observations.ObservationSet.from_molecule`).
        bond_tolerance (float): Multiplier for covalent-radii bond
            detection. Use 1.4+ for TS.
        charge (int | None): Molecular charge override. ``None`` preserves
            the value parsed from the Gaussian archive.
        multiplicity (int | None): Spin multiplicity override. ``None``
            preserves the value parsed from the Gaussian archive.
        include_frequencies (bool): Whether to add frequency data
            from the log file. Default is ``False``.
        skip_imaginary (bool): If ``True``, negative frequencies are
            skipped.
        au_hessian (bool): Keep Hessian in atomic units
            (Hartree/Bohr²).
        include_eigenmatrix (bool): If ``True`` (the default) and a
            Hessian is available, add eigenmatrix training data.

    Returns:
        tuple[ObservationSet, Molecule]: Populated reference data
            and the parsed molecule (with Hessian attached if
            available).

    """
    log = GaussLog(str(path), au_hessian=au_hessian)

    # Build molecule from the last (optimised) structure
    mol = log.molecules[-1]
    mol = mol.with_overrides(charge=charge, multiplicity=multiplicity, bond_tolerance=bond_tolerance)

    # ``mol.hessian`` carries the archive Cartesian Hessian (Hartree/Bohr²,
    # full rank 3N, imaginary mode intact) in a frame consistent with the
    # geometry.  Do NOT override it with a reconstruction from
    # ``log.evals``/``log.evecs`` — those come from Gaussian's mass-weighted
    # frequency analysis and would reintroduce a ~√(mᵢmⱼ) error into every
    # heavy-atom force constant.

    # Frequencies in cm⁻¹ from the Gaussian log
    # Note: log.evals are eigenvalues (mass-weighted force constants in
    # atomic units), NOT frequencies.  Use log.frequencies for cm⁻¹ values.
    frequencies = None
    if include_frequencies and log.frequencies is not None and len(log.frequencies):
        frequencies = np.array(log.frequencies)

    ref = ObservationSet.from_molecule(
        mol,
        weights=weights,
        frequencies=frequencies,
        skip_imaginary=skip_imaginary,
        include_eigenmatrix=include_eigenmatrix,
    )

    return ref, mol
