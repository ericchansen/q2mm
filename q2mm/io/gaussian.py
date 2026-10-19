"""Parser for Gaussian log files.

Extracts structures, Hessians, eigenvectors, eigenvalues, frequencies,
and ESP data from Gaussian output files.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from q2mm import constants as co
from q2mm.constants import DEFAULT_BOND_TOLERANCE
from q2mm.io._helpers import _TextFileBase
from q2mm.models.hessian import HessianProvenance, HessianUnits, hessian_to_atomic_units
from q2mm.models.identifiers import _extract_element
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet

logger = logging.getLogger(__name__)
_MASS_SYMBOLS: tuple[str, ...] = tuple(co.MASSES.keys())
_MASS_VALUES: tuple[float, ...] = tuple(co.MASSES.values())


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
            return _MASS_SYMBOLS[self.atomic_num - 1]
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


class GaussLog(_TextFileBase):
    """Retrieves data from Gaussian log files.

    If you are extracting frequencies/Hessian data from this file, use
    the keyword NoSymmetry when running the Gaussian calculation.
    """

    __slots__ = (
        "_body_parsed",
        "_evals",
        "_evecs",
        "_frequencies_cm",
        "_records",
        "_esp_rms",
        "_au_hessian",
    )

    def __init__(self, path: str, au_hessian: bool = False) -> None:
        """Instantiate a file object for the file at the location path passed.

        Populates the directory and filename properties as well.

        Args:
            path (str): Location of the Gaussian log file.
            au_hessian (bool, optional): If True, the Hessian will not be
                converted to kJ/(mol·Å²) but rather left in atomic units
                (Hartree/Bohr²). Defaults to False.

        """
        super().__init__(path)
        self._body_parsed = False
        self._evals: np.ndarray | None = None
        self._evecs: np.ndarray | None = None
        self._frequencies_cm: np.ndarray | None = None
        self._records: list[_GaussianRecord] | None = None
        self._esp_rms: float | None = None
        self._au_hessian = au_hessian

    @property
    def evecs(self) -> np.ndarray | None:
        """Returns eigenvectors of frequency analysis if applicable.

        If not yet parsed, parses them from the log body, not the archive.

        Returns:
            (np.ndarray | None): Mass-weighted, normalized eigenvectors of the
                Gaussian frequency analysis, or None if not a frequency job.

        """
        if not self._body_parsed:
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
        if not self._body_parsed:
            self.read_out()
        return self._evals

    @property
    def frequencies(self) -> np.ndarray | None:
        """Vibrational frequencies in cm⁻¹ from the Gaussian frequency analysis.

        Negative values indicate imaginary modes (transition states).

        Returns:
            (np.ndarray | None): Frequencies in cm⁻¹, or None if not a frequency job.

        """
        if not self._body_parsed:
            self.read_out()
        return self._frequencies_cm

    @property
    def _records_parsed(self) -> list[_GaussianRecord]:
        """Records parsed from the Gaussian log file archive (private staging).

        Not part of the public API — see :attr:`molecules`.
        """
        if self._records is None:
            self.read_archive()
        assert self._records is not None
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
        self._evals = None
        self._evecs = None
        self._frequencies_cm = None
        evals_list: list[float] = []
        evecs_list: list[list[float]] = []
        frequencies_cm_list: list[float] = []
        records: list[_GaussianRecord] = []
        force_constants: list[float] = []
        evecs: list[list[float]] = []
        with open(self.path) as f:
            past_first_harm = False
            hpmodes = False
            file_iterator = iter(f)
            while True:
                try:
                    line = next(file_iterator)
                except StopIteration:
                    break
                if "Charges from ESP fit" in line:
                    pattern = re.compile(rf"RMS=\s+({co.RE_FLOAT})")
                    match = pattern.search(line)
                    if match is not None:
                        self._esp_rms = float(match.group(1))
                elif "Standard orientation:" in line:
                    records.append(_GaussianRecord(self.filename, self.path))
                    next(file_iterator)
                    next(file_iterator)
                    next(file_iterator)
                    next(file_iterator)
                    line = next(file_iterator)
                    while "---" not in line:
                        cols = line.split()
                        records[-1].atoms.append(
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
                        f"  -- Found {len(records[-1].atoms)} atoms.",
                    )
                elif "Harmonic" in line:
                    if past_first_harm:
                        break
                    past_first_harm = True
                elif "Frequencies" in line:
                    del force_constants[:]
                    del evecs[:]
                    cols = line.split()
                    for frequency in map(float, cols[2:]):
                        frequencies_cm_list.append(frequency)
                        force_constants.append(-1.0 if frequency < 0.0 else 1.0)
                        evecs.append([])

                    line = next(file_iterator)
                    cols = line.split()
                    for i, mass in enumerate(map(float, cols[3:])):
                        force_constants[i] /= mass

                    line = next(file_iterator)
                    cols = line.split()
                    for i, force_constant in enumerate(map(float, cols[3:])):
                        force_constants[i] *= force_constant / co.AU_TO_MDYNA

                    next(file_iterator)
                    line = next(file_iterator)
                    if "Coord" in line:
                        hpmodes = True
                    line = next(file_iterator)
                    if "IRC Coupling" in line:
                        line = next(file_iterator)
                    if "Depolar" in line:
                        line = next(file_iterator)
                    if "Atom" in line:
                        line = next(file_iterator)
                    cols = line.split()
                    cols_len = len(cols)

                    while len(cols) == cols_len:
                        if "Harmonic" in line:
                            break
                        if hpmodes:
                            cols = cols[1:]
                        mass_sqrt = np.sqrt(_MASS_VALUES[int(cols[1]) - 1])
                        cols = cols[2:]
                        for i in range(len(evecs)):
                            if hpmodes:
                                evecs[i].append(float(cols[i]) * mass_sqrt)
                            else:
                                for _ in range(3):
                                    x = float(cols.pop(0))
                                    evecs[i].append(x * mass_sqrt)
                        line = next(file_iterator)
                        cols = line.split()

                    for i in range(len(evecs)):
                        evals_list.append(force_constants[i])
                        evecs_list.append(evecs[i])
                    if "Harmonic" in line:
                        break
        self._records = records
        if evals_list and evecs_list:
            for evec in evecs_list:
                inv_norm = 1.0 / float(np.sqrt(sum(element * element for element in evec)))
                for i in range(len(evec)):
                    evec[i] *= inv_norm
            self._evals = np.array(evals_list)
            self._evecs = np.array(evecs_list)
            self._frequencies_cm = np.array(frequencies_cm_list)
            logger.log(1, f">>> self._evals: {self._evals}")
            logger.log(1, f">>> self._evecs: {self._evecs}")
            logger.log(5, f"  -- {len(self._records)} structures found.")
        self._body_parsed = True

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
