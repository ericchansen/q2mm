"""Parser for Gaussian log files.

Extracts structures, Hessians, eigenvectors, eigenvalues, frequencies,
and ESP data from Gaussian output files.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np

from q2mm import constants as co
from q2mm.parsers.base import File
from q2mm.models.structure import Atom, Structure

if TYPE_CHECKING:
    from q2mm.models.molecule import Q2MMMolecule

logger = logging.getLogger(__name__)


class GaussLog(File):
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
        "_structures",
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
        super().__init__(path)
        self._evals = None
        self._evecs = None
        self._frequencies_cm = None
        self._structures = None
        self._esp_rms = None
        self._au_hessian = au_hessian

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
    def structures(self) -> list[Structure]:
        """Returns Structure objects parsed from the Gaussian log file.

        If None, parses the archive of the log file for structures.

        Returns:
            (list[Structure]): Structures parsed from log file archive.

        .. deprecated::
            Use :attr:`molecules` instead for ``Q2MMMolecule`` objects.

        """
        if self._structures is None:
            self.read_archive()
        return self._structures

    @property
    def molecules(self) -> list[Q2MMMolecule]:
        """Parsed structures as :class:`~q2mm.models.molecule.Q2MMMolecule` objects.

        Each structure is converted via
        :meth:`Q2MMMolecule.from_structure`, preserving any Hessian data
        attached to the underlying ``Structure`` when it is stored in
        atomic units (Hartree/Bohr²).

        Note:
            ``Q2MMMolecule.hessian`` is documented in Hartree/Bohr².  When
            this parser was constructed with ``au_hessian=False`` (the
            default), any attached Hessian has already been converted to
            kJ/(mol·Å²) and will **not** be forwarded to avoid silently
            mixing unit systems.

        """
        from q2mm.models.molecule import Q2MMMolecule

        molecules: list[Q2MMMolecule] = []
        for s in self.structures:
            hess = getattr(s, "hess", None)
            if hess is not None and not self._au_hessian:
                logger.warning(
                    "Non-atomic-unit Hessian detected (au_hessian=False); "
                    "not attaching Hessian to Q2MMMolecule, which expects "
                    "Hartree/Bohr²."
                )
                hess = None
            molecules.append(Q2MMMolecule.from_structure(s, hessian=hess))
        return molecules

    def read_out(self) -> None:
        """Read force constant and eigenvector data from a frequency calculation.

        Populates ``_evals``, ``_evecs``, ``_frequencies_cm``, ``_structures``,
        and ``_esp_rms`` from the Gaussian log file body.
        """
        logger.log(5, f"READING: {self.filename}")
        self._evals = []
        self._evecs = []
        self._frequencies_cm = []
        self._structures = []
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
                    self._structures.append(Structure(self.filename))
                    next(file_iterator)
                    next(file_iterator)
                    next(file_iterator)
                    next(file_iterator)
                    line = next(file_iterator)
                    while "---" not in line:
                        cols = line.split()
                        self._structures[-1].atoms.append(
                            Atom(
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
                        f"  -- Found {len(self._structures[-1].atoms)} atoms.",
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
            logger.log(5, f"  -- {len(self.structures)} structures found.")

    # May want to move some attributes assigned to the structure class onto
    # this filetype class.
    def read_archive(self) -> None:
        r"""Read the last archive section from the Gaussian log file.

        Extracts atoms, coordinates, properties, and the Hessian (converted
        to kJ/(mol·Å²) unless ``au_hessian`` is True) from the archive block.

        The Gaussian archive format uses ``\\\\`` as section separators and
        ``\\`` within sections.  The fields always present are: user, date,
        route/command, title, charge/multiplicity, and atom coordinates
        (``element,x,y,z`` entries separated by ``\\``).  Optional fields
        that may follow include ``HF``, ``ZeroPoint``, ``Thermal``,
        ``NImag``, the Hessian (lower-triangular, comma-separated), and
        eigenvalues.

        Raises:
            IndexError: If no archive section is found in the log file.

        """
        logger.log(5, f"READING: {self.filename}")
        struct = Structure(self.filename)
        self._structures = [struct]
        # Matches everything in between the start and end.
        # (?s)  - Flag for re.compile which says that . matches all.
        # \\\\  - One single \
        # Start - " 1\1\".
        # End   - Some number of \ followed by @. Not sure how many \ there
        #         are, so this matches as many as possible. Also, this could
        #         get separated by a line break (which would also include
        #         adding in a space since that's how Gaussian starts new lines
        #         in the archive).
        # We pull out the last one [-1] in case there are multiple archives
        # in a file.
        try:
            with open(self.path) as fh:
                arch = re.findall("(?s)(\\s1\\\\1\\\\.*?[\\\\\n\\s]+@)", fh.read())[-1]
            logger.log(5, "  -- Located last archive.")
        except IndexError:
            logger.warning("  -- Couldn't locate archive.")
            raise
        # Make it into one string.
        arch = arch.replace("\n ", "")
        # Separate it by Gaussian's section divider.
        arch = arch.split("\\\\")
        # Helps us iterate over sections of the archive.
        section_counter = 0
        # SECTION 0
        # General job information.
        arch_general = arch[section_counter]
        section_counter += 1
        stuff = re.search(
            "\\s1\\\\1\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\(?P<user>.*?)\\\\(?P<date>.*?)\\\\.*?",
            arch_general,
        )
        struct.props["user"] = stuff.group("user")
        struct.props["date"] = stuff.group("date")
        # SECTION 1
        # The commands you wrote.
        _arch_commands = arch[section_counter]
        section_counter += 1
        # SECTION 2
        # The comment line.
        _arch_comment = arch[section_counter]
        section_counter += 1
        # SECTION 3
        # Actually has charge, multiplicity and coords.
        arch_coords = arch[section_counter]
        section_counter += 1
        stuff = re.search("(?P<charge>.*?),(?P<multiplicity>.*?)\\\\(?P<atoms>.*)", arch_coords)
        struct.props["charge"] = stuff.group("charge")
        struct.props["multiplicity"] = stuff.group("multiplicity")
        # We want to do more fancy stuff with the atoms than simply add to
        # the properties dictionary.
        atoms = stuff.group("atoms")
        atoms = atoms.split("\\")
        # Z-matrix coordinates adds another section. We need to be aware of
        # this.
        struct._atoms = []
        for atom in atoms:
            stuff = atom.split(",")
            # An atom typically looks like this:
            #    C,0.1135,0.13135,0.63463
            if len(stuff) == 4:
                ele, x, y, z = stuff
            # But sometimes they look like this (notice the extra zero):
            #    C,0,0.1135,0.13135,0.63463
            # I'm not sure what that extra zero is for. Anyway, ignore
            # that extra whatever if it's there.
            elif len(stuff) == 5:
                ele, x, y, z = stuff[0], stuff[2], stuff[3], stuff[4]
            # And this would be really bad. Haven't seen anything else like
            # this yet.
            # 160613 - So, not sure when I wrote that comment, but something
            # like this definitely happens when using scans and z-matrices.
            # I'm going to ignore grabbing any atoms in this case.
            else:
                logger.warning("Not sure how to read coordinates from Gaussian acrhive!")
                section_counter += 1
                # Let's have it stop looping over atoms, but not fail anymore.
                break
                # raise Exception(
                #     'Not sure how to read coordinates from Gaussian archive!')
            struct._atoms.append(Atom(element=ele, x=float(x), y=float(y), z=float(z)))
        logger.log(20, f"  -- Read {len(struct._atoms)} atoms.")
        # SECTION 4
        # All sorts of information here. This area looks like:
        #     prop1=value1\prop2=value2\prop3=value3
        arch_info = arch[section_counter]
        section_counter += 1
        arch_info = arch_info.split("\\")
        for thing in arch_info:
            prop_name, prop_value = thing.split("=")
            struct.props[prop_name] = prop_value
        # SECTION 5
        # The Hessian. Only exists if you did a frequency calculation.
        # Appears in lower triangular form, not mass-weighted.
        if arch[section_counter] != "@":
            hess_tri = arch[section_counter]
            hess_tri = hess_tri.split(",")
            logger.log(
                5,
                f"  -- Read {len(hess_tri)} Hessian elements in lower triangular form.",
            )
            hess = np.zeros([len(atoms) * 3, len(atoms) * 3], dtype=float)
            logger.log(5, f"  -- Created {hess.shape} Hessian matrix.")
            # Hessian is stored as lower triangle in the archive.
            hess[np.tril_indices_from(hess)] = hess_tri
            hess += np.tril(hess, -1).T
            if not self._au_hessian:
                hess *= co.HESSIAN_CONVERSION
            struct.hess = hess
            # SECTION 6
            # Not sure what this is.
