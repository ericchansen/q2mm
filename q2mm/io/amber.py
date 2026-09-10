"""AMBER .frcmod file format I/O."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from q2mm.io._helpers import _validate_form_for_format
from q2mm.models.forcefield import (
    AngleParam,
    BondParam,
    ForceField,
    FunctionalForm,
    TorsionParam,
    VdwParam,
)
from q2mm.models.identifiers import canonicalize_angle_env_id, canonicalize_bond_env_id

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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


def _valid_amber_dihe_types(types: Sequence[str]) -> bool:
    """Check the existing four-native-type grammar without choosing a row interpretation."""
    return (
        len(types) == 4
        and not types[0].startswith("#")
        and not any(not 1 <= len(t) <= 2 or any(c.isspace() or c == "-" or not c.isascii() for c in t) for t in types)
    )


def _amber_dihe_key(types: Sequence[str]) -> tuple[str, ...]:
    """Identify a proper torsion by its native types, up to full reversal."""
    if not _valid_amber_dihe_types(types):
        raise ValueError("AMBER DIHE requires four explicit one- or two-character atom types")
    forward = tuple(types)
    return min(forward, forward[::-1])


@dataclass(frozen=True)
class _AmberDihedralRow:
    """File-only DIHE values; PN's sign is not physical parameter identity."""

    atom_types: tuple[str, ...]
    idivf: int
    barrier: float
    phase: float
    pn: int
    type_prefix: str

    @property
    def effective_idivf(self) -> int:
        """Use LEaP's zero-as-one divisor without losing the encoded value."""
        return self.idivf or 1


def _amber_dihe_rows(lines: Sequence[str]) -> dict[int, _AmberDihedralRow]:
    """Read contiguous positive-terminated chains, inheriting omitted native types."""
    rows = {}
    in_dihe = False
    pending: tuple[str, ...] | None = None
    seen: set[tuple[str, ...]] = set()
    folds: set[int] = set()
    for row, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped in _FRCMOD_SECTIONS or not stripped:
            if pending is not None:
                raise ValueError(f"AMBER DIHE row {row}: continuation requires a positive-PN final component")
            in_dihe = stripped == "DIHE"
            continue
        if not in_dihe or stripped.startswith("#"):
            continue
        types, rest = _parse_amber_types(line, 4)
        explicit_values = (
            _parse_floats(rest)
            if len(line) >= 11 and all(line[i] == "-" for i in (2, 5, 8)) and _valid_amber_dihe_types(types)
            else []
        )
        compact_values = _parse_floats(line)
        if len(explicit_values) >= 4:
            if pending is not None and len(compact_values) >= 4:
                raise ValueError(f"AMBER DIHE row {row}: ambiguous explicit atom types and implicit numeric values")
            type_prefix = line[:11]
            values = explicit_values
        elif len(compact_values) >= 4:
            if pending is None:
                raise ValueError(f"AMBER DIHE row {row}: implicit types require a preceding negative-PN component")
            types = list(pending)
            type_prefix = line[: len(line) - len(line.lstrip())]
            values = compact_values
        else:
            raise ValueError(f"AMBER DIHE row {row}: requires atom types or a continuation and IDIVF, PK, PHASE, PN")
        key = _amber_dihe_key(types)
        idivf, barrier, phase, pn = values[:4]
        if not all(math.isfinite(v) for v in values[:4]) or idivf != int(idivf) or pn == 0 or pn != int(pn):
            raise ValueError(f"AMBER DIHE row {row}: requires finite values, integer IDIVF and nonzero integer PN")
        if barrier != 0.0 and barrier / (idivf or 1.0) == 0.0:
            raise ValueError(f"AMBER DIHE row {row}: amplitude scaling underflow")
        if pending is not None and _amber_dihe_key(pending) != key:
            raise ValueError(f"AMBER DIHE row {row}: interleaved continuation changes atom types")
        if pending is None:
            if key in seen:
                raise ValueError(f"AMBER DIHE row {row}: multiple completed definitions for the same atom types")
            seen.add(key)
            folds = set()
        fold = abs(int(pn))
        if fold in folds:
            raise ValueError(f"AMBER DIHE row {row}: duplicate periodicity in a continuation group")
        folds.add(fold)
        rows[row] = _AmberDihedralRow(tuple(types), int(idivf), barrier, phase, int(pn), type_prefix)
        pending = tuple(types) if pn < 0 else None
    if pending is not None:
        raise ValueError("AMBER DIHE continuation requires a positive-PN final component before end of file")
    return rows


# ---------------------------------------------------------------------------
# Public load / save
# ---------------------------------------------------------------------------


def load_amber_frcmod(path: str | Path) -> ForceField:
    """Load from standard AMBER .frcmod file.

    Parses MASS, BOND, ANGLE/ANGL, DIHE, IMPROPER, and NONBON sections.
    Atom type → element mapping uses the MASS section when present,
    falling back to the GAFF convention (first character).

    Negative DIHE PN marks another component, not negative physical
    periodicity. Chains start with explicit atom types (allowing full
    reversal); numeric-only continuations inherit the preceding negative-PN
    component's type orientation. Components have distinct positive absolute
    periodicities and a positive final PN. Integer IDIVF values retain their
    sign; encoded zero uses an effective divisor of one, as in LEaP.
    Ambiguous redefinitions, orphan implicit rows, and interrupted or
    unterminated chains raise ``ValueError``. A pending row that can be read
    as both explicit numeric atom types and compact values is rejected rather
    than guessed. Raw IMPROPER PN must be a finite positive integer: unlike
    DIHE continuation signs or zero IDIVF, no absolute-value/default-one
    normalization is applied to improper source periodicities.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    dihe_rows = _amber_dihe_rows(lines)

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
                        env_id=canonicalize_bond_env_id(types),
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
                        env_id=canonicalize_angle_env_id(types),
                        ff_row=row,
                        label=f"frcmod row {row}",
                    )
                )

        elif section == "DIHE":
            record = dihe_rows[row]
            elems = tuple(_amber_type_to_element(t, mass_map) for t in record.atom_types)
            torsions.append(
                TorsionParam(
                    elements=elems,
                    periodicity=abs(record.pn),
                    force_constant=record.barrier / record.effective_idivf,
                    phase=record.phase,
                    env_id="-".join(record.atom_types),
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
                pn = vals[2]
                # IMPROPER has neither DIHE's continuation sign nor a zero-to-one convention.
                if not math.isfinite(pn) or pn <= 0 or pn != int(pn):
                    raise ValueError(f"AMBER IMPROPER row {row}: raw PN must be a finite positive integer")
                periodicity = int(pn)
                torsions.append(
                    TorsionParam(
                        elements=elems,
                        periodicity=periodicity,
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

    Standalone DIHE components are grouped by atom types up to full reversal,
    preserving each component's orientation, amplitude and explicit phase.
    All but the last component receive negative PN; the final PN is positive,
    regardless of periodicity order. Source rows and labels do not define
    groups. Templates retain their valid continuation signs and IDIVF;
    proper and improper components must bind one-to-one. Additions/removals
    or ambiguous bindings raise ``ValueError`` before the destination is
    opened. Proper template bindings recognize full
    reversal of types and elements without changing source-column orientation;
    source rows, when present, must still match exactly. Improper bindings
    retain native atom-type/element ordering rather than applying DIHE
    reversal equivalence. Without a source row, each component needs a unique
    type/periodicity binding. Both proper and improper torsions require
    positive integer canonical periodicities and finite amplitudes/phases,
    which are emitted without exponent tokens or precision truncation.
    Templates preserve implicit type prefixes and encoded IDIVF, including
    zero's effective-one convention. Unchanged improper rows retain their text
    only after raw source PN validation; zero, negative, fractional or
    nonfinite improper periods are not silently mapped to canonical values.

    This is file-format preservation, not validation of AMBER engine energies
    or a change to signed-dihedral conventions.

    References:
        https://ambermd.org/FileFormats.php (parameter card 6 and frcmod DIHE)
        https://github.com/ParmEd/ParmEd/blob/4.3.1/parmed/amber/parameters.py
        https://github.com/Amber-MD/AmberClassic/blob/8e55e97ada48b96eefaec2e6a3fa849018aaeea5/src/leap/amber.c
        https://github.com/Amber-MD/AmberClassic/blob/8e55e97ada48b96eefaec2e6a3fa849018aaeea5/src/msander/set.F90
        https://github.com/ParmEd/ParmEd/blob/4.3.1/parmed/topologyobjects.py

    """
    _validate_form_for_format(ff, "amber_frcmod")
    output_path = Path(path)
    template = Path(template_path) if template_path is not None else None
    if template is None and ff.source_format == "amber_frcmod" and ff.source_path is not None:
        template = ff.source_path

    if template is not None:
        return _save_amber_frcmod_template(ff, output_path, template)

    return _save_amber_frcmod_standalone(ff, output_path, remark)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_amber_trailing(line: str, n_types: int, n_values: int, *, prefix_length: int | None = None) -> str:
    """Extract trailing comment/text after the numeric fields in a frcmod line.

    *n_types* is the number of atom types (2 for BOND, 3 for ANGLE, etc.)
    and *n_values* is the expected count of numeric columns.  Returns the
    trailing text (including any leading whitespace) or empty string.
    """
    if prefix_length is None:
        _, rest = _parse_amber_types(line, n_types)
    else:
        rest = line[prefix_length:]
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


def _amber_torsion_decimal_values(k: float, phase: float, *, section: str) -> tuple[str, str]:
    """Retain finite float precision in the readers' exponent-free decimal syntax."""
    if not math.isfinite(k) or not math.isfinite(phase):
        raise ValueError(f"AMBER {section} exported amplitude and phase must be finite")
    # Decimal notation retains float precision without exponent tokens that
    # some frcmod readers (including ParmEd) do not accept. These are minimum
    # display widths, not numeric limits for their whitespace-delimited fields.
    return format(Decimal(str(float(k))), "f"), format(Decimal(str(float(phase))), "f")


def _format_amber_dihe_line(
    types: list[str],
    k: float,
    phase: float,
    periodicity: int,
    suffix: str = "",
    *,
    idivf: int = 1,
    type_prefix: str | None = None,
) -> str:
    barrier, phase_text = _amber_torsion_decimal_values(k, phase, section="DIHE")
    prefix = "-".join(f"{atom:<2}" for atom in types) if type_prefix is None else type_prefix
    separator = "   " if prefix.strip() else ""
    return f"{prefix}{separator}{idivf} {barrier:>10} {phase_text:>8} {periodicity}.0{suffix}\n"


def _format_amber_improper_line(types: list[str], k: float, phase: float, periodicity: int, suffix: str = "") -> str:
    barrier, phase_text = _amber_torsion_decimal_values(k, phase, section="IMPROPER")
    # LEaP skips the first 15 characters before scanning improper numeric fields.
    return f"{types[0]:<2}-{types[1]:<2}-{types[2]:<2}-{types[3]:<2}    {barrier:>10} {phase_text:>8} {int(periodicity)}.0{suffix}\n"


def _format_amber_nonbon_line(atom_type: str, radius: float, epsilon: float, suffix: str = "") -> str:
    return f"{atom_type:<2} {radius:10.4f} {epsilon:10.4f}{suffix}\n"


def _amber_env_types(env_id: str, elements: tuple[str, ...]) -> list[str]:
    """Get AMBER-style atom types from env_id, falling back to element symbols."""
    parts = [p.strip() for p in env_id.split("-") if p.strip()] if env_id else []
    if len(parts) == len(elements):
        return parts
    return [e.lower() for e in elements]


def _amber_proper_groups(torsions: Sequence[TorsionParam]) -> dict[tuple[str, ...], list[TorsionParam]]:
    """Group native proper types without conflating source-row provenance."""
    groups: dict[tuple[str, ...], list[TorsionParam]] = {}
    for tor in torsions:
        section = "IMPROPER" if tor.is_improper else "DIHE"
        if len(tor.elements) != 4:
            raise ValueError(f"AMBER {section} requires four elements.")
        if not math.isfinite(tor.force_constant) or not math.isfinite(tor.phase):
            raise ValueError(f"AMBER {section} amplitude and phase must be finite")
        if (
            isinstance(tor.periodicity, bool)
            or not math.isfinite(tor.periodicity)
            or tor.periodicity <= 0
            or tor.periodicity != int(tor.periodicity)
        ):
            raise ValueError(f"AMBER {section} canonical periodicity must be a positive integer")
        types = [part.strip() for part in tor.env_id.split("-")] if tor.env_id else _amber_env_types("", tor.elements)
        if not _valid_amber_dihe_types(types):
            raise ValueError(f"AMBER {section} requires four explicit one- or two-character atom types")
        if tor.is_improper:
            continue
        key = _amber_dihe_key(types)
        group = groups.setdefault(key, [])
        if any(t.periodicity == tor.periodicity for t in group):
            raise ValueError("AMBER DIHE cannot represent duplicate periodicities for the same atom types")
        group.append(tor)
    return groups


def _save_amber_frcmod_standalone(ff: ForceField, output_path: Path, remark: str) -> Path:
    """Generate a standalone .frcmod file from scratch."""
    proper_groups = _amber_proper_groups(ff.torsions)
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
        improper = [t for t in ff.torsions if t.is_improper]
        if proper_groups:
            lines.append("DIHE\n")
            for group in proper_groups.values():
                for index, tor in enumerate(group):
                    types = _amber_env_types(tor.env_id, tor.elements)
                    pn = int(tor.periodicity) if index == len(group) - 1 else -int(tor.periodicity)
                    lines.append(_format_amber_dihe_line(types, tor.force_constant, tor.phase, pn))
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


def _bind_amber_template_torsions(
    torsions: Sequence[TorsionParam], originals: Sequence[TorsionParam], *, section: str
) -> dict[int, TorsionParam]:
    if len(torsions) != len(originals):
        raise ValueError(f"AMBER {section} template cannot represent component additions or removals")
    reversible = section == "DIHE"

    def key(term: TorsionParam) -> tuple[str, ...]:
        types = _amber_env_types(term.env_id, term.elements)
        return _amber_dihe_key(types) if reversible else tuple(types)

    bindings: dict[int, TorsionParam] = {}
    for tor in torsions:
        target_key = key(tor)
        elements = tuple(tor.elements)
        candidates = [
            before
            for before in originals
            if target_key == key(before)
            and (elements == tuple(before.elements) or (reversible and elements == tuple(before.elements)[::-1]))
            and (tor.ff_row == before.ff_row if tor.ff_row is not None else tor.periodicity == before.periodicity)
        ]
        if len(candidates) != 1:
            raise ValueError(f"AMBER {section} template component has missing or ambiguous source-row binding")
        row = candidates[0].ff_row
        assert row is not None
        if row in bindings:
            raise ValueError(f"AMBER {section} template component has duplicate source-row binding")
        bindings[row] = tor
    return bindings


def _save_amber_frcmod_template(ff: ForceField, output_path: Path, template: Path) -> Path:
    """Update parameter values in an existing .frcmod template."""
    src_lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
    dihe_rows = _amber_dihe_rows(src_lines)
    original = load_amber_frcmod(template)
    original_impropers = {t.ff_row: t for t in original.improper_torsions}
    _amber_proper_groups(ff.torsions)
    bond_by_row = {b.ff_row: b for b in ff.bonds if b.ff_row is not None}
    angle_by_row = {a.ff_row: a for a in ff.angles if a.ff_row is not None}
    torsion_by_row = _bind_amber_template_torsions(ff.proper_torsions, original.proper_torsions, section="DIHE")
    torsion_by_row.update(
        _bind_amber_template_torsions(ff.improper_torsions, original.improper_torsions, section="IMPROPER")
    )
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
            if section == "IMPROPER":
                before = original_impropers[row]
                if (t.force_constant, t.phase, t.periodicity) == (
                    before.force_constant,
                    before.phase,
                    before.periodicity,
                ):
                    out_lines.append(line)
                else:
                    suffix = _extract_amber_trailing(line, 4, 3)
                    out_lines.append(
                        _format_amber_improper_line(types, t.force_constant, t.phase, t.periodicity, suffix)
                    )
            else:
                # Preserve the template's IDIVF and reconstruct the barrier
                # before the formatter checks finiteness: even a finite
                # canonical amplitude can overflow when multiplied by IDIVF.
                record = dihe_rows[row]
                suffix = _extract_amber_trailing(line, 4, 4, prefix_length=len(record.type_prefix))
                barrier = t.force_constant * record.effective_idivf
                pn = -int(t.periodicity) if record.pn < 0 else int(t.periodicity)
                out_lines.append(
                    _format_amber_dihe_line(
                        list(record.atom_types),
                        barrier,
                        t.phase,
                        pn,
                        suffix,
                        idivf=record.idivf,
                        type_prefix=record.type_prefix,
                    )
                )
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
