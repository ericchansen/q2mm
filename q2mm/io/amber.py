"""AMBER .frcmod file format I/O."""

from __future__ import annotations

import contextlib
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


# ---------------------------------------------------------------------------
# Public load / save
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
