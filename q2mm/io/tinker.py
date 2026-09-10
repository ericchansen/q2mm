"""Tinker .prm file format I/O."""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from q2mm.io._helpers import (
    _clean_atom_types,
    _normalize_equilibrium_angle,
    _split_env_id,
    _validate_form_for_format,
)
from q2mm.models.forcefield import (
    AngleParam,
    BondParam,
    ForceField,
    FunctionalForm,
    TorsionParam,
    VdwParam,
)
from q2mm.models.identifiers import (
    _extract_element,
    canonicalize_angle_env_id,
    canonicalize_bond_env_id,
)
from q2mm.models.units import (
    canonical_to_mm3_angle_k,
    canonical_to_mm3_bond_k,
    mm3_angle_k_to_canonical,
    mm3_bond_k_to_canonical,
)


@dataclasses.dataclass(slots=True)
class _TinkerParameterRow:
    """One Tinker ``.prm`` file row, staged during parsing/serialization.

    Represents a single physical value read from (or to be written back
    to) a Tinker ``.prm`` file — one bond/angle/torsion/vdW/dipole/pibond/
    out-of-plane-bend scalar — before it is converted into (or matched
    against) an immutable :class:`~q2mm.models.forcefield.BondParam` /
    :class:`~q2mm.models.forcefield.AngleParam` / etc. record.
    Parser-private to this module; never exported. Carries no
    optimizer-facing state (step sizes, allowed ranges, active/frozen
    partition). Unlike the MM3 ``.fld`` row,
    Tinker ``.prm`` records carry no bond-order/context columns.
    ``slots=True`` enforces this exact field set — no arbitrary
    attribute (e.g. a MM3-only ``bond_order``/``context``) can be
    attached later.

    Attributes:
        ptype: Parameter type (``"bf"``, ``"be"``, ``"af"``, ``"ae"``,
            ``"df"``, ``"q"``, ``"q_p"``, ``"pi_b"``, ``"pi_t"``,
            ``"op_b"``, ``"vdw"``).
        value: The row's numeric value, already in ``.prm`` (file)
            convention/units.
        ff_row: 1-based row number in the ``.prm`` file.
        ff_col: Column index within the row (used to distinguish e.g.
            multiple equilibrium angles, or torsion triplet position).
        atom_types: Atom-type strings for this row, in file order.

    """

    ptype: str
    value: float
    ff_row: int
    ff_col: int
    atom_types: list[str] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Atom-type helpers
# ---------------------------------------------------------------------------


def _default_tinker_atom_types(elements: tuple[str, ...]) -> list[str]:
    counts: dict[str, int] = {}
    atom_types = []
    for element in elements:
        count = counts.get(element, 0) + 1
        counts[element] = count
        atom_types.append(f"{element}{count}")
    return atom_types


def _tinker_atom_types(env_id: str, elements: tuple[str, ...]) -> list[str]:
    return _split_env_id(env_id, len(elements)) or _default_tinker_atom_types(elements)


# ---------------------------------------------------------------------------
# Line formatters
# ---------------------------------------------------------------------------


def _format_tinker_bond_line(atom_types: list[str], force_constant: float, equilibrium: float) -> str:
    return f"bond   {atom_types[0]:>4} {atom_types[1]:>4} {force_constant:10.4f} {equilibrium:10.4f}\n"


def _format_tinker_angle_line(atom_types: list[str], force_constant: float, equilibrium: float) -> str:
    return (
        f"angle  {atom_types[0]:>4} {atom_types[1]:>4} {atom_types[2]:>4} {force_constant:10.4f} {equilibrium:10.4f}\n"
    )


def _format_tinker_vdw_line(atom_type: str, radius: float, epsilon: float, reduction: float = 0.0) -> str:
    return f"vdw    {atom_type:>4} {radius:10.4f} {epsilon:10.4f} {reduction:10.4f}\n"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _tinker_data(line: str) -> str:
    """Return data before the first unquoted # or ! comment delimiter."""
    for match in re.finditer(r""""[^"]*"|'[^']*'|[#!]""", line):
        if match.group() in ("#", "!"):
            return line[: match.start()]
    return line


def _tinker_tokens(line: str) -> list[str]:
    return _tinker_data(line).split()


def _validate_tinker_record_lengths(lines: Sequence[str]) -> None:
    # getprm.f reads A240 records; readprm.f also uses CHARACTER*240.
    # Comments may extend beyond that limit, but required data must not.
    for row, line in enumerate(lines, start=1):
        if len(_tinker_data(line).rstrip().encode("utf-8")) > 240:
            raise ValueError(f"Tinker row {row}: data exceeds the native 240-byte record limit")


def _tinker_float(token: str, row: int) -> float:
    if "_" in token:
        raise ValueError(f"Tinker row {row}: invalid numeric value {token!r}")
    try:
        value = float(token.replace("D", "e").replace("d", "e"))
    except ValueError as exc:
        raise ValueError(f"Tinker row {row}: invalid numeric value {token!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Tinker row {row}: numeric values must be finite")
    return value


def _tinker_torsion_unit(lines: Sequence[str]) -> float:
    """Read the file-wide scale, with Tinker's default and last-value precedence."""
    unit = 1.0
    for row, line in enumerate(lines, start=1):
        parts = _tinker_tokens(line)
        if parts and parts[0].lower() == "torsionunit":
            if len(parts) != 2:
                raise ValueError(f"Tinker row {row}: torsionunit requires one finite nonzero value")
            unit = _tinker_float(parts[1], row)
            if unit == 0.0:
                raise ValueError(f"Tinker row {row}: zero torsionunit is not invertible")
    return unit


def _tinker_scaled_torsion(amplitude: float, unit: float, row: int) -> float:
    coefficient = unit * amplitude
    if not math.isfinite(coefficient) or (amplitude != 0.0 and coefficient == 0.0):
        raise ValueError(f"Tinker row {row}: torsion scaling overflow or underflow")
    return coefficient


def _tinker_torsion_terms(parts: list[str], row: int) -> list[tuple[float, float, int]]:
    """Read amplitude/phase/fold triples, not positional MM3 V1/V2/V3 columns.

    Tinker's readprm.f / torphase.f accept up to six folds (1..6).
    Duplicate folds overwrite earlier values in Tinker; reject them here
    rather than expose independent canonical terms with different semantics.
    """
    if len(parts) < 8 or len(parts) > 23 or (len(parts) - 5) % 3:
        raise ValueError(f"Tinker row {row}: torsion requires one to six complete amplitude/phase/periodicity triplets")
    terms = []
    folds: set[int] = set()
    for offset in range(5, len(parts), 3):
        amplitude = _tinker_float(parts[offset], row)
        phase = _tinker_float(parts[offset + 1], row)
        try:
            fold = int(parts[offset + 2])
        except ValueError as exc:
            raise ValueError(f"Tinker row {row}: torsion periodicity must be an integer from 1 to 6") from exc
        if fold not in range(1, 7) or fold in folds:
            raise ValueError(f"Tinker row {row}: torsion periodicities must be distinct integers from 1 to 6")
        folds.add(fold)
        terms.append((amplitude, phase, fold))
    return terms


def _parse_tinker_atom_elements(path: Path) -> dict[str, str]:
    """Map Tinker atom-type numbers to element symbols from ``atom`` records.

    The element comes from each ``atom`` record's symbol column, which is
    authoritative — unlike guessing from downstream atom-type *labels*,
    where ``_extract_element`` would misread a two-letter label that
    title-cases to a real element (``"CO"``/``"CA"`` → cobalt/calcium).
    """
    atom_elements: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = raw_line.split()
        # Standard Tinker: atom <type> <symbol> "desc" <anum> <mass> <val>
        # AMOEBA-style:    atom <type> <class> <symbol> "desc" ...
        # Distinguish: if parts[2] is purely numeric, it's a class field.
        if parts[0].lower() == "atom" and len(parts) >= 3:
            symbol_col = 2
            if parts[2].isdigit() and len(parts) >= 4:
                symbol_col = 3
            atom_elements[parts[1]] = _extract_element(parts[symbol_col])
    return atom_elements


# ---------------------------------------------------------------------------
# Tinker FF import / export
# ---------------------------------------------------------------------------

_BONDS = ["bond", "bond3", "bond4", "bond5"]
_PIBONDS = ["pibond", "pibond3", "pibond4", "pibond5"]
_ANGLES = ["angle", "angle3", "angle4", "angle5"]
_DIPOLES = ["dipole", "dipole3", "dipole4", "dipole5"]

logger = logging.getLogger(__name__)


def _tinker_import_ff(path: str | Path) -> tuple[list[_TinkerParameterRow], list[str]]:
    """Read OPT rows in marked files, or supported rows in unmarked files."""
    with open(path, encoding="utf-8", newline="") as f:
        lines = f.readlines()
    rows: list[_TinkerParameterRow] = []
    q2mm_sec = False
    gather_data = not any("# Q2MM" in line for line in lines)
    for row, line in enumerate(lines, start=1):
        if not q2mm_sec and "# Q2MM" in line:
            q2mm_sec = True
            gather_data = False
        elif q2mm_sec and line.lstrip().startswith("#"):
            gather_data = "OPT" in line
        parts = _tinker_tokens(line)
        if not gather_data or not parts:
            continue
        record = parts[0].lower()
        if record in ("torsion4", "torsion5", "anglep", "anglef"):
            raise ValueError(f"Tinker row {row}: unsupported functional form {record!r}")
        if record == "torsion":
            for slot, (amplitude, _phase, _fold) in enumerate(_tinker_torsion_terms(parts, row), start=1):
                rows.append(_TinkerParameterRow("df", amplitude, row, slot, parts[1:5]))
            continue
        if record in _BONDS:
            atom_count, ptypes, allowed_lengths = 2, ("bf", "be"), (5,)
        elif record in _ANGLES:
            atom_count, ptypes, allowed_lengths = 3, ("af", "ae", "ae", "ae"), (6, 7, 8)
        elif record == "vdw":
            atom_count, ptypes, allowed_lengths = 1, ("vdw",), (4, 5)
        elif record in _DIPOLES:
            atom_count, ptypes, allowed_lengths = 2, ("q", "q_p"), (5,)
        elif record in _PIBONDS:
            atom_count, ptypes, allowed_lengths = 2, ("pi_b", "pi_t"), (5,)
        elif record == "opbend":
            atom_count, ptypes, allowed_lengths = 4, ("op_b",), (6,)
        else:
            continue
        if len(parts) not in allowed_lengths:
            raise ValueError(f"Tinker row {row}: malformed or unsupported {record} fields")
        for col, token in enumerate(parts[atom_count + 1 :], start=1):
            value = _tinker_float(token, row)
            if col > len(ptypes):
                continue
            ptype = ptypes[col - 1]
            if ptype == "ae":
                value = _normalize_equilibrium_angle(value)
            rows.append(_TinkerParameterRow(ptype, value, row, col, parts[1 : atom_count + 1]))
    logger.log(15, "Read %s Tinker parameters from %s", len(rows), path)
    return rows, lines


# ---------------------------------------------------------------------------
# Public load / save
# ---------------------------------------------------------------------------


def load_tinker_prm(path: str | Path) -> ForceField:
    """Load supported bond, angle, proper torsion and vdW parameters.

    Marked files expose only OPT sections; unmarked files expose supported
    rows throughout. Bond/angle conversions retain the MM3 convention.
    Torsion coefficients are ``torsionunit * amplitude`` in kcal/mol, with
    the source phase (degrees) and fold retained. The default scale is 1;
    finite nonzero file overrides are supported, but keyfiles are not read.
    Ring-specific torsions, duplicate folds and malformed triples fail.
    Other template records are retained on save, not modeled here.

    References:
        https://tinkerdoc.readthedocs.io/en/latest/text/key/index.html#key-torsion
        https://tinkerdoc.readthedocs.io/en/latest/text/key/index.html#key-torsionunit

    """
    rows, lines = _tinker_import_ff(path)
    torsion_unit = _tinker_torsion_unit(lines)
    bonds = []
    angles = []
    torsions = []
    vdws = []
    atom_elements = _parse_tinker_atom_elements(Path(path))

    def _elem(atom_type: str) -> str:
        return atom_elements.get(atom_type.strip(), _extract_element(atom_type))

    eq_lookup: dict[tuple[str, int], float] = {}
    for row in rows:
        if row.ptype == "be" or (row.ptype == "ae" and row.ff_col == 2):
            eq_lookup[(row.ptype, row.ff_row)] = row.value

    for row in rows:
        atom_types = _clean_atom_types(row.atom_types, 4)

        if row.ptype == "bf" and len(atom_types) >= 2:
            elems = tuple(_elem(t) for t in atom_types[:2])
            env_id = canonicalize_bond_env_id(atom_types[:2])
            eq_val = eq_lookup.get(("be", row.ff_row), 0.0)
            bonds.append(
                BondParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=mm3_bond_k_to_canonical(row.value),
                    label=f"Tinker row {row.ff_row}",
                    env_id=env_id,
                    ff_row=row.ff_row,
                )
            )
        elif row.ptype == "af" and len(atom_types) >= 3:
            elems = tuple(_elem(t) for t in atom_types[:3])
            env_id = canonicalize_angle_env_id(atom_types[:3])
            eq_val = eq_lookup.get(("ae", row.ff_row), 0.0)
            angles.append(
                AngleParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=mm3_angle_k_to_canonical(row.value),
                    label=f"Tinker row {row.ff_row}",
                    env_id=env_id,
                    ff_row=row.ff_row,
                )
            )
        elif row.ptype == "df" and len(atom_types) >= 4:
            elems = tuple(_elem(t) for t in atom_types[:4])
            env_id = "-".join(t.strip() for t in atom_types[:4])
            amplitude, phase, periodicity = _tinker_torsion_terms(_tinker_tokens(lines[row.ff_row - 1]), row.ff_row)[
                row.ff_col - 1
            ]
            coefficient = _tinker_scaled_torsion(amplitude, torsion_unit, row.ff_row)
            torsions.append(
                TorsionParam(
                    elements=elems,
                    periodicity=periodicity,
                    force_constant=coefficient,
                    phase=phase,
                    label=f"Tinker row {row.ff_row} V{periodicity}",
                    env_id=env_id,
                    ff_row=row.ff_row,
                )
            )
        elif row.ptype == "vdw":
            parts = _tinker_tokens(lines[row.ff_row - 1])
            vdws.append(
                VdwParam(
                    atom_type=parts[1],
                    radius=row.value,
                    epsilon=_tinker_float(parts[3], row.ff_row),
                    reduction=_tinker_float(parts[4], row.ff_row) if len(parts) == 5 else 0.0,
                    element=_elem(parts[1]),
                    label=f"Tinker row {row.ff_row}",
                    ff_row=row.ff_row,
                )
            )

    return ForceField(
        name=f"Tinker from {Path(path).name}",
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        vdws=vdws,
        source_path=Path(path),
        source_format="tinker_prm",
        functional_form=FunctionalForm.MM3,
    )


_TemplateParam = TypeVar("_TemplateParam", BondParam, AngleParam, TorsionParam, VdwParam)


def _tinker_template_pairs(
    originals: Sequence[_TemplateParam],
    updates: Sequence[_TemplateParam],
    identity: Callable[[_TemplateParam], object],
    editable: tuple[str, ...],
) -> list[tuple[_TemplateParam, _TemplateParam]]:
    """Bind every update once, rejecting lossy edits and ambiguous fallback."""
    if len(originals) != len(updates):
        raise ValueError("Tinker template cannot represent parameter additions or removals")
    pairs = []
    matched: set[int] = set()
    for update in updates:
        candidates = [
            (index, original)
            for index, original in enumerate(originals)
            if identity(original) == identity(update) and (update.ff_row is None or update.ff_row == original.ff_row)
        ]
        if len(candidates) != 1 or candidates[0][0] in matched:
            raise ValueError("Tinker template parameter has missing or ambiguous source-row identity")
        index, original = candidates[0]
        expected = dataclasses.replace(original, **{field: getattr(update, field) for field in editable})
        if dataclasses.replace(update, label=original.label, ff_row=original.ff_row) != expected:
            raise ValueError(f"Tinker row {original.ff_row}: template cannot represent non-scalar parameter edits")
        for field in editable:
            if not math.isfinite(getattr(update, field)):
                raise ValueError(f"Tinker row {original.ff_row}: edited {field} must be finite")
        matched.add(index)
        pairs.append((original, update))
    return pairs


def _tinker_replace_token(lines: list[str], row: int, column: int, value: float) -> None:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Tinker row {row}: exported value must be finite")
    line = lines[row - 1]
    tokens = list(re.finditer(r"\S+", _tinker_data(line)))
    if column == len(tokens):
        # An omitted vdW reduction can be appended without moving the comment.
        end = tokens[-1].end()
        lines[row - 1] = line[:end] + f" {value!r}" + line[end:]
    else:
        token = tokens[column]
        lines[row - 1] = line[: token.start()] + repr(value) + line[token.end() :]


def _tinker_template_lines(ff: ForceField, template: Path) -> list[str]:
    original = load_tinker_prm(template)
    with template.open(encoding="utf-8", newline="") as f:
        lines = f.readlines()
    unit = _tinker_torsion_unit(lines)
    if ff.stretch_bends or ff.cmaps or ff.nonbonded_excluded_atom_types:
        raise ValueError("Tinker template cannot represent stretch-bend, CMAP or nonbonded exclusion edits")

    for before, after in _tinker_template_pairs(
        original.bonds, ff.bonds, lambda p: p.env_id, ("force_constant", "equilibrium")
    ):
        assert before.ff_row is not None
        if before.force_constant != after.force_constant:
            _tinker_replace_token(lines, before.ff_row, 3, canonical_to_mm3_bond_k(after.force_constant))
        if before.equilibrium != after.equilibrium:
            _tinker_replace_token(lines, before.ff_row, 4, after.equilibrium)
    for before, after in _tinker_template_pairs(
        original.angles, ff.angles, lambda p: p.env_id, ("force_constant", "equilibrium")
    ):
        assert before.ff_row is not None
        if before.force_constant != after.force_constant:
            _tinker_replace_token(lines, before.ff_row, 4, canonical_to_mm3_angle_k(after.force_constant))
        if before.equilibrium != after.equilibrium:
            _tinker_replace_token(lines, before.ff_row, 5, _normalize_equilibrium_angle(after.equilibrium))
    for before, after in _tinker_template_pairs(
        original.torsions, ff.torsions, lambda p: (p.env_id, p.periodicity), ("force_constant", "phase")
    ):
        assert before.ff_row is not None
        terms = _tinker_torsion_terms(_tinker_tokens(lines[before.ff_row - 1]), before.ff_row)
        slot = next(index for index, term in enumerate(terms) if term[2] == before.periodicity)
        if before.force_constant != after.force_constant:
            amplitude = after.force_constant / unit
            if after.force_constant != 0.0 and amplitude == 0.0:
                raise ValueError(f"Tinker row {before.ff_row}: torsion scaling underflow")
            _tinker_replace_token(lines, before.ff_row, 5 + 3 * slot, amplitude)
            serialized = _tinker_tokens(lines[before.ff_row - 1])[5 + 3 * slot]
            _tinker_scaled_torsion(_tinker_float(serialized, before.ff_row), unit, before.ff_row)
        if before.phase != after.phase:
            _tinker_replace_token(lines, before.ff_row, 6 + 3 * slot, after.phase)
    for before, after in _tinker_template_pairs(
        original.vdws, ff.vdws, lambda p: p.atom_type, ("radius", "epsilon", "reduction")
    ):
        assert before.ff_row is not None
        for column, field in enumerate(("radius", "epsilon", "reduction"), start=2):
            if getattr(before, field) != getattr(after, field):
                _tinker_replace_token(lines, before.ff_row, column, getattr(after, field))
    return lines


def save_tinker_prm(
    ff: ForceField,
    path: str | Path,
    template_path: str | Path | None = None,
    *,
    section_name: str = "Generated",
) -> Path:
    """Write the force field to Tinker .prm format.

    If a template path is provided, or this force field came from
    :func:`load_tinker_prm`, only changed scalar tokens are replaced.
    Unmodified bytes, comments and extra angle equilibria are preserved.
    Source rows disambiguate repeated environments; missing or ambiguous
    bindings, additions/removals and non-scalar edits (including changing
    a torsion fold) fail before output is opened. Torsion phase edits are
    supported, and amplitudes are divided by the template's torsionunit.
    Serialized amplitudes must reconstruct finite coefficients without
    underflow; required data must fit Tinker's 240-byte records. Comment
    tails beginning with # or ! may extend beyond that limit.
    Otherwise, a minimal Q2MM bond/angle/vdW section is written; proper
    torsions require a template. This is not a complete Tinker FF writer.
    """
    _validate_form_for_format(ff, "tinker_prm")
    output_path = Path(path)
    template = Path(template_path) if template_path is not None else None
    if template is None and ff.source_format == "tinker_prm" and ff.source_path is not None:
        template = ff.source_path

    if template is not None:
        lines = _tinker_template_lines(ff, template)
        _validate_tinker_record_lengths(lines)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
        return output_path

    if ff.torsions:
        raise ValueError("Tinker torsion export requires a source template")
    lines = ["# Q2MM\n", f"# OPT {section_name}\n"]
    for bond in ff.bonds:
        lines.append(
            _format_tinker_bond_line(
                _tinker_atom_types(bond.env_id, bond.elements),
                canonical_to_mm3_bond_k(bond.force_constant),
                bond.equilibrium,
            )
        )
    for angle in ff.angles:
        lines.append(
            _format_tinker_angle_line(
                _tinker_atom_types(angle.env_id, angle.elements),
                canonical_to_mm3_angle_k(angle.force_constant),
                angle.equilibrium,
            )
        )
    for vdw in ff.vdws:
        lines.append(_format_tinker_vdw_line(vdw.atom_type, vdw.radius, vdw.epsilon, vdw.reduction))
    _validate_tinker_record_lengths(lines)
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
