"""Explicit file-format bridges to canonical immutable molecules."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from q2mm.constants import DEFAULT_BOND_TOLERANCE
from q2mm.models.hessian import HessianProvenance, HessianUnits
from q2mm.models.molecule import Molecule

PathLike = str | Path


def _paths(values: Sequence[PathLike]) -> tuple[Path, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("paths must be a sequence of paths, not one path string.")
    paths = tuple(Path(value) for value in values)
    if not paths:
        raise ValueError("paths must contain at least one input.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files do not exist: {missing}")
    return paths


def _select(molecules: Sequence[Molecule], structure_index: int, path: Path) -> Molecule:
    if not isinstance(structure_index, int) or isinstance(structure_index, bool):
        raise TypeError("structure_index must be an integer.")
    if not molecules:
        raise ValueError(f"No structures were parsed from {path}.")
    if not -len(molecules) <= structure_index < len(molecules):
        raise IndexError(
            f"structure_index={structure_index} is out of range for {path.name} ({len(molecules)} parsed structures)."
        )
    return molecules[structure_index]


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _parser_version() -> str | None:
    try:
        return version("q2mm")
    except PackageNotFoundError:
        return None


def _with_hessian_provenance(
    molecule: Molecule,
    path: Path,
    *,
    source: str,
    parser: str,
    source_units: str,
    conversion: str,
) -> Molecule:
    if molecule.hessian is None:
        return molecule
    details: dict[str, object] = {
        "parser": parser,
        "source_units": source_units,
        "canonical_units": HessianUnits.ATOMIC.value,
        "conversion": conversion,
        "file_content_sha256": _content_sha256(path),
    }
    parser_version = _parser_version()
    if parser_version is not None:
        details["parser_version"] = parser_version
    return molecule.with_hessian(
        molecule.hessian,
        HessianProvenance(
            units=HessianUnits.ATOMIC,
            source=source,
            path=path.name,
            source_details=details,
        ),
    )


def load_fchk_molecule(
    path: PathLike,
    *,
    bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
) -> Molecule:
    """Load one Gaussian FCHK molecule with canonical Hessian provenance."""
    from q2mm.io.fchk import load_fchk

    resolved = Path(path)
    molecule = load_fchk(resolved, bond_tolerance=bond_tolerance)
    return _with_hessian_provenance(
        molecule,
        resolved,
        source="fchk",
        parser="q2mm.io.fchk.load_fchk",
        source_units=HessianUnits.ATOMIC.value,
        conversion="identity",
    )


def load_gaussian_molecules(
    paths: Sequence[PathLike],
    *,
    structure_index: int,
    require_hessian: bool = True,
    bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
) -> tuple[Molecule, ...]:
    """Load one explicitly selected structure from every Gaussian log."""
    from q2mm.io.gaussian import GaussLog

    staged: list[tuple[Path, Molecule]] = []
    for path in _paths(paths):
        try:
            parser = GaussLog(str(path), au_hessian=True)
            parser.read_archives()
            parsed = tuple(parser.molecules)
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Could not parse Gaussian structures from {path.name}: {exc}") from exc
        molecule = _select(parsed, structure_index, path).with_overrides(bond_tolerance=bond_tolerance)
        if require_hessian and molecule.hessian is None:
            raise ValueError(f"Selected Gaussian structure {structure_index} in {path.name} has no Hessian.")
        staged.append((path, molecule))
    return tuple(
        _with_hessian_provenance(
            molecule,
            path,
            source="gaussian",
            parser="q2mm.io.gaussian.GaussLog",
            source_units=HessianUnits.ATOMIC.value,
            conversion="identity (GaussLog au_hessian=True)",
        )
        for path, molecule in staged
    )


def load_jaguar_molecules(
    paths: Sequence[PathLike],
    *,
    structure_index: int,
    require_hessian: bool = True,
    bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
) -> tuple[Molecule, ...]:
    """Load one explicitly selected structure from every Jaguar input/output."""
    from q2mm.io.jaguar import JaguarIn, JaguarOut

    staged: list[tuple[Path, Molecule, str]] = []
    for path in _paths(paths):
        suffix = path.suffix.lower()
        if suffix == ".in":
            parser = JaguarIn(str(path))
            molecule = _select(tuple(parser.molecules), structure_index, path)
            try:
                hessian = parser.get_hessian(molecule.n_atoms)
            except ValueError as exc:
                if require_hessian:
                    raise ValueError(
                        f"Selected Jaguar structure {structure_index} in {path.name} has no complete Hessian."
                    ) from exc
            else:
                molecule = molecule.with_hessian(
                    hessian,
                    HessianProvenance(units=HessianUnits.ATOMIC, source="jaguar", path=path.name),
                )
            parser_identity = "q2mm.io.jaguar.JaguarIn"
        elif suffix == ".out":
            parser_out = JaguarOut(str(path))
            molecule = _select(tuple(parser_out.molecules), structure_index, path)
            if require_hessian:
                raise ValueError(
                    f"Jaguar output {path.name} does not expose a full Cartesian Hessian; "
                    "use the corresponding Jaguar .in file or set require_hessian=False."
                )
            parser_identity = "q2mm.io.jaguar.JaguarOut"
        else:
            raise ValueError(f"Jaguar bridge requires .in or .out files, got {path.name!r}.")
        molecule = molecule.with_overrides(bond_tolerance=bond_tolerance)
        if require_hessian and molecule.hessian is None:
            raise ValueError(f"Selected Jaguar structure {structure_index} in {path.name} has no Hessian.")
        staged.append((path, molecule, parser_identity))
    return tuple(
        _with_hessian_provenance(
            molecule,
            path,
            source="jaguar",
            parser=parser_identity,
            source_units=HessianUnits.ATOMIC.value,
            conversion="identity",
        )
        for path, molecule, parser_identity in staged
    )


def load_macromodel_molecules(
    paths: Sequence[PathLike],
    *,
    structure_index: int,
    bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
) -> tuple[Molecule, ...]:
    """Load one explicitly selected structure from every MacroModel MMO file."""
    from q2mm.io.macromodel import MacroModel

    staged: list[Molecule] = []
    for path in _paths(paths):
        molecule = _select(tuple(MacroModel(str(path)).molecules), structure_index, path)
        staged.append(molecule.with_overrides(bond_tolerance=bond_tolerance))
    return tuple(staged)


__all__ = [
    "load_fchk_molecule",
    "load_gaussian_molecules",
    "load_jaguar_molecules",
    "load_macromodel_molecules",
]
