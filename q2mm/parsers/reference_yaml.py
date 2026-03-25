"""YAML-based reference data file format for Q2MM.

Provides :func:`load_reference_yaml` and :func:`save_reference_yaml` for a
human-friendly YAML schema that describes molecules and their reference
observables (energies, geometries, frequencies, eigenmatrix data).

Example YAML
-------------
.. code-block:: yaml

    molecules:
      - name: ch3f
        xyz: ch3f-optimized.xyz
        charge: 0
        multiplicity: 1
        data:
          - kind: energy
            value: -139.7621
            weight: 1.0
          - kind: bond_length
            atoms: [0, 1]
            value: 1.383
            weight: 10.0

Paths to external files (``xyz``, ``hessian``) are resolved relative to
the YAML file's directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from q2mm.models.molecule import Q2MMMolecule
from q2mm.optimizers.objective import ReferenceData, ReferenceValue

# Reference-value kinds accepted by the schema.
# Note: ``eigenmatrix`` is also supported as a special bulk-loading directive
# handled directly in :func:`_load_molecule` (not via :func:`_parse_datum`).
_VALID_KINDS: frozenset[str] = frozenset(
    {
        "energy",
        "frequency",
        "bond_length",
        "bond_angle",
        "torsion_angle",
        "eig_diagonal",
        "eig_offdiagonal",
    }
)

# Minimum atom counts required for geometry kinds.
_ATOMS_REQUIRED: dict[str, int] = {
    "bond_length": 2,
    "bond_angle": 3,
    "torsion_angle": 4,
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class ReferenceYAMLError(ValueError):
    """Raised for invalid or malformed reference YAML files."""


def _require_key(mapping: dict[str, Any], key: str, context: str) -> Any:
    """Return *mapping[key]* or raise with a human-friendly message."""
    if key not in mapping:
        raise ReferenceYAMLError(f"Missing required key '{key}' in {context}.")
    return mapping[key]


def _validate_kind(kind: str, context: str) -> str:
    """Validate that *kind* is one of the accepted reference value types."""
    if kind not in _VALID_KINDS:
        raise ReferenceYAMLError(f"Unknown kind '{kind}' in {context}. Must be one of: {sorted(_VALID_KINDS)}")
    return kind


def _as_float(value: Any, name: str, context: str) -> float:
    """Coerce *value* to float with a clear error on failure."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ReferenceYAMLError(f"'{name}' must be a number in {context}, got {value!r}.") from exc


def _as_int(value: Any, name: str, context: str) -> int:
    """Coerce *value* to int with a clear error on failure."""
    if isinstance(value, bool):
        raise ReferenceYAMLError(f"'{name}' must be an integer in {context}, got {value!r}.")
    if isinstance(value, float):
        if value != int(value):
            raise ReferenceYAMLError(f"'{name}' must be an integer in {context}, got {value!r}.")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ReferenceYAMLError(f"'{name}' must be an integer in {context}, got {value!r}.") from exc


def _as_int_list(value: Any, name: str, context: str) -> list[int]:
    """Coerce *value* to a list of ints with per-element validation."""
    if not isinstance(value, list):
        raise ReferenceYAMLError(f"'{name}' must be a list of integers in {context}, got {type(value).__name__}.")
    return [_as_int(v, f"{name}[{idx}]", context) for idx, v in enumerate(value)]


def _as_float_list(value: Any, name: str, context: str) -> list[float]:
    """Coerce *value* to a list of floats."""
    if not isinstance(value, list):
        raise ReferenceYAMLError(f"'{name}' must be a list of numbers in {context}, got {type(value).__name__}.")
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ReferenceYAMLError(f"'{name}' must be a list of numbers in {context}.") from exc


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _parse_datum(
    datum: dict[str, Any],
    molecule_idx: int,
    context: str,
) -> list[ReferenceValue]:
    """Parse a single ``data`` entry into one or more :class:`ReferenceValue` objects.

    Args:
        datum: A single mapping from the ``data`` list.
        molecule_idx: Index of the parent molecule.
        context: Human-readable location for error messages.

    Returns:
        List of :class:`ReferenceValue` instances.

    """
    kind = _validate_kind(str(_require_key(datum, "kind", context)), context)
    weight = _as_float(datum.get("weight", 1.0), "weight", context)
    label = str(datum.get("label", ""))

    # ---- Bulk array kinds -------------------------------------------------
    if kind == "frequency" and "values" in datum:
        values = _as_float_list(datum["values"], "values", context)
        indices = datum.get("indices")
        if indices is not None:
            indices = _as_int_list(indices, "indices", context)
            if len(indices) != len(values):
                raise ReferenceYAMLError(
                    f"'indices' length ({len(indices)}) must match 'values' length ({len(values)}) in {context}."
                )
            if any(i < 0 for i in indices):
                raise ReferenceYAMLError(f"'indices' must be non-negative in {context}, got {indices}.")
        else:
            indices = list(range(len(values)))
        return [
            ReferenceValue(
                kind="frequency",
                value=v,
                weight=weight,
                label=label or f"mode {idx}",
                molecule_idx=molecule_idx,
                data_idx=idx,
            )
            for idx, v in zip(indices, values)
        ]

    # ---- Scalar kinds -----------------------------------------------------
    if kind in ("energy",):
        value = _as_float(_require_key(datum, "value", context), "value", context)
        return [
            ReferenceValue(
                kind=kind,
                value=value,
                weight=weight,
                label=label,
                molecule_idx=molecule_idx,
            )
        ]

    if kind in ("bond_length", "bond_angle", "torsion_angle"):
        value = _as_float(_require_key(datum, "value", context), "value", context)
        atoms_raw = datum.get("atoms")
        data_idx_raw = datum.get("data_idx")

        if atoms_raw is not None:
            atoms = _as_int_list(atoms_raw, "atoms", context)
            if any(a < 0 for a in atoms):
                raise ReferenceYAMLError(f"'atoms' indices must be non-negative in {context}, got {atoms}.")
            required = _ATOMS_REQUIRED[kind]
            if len(atoms) != required:
                raise ReferenceYAMLError(
                    f"'{kind}' requires exactly {required} atom indices, got {len(atoms)} in {context}."
                )
            return [
                ReferenceValue(
                    kind=kind,
                    value=value,
                    weight=weight,
                    label=label,
                    molecule_idx=molecule_idx,
                    atom_indices=tuple(atoms),
                )
            ]

        if data_idx_raw is not None:
            data_idx = _as_int(data_idx_raw, "data_idx", context)
            if data_idx < 0:
                raise ReferenceYAMLError(f"'data_idx' must be non-negative in {context}, got {data_idx}.")
            return [
                ReferenceValue(
                    kind=kind,
                    value=value,
                    weight=weight,
                    label=label,
                    molecule_idx=molecule_idx,
                    data_idx=data_idx,
                )
            ]

        raise ReferenceYAMLError(f"'{kind}' requires either 'atoms' or 'data_idx' in {context}.")

    if kind == "eig_diagonal":
        value = _as_float(_require_key(datum, "value", context), "value", context)
        mode_idx = _as_int(_require_key(datum, "mode_idx", context), "mode_idx", context)
        return [
            ReferenceValue(
                kind="eig_diagonal",
                value=value,
                weight=weight,
                label=label or f"eig[{mode_idx}]",
                molecule_idx=molecule_idx,
                data_idx=mode_idx,
            )
        ]

    if kind == "eig_offdiagonal":
        value = _as_float(_require_key(datum, "value", context), "value", context)
        row = _as_int(_require_key(datum, "row", context), "row", context)
        col = _as_int(_require_key(datum, "col", context), "col", context)
        return [
            ReferenceValue(
                kind="eig_offdiagonal",
                value=value,
                weight=weight,
                label=label or f"eig[{row},{col}]",
                molecule_idx=molecule_idx,
                atom_indices=(row, col),
            )
        ]

    if kind == "frequency":
        # Single frequency entry (no bulk 'values')
        value = _as_float(_require_key(datum, "value", context), "value", context)
        data_idx = _as_int(datum.get("data_idx", datum.get("index", 0)), "data_idx", context)
        if data_idx < 0:
            raise ReferenceYAMLError(f"'data_idx' must be non-negative in {context}, got {data_idx}.")
        return [
            ReferenceValue(
                kind="frequency",
                value=value,
                weight=weight,
                label=label or f"mode {data_idx}",
                molecule_idx=molecule_idx,
                data_idx=data_idx,
            )
        ]

    # Should not reach here since _validate_kind already checked.
    raise ReferenceYAMLError(f"Unhandled kind '{kind}' in {context}.")  # pragma: no cover


def _load_molecule(
    mol_dict: dict[str, Any],
    base_dir: Path,
    molecule_idx: int,
) -> tuple[Q2MMMolecule, list[ReferenceValue]]:
    """Parse one molecule entry from the YAML.

    Args:
        mol_dict: Mapping from the ``molecules`` list.
        base_dir: Directory of the YAML file for relative path resolution.
        molecule_idx: 0-based molecule index.

    Returns:
        A ``(molecule, ref_values)`` tuple.

    Raises:
        ReferenceYAMLError: If the molecule has no geometry source.

    """
    ctx = f"molecule[{molecule_idx}]"
    name = str(mol_dict.get("name", f"mol_{molecule_idx}"))
    charge = _as_int(mol_dict.get("charge", 0), "charge", ctx)
    multiplicity = _as_int(mol_dict.get("multiplicity", 1), "multiplicity", ctx)
    bond_tolerance = _as_float(mol_dict.get("bond_tolerance", 1.3), "bond_tolerance", ctx)

    # ---- Geometry (required) ----------------------------------------------
    if "xyz" not in mol_dict and "geometry" not in mol_dict:
        raise ReferenceYAMLError(
            f"Molecule '{name}' at index {molecule_idx} has no geometry ('xyz' or 'geometry' key required)."
        )

    mol: Q2MMMolecule
    if "xyz" in mol_dict:
        xyz_path = base_dir / mol_dict["xyz"]
        if not xyz_path.exists():
            raise ReferenceYAMLError(f"XYZ file not found: {xyz_path} (referenced in {ctx}).")
        mol = Q2MMMolecule.from_xyz(
            xyz_path,
            charge=charge,
            multiplicity=multiplicity,
            name=name,
            bond_tolerance=bond_tolerance,
        )
    else:  # "geometry" in mol_dict
        geo = mol_dict["geometry"]
        if not isinstance(geo, dict):
            raise ReferenceYAMLError(f"'geometry' must be a mapping in {ctx}, got {type(geo).__name__}.")
        symbols_raw = _require_key(geo, "symbols", f"{ctx}.geometry")
        if not isinstance(symbols_raw, list):
            raise ReferenceYAMLError(
                f"'symbols' must be a list of element symbols, not a {type(symbols_raw).__name__} in {ctx}.geometry."
            )
        symbols = [str(s) for s in symbols_raw]
        coords = _require_key(geo, "coordinates", f"{ctx}.geometry")
        try:
            coords_arr = np.array(coords, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ReferenceYAMLError(f"'coordinates' contains non-numeric entries in {ctx}.geometry: {exc}") from exc
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
            raise ReferenceYAMLError(
                f"'coordinates' must be an Nx3 array in {ctx}.geometry, got shape {coords_arr.shape}."
            )
        if len(symbols) != coords_arr.shape[0]:
            raise ReferenceYAMLError(
                f"Number of symbols ({len(symbols)}) must match rows in "
                f"coordinates ({coords_arr.shape[0]}) in {ctx}.geometry."
            )
        atom_types = geo.get("atom_types")
        if atom_types is not None:
            if not isinstance(atom_types, list):
                raise ReferenceYAMLError(
                    f"'atom_types' must be a list of type labels, not a {type(atom_types).__name__} in {ctx}.geometry."
                )
            atom_types = [str(t) for t in atom_types]
        mol = Q2MMMolecule(
            symbols=symbols,
            geometry=coords_arr,
            atom_types=atom_types,
            charge=charge,
            multiplicity=multiplicity,
            name=name,
            bond_tolerance=bond_tolerance,
        )

    # ---- Hessian ----------------------------------------------------------
    if "hessian" in mol_dict:
        hessian_path = base_dir / mol_dict["hessian"]
        if not hessian_path.exists():
            raise ReferenceYAMLError(f"Hessian file not found: {hessian_path} (referenced in {ctx}).")
        hessian = np.load(str(hessian_path))
        mol = mol.with_hessian(hessian)

    # ---- Data entries -----------------------------------------------------
    ref_values: list[ReferenceValue] = []
    data_list = mol_dict.get("data", [])
    if not isinstance(data_list, list):
        raise ReferenceYAMLError(f"'data' must be a list in {ctx}, got {type(data_list).__name__}.")

    for i, datum in enumerate(data_list):
        if not isinstance(datum, dict):
            raise ReferenceYAMLError(f"data[{i}] must be a mapping in {ctx}, got {type(datum).__name__}.")
        datum_ctx = f"{ctx}.data[{i}]"

        # Handle bulk eigenmatrix loading from hessian
        if datum.get("kind") == "eigenmatrix":
            if mol.hessian is None:
                raise ReferenceYAMLError(f"'eigenmatrix' data requires a molecule with a hessian in {datum_ctx}.")
            diag_weight = _as_float(datum.get("diagonal_weight", 0.1), "diagonal_weight", datum_ctx)
            offdiag_weight = _as_float(datum.get("offdiagonal_weight", 0.05), "offdiagonal_weight", datum_ctx)
            skip_first = datum.get("skip_first", True)
            diagonal_only = datum.get("diagonal_only", False)
            temp_ref = ReferenceData()
            temp_ref.add_eigenmatrix_from_hessian(
                mol.hessian,
                diagonal_only=diagonal_only,
                molecule_idx=molecule_idx,
                weights={
                    "eig_i": 0.0 if skip_first else diag_weight,
                    "eig_d_low": diag_weight,
                    "eig_d_high": diag_weight,
                    "eig_o": offdiag_weight,
                },
                skip_first=bool(skip_first),
            )
            ref_values.extend(temp_ref.values)
            continue

        ref_values.extend(_parse_datum(datum, molecule_idx, datum_ctx))

    return mol, ref_values


def load_reference_yaml(path: str | Path) -> tuple[ReferenceData, list[Q2MMMolecule]]:
    """Load reference data and molecules from a YAML file.

    Args:
        path: Path to the YAML reference file.

    Returns:
        A ``(reference_data, molecules)`` tuple.

    Raises:
        ReferenceYAMLError: On malformed YAML or missing required fields.
        FileNotFoundError: If *path* does not exist.

    """
    if yaml is None:
        raise ImportError("pyyaml is required for YAML support: pip install pyyaml")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference YAML file not found: {path}")

    with open(path, encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ReferenceYAMLError(f"Failed to parse YAML file {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ReferenceYAMLError(f"Top-level YAML must be a mapping, got {type(raw).__name__}.")

    if "molecules" not in raw:
        raise ReferenceYAMLError("YAML file must contain a top-level 'molecules' key")
    mol_list = raw["molecules"]
    if not isinstance(mol_list, list):
        raise ReferenceYAMLError(f"'molecules' must be a list, got {type(mol_list).__name__}.")

    base_dir = path.parent
    molecules: list[Q2MMMolecule] = []
    all_values: list[ReferenceValue] = []

    for idx, mol_dict in enumerate(mol_list):
        if not isinstance(mol_dict, dict):
            raise ReferenceYAMLError(f"molecules[{idx}] must be a mapping, got {type(mol_dict).__name__}.")
        mol, values = _load_molecule(mol_dict, base_dir, molecule_idx=idx)
        molecules.append(mol)
        all_values.extend(values)

    return ReferenceData(values=all_values), molecules


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _reference_value_to_dict(rv: ReferenceValue) -> dict[str, Any]:
    """Convert a single :class:`ReferenceValue` to a YAML-friendly dict.

    Args:
        rv: The reference value to serialise.

    Returns:
        Dictionary suitable for YAML output.

    """
    d: dict[str, Any] = {"kind": rv.kind, "value": float(rv.value)}

    if rv.weight != 1.0:
        d["weight"] = float(rv.weight)
    if rv.label:
        d["label"] = rv.label

    if rv.kind in ("bond_length", "bond_angle", "torsion_angle"):
        if rv.atom_indices is not None:
            d["atoms"] = list(rv.atom_indices)
        else:
            d["data_idx"] = rv.data_idx
    elif rv.kind == "frequency":
        d["data_idx"] = rv.data_idx
    elif rv.kind == "eig_diagonal":
        d["mode_idx"] = rv.data_idx
    elif rv.kind == "eig_offdiagonal":
        if rv.atom_indices is not None:
            d["row"] = rv.atom_indices[0]
            d["col"] = rv.atom_indices[1]
        else:
            raise ReferenceYAMLError("eig_offdiagonal requires row/col (atom_indices)")

    return d


def _molecule_to_dict(mol: Q2MMMolecule) -> dict[str, Any]:
    """Convert a :class:`Q2MMMolecule` to a YAML-friendly dict (inline geometry).

    Args:
        mol: The molecule to serialise.

    Returns:
        Dictionary suitable for YAML output.

    """
    d: dict[str, Any] = {"name": mol.name}

    # Inline geometry — always serialise coordinates inline.
    geo: dict[str, Any] = {
        "symbols": list(mol.symbols),
        "coordinates": mol.geometry.tolist(),
    }
    if mol.atom_types is not None and mol.atom_types != mol.symbols:
        geo["atom_types"] = list(mol.atom_types)
    d["geometry"] = geo

    if mol.charge != 0:
        d["charge"] = mol.charge
    if mol.multiplicity != 1:
        d["multiplicity"] = mol.multiplicity
    if mol.bond_tolerance != 1.3:
        d["bond_tolerance"] = mol.bond_tolerance

    return d


def save_reference_yaml(
    path: str | Path,
    ref: ReferenceData,
    molecules: list[Q2MMMolecule],
) -> None:
    """Save reference data and molecules to a YAML file.

    Each molecule's geometry is serialised inline. Reference values are
    grouped under their parent molecule's ``data`` list.

    Args:
        path: Output file path.
        ref: Reference data to save.
        molecules: Molecules corresponding to the reference data.

    """
    if yaml is None:
        raise ImportError("pyyaml is required for YAML support: pip install pyyaml")
    path = Path(path)

    # Group reference values by molecule_idx.
    grouped: dict[int, list[dict[str, Any]]] = {}
    for rv in ref.values:
        grouped.setdefault(rv.molecule_idx, []).append(_reference_value_to_dict(rv))

    mol_dicts: list[dict[str, Any]] = []
    for idx, mol in enumerate(molecules):
        mol_d = _molecule_to_dict(mol)
        data = grouped.get(idx, [])
        if data:
            mol_d["data"] = data
        mol_dicts.append(mol_d)

    # Reject orphaned molecule indices rather than writing invalid placeholders.
    orphaned = sorted(idx for idx in grouped if idx >= len(molecules))
    if orphaned:
        raise ReferenceYAMLError(
            f"Reference values point to molecule indices {orphaned} but only {len(molecules)} molecule(s) provided."
        )

    output: dict[str, Any] = {"molecules": mol_dicts}

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(output, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
