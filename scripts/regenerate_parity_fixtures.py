#!/usr/bin/env python3
"""Regenerate legacy Seminario parity fixtures from an upstream worktree."""

import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import logging.config
from pathlib import Path
import subprocess
import sys
import types
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKTREE = REPO_ROOT.parent / f"{REPO_ROOT.name}-upstream-worktree"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "test" / "fixtures" / "seminario_parity"
DFT_SCALING = 0.963


def _git_stdout(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _load_upstream_modules(worktree_root: Path) -> dict[str, Any]:
    if not (worktree_root / "q2mm" / "seminario.py").exists():
        raise FileNotFoundError(f"Upstream worktree does not contain q2mm/seminario.py: {worktree_root}")

    for module_name in list(sys.modules):
        if (
            module_name == "q2mm"
            or module_name.startswith("q2mm.")
            or module_name in {"constants", "utilities", "linear_algebra"}
        ):
            del sys.modules[module_name]

    sys.path.insert(0, str(worktree_root))
    sys.path.insert(0, str(worktree_root / "q2mm"))
    if importlib.util.find_spec("parmed") is None and "parmed" not in sys.modules:
        sys.modules["parmed"] = types.ModuleType("parmed")
    return {
        "constants": importlib.import_module("q2mm.constants"),
        "filetypes": importlib.import_module("q2mm.filetypes"),
        "seminario": importlib.import_module("q2mm.seminario"),
    }


def _build_sn2_atoms(atom_cls: type, symbols: list[str], coords: np.ndarray) -> list[Any]:
    atomic_numbers = {"H": 1, "C": 6, "F": 9}
    atomic_masses = {"H": 1.008, "C": 12.0, "F": 19.0}

    atoms = []
    for index, (symbol, xyz) in enumerate(zip(symbols, coords), start=1):
        atom = atom_cls.__new__(atom_cls)
        atom.index = index
        atom.coords = np.array(xyz, dtype=float)
        atom.x, atom.y, atom.z = xyz
        atom.element = symbol
        atom.atomic_num = atomic_numbers[symbol]
        atom.atomic_mass = atomic_masses[symbol]
        atoms.append(atom)
    return atoms


def _generate_sn2_fixture(modules: dict[str, Any], upstream_commit: str) -> dict[str, Any]:
    qm_ref = REPO_ROOT / "examples" / "sn2-test" / "qm-reference"
    xyz_path = qm_ref / "sn2-ts-optimized.xyz"
    hessian_path = qm_ref / "sn2-ts-hessian.npy"

    with xyz_path.open() as handle:
        lines = handle.readlines()
    n_atoms = int(lines[0].strip())
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(value) for value in parts[1:4]])

    coordinates = np.array(coords, dtype=float)
    hessian = np.load(str(hessian_path))
    atom_cls = modules["filetypes"].Atom
    legacy_atoms = _build_sn2_atoms(atom_cls, symbols, coordinates)
    legacy_bond = modules["seminario"].seminario_bond
    constants = modules["constants"]

    bonds = []
    for atom_i, atom_j in [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]:
        legacy_au = legacy_bond(
            atoms=[legacy_atoms[atom_i], legacy_atoms[atom_j]],
            hessian=hessian,
            ang_to_bohr=True,
            scaling=DFT_SCALING,
        )
        bonds.append(
            {
                "atom_i": atom_i,
                "atom_j": atom_j,
                "label": f"{symbols[atom_i]}{atom_i + 1}-{symbols[atom_j]}{atom_j + 1}",
                "elements": [symbols[atom_i], symbols[atom_j]],
                "legacy_force_constant_au": float(legacy_au),
                "legacy_force_constant_mdyn_a": float(legacy_au * constants.AU_TO_MDYNA),
            }
        )

    return {
        "metadata": {
            "system": "sn2-test",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "scripts/regenerate_parity_fixtures.py",
            "upstream_commit": upstream_commit,
            "xyz_path": _relative(xyz_path),
            "hessian_path": _relative(hessian_path),
            "dft_scaling": DFT_SCALING,
        },
        "bonds": bonds,
    }


def _generate_rh_fixture(modules: dict[str, Any], upstream_commit: str) -> dict[str, Any]:
    filetypes = modules["filetypes"]
    seminario = modules["seminario"]
    constants = modules["constants"]

    rh_dir = REPO_ROOT / "examples" / "rh-enamide"
    training_dir = rh_dir / "rh_enamide_training_set"
    mm3_path = rh_dir / "mm3.fld"
    mmo_path = training_dir / "rh_enamide_training_set.mmo"
    jag_dir = training_dir / "jaguar_spe_freq_in_out"
    mol2_path = training_dir / "mol2" / "1_zdmp.mol2"
    first_hessian_path = jag_dir / "1ZDMPfromJCTCSI_loner1.01.in"

    legacy_ff = filetypes.MM3(str(mm3_path))
    legacy_ff.import_ff()
    structures = filetypes.MacroModel(str(mmo_path)).structures
    hessian_files = sorted(jag_dir.glob("*.in"))
    if len(structures) != len(hessian_files):
        raise ValueError(f"Structure/hessian mismatch: {len(structures)} structures vs {len(hessian_files)} Hessians")

    hessians = [
        filetypes.JaguarIn(str(path)).get_hessian(len(structure.atoms))
        for structure, path in zip(structures, hessian_files)
    ]

    bond_force_constants_au: dict[str, float | None] = {}
    bond_force_constants_mdyn_a: dict[str, float | None] = {}
    bond_equilibria_angstrom: dict[str, float] = {}
    angle_force_constants_au: dict[str, float | None] = {}
    angle_force_constants_mdyn_a_rad2: dict[str, float | None] = {}
    angle_equilibria_degrees: dict[str, float] = {}

    for param in legacy_ff.params:
        row_key = str(param.ff_row)
        if param.ptype == "bf":
            value = seminario.estimate_bf_param(
                param,
                structures,
                hessians,
                ang_to_bohr=True,
            )
            bond_force_constants_au[row_key] = None if value is None else float(value)
            bond_force_constants_mdyn_a[row_key] = None if value is None else float(value * constants.AU_TO_MDYNA)
        elif param.ptype == "be":
            bond_equilibria_angstrom[row_key] = float(seminario.average_be_param(param, structures))
        elif param.ptype == "af":
            value = seminario.estimate_af_param(
                param,
                structures,
                hessians,
                ang_to_bohr=True,
            )
            angle_force_constants_au[row_key] = None if value is None else float(value)
            angle_force_constants_mdyn_a_rad2[row_key] = (
                None if value is None else float(value * constants.AU_TO_MDYN_ANGLE)
            )
        elif param.ptype == "ae":
            angle_equilibria_degrees[row_key] = float(seminario.average_ae_param(param, structures))

    first_hessian = filetypes.JaguarIn(str(first_hessian_path)).get_hessian(36)
    first_structure = filetypes.Mol2(str(mol2_path)).structures[0]
    direct_bonds = []
    for bond in first_structure.bonds:
        atom_i = bond.atom_nums[0] - 1
        atom_j = bond.atom_nums[1] - 1
        legacy_au = seminario.seminario_bond(
            atoms=[first_structure.atoms[atom_i], first_structure.atoms[atom_j]],
            hessian=first_hessian,
            ang_to_bohr=True,
            scaling=DFT_SCALING,
        )
        direct_bonds.append(
            {
                "atom_i": atom_i,
                "atom_j": atom_j,
                "label": (
                    f"{first_structure.atoms[atom_i].element}{atom_i + 1}-"
                    f"{first_structure.atoms[atom_j].element}{atom_j + 1}"
                ),
                "legacy_force_constant_au": float(legacy_au),
                "legacy_force_constant_mdyn_a": float(legacy_au * constants.AU_TO_MDYNA),
            }
        )

    return {
        "metadata": {
            "system": "rh-enamide",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "scripts/regenerate_parity_fixtures.py",
            "upstream_commit": upstream_commit,
            "mm3_path": _relative(mm3_path),
            "mmo_path": _relative(mmo_path),
            "mol2_path": _relative(mol2_path),
            "hessian_dir": _relative(jag_dir),
            "direct_hessian_path": _relative(first_hessian_path),
            "dft_scaling": DFT_SCALING,
            "structure_count": len(structures),
            "structure_names": [structure.origin_name for structure in structures],
            "hessian_files": [path.name for path in hessian_files],
        },
        "parameters": {
            "bond_force_constants_au": bond_force_constants_au,
            "bond_force_constants_mdyn_a": bond_force_constants_mdyn_a,
            "bond_equilibria_angstrom": bond_equilibria_angstrom,
            "angle_force_constants_au": angle_force_constants_au,
            "angle_force_constants_mdyn_a_rad2": angle_force_constants_mdyn_a_rad2,
            "angle_equilibria_degrees": angle_equilibria_degrees,
        },
        "direct_bonds": direct_bonds,
    }


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _map_diff(old_map: dict[str, Any], new_map: dict[str, Any]) -> tuple[int, float]:
    changed = 0
    max_abs_diff = 0.0
    for key in sorted(set(old_map) | set(new_map)):
        old_value = old_map.get(key)
        new_value = new_map.get(key)
        if old_value != new_value:
            changed += 1
            if old_value is not None and new_value is not None:
                max_abs_diff = max(max_abs_diff, abs(float(new_value) - float(old_value)))
    return changed, max_abs_diff


def _sn2_diff_summary(old: dict[str, Any], new: dict[str, Any]) -> str:
    old_bonds = {f"{item['atom_i']}-{item['atom_j']}": item for item in old.get("bonds", [])}
    new_bonds = {f"{item['atom_i']}-{item['atom_j']}": item for item in new.get("bonds", [])}
    changed = 0
    max_abs_diff = 0.0
    for key in sorted(set(old_bonds) | set(new_bonds)):
        old_value = old_bonds.get(key, {}).get("legacy_force_constant_mdyn_a")
        new_value = new_bonds.get(key, {}).get("legacy_force_constant_mdyn_a")
        if old_value != new_value:
            changed += 1
            if old_value is not None and new_value is not None:
                max_abs_diff = max(max_abs_diff, abs(float(new_value) - float(old_value)))
    return f"{changed} bond entries changed; max |Δ| = {max_abs_diff:.6g}"


def _rh_diff_summary(old: dict[str, Any], new: dict[str, Any]) -> str:
    summaries = []
    for label, key in (
        ("bond FC", "bond_force_constants_mdyn_a"),
        ("bond eq", "bond_equilibria_angstrom"),
        ("angle FC", "angle_force_constants_mdyn_a_rad2"),
        ("angle eq", "angle_equilibria_degrees"),
    ):
        changed, max_abs_diff = _map_diff(
            old.get("parameters", {}).get(key, {}),
            new.get("parameters", {}).get(key, {}),
        )
        summaries.append(f"{label}: {changed} changed, max |Δ| = {max_abs_diff:.6g}")
    return "; ".join(summaries)


def _write_fixture(path: Path, payload: dict[str, Any], system_name: str) -> None:
    previous = _load_json_if_exists(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    try:
        display_path = path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = path
    if previous is None:
        print(f"[create] {display_path}")
        return
    if system_name == "sn2":
        print(f"[update] {display_path} :: {_sn2_diff_summary(previous, payload)}")
    else:
        print(f"[update] {display_path} :: {_rh_diff_summary(previous, payload)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worktree",
        type=Path,
        default=DEFAULT_WORKTREE,
        help="Path to the upstream worktree containing legacy q2mm code.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where fixture JSON files should be written.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=("rh-enamide", "sn2"),
        default=("rh-enamide", "sn2"),
        help="Fixture sets to regenerate.",
    )
    args = parser.parse_args(argv)

    worktree_root = args.worktree.resolve()
    upstream_commit = _git_stdout("git", "-C", str(worktree_root), "rev-parse", "HEAD")
    modules = _load_upstream_modules(worktree_root)

    if "rh-enamide" in args.systems:
        rh_fixture = _generate_rh_fixture(modules, upstream_commit)
        _write_fixture(args.output_dir / "rh_enamide_reference.json", rh_fixture, "rh-enamide")

    if "sn2" in args.systems:
        sn2_fixture = _generate_sn2_fixture(modules, upstream_commit)
        _write_fixture(args.output_dir / "sn2_reference.json", sn2_fixture, "sn2")

    print(f"Using upstream worktree: {worktree_root}")
    print(f"Pinned upstream commit: {upstream_commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
