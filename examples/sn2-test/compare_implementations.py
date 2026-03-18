"""Compare SN2 bond projections against the pinned upstream parity fixture."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import seminario_bond_fc

FIXTURE_PATH = REPO_ROOT / "test" / "fixtures" / "seminario_parity" / "sn2_reference.json"
QM_REF = Path(__file__).parent / "qm-reference"
XYZ_PATH = QM_REF / "sn2-ts-optimized.xyz"
HESSIAN_PATH = QM_REF / "sn2-ts-hessian.npy"


def main(title: str = "Fixture-backed SN2 Seminario Comparison") -> int:
    fixture = json.loads(FIXTURE_PATH.read_text())
    molecule = Q2MMMolecule.from_xyz(XYZ_PATH, name="sn2_ts", bond_tolerance=1.5)
    hessian = np.load(str(HESSIAN_PATH))
    scaling = float(fixture["metadata"]["dft_scaling"])

    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Pinned upstream commit: {fixture['metadata']['upstream_commit']}")
    print(f"System: {molecule.n_atoms} atoms ({', '.join(molecule.symbols)})")
    print(f"Hessian: {hessian.shape}")
    print()
    print(f"{'Bond':<10} {'Fixture':>15} {'New':>15} {'Diff':>12} {'Match':>8}")
    print("-" * 70)

    all_match = True
    max_diff = 0.0
    for bond in fixture["bonds"]:
        actual = seminario_bond_fc(
            bond["atom_i"],
            bond["atom_j"],
            molecule.geometry,
            hessian,
            au_units=True,
            dft_scaling=scaling,
        )
        expected = float(bond["legacy_force_constant_mdyn_a"])
        diff = abs(actual - expected)
        max_diff = max(max_diff, diff)
        match = "OK" if diff < 1e-8 else "DIFF"
        if diff >= 1e-8:
            all_match = False
        print(f"{bond['label']:<10} {expected:>15.8f} {actual:>15.8f} {diff:>12.3e} {match:>8}")

    print()
    if all_match:
        print(f"OK: all bond force constants match the pinned upstream fixture (max diff {max_diff:.3e}).")
    else:
        print(f"DIFF: at least one bond differs from the pinned upstream fixture (max diff {max_diff:.3e}).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
