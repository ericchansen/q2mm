"""Direct function-to-function comparison: upstream seminario_bond vs new seminario_bond_fc.

Calls both functions on the same inputs and compares outputs.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

QM_REF = Path(__file__).parent / "qm-reference"
hessian = np.load(str(QM_REF / "sn2-ts-hessian.npy"))

with open(QM_REF / "sn2-ts-optimized.xyz") as f:
    lines = f.readlines()
symbols, coords = [], []
for line in lines[2:2 + int(lines[0])]:
    p = line.split()
    symbols.append(p[0])
    coords.append([float(x) for x in p[1:4]])
coords = np.array(coords)

# Build old-style Atoms for upstream
from q2mm.schrod_indep_filetypes import Atom as OldAtom
old_atoms = []
for i, (sym, xyz) in enumerate(zip(symbols, coords)):
    atom = OldAtom.__new__(OldAtom)
    atom.index = i + 1
    atom.coords = np.array(xyz)
    atom.x, atom.y, atom.z = xyz
    atom.element = sym
    atom.atomic_num = {"C": 6, "H": 1, "F": 9}[sym]
    atom.atomic_mass = {"C": 12.0, "H": 1.008, "F": 19.0}[sym]
    old_atoms.append(atom)

# Import both implementations
from q2mm.seminario import seminario_bond as upstream_seminario_bond
from q2mm.models.seminario import seminario_bond_fc as new_seminario_bond_fc
from q2mm import constants as co

print("=" * 70)
print("FUNCTION-TO-FUNCTION COMPARISON")
print("upstream: q2mm.seminario.seminario_bond()")
print("new:      q2mm.models.seminario.seminario_bond_fc()")
print("=" * 70)

bond_pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]

print(f"\n{'Bond':<10} {'Upstream (AU)':>15} {'New (mdyn/A)':>15} {'Up->mdyn/A':>15} {'Diff':>10} {'Match?':>8}")
print("-" * 73)

all_match = True
for i, j in bond_pairs:
    label = f"{symbols[i]}{i+1}-{symbols[j]}{j+1}"

    # Upstream returns in AU (Hartree/Bohr^2), scaled by 0.963
    up_au = upstream_seminario_bond(
        atoms=[old_atoms[i], old_atoms[j]],
        hessian=hessian,
        ang_to_bohr=True,
        scaling=0.963,
    )
    # Convert upstream AU to mdyn/A for comparison
    up_mdyn = up_au * co.AU_TO_MDYNA

    # New returns directly in mdyn/A, scaled by 0.963
    nw_mdyn = new_seminario_bond_fc(i, j, coords, hessian, au_units=True, dft_scaling=0.963)

    diff = abs(up_mdyn - nw_mdyn)
    match = "✓" if diff < 0.01 else "✗"
    if diff >= 0.01:
        all_match = False

    print(f"  {label:<8} {up_au:>15.6f} {nw_mdyn:>15.6f} {up_mdyn:>15.6f} {diff:>10.6f} {match:>8}")

print()
if all_match:
    print("✅ All bond force constants match within 0.01 mdyn/A!")
else:
    print("⚠️  Differences found — need investigation")
    print("   Note: Sign differences are expected (upstream uses abs(), new preserves sign)")
    print("   Magnitude differences suggest algorithm mismatch")
