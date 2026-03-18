"""Compare upstream seminario vs new models/seminario on the same Hessian.

Both implementations should produce identical (or numerically equivalent)
force constants when given the same molecule + Hessian.

The upstream code (q2mm/seminario.py) uses the old Structure/FF/Param objects.
The new code (q2mm/models/seminario.py) uses Q2MMMolecule/ForceField.

This script runs both and compares results.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

QM_REF = Path(__file__).parent / "qm-reference"

print("=" * 70)
print("Upstream vs New Seminario Implementation Comparison")
print("=" * 70)

# =====================================================================
# Load common data
# =====================================================================
hessian = np.load(str(QM_REF / "sn2-ts-hessian.npy"))

# Read XYZ
with open(QM_REF / "sn2-ts-optimized.xyz") as f:
    lines = f.readlines()
n_atoms = int(lines[0])
symbols = []
coords = []
for line in lines[2:2 + n_atoms]:
    parts = line.split()
    symbols.append(parts[0])
    coords.append([float(x) for x in parts[1:4]])
coords = np.array(coords)

print(f"\nSystem: {n_atoms} atoms ({', '.join(symbols)})")
print(f"Hessian: {hessian.shape}")

# =====================================================================
# Run UPSTREAM sub_hessian + seminario_bond on each atom pair
# =====================================================================
print("\n--- Upstream (q2mm/seminario.py) ---")
from q2mm.seminario import sub_hessian, seminario_bond
from q2mm.schrod_indep_filetypes import Atom as OldAtom
from q2mm import constants as co

# Build old-style Atom objects
old_atoms = []
for i, (sym, xyz) in enumerate(zip(symbols, coords)):
    atom = OldAtom.__new__(OldAtom)
    atom.index = i + 1  # 1-based!
    atom.coords = np.array(xyz)
    atom.x, atom.y, atom.z = xyz
    atom.element = sym
    atom.atomic_num = {"C": 6, "H": 1, "F": 9}.get(sym, 1)
    atom.atomic_mass = {"C": 12.0, "H": 1.008, "F": 19.0}.get(sym, 1.0)
    old_atoms.append(atom)

# Test each bond pair using upstream sub_hessian
upstream_bond_fcs = {}
bond_pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]  # C-F1, C-F2, C-H1, C-H2, C-H3
for i, j in bond_pairs:
    vec12, eigval, eigvec = sub_hessian(hessian, old_atoms[i], old_atoms[j], ang_to_bohr=True)
    # Compute FC same as seminario_bond does
    k_bond = 0.0
    for n in range(3):
        proj = np.dot(eigvec[:, n], vec12) ** 2
        k_bond += abs(eigval[n]) * proj
    # Convert to mdyn/A (upstream uses co.AU_TO_MDYNA)
    k_bond_mdyn = k_bond * co.AU_TO_MDYNA
    upstream_bond_fcs[(i, j)] = k_bond_mdyn
    label = f"{symbols[i]}{i+1}-{symbols[j]}{j+1}"
    print(f"  {label}: eigvals={eigval.real}, k={k_bond_mdyn:.6f} mdyn/A")

# Also get raw seminario_bond result
print("\n  Using seminario_bond() directly:")
for i, j in bond_pairs:
    k = seminario_bond(
        atoms=[old_atoms[i], old_atoms[j]],
        hessian=hessian,
        ang_to_bohr=True,
    )
    label = f"{symbols[i]}{i+1}-{symbols[j]}{j+1}"
    print(f"  {label}: k={k:.6f} (raw, kJ/mol/A^2) = {k * co.AU_TO_MDYNA / co.HARTREE_TO_KJMOL:.6f} mdyn/A")
    upstream_bond_fcs[(i, j, "raw")] = k

# =====================================================================
# Run NEW models/seminario on the same data
# =====================================================================
print("\n--- New (q2mm/models/seminario.py) ---")
from q2mm.models.seminario import seminario_bond_fc

new_bond_fcs = {}
for i, j in bond_pairs:
    k = seminario_bond_fc(i, j, coords, hessian, au_units=True)
    new_bond_fcs[(i, j)] = k
    label = f"{symbols[i]}{i+1}-{symbols[j]}{j+1}"
    print(f"  {label}: k={k:.6f} mdyn/A")

# =====================================================================
# Compare
# =====================================================================
print("\n" + "=" * 70)
print("COMPARISON: Upstream vs New")
print("=" * 70)
print(f"{'Bond':<10} {'Upstream':>15} {'New':>15} {'Diff':>12} {'Match?':>8}")
print("-" * 60)

all_match = True
for i, j in bond_pairs:
    label = f"{symbols[i]}{i+1}-{symbols[j]}{j+1}"
    up = upstream_bond_fcs.get((i, j), 0.0)
    nw = new_bond_fcs.get((i, j), 0.0)
    diff = abs(up - nw)
    match = "✓" if diff < 0.01 else "✗"
    if diff >= 0.01:
        all_match = False
    print(f"  {label:<8} {up:>15.6f} {nw:>15.6f} {diff:>12.6f} {match:>8}")

print()
if all_match:
    print("✅ All bond force constants match within tolerance!")
else:
    print("⚠️  Some force constants differ — investigate!")

# =====================================================================
# Also compare the raw sub-block Hessian eigenvalues
# =====================================================================
print("\n--- Sub-block Hessian eigenvalue comparison ---")
for i, j in bond_pairs:
    label = f"{symbols[i]}{i+1}-{symbols[j]}{j+1}"

    # Upstream: uses eig (general)
    i3_up, j3_up = 3 * (old_atoms[i].index - 1), 3 * (old_atoms[j].index - 1)
    h_up = -hessian[i3_up:i3_up+3, j3_up:j3_up+3]
    evals_up = np.sort(np.linalg.eig(h_up)[0].real)

    # New: also uses eig now (after fix)
    i3_nw, j3_nw = 3 * i, 3 * j
    h_nw = -hessian[i3_nw:i3_nw+3, j3_nw:j3_nw+3]
    evals_nw = np.sort(np.linalg.eig(h_nw)[0].real)

    # Check sub-block symmetry
    asym = np.max(np.abs(h_up - h_up.T))

    print(f"  {label}: asymmetry={asym:.2e}")
    print(f"    upstream eig: {evals_up}")
    print(f"    new eig:      {evals_nw}")
    print(f"    match: {np.allclose(evals_up, evals_nw, atol=1e-10)}")
