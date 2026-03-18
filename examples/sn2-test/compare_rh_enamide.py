"""Compare upstream vs new Seminario on the REAL Rh-enamide system from the QFUERZA paper.

This is the actual organometallic TS from Farrugia et al., JCTC 2026.
36 atoms including Rh, P, C, N, O, H — much more complex than SN2.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from q2mm.schrod_indep_filetypes import JaguarIn, Mol2
from q2mm.seminario import seminario_bond as upstream_bond
from q2mm.seminario import sub_hessian as upstream_sub_hessian
from q2mm.models.seminario import seminario_bond_fc as new_bond_fc
from q2mm import constants as co

RH_DIR = Path("examples/rh-enamide/rh_enamide_training_set")

print("=" * 70)
print("Rh-Enamide TSFF: Upstream vs New Seminario Comparison")
print("(From Farrugia et al., JCTC 2026)")
print("=" * 70)

# Load Jaguar Hessian
jag_path = RH_DIR / "jaguar_spe_freq_in_out" / "1ZDMPfromJCTCSI_loner1.01.in"
jag = JaguarIn(str(jag_path))
hessian = jag.get_hessian(36)
print(f"\nJaguar Hessian: {hessian.shape} from {jag_path.name}")

# Load mol2 structure for coordinates
mol2 = Mol2(str(RH_DIR / "mol2" / "1_zdmp.mol2"))
struct = mol2.structures[0]
print(f"Mol2 structure: {len(struct.atoms)} atoms")

# Extract coordinates for new implementation (0-based numpy array)
coords = np.array([[a.x, a.y, a.z] for a in struct.atoms])
symbols = [a.element for a in struct.atoms]
print(f"Elements: {set(symbols)}")

# Build old-style Atom list (1-based index)
old_atoms = struct.atoms

# Test on all detected bonds
print(f"Bonds: {len(struct.bonds)}")

print(f"\n{'Bond':<20} {'Upstream (AU)':>15} {'New (mdyn/A)':>15} {'Up->mdyn/A':>15} {'Diff':>10} {'Match?':>8}")
print("-" * 83)

max_diff = 0.0
n_bonds = 0
n_match = 0

for bond in struct.bonds:
    i_old, j_old = bond.atom_nums  # 1-based
    i_new, j_new = i_old - 1, j_old - 1  # 0-based

    if i_new >= len(coords) or j_new >= len(coords):
        continue

    elem_i = symbols[i_new] if i_new < len(symbols) else "?"
    elem_j = symbols[j_new] if j_new < len(symbols) else "?"
    label = f"{elem_i}{i_old}-{elem_j}{j_old}"

    # Upstream
    try:
        up_au = upstream_bond(
            atoms=[old_atoms[i_old - 1], old_atoms[j_old - 1]],
            hessian=hessian,
            ang_to_bohr=True,
            scaling=0.963,
        )
        up_mdyn = up_au * co.AU_TO_MDYNA
    except Exception as e:
        print(f"  {label:<18} UPSTREAM ERROR: {e}")
        continue

    # New
    try:
        nw_mdyn = new_bond_fc(i_new, j_new, coords, hessian, au_units=True, dft_scaling=0.963)
    except Exception as e:
        print(f"  {label:<18} NEW ERROR: {e}")
        continue

    diff = abs(up_mdyn - nw_mdyn)
    match = "✓" if diff < 0.01 else "✗"
    max_diff = max(max_diff, diff)
    n_bonds += 1
    if diff < 0.01:
        n_match += 1

    print(f"  {label:<18} {up_au:>15.6f} {nw_mdyn:>15.6f} {up_mdyn:>15.6f} {diff:>10.6f} {match:>8}")

print(f"\n{'='*70}")
print(f"SUMMARY: {n_match}/{n_bonds} bonds match within 0.01 mdyn/A")
print(f"Max difference: {max_diff:.6f} mdyn/A")
if n_match == n_bonds:
    print("✅ PERFECT PARITY on real organometallic system!")
else:
    print(f"⚠️  {n_bonds - n_match} bonds differ — investigate")
