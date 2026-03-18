"""Test the full TSFF pipeline with Q2MM's new clean data models."""

import numpy as np
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
logging.basicConfig(level=logging.INFO, format="%(message)s")

from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.forcefield import ForceField
from q2mm.models.seminario import estimate_force_constants

QM_REF = Path(__file__).parent / "qm-reference"

# Load SN2 TS (use 1.4x tolerance for partially broken C-F bonds at TS)
mol = Q2MMMolecule.from_xyz(QM_REF / "sn2-ts-optimized.xyz", charge=-1, name="SN2_TS", bond_tolerance=1.4)
hess = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
mol = mol.with_hessian(hess)

print(mol)
print(f"Bonds: {[(b.element_pair, f'{b.length:.3f} A') for b in mol.bonds]}")
print(f"Angles: {len(mol.angles)} detected")
for a in mol.angles:
    print(f"  {a.elements}: {a.value:.1f} deg")

# Auto-create force field
ff = ForceField.create_for_molecule(mol)
print(f"\n{ff}")
for b in ff.bonds:
    print(f"  Bond {b.key}: r0={b.equilibrium:.4f} A, k={b.force_constant:.2f} mdyn/A")
for a in ff.angles:
    print(f"  Angle {a.key}: theta0={a.equilibrium:.1f} deg, k={a.force_constant:.4f}")

# Run Seminario
print("\n" + "=" * 60)
print("Running Seminario/QFUERZA estimation...")
print("=" * 60)
estimated_ff = estimate_force_constants(mol)

print(f"\n{'=' * 60}")
print("RESULTS: Estimated Force Constants from QM Hessian")
print(f"{'=' * 60}")
for b in estimated_ff.bonds:
    print(f"  Bond {b.key}: r0={b.equilibrium:.4f} A, k={b.force_constant:.4f} mdyn/A  {b.label}")
for a in estimated_ff.angles:
    print(f"  Angle {a.key}: theta0={a.equilibrium:.1f} deg, k={a.force_constant:.6f} mdyn*A/rad^2  {a.label}")
