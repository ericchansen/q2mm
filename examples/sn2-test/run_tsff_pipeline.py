"""Full TSFF pipeline using the clean Q2MM models API."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from q2mm.models.hessian import decompose, reform_hessian
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import estimate_force_constants

QM_REF = Path(__file__).parent / "qm-reference"


def create_sn2_ff(path):
    """Create a minimal MM3 force field for the SN2 F-CH3-F system.

    Parameters are rough initial guesses that Seminario will improve.
    """
    # MM3 .fld format for F-CH3-F with substructure marked OPT
    fld_content = """\
 MM3 Force Field - SN2 TS Minimal
 Custom parameters for F- + CH3F transition state
 C
 C  SN2 F-CH3-F Transition State Force Field
 C
-1
  0    STR     3      601.99392
  0    BND     3      601.99392
  0    TOR     1        2.09200
  0    IMP     1       60.19939
  0    S-B     1      601.99392
 C
 C
 C  Substructure section
 C  --------------------
 C SN2 TS OPT
 9 C1(-F1)(-F1)(-H5)(-H5)(-H5)
 1  C1 - F1                1.8500     3.0000     0.0000 0000 0000              A 3             C-F TS bond
 1  C1 - H5                1.0760     4.7400     0.0000 0000 0000              A 3             C-H bond
 2  F1 - C1 - F1         180.0000     0.3000                                  A 3             F-C-F TS angle
 2  F1 - C1 - H5          90.0000     0.5000                                  A 3             F-C-H angle
 2  H5 - C1 - H5         120.0000     0.4500                                  A 3             H-C-H angle
 C End of SN2 TS
"""
    with open(path, "w") as f:
        f.write(fld_content)
    return path


def main():
    print("=" * 70)
    print("Q2MM TSFF Pipeline: SN2 F- + CH3F Transition State")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load QM data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading QM reference data...")

    molecule = Q2MMMolecule.from_xyz(
        QM_REF / "sn2-ts-optimized.xyz",
        name="SN2_TS",
        bond_tolerance=1.5,
    )
    hessian = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
    molecule = molecule.with_hessian(hessian)
    ts_freqs = np.loadtxt(str(QM_REF / "sn2-ts-frequencies.txt"))

    print(f"  Structure: {molecule.n_atoms} atoms ({', '.join(molecule.symbols)})")
    print(f"  Hessian: {hessian.shape}")
    print(f"  Frequencies: {len(ts_freqs)} modes, {sum(1 for f in ts_freqs if f < 0)} imaginary")
    print(f"  Imaginary freq: {min(ts_freqs):.1f} cm^-1")

    # ------------------------------------------------------------------
    # Step 2: Build molecular structure
    # ------------------------------------------------------------------
    print("\n[2/5] Building clean molecule model...")

    print(f"  Atoms: {molecule.n_atoms}")
    print(f"  Bonds: {len(molecule.bonds)}")
    for bond in molecule.bonds:
        print(
            "    "
            f"{molecule.symbols[bond.atom_i]}{bond.atom_i + 1}-"
            f"{molecule.symbols[bond.atom_j]}{bond.atom_j + 1}: "
            f"{bond.length:.4f} A"
        )
    print(f"  Angles: {len(molecule.angles)}")
    for angle in molecule.angles:
        print(
            "    "
            f"{molecule.symbols[angle.atom_i]}{angle.atom_i + 1}-"
            f"{molecule.symbols[angle.atom_j]}{angle.atom_j + 1}-"
            f"{molecule.symbols[angle.atom_k]}{angle.atom_k + 1}: "
            f"{angle.value:.1f} deg"
        )

    # ------------------------------------------------------------------
    # Step 3: Create force field with initial guesses
    # ------------------------------------------------------------------
    print("\n[3/5] Creating initial force field...")

    fld_path = str(Path(__file__).parent / "sn2-ts-initial.fld")
    create_sn2_ff(fld_path)

    initial_ff = ForceField.from_mm3_fld(fld_path)

    print(f"  Parameters: {len(initial_ff.bonds) + len(initial_ff.angles)}")
    print("  Initial values:")
    for bond in initial_ff.bonds:
        print(
            f"    bf   {bond.elements!s:20s} row={bond.ff_row:4d} "
            f"k={bond.force_constant:10.4f} r0={bond.equilibrium:8.4f}"
        )
    for angle in initial_ff.angles:
        print(
            f"    af   {angle.elements!s:20s} row={angle.ff_row:4d} "
            f"k={angle.force_constant:10.4f} a0={angle.equilibrium:8.4f}"
        )

    # ------------------------------------------------------------------
    # Step 4: Run Seminario / QFUERZA
    # ------------------------------------------------------------------
    print("\n[4/5] Running Seminario/QFUERZA estimation...")

    estimated_ff = estimate_force_constants(
        molecule,
        forcefield=initial_ff,
        zero_torsions=True,
        au_hessian=True,
        invalid_policy="skip",
    )

    print("  Seminario completed successfully.")
    print("  Negative or complex TS estimates are skipped to preserve legacy semantics.")
    print("\n  Parameter comparison (initial -> estimated):")
    print(f"  {'Type':4s} {'Elements':20s} {'Initial':>10s} {'Estimated':>10s} {'Change':>10s}")
    print(f"  {'-' * 4} {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")

    results = {}
    initial_bonds = {param.ff_row: param for param in initial_ff.bonds}
    initial_angles = {param.ff_row: param for param in initial_ff.angles}
    for bond in estimated_ff.bonds:
        initial = initial_bonds[bond.ff_row].force_constant if bond.ff_row is not None else 0.0
        change = bond.force_constant - initial
        results[("bf", bond.ff_row)] = {
            "initial": initial,
            "estimated": bond.force_constant,
            "change": change,
        }
        print(f"  bf   {str(bond.elements):20s} {initial:10.4f} {bond.force_constant:10.4f} {change:+10.4f}")
    for angle in estimated_ff.angles:
        initial = initial_angles[angle.ff_row].force_constant if angle.ff_row is not None else 0.0
        change = angle.force_constant - initial
        results[("af", angle.ff_row)] = {
            "initial": initial,
            "estimated": angle.force_constant,
            "change": change,
        }
        print(f"  af   {str(angle.elements):20s} {initial:10.4f} {angle.force_constant:10.4f} {change:+10.4f}")

    # ------------------------------------------------------------------
    # Step 5: Hessian eigenvalue analysis
    # ------------------------------------------------------------------
    print("\n[5/5] Hessian eigenvalue analysis...")

    eigenvalues, eigenvectors = decompose(hessian)

    # Sort by magnitude
    sorted_idx = np.argsort(eigenvalues)
    sorted_evals = eigenvalues[sorted_idx]

    print("  Eigenvalues (Hartree/Bohr^2):")
    for i, ev in enumerate(sorted_evals):
        label = "  <-- reaction coordinate" if ev < -0.001 else ""
        if abs(ev) < 1e-4:
            label = "  (translation/rotation)"
        print(f"    [{i + 1:2d}] {ev:12.6f}{label}")

    # Reform and verify
    reformed = reform_hessian(eigenvalues, eigenvectors)
    roundtrip_err = np.max(np.abs(hessian - reformed))
    print(f"\n  Decompose/reform roundtrip error: {roundtrip_err:.2e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print("  System:           F- + CH3F SN2 transition state")
    print("  QM Method:        B3LYP/6-31+G(d)")
    print(f"  Atoms:            {molecule.n_atoms}")
    print(f"  Bonds detected:   {len(molecule.bonds)}")
    print(f"  Angles detected:  {len(molecule.angles)}")
    print(f"  FF parameters:    {len(initial_ff.bonds) + len(initial_ff.angles)}")
    print(f"  Imaginary freq:   {min(ts_freqs):.1f} cm^-1")
    print(f"  Hessian eigenvals: {len(eigenvalues)} ({sum(1 for e in eigenvalues if e < -0.001)} negative)")

    if results:
        print("\n  Force constant estimates from the clean Seminario pipeline:")
        for key, val in results.items():
            if key[0] == "bf":
                print(f"    Bond row {key[1]}: {val['initial']:.2f} -> {val['estimated']:.2f} mdyn/A")
            elif key[0] == "af":
                print(f"    Angle row {key[1]}: {val['initial']:.4f} -> {val['estimated']:.4f} mdyn*A/rad^2")

    # Save results
    results_path = Path(__file__).parent / "pipeline-results.txt"
    with open(results_path, "w") as f:
        f.write("Q2MM TSFF Pipeline Results (clean models API)\n")
        f.write("System: SN2 F- + CH3F\n")
        f.write("Method: B3LYP/6-31+G(d)\n")
        f.write(f"Imaginary freq: {min(ts_freqs):.1f} cm^-1\n\n")
        if results:
            f.write("Parameter estimates:\n")
            for key, val in results.items():
                f.write(f"  {key[0]} row {key[1]}: {val['initial']:.4f} -> {val['estimated']:.4f}\n")

    print(f"\n  Results saved to: {results_path}")

    # Clean up
    if os.path.exists(fld_path):
        os.remove(fld_path)

    return results


if __name__ == "__main__":
    main()
