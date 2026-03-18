"""Full TSFF Pipeline: QM Hessian -> Seminario -> Force Constants.

This script demonstrates the complete Q2MM QFUERZA workflow:
1. Load QM Hessian from Psi4 calculation (saved fixture)
2. Load molecular structure
3. Create/load MM3 force field with initial parameters
4. Run Seminario method to estimate force constants from QM Hessian
5. Compare estimated vs initial parameters
6. Validate against literature

Run with: python examples/sn2-test/run_tsff_pipeline.py
"""
import copy
import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from q2mm.schrod_indep_filetypes import (
    MM3, Structure, Atom, Bond, Angle, ParamMM3,
    mass_weight_hessian,
)
from q2mm.seminario import seminario
from q2mm import constants as co
from q2mm import linear_algebra

QM_REF = Path(__file__).parent / "qm-reference"

logging.basicConfig(level=logging.WARNING)


def load_xyz(path):
    """Load atoms and coordinates from XYZ file."""
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    atoms = []
    coords = []
    for line in lines[2:2 + n]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return atoms, np.array(coords)


def build_structure(atoms, coords, atom_type_map, origin_name="SN2_TS"):
    """Build a Structure object from atoms and coordinates.

    Automatically detects bonds and angles based on distances.
    """
    struct = Structure(origin_name=origin_name)

    # Create Atom objects
    atom_objs = []
    for i, (symbol, xyz) in enumerate(zip(atoms, coords)):
        atom = Atom.__new__(Atom)
        atom.index = i + 1  # 1-based
        atom.coords = list(xyz)
        atom.x, atom.y, atom.z = xyz
        atom.atom_type = atom_type_map.get(symbol, "1")
        atom.atom_type_name = atom.atom_type
        atom.element = symbol
        atom.atomic_num = {"C": 6, "H": 1, "F": 9, "Cl": 17, "N": 7, "O": 8}.get(symbol, 1)
        atom.atomic_mass = {"C": 12.0, "H": 1.008, "F": 19.0, "Cl": 35.45}.get(symbol, 1.0)
        atom.partial_charge = 0.0
        atom_objs.append(atom)
    struct._atoms = atom_objs

    # Detect bonds by covalent radii
    cov_radii = {"H": 0.31, "C": 0.76, "F": 0.57, "Cl": 0.99, "N": 0.71, "O": 0.66}
    bond_objs = []
    bond_pairs = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            ri = cov_radii.get(atoms[i], 0.76)
            rj = cov_radii.get(atoms[j], 0.76)
            dist = np.linalg.norm(coords[i] - coords[j])
            # For SN2 TS, C-F bonds are ~1.85A, use generous cutoff
            if dist < 1.5 * (ri + rj):
                bond = Bond.__new__(Bond)
                bond.atom_nums = [i + 1, j + 1]
                bond.value = dist
                bond.ff_row = None  # Will be set below
                bond_objs.append(bond)
                bond_pairs.append((i, j))
    struct._bonds = bond_objs

    # Detect angles from bonds
    angle_objs = []
    adjacency = {i: [] for i in range(len(atoms))}
    for i, j in bond_pairs:
        adjacency[i].append(j)
        adjacency[j].append(i)

    for center in range(len(atoms)):
        neighbors = adjacency[center]
        for ii in range(len(neighbors)):
            for jj in range(ii + 1, len(neighbors)):
                a, b = neighbors[ii], neighbors[jj]
                # Calculate angle
                v1 = coords[a] - coords[center]
                v2 = coords[b] - coords[center]
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle_val = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

                angle = Angle.__new__(Angle)
                angle.atom_nums = [a + 1, center + 1, b + 1]
                angle.value = angle_val
                angle.ff_row = None  # Will be set below
                angle_objs.append(angle)
    struct._angles = angle_objs

    return struct


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


def assign_ff_rows(struct, ff):
    """Match structure bonds/angles to FF parameter rows.

    Uses atom element matching since the .fld parser produces
    different atom type string formats than the structure builder.
    """
    # Map FF param ff_row to the element pair it represents
    # Bond params: extract actual elements from the atom_types
    bond_rows = {}  # (elem1, elem2) -> ff_row
    angle_rows = {}  # (elem1, elem2, elem3) -> ff_row

    for param in ff.params:
        if param.ptype == "bf":
            # Extract element letters from atom type (e.g., 'C1' -> 'C', ' F' -> 'F')
            elems = tuple(sorted(t.strip()[0] for t in param.atom_types))
            bond_rows[elems] = param.ff_row
        elif param.ptype == "af":
            # For angles, order matters: outer-center-outer
            types = [t.strip() for t in param.atom_types if t.strip() and t.strip() != '-']
            if len(types) >= 2:
                # Extract just the element letters
                elems = tuple(t[0] for t in types[:3] if t)
                if elems not in angle_rows:
                    angle_rows[elems] = param.ff_row

    # Map structure atoms to elements
    elem_map = {a.index: a.element for a in struct.atoms}

    for bond in struct.bonds:
        i, j = bond.atom_nums
        pair = tuple(sorted([elem_map[i], elem_map[j]]))
        if pair in bond_rows:
            bond.ff_row = bond_rows[pair]

    for angle in struct.angles:
        i, j, k = angle.atom_nums
        # Try both orderings
        triple = (elem_map[i], elem_map[j], elem_map[k])
        triple_rev = (elem_map[k], elem_map[j], elem_map[i])
        for key in angle_rows:
            if triple == key or triple_rev == key:
                angle.ff_row = angle_rows[key]
                break


# =====================================================================
# Main Pipeline
# =====================================================================
def main():
    print("=" * 70)
    print("Q2MM TSFF Pipeline: SN2 F- + CH3F Transition State")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load QM data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading QM reference data...")

    ts_atoms, ts_coords = load_xyz(QM_REF / "sn2-ts-optimized.xyz")
    hessian = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
    ts_freqs = np.loadtxt(str(QM_REF / "sn2-ts-frequencies.txt"))

    print(f"  Structure: {len(ts_atoms)} atoms ({', '.join(ts_atoms)})")
    print(f"  Hessian: {hessian.shape}")
    print(f"  Frequencies: {len(ts_freqs)} modes, {sum(1 for f in ts_freqs if f < 0)} imaginary")
    print(f"  Imaginary freq: {min(ts_freqs):.1f} cm^-1")

    # ------------------------------------------------------------------
    # Step 2: Build molecular structure
    # ------------------------------------------------------------------
    print("\n[2/5] Building molecular structure...")

    # MM3 atom types: C=C1, F=F1, H=H5
    atom_type_map = {"C": "C1", "F": "F1", "H": "H5"}
    struct = build_structure(ts_atoms, ts_coords, atom_type_map, "SN2_TS")

    print(f"  Atoms: {len(struct.atoms)}")
    print(f"  Bonds: {len(struct.bonds)}")
    for b in struct.bonds:
        i, j = b.atom_nums
        print(f"    {ts_atoms[i-1]}{i}-{ts_atoms[j-1]}{j}: {b.value:.4f} A")
    print(f"  Angles: {len(struct.angles)}")
    for a in struct.angles:
        i, j, k = a.atom_nums
        print(f"    {ts_atoms[i-1]}{i}-{ts_atoms[j-1]}{j}-{ts_atoms[k-1]}{k}: {a.value:.1f} deg")

    # ------------------------------------------------------------------
    # Step 3: Create force field with initial guesses
    # ------------------------------------------------------------------
    print("\n[3/5] Creating initial force field...")

    fld_path = str(Path(__file__).parent / "sn2-ts-initial.fld")
    create_sn2_ff(fld_path)

    ff = MM3(path=fld_path)
    ff.import_ff()

    print(f"  Parameters: {len(ff.params)}")
    print("  Initial values:")
    for p in ff.params:
        print(f"    {p.ptype:4s} {str(p.atom_types):20s} row={p.ff_row:3d} col={p.ff_col} value={p.value:.4f}")

    # Assign ff_rows to structure bonds/angles
    assign_ff_rows(struct, ff)

    matched_bonds = sum(1 for b in struct.bonds if b.ff_row is not None)
    matched_angles = sum(1 for a in struct.angles if a.ff_row is not None)
    print(f"  Matched: {matched_bonds}/{len(struct.bonds)} bonds, {matched_angles}/{len(struct.angles)} angles")

    # ------------------------------------------------------------------
    # Step 4: Run Seminario / QFUERZA
    # ------------------------------------------------------------------
    print("\n[4/5] Running Seminario/QFUERZA estimation...")

    initial_params = {(p.ptype, tuple(p.atom_types)): p.value for p in ff.params}

    try:
        optimized_ff = seminario(
            force_field=ff,
            structures=[struct],
            hessians=[hessian],
            zero_out=True,
            hessian_units=co.GAUSSIAN,
        )

        print("  Seminario completed successfully!")
        print("\n  Parameter comparison (initial -> estimated):")
        print(f"  {'Type':4s} {'Atoms':20s} {'Initial':>10s} {'Estimated':>10s} {'Change':>10s}")
        print(f"  {'-'*4} {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

        results = {}
        for p in optimized_ff.params:
            key = (p.ptype, tuple(p.atom_types))
            initial = initial_params.get(key, 0.0)
            change = p.value - initial
            pct = (change / initial * 100) if initial != 0 else float('inf')
            print(f"  {p.ptype:4s} {str(p.atom_types):20s} {initial:10.4f} {p.value:10.4f} {change:+10.4f} ({pct:+.1f}%)")
            results[key] = {"initial": initial, "estimated": p.value, "change": change}

    except Exception as e:
        print(f"  Seminario failed: {e}")
        import traceback
        traceback.print_exc()
        optimized_ff = None
        results = {}

    # ------------------------------------------------------------------
    # Step 5: Hessian eigenvalue analysis
    # ------------------------------------------------------------------
    print("\n[5/5] Hessian eigenvalue analysis...")

    eigenvalues, eigenvectors = linear_algebra.decompose(hessian)

    # Sort by magnitude
    sorted_idx = np.argsort(eigenvalues)
    sorted_evals = eigenvalues[sorted_idx]

    print("  Eigenvalues (Hartree/Bohr^2):")
    for i, ev in enumerate(sorted_evals):
        label = "  <-- reaction coordinate" if ev < -0.001 else ""
        if abs(ev) < 1e-4:
            label = "  (translation/rotation)"
        print(f"    [{i+1:2d}] {ev:12.6f}{label}")

    # Reform and verify
    reformed = linear_algebra.reform_hessian(eigenvalues, eigenvectors)
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
    print(f"  Atoms:            {len(ts_atoms)}")
    print(f"  Bonds detected:   {len(struct.bonds)}")
    print(f"  Angles detected:  {len(struct.angles)}")
    print(f"  FF parameters:    {len(ff.params)}")
    print(f"  Imaginary freq:   {min(ts_freqs):.1f} cm^-1")
    print(f"  Hessian eigenvals: {len(eigenvalues)} ({sum(1 for e in eigenvalues if e < -0.001)} negative)")

    if results:
        print("\n  Force constant estimates from QFUERZA:")
        for key, val in results.items():
            if key[0] == "bf":
                print(f"    Bond FC {list(key[1])}: {val['initial']:.2f} -> {val['estimated']:.2f} mdyn/A")
            elif key[0] == "af":
                print(f"    Angle FC {list(key[1])}: {val['initial']:.4f} -> {val['estimated']:.4f} mdyn*A/rad^2")

    # Save results
    results_path = Path(__file__).parent / "pipeline-results.txt"
    with open(results_path, "w") as f:
        f.write("Q2MM TSFF Pipeline Results\n")
        f.write("System: SN2 F- + CH3F\n")
        f.write("Method: B3LYP/6-31+G(d)\n")
        f.write(f"Imaginary freq: {min(ts_freqs):.1f} cm^-1\n\n")
        if results:
            f.write("Parameter estimates:\n")
            for key, val in results.items():
                f.write(f"  {key[0]} {list(key[1])}: {val['initial']:.4f} -> {val['estimated']:.4f}\n")

    print(f"\n  Results saved to: {results_path}")

    # Clean up
    if os.path.exists(fld_path):
        os.remove(fld_path)

    return results


if __name__ == "__main__":
    main()
