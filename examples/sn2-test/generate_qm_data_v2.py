"""Re-generate QM reference data with corrected basis set.

6-31+G(d) adds diffuse functions critical for F- anion.
Also computes the ion-dipole complex for standard barrier height.

Run with: conda run -n q2mm python examples/sn2-test/generate_qm_data_v2.py
"""

import os
import sys
import numpy as np

try:
    import psi4
except ImportError:
    print("Psi4 not available. Install via: conda install psi4 -c conda-forge")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "qm-reference")
os.makedirs(OUTPUT_DIR, exist_ok=True)

psi4.set_memory("2 GB")
psi4.set_num_threads(4)
psi4.core.set_output_file(os.path.join(OUTPUT_DIR, "psi4-output-v2.dat"), False)

METHOD = "b3lyp"
BASIS = "6-31+G(d)"

print("=" * 60)
print(f"SN2 F- + CH3F — QM Reference Data (v2, {BASIS})")
print("=" * 60)

# =====================================================================
# 1. F- energy
# =====================================================================
print(f"\n[1/5] F- energy at {METHOD}/{BASIS}...")
f_minus = psi4.geometry("""
    -1 1
    F 0.0 0.0 0.0
""")
psi4.set_options({"basis": BASIS, "reference": "rhf"})
f_energy = psi4.energy(METHOD, molecule=f_minus)
print(f"  F- energy: {f_energy:.12f} Ha")

# =====================================================================
# 2. CH3F ground state
# =====================================================================
print(f"\n[2/5] CH3F ground state optimization at {METHOD}/{BASIS}...")
ch3f = psi4.geometry("""
    0 1
    C     0.000000    0.000000    0.000000
    F     0.000000    0.000000    1.383000
    H     1.026720    0.000000   -0.363000
    H    -0.513360    0.889165   -0.363000
    H    -0.513360   -0.889165   -0.363000
""")
psi4.set_options({"basis": BASIS, "opt_type": "min", "geom_maxiter": 100})
ch3f_energy = psi4.optimize(METHOD, molecule=ch3f)
print(f"  CH3F energy: {ch3f_energy:.12f} Ha")
ch3f.save_xyz_file(os.path.join(OUTPUT_DIR, "ch3f-optimized.xyz"), True)

# CH3F frequencies
ch3f_e2, ch3f_wfn = psi4.frequency(METHOD, molecule=ch3f, return_wfn=True)
np.save(os.path.join(OUTPUT_DIR, "ch3f-hessian.npy"), np.array(ch3f_wfn.hessian()))
ch3f_freqs = np.array(ch3f_wfn.frequencies())
np.savetxt(
    os.path.join(OUTPUT_DIR, "ch3f-frequencies.txt"), ch3f_freqs, header=f"CH3F frequencies (cm^-1) at {METHOD}/{BASIS}"
)

# Extract CH3F geometry
ch3f_coords = ch3f.geometry().np
cf_dist = np.linalg.norm(ch3f_coords[0] - ch3f_coords[1]) * 0.529177  # Bohr to Angstrom
ch_dist = np.linalg.norm(ch3f_coords[0] - ch3f_coords[2]) * 0.529177
print(f"  C-F: {cf_dist:.4f} A, C-H: {ch_dist:.4f} A")

# =====================================================================
# 3. TS optimization (start from slightly asymmetric guess to help optimizer)
# =====================================================================
print(f"\n[3/5] TS optimization at {METHOD}/{BASIS}...")
# Use 1.85 A as a better starting guess based on literature
ts_mol = psi4.geometry("""
    -1 1
    C     0.000000    0.000000    0.000000
    F     0.000000    0.000000    1.850000
    F     0.000000    0.000000   -1.850000
    H     1.026720    0.000000    0.000000
    H    -0.513360    0.889165    0.000000
    H    -0.513360   -0.889165    0.000000
""")

psi4.set_options(
    {
        "basis": BASIS,
        "reference": "rhf",
        "opt_type": "ts",
        "geom_maxiter": 150,
        "full_hess_every": 5,
    }
)

ts_energy = psi4.optimize(METHOD, molecule=ts_mol)
print(f"  TS energy: {ts_energy:.12f} Ha")
ts_mol.save_xyz_file(os.path.join(OUTPUT_DIR, "sn2-ts-optimized.xyz"), True)

# Extract TS geometry
ts_coords = ts_mol.geometry().np
cf1 = np.linalg.norm(ts_coords[0] - ts_coords[1]) * 0.529177
cf2 = np.linalg.norm(ts_coords[0] - ts_coords[2]) * 0.529177
ch1 = np.linalg.norm(ts_coords[0] - ts_coords[3]) * 0.529177
print(f"  C-F1: {cf1:.4f} A, C-F2: {cf2:.4f} A, C-H: {ch1:.4f} A")

# =====================================================================
# 4. Hessian at TS
# =====================================================================
print("\n[4/5] Hessian at TS...")
ts_e2, ts_wfn = psi4.frequency(METHOD, molecule=ts_mol, return_wfn=True)

hessian = np.array(ts_wfn.hessian())
np.save(os.path.join(OUTPUT_DIR, "sn2-ts-hessian.npy"), hessian)

freqs = np.array(ts_wfn.frequencies())
np.savetxt(
    os.path.join(OUTPUT_DIR, "sn2-ts-frequencies.txt"), freqs, header=f"SN2 TS frequencies (cm^-1) at {METHOD}/{BASIS}"
)

n_imag = np.sum(freqs < 0)
print(f"  Hessian shape: {hessian.shape}")
print(f"  Frequencies: {freqs}")
print(f"  Imaginary: {n_imag} (must be 1)")
if n_imag == 1:
    print(f"  OK: imaginary freq = {freqs[freqs < 0][0]:.1f} cm^-1")
else:
    print(f"  WARNING: expected 1 imaginary, got {n_imag}")

# =====================================================================
# 5. Ion-dipole complex (F-...CH3F)
# =====================================================================
print("\n[5/5] Ion-dipole complex optimization...")
# F- approaching CH3F from the backside, ~2.5 A away
complex_mol = psi4.geometry("""
    -1 1
    C     0.000000    0.000000    0.000000
    F     0.000000    0.000000    1.383000
    H     1.026720    0.000000   -0.363000
    H    -0.513360    0.889165   -0.363000
    H    -0.513360   -0.889165   -0.363000
    F     0.000000    0.000000   -2.500000
""")

psi4.set_options({"opt_type": "min", "geom_maxiter": 100, "full_hess_every": -1})
complex_energy = psi4.optimize(METHOD, molecule=complex_mol)
print(f"  Complex energy: {complex_energy:.12f} Ha")
complex_mol.save_xyz_file(os.path.join(OUTPUT_DIR, "complex-optimized.xyz"), True)

# =====================================================================
# Summary
# =====================================================================
barrier_vs_reactants = (ts_energy - (ch3f_energy + f_energy)) * 627.509
barrier_vs_complex = (ts_energy - complex_energy) * 627.509

with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
    f.write(f"SN2 F- + CH3F Reference Data at {METHOD}/{BASIS}\n")
    f.write(f"{'=' * 60}\n\n")
    f.write(f"F- energy:       {f_energy:.12f} Ha\n")
    f.write(f"CH3F energy:     {ch3f_energy:.12f} Ha\n")
    f.write(f"Complex energy:  {complex_energy:.12f} Ha\n")
    f.write(f"TS energy:       {ts_energy:.12f} Ha\n\n")
    f.write(f"C-F (TS):        {cf1:.4f} / {cf2:.4f} A\n")
    f.write(f"C-H (TS):        {ch1:.4f} A\n")
    f.write(f"C-F (CH3F):      {cf_dist:.4f} A\n\n")
    f.write(f"Barrier (TS - reactants):  {barrier_vs_reactants:.2f} kcal/mol\n")
    f.write(f"Barrier (TS - complex):    {barrier_vs_complex:.2f} kcal/mol\n")
    f.write("  Literature expected:     ~13-15 kcal/mol\n\n")
    f.write(f"Imaginary freq:  {freqs[freqs < 0][0]:.1f} cm^-1\n")
    f.write(f"Total freqs:     {len(freqs)} ({n_imag} imaginary)\n")

with open(os.path.join(OUTPUT_DIR, "sn2-ts-energy.txt"), "w") as f:
    f.write(f"# SN2 TS energy at {METHOD}/{BASIS}\n{ts_energy:.12f}\n")
with open(os.path.join(OUTPUT_DIR, "ch3f-energy.txt"), "w") as f:
    f.write(f"# CH3F energy at {METHOD}/{BASIS}\n{ch3f_energy:.12f}\n")

print(f"\n{'=' * 60}")
print(f"RESULTS at {METHOD}/{BASIS}")
print(f"{'=' * 60}")
print(f"  C-F (TS):             {cf1:.4f} / {cf2:.4f} A  (lit: ~1.83-1.85)")
print(f"  C-F (CH3F):           {cf_dist:.4f} A  (expt: 1.382)")
print(f"  Imaginary freq:       {freqs[freqs < 0][0]:.1f} cm^-1")
print(f"  Barrier (vs reactants): {barrier_vs_reactants:.2f} kcal/mol")
print(f"  Barrier (vs complex):   {barrier_vs_complex:.2f} kcal/mol  (lit: ~13-15)")
