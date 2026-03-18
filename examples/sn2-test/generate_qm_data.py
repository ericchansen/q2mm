"""Generate QM reference data for SN2 F- + CH3F transition state.

Run with: conda run -n q2mm python examples/sn2-test/generate_qm_data.py

Outputs saved to examples/sn2-test/qm-reference/
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

# Configure Psi4
psi4.set_memory("2 GB")
psi4.set_num_threads(4)
psi4.core.set_output_file(os.path.join(OUTPUT_DIR, "psi4-output.dat"), False)

print("=" * 60)
print("SN2 F- + CH3F Transition State — QM Reference Data Generation")
print("=" * 60)

# =====================================================================
# 1. TS Optimization (saddle point search)
# =====================================================================
print("\n[1/4] Setting up SN2 TS geometry...")

# SN2 TS: F...CH3...F  (charge -1, singlet)
ts_mol = psi4.geometry("""
    -1 1
    C     0.000000    0.000000    0.000000
    F     0.000000    0.000000    1.800000
    F     0.000000    0.000000   -1.800000
    H     1.026720    0.000000    0.000000
    H    -0.513360    0.889165    0.000000
    H    -0.513360   -0.889165    0.000000
""")

psi4.set_options(
    {
        "basis": "6-31G*",
        "reference": "rhf",
        "opt_type": "ts",
        "geom_maxiter": 100,
    }
)

print("[2/4] Optimizing TS geometry (saddle point search)...")
ts_energy = psi4.optimize("b3lyp", molecule=ts_mol)
print(f"  TS Energy: {ts_energy:.10f} Hartree")

# Save optimized TS geometry as XYZ
ts_xyz_path = os.path.join(OUTPUT_DIR, "sn2-ts-optimized.xyz")
ts_mol.save_xyz_file(ts_xyz_path, True)
print(f"  Saved: {ts_xyz_path}")

# =====================================================================
# 2. Frequency calculation at TS (Hessian)
# =====================================================================
print("\n[3/4] Computing Hessian (frequency calculation) at TS...")

ts_energy_freq, ts_wfn = psi4.frequency("b3lyp", molecule=ts_mol, return_wfn=True)

# Extract Hessian matrix
hessian = np.array(ts_wfn.hessian())
hess_path = os.path.join(OUTPUT_DIR, "sn2-ts-hessian.npy")
np.save(hess_path, hessian)
print(f"  Hessian shape: {hessian.shape}")
print(f"  Saved: {hess_path}")

# Extract frequencies
freqs = np.array(ts_wfn.frequencies())
freqs_path = os.path.join(OUTPUT_DIR, "sn2-ts-frequencies.txt")
np.savetxt(freqs_path, freqs, header="Vibrational frequencies (cm^-1)")
print(f"  Frequencies: {freqs}")

# Check for exactly 1 imaginary frequency
n_imaginary = np.sum(freqs < 0)
print(f"  Imaginary frequencies: {n_imaginary}")
if n_imaginary == 1:
    print(f"  ✓ Valid transition state! Imaginary freq: {freqs[freqs < 0][0]:.1f} cm^-1")
else:
    print(f"  ✗ WARNING: Expected 1 imaginary frequency, got {n_imaginary}")

# Save energy
energy_path = os.path.join(OUTPUT_DIR, "sn2-ts-energy.txt")
with open(energy_path, "w") as f:
    f.write("# SN2 TS energy at B3LYP/6-31G*\n")
    f.write(f"{ts_energy:.12f}\n")

# =====================================================================
# 3. CH3F ground state (reactant)
# =====================================================================
print("\n[4/4] Optimizing CH3F ground state...")

ch3f_mol = psi4.geometry("""
    0 1
    C     0.000000    0.000000    0.000000
    F     0.000000    0.000000    1.383000
    H     1.026720    0.000000   -0.363000
    H    -0.513360    0.889165   -0.363000
    H    -0.513360   -0.889165   -0.363000
""")

psi4.set_options({"opt_type": "min"})
ch3f_energy = psi4.optimize("b3lyp", molecule=ch3f_mol)
print(f"  CH3F Energy: {ch3f_energy:.10f} Hartree")

ch3f_xyz_path = os.path.join(OUTPUT_DIR, "ch3f-optimized.xyz")
ch3f_mol.save_xyz_file(ch3f_xyz_path, True)

# Also do frequency calc on CH3F
ch3f_energy_freq, ch3f_wfn = psi4.frequency("b3lyp", molecule=ch3f_mol, return_wfn=True)
ch3f_hessian = np.array(ch3f_wfn.hessian())
np.save(os.path.join(OUTPUT_DIR, "ch3f-hessian.npy"), ch3f_hessian)
ch3f_freqs = np.array(ch3f_wfn.frequencies())
np.savetxt(os.path.join(OUTPUT_DIR, "ch3f-frequencies.txt"), ch3f_freqs, header="CH3F vibrational frequencies (cm^-1)")

with open(os.path.join(OUTPUT_DIR, "ch3f-energy.txt"), "w") as f:
    f.write("# CH3F ground state energy at B3LYP/6-31G*\n")
    f.write(f"{ch3f_energy:.12f}\n")

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 60)
print("Done! Reference data saved to:", OUTPUT_DIR)
print("=" * 60)
print(f"  TS energy:     {ts_energy:.10f} Ha")
print(f"  CH3F energy:   {ch3f_energy:.10f} Ha")
print(f"  Hessian shape: {hessian.shape}")
print(f"  TS freqs:      {len(freqs)} modes, {n_imaginary} imaginary")
print("\nFiles:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f:40s} {size:>10,} bytes")
