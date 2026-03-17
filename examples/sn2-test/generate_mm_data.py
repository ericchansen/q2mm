"""Run Tinker MM3 calculations on the SN2 TS geometry.

Run with: python examples/sn2-test/generate_mm_data.py
Requires Tinker binaries on PATH or at C:\\Users\\ericc\\tinker\\bin-windows
"""
import os
import subprocess
import numpy as np

TINKER_BIN = r"C:\Users\ericc\tinker\bin-windows"
TINKER_PARAMS = r"C:\Users\ericc\tinker\params\mm3.prm"
QM_DIR = os.path.join(os.path.dirname(__file__), "qm-reference")
MM_DIR = os.path.join(os.path.dirname(__file__), "mm-reference")
os.makedirs(MM_DIR, exist_ok=True)

print("=" * 60)
print("Tinker MM3 Calculations on SN2 TS")
print("=" * 60)

# Read the QM-optimized TS geometry
with open(os.path.join(QM_DIR, "sn2-ts-optimized.xyz")) as f:
    lines = f.readlines()

atoms = []
coords = []
for line in lines[2:]:
    parts = line.split()
    if len(parts) == 4:
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:]])

# MM3 atom types: C=1, F=11, H=5
type_map = {"C": 1, "F": 11, "H": 5}

# Write Tinker XYZ format
# Format: natoms title
# atom# symbol x y z type connectivity...
tinker_xyz = os.path.join(MM_DIR, "sn2-ts.xyz")
with open(tinker_xyz, "w") as f:
    f.write(f"     {len(atoms)}  SN2 TS F-CH3-F from Psi4 B3LYP/6-31+G(d)\n")
    # C is atom 1, bonded to F(2), F(3), H(4), H(5), H(6)
    # In the TS, C has 5 "bonds" (trigonal bipyramidal)
    f.write(f"     1  C   {coords[0][0]:12.6f} {coords[0][1]:12.6f} {coords[0][2]:12.6f}     1     2     3     4     5     6\n")
    f.write(f"     2  F   {coords[1][0]:12.6f} {coords[1][1]:12.6f} {coords[1][2]:12.6f}    11     1\n")
    f.write(f"     3  F   {coords[2][0]:12.6f} {coords[2][1]:12.6f} {coords[2][2]:12.6f}    11     1\n")
    f.write(f"     4  H   {coords[3][0]:12.6f} {coords[3][1]:12.6f} {coords[3][2]:12.6f}     5     1\n")
    f.write(f"     5  H   {coords[4][0]:12.6f} {coords[4][1]:12.6f} {coords[4][2]:12.6f}     5     1\n")
    f.write(f"     6  H   {coords[5][0]:12.6f} {coords[5][1]:12.6f} {coords[5][2]:12.6f}     5     1\n")

# Write a key file pointing to MM3 parameters
key_file = os.path.join(MM_DIR, "sn2-ts.key")
with open(key_file, "w") as f:
    f.write(f"parameters {TINKER_PARAMS}\n")

print(f"\n[1/3] Tinker XYZ: {tinker_xyz}")
print(f"      Key file:   {key_file}")

# Run analyze for energy breakdown
print("\n[2/3] Running Tinker analyze (energy)...")
analyze_exe = os.path.join(TINKER_BIN, "analyze.exe")
result = subprocess.run(
    [analyze_exe, tinker_xyz, "-k", key_file, "E"],
    capture_output=True, text=True, timeout=60
)

energy_output = os.path.join(MM_DIR, "analyze-energy.txt")
with open(energy_output, "w") as f:
    f.write(result.stdout)
    if result.stderr:
        f.write("\n--- STDERR ---\n")
        f.write(result.stderr)

# Parse total energy
for line in result.stdout.split("\n"):
    if "Total Potential Energy" in line or "Bond Stretching" in line or "Angle Bending" in line or "Torsional" in line:
        print(f"  {line.strip()}")

# Run vibrate for frequencies and Hessian
print("\n[3/3] Running Tinker vibrate (frequencies)...")
vibrate_exe = os.path.join(TINKER_BIN, "vibrate.exe")
result_vib = subprocess.run(
    [vibrate_exe, tinker_xyz, "-k", key_file],
    capture_output=True, text=True, timeout=120,
    input="A\n"  # 'A' for all vibrations
)

vib_output = os.path.join(MM_DIR, "vibrate-output.txt")
with open(vib_output, "w") as f:
    f.write(result_vib.stdout)
    if result_vib.stderr:
        f.write("\n--- STDERR ---\n")
        f.write(result_vib.stderr)

# Parse frequencies
mm_freqs = []
for line in result_vib.stdout.split("\n"):
    if "Frequency" in line and "cm-1" in line:
        parts = line.split()
        for p in parts:
            try:
                freq = float(p)
                mm_freqs.append(freq)
            except ValueError:
                pass

if mm_freqs:
    np.savetxt(os.path.join(MM_DIR, "mm-frequencies.txt"), mm_freqs,
               header="MM3 frequencies (cm^-1)")
    print(f"  MM3 frequencies: {mm_freqs}")
else:
    print("  No frequencies parsed — check vibrate-output.txt")
    # Print last 20 lines of output for debugging
    for line in result_vib.stdout.split("\n")[-20:]:
        print(f"    {line}")

print(f"\n{'='*60}")
print(f"MM3 outputs saved to: {MM_DIR}")
print(f"{'='*60}")
