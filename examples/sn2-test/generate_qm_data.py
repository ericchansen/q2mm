"""Regenerate the packaged QM reference data for the SN2 test case.

This repository-only script writes to ``q2mm/data/sn2`` in the same checkout,
derives deterministic normal-mode archives, and refreshes ``manifest.json``.

Run with: conda run -n q2mm python examples/sn2-test/generate_qm_data.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from q2mm.constants import MASSES
from q2mm.models.hessian import mass_weighted_normal_modes

METHOD = "b3lyp"
BASIS = "6-31+G(d)"
TEXT_RESOURCE_NAMES = (
    "ch3f-energy.txt",
    "ch3f-frequencies.txt",
    "ch3f-optimized.xyz",
    "complex-optimized.xyz",
    "sn2-ts-energy.txt",
    "sn2-ts-frequencies.txt",
    "sn2-ts-optimized.xyz",
    "summary.txt",
)


def checkout_resource_dir(script_path: Path = Path(__file__)) -> Path:
    """Return this checkout's canonical resource directory.

    Raises:
        RuntimeError: If *script_path* is not inside a Q2MM source checkout.

    """
    repository = script_path.resolve().parents[2]
    resource_dir = repository / "q2mm" / "data" / "sn2"
    if not (repository / "pyproject.toml").is_file() or not (resource_dir / "manifest.json").is_file():
        raise RuntimeError(
            "SN2 reference regeneration must run from the checked-out examples/sn2-test/generate_qm_data.py script."
        )
    return resource_dir


OUTPUT_DIR = checkout_resource_dir()
LOG_PATH = REPO_ROOT / "examples" / "sn2-test" / "psi4-output.dat"


def write_normal_modes(
    path: Path,
    hessian: np.ndarray,
    symbols: list[str],
) -> None:
    """Derive and write a deterministic mass-weighted normal-mode archive."""
    eigenvalues, eigenvectors = mass_weighted_normal_modes(hessian, symbols)
    masses_amu = np.array([MASSES[symbol] for symbol in symbols], dtype=float)
    np.savez(
        path,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        masses_amu=masses_amu,
        symbols=np.array(symbols),
    )


def refresh_manifest(resource_dir: Path) -> None:
    """Normalize generated text and refresh every size and SHA-256 entry."""
    for name in TEXT_RESOURCE_NAMES:
        path = resource_dir / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))

    manifest_path = resource_dir / "manifest.json"
    metadata: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = metadata.get("files")
    if not isinstance(entries, list):
        raise RuntimeError(f"Invalid SN2 resource manifest: {manifest_path}")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise RuntimeError(f"Invalid SN2 resource manifest entry: {entry!r}")
        path = resource_dir / entry["name"]
        content = path.read_bytes()
        entry["size"] = len(content)
        entry["sha256"] = hashlib.sha256(content).hexdigest()

    manifest_path.write_bytes((json.dumps(metadata, indent=2) + "\n").encode())


def main() -> int:
    """Run all Psi4 calculations and refresh the packaged resource manifest."""
    try:
        import psi4
    except ImportError:
        print("Psi4 not available. Install via: conda install psi4 -c conda-forge")
        return 1

    psi4.set_memory("2 GB")
    psi4.set_num_threads(4)
    psi4.core.set_output_file(str(LOG_PATH), False)

    print("=" * 60)
    print(f"SN2 F- + CH3F - QM Reference Data ({BASIS})")
    print("=" * 60)

    print(f"\n[1/5] F- energy at {METHOD}/{BASIS}...")
    f_minus = psi4.geometry("""
        -1 1
        F 0.0 0.0 0.0
    """)
    psi4.set_options({"basis": BASIS, "reference": "rhf"})
    f_energy = psi4.energy(METHOD, molecule=f_minus)
    print(f"  F- energy: {f_energy:.12f} Ha")

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
    ch3f.save_xyz_file(str(OUTPUT_DIR / "ch3f-optimized.xyz"), True)

    _, ch3f_wfn = psi4.frequency(METHOD, molecule=ch3f, return_wfn=True)
    ch3f_hessian = np.array(ch3f_wfn.hessian())
    ch3f_symbols = ["C", "F", "H", "H", "H"]
    np.save(OUTPUT_DIR / "ch3f-hessian.npy", ch3f_hessian)
    write_normal_modes(OUTPUT_DIR / "ch3f-normal-modes.npz", ch3f_hessian, ch3f_symbols)
    ch3f_freqs = np.array(ch3f_wfn.frequencies())
    np.savetxt(
        OUTPUT_DIR / "ch3f-frequencies.txt",
        ch3f_freqs,
        header=f"CH3F frequencies (cm^-1) at {METHOD}/{BASIS}",
    )

    ch3f_coords = ch3f.geometry().np
    cf_dist = np.linalg.norm(ch3f_coords[0] - ch3f_coords[1]) * 0.529177
    ch_dist = np.linalg.norm(ch3f_coords[0] - ch3f_coords[2]) * 0.529177
    print(f"  C-F: {cf_dist:.4f} A, C-H: {ch_dist:.4f} A")

    print(f"\n[3/5] TS optimization at {METHOD}/{BASIS}...")
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
    ts_mol.save_xyz_file(str(OUTPUT_DIR / "sn2-ts-optimized.xyz"), True)

    ts_coords = ts_mol.geometry().np
    cf1 = np.linalg.norm(ts_coords[0] - ts_coords[1]) * 0.529177
    cf2 = np.linalg.norm(ts_coords[0] - ts_coords[2]) * 0.529177
    ch1 = np.linalg.norm(ts_coords[0] - ts_coords[3]) * 0.529177
    print(f"  C-F1: {cf1:.4f} A, C-F2: {cf2:.4f} A, C-H: {ch1:.4f} A")

    print("\n[4/5] Hessian at TS...")
    _, ts_wfn = psi4.frequency(METHOD, molecule=ts_mol, return_wfn=True)
    ts_hessian = np.array(ts_wfn.hessian())
    ts_symbols = ["C", "F", "F", "H", "H", "H"]
    np.save(OUTPUT_DIR / "sn2-ts-hessian.npy", ts_hessian)
    write_normal_modes(OUTPUT_DIR / "sn2-ts-normal-modes.npz", ts_hessian, ts_symbols)
    freqs = np.array(ts_wfn.frequencies())
    np.savetxt(
        OUTPUT_DIR / "sn2-ts-frequencies.txt",
        freqs,
        header=f"SN2 TS frequencies (cm^-1) at {METHOD}/{BASIS}",
    )

    n_imag = np.sum(freqs < 0)
    print(f"  Hessian shape: {ts_hessian.shape}")
    print(f"  Frequencies: {freqs}")
    print(f"  Imaginary: {n_imag} (must be 1)")
    if n_imag == 1:
        print(f"  OK: imaginary freq = {freqs[freqs < 0][0]:.1f} cm^-1")
    else:
        print(f"  WARNING: expected 1 imaginary, got {n_imag}")

    print("\n[5/5] Ion-dipole complex optimization...")
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
    complex_mol.save_xyz_file(str(OUTPUT_DIR / "complex-optimized.xyz"), True)

    barrier_vs_reactants = (ts_energy - (ch3f_energy + f_energy)) * 627.509
    barrier_vs_complex = (ts_energy - complex_energy) * 627.509
    summary = (
        f"SN2 F- + CH3F Reference Data at {METHOD}/{BASIS}\n"
        f"{'=' * 60}\n\n"
        f"F- energy:       {f_energy:.12f} Ha\n"
        f"CH3F energy:     {ch3f_energy:.12f} Ha\n"
        f"Complex energy:  {complex_energy:.12f} Ha\n"
        f"TS energy:       {ts_energy:.12f} Ha\n\n"
        f"C-F (TS):        {cf1:.4f} / {cf2:.4f} A\n"
        f"C-H (TS):        {ch1:.4f} A\n"
        f"C-F (CH3F):      {cf_dist:.4f} A\n\n"
        f"Barrier (TS - reactants):  {barrier_vs_reactants:.2f} kcal/mol\n"
        f"Barrier (TS - complex):    {barrier_vs_complex:.2f} kcal/mol\n"
        "  Literature expected:     ~13-15 kcal/mol\n\n"
        f"Imaginary freq:  {freqs[freqs < 0][0]:.1f} cm^-1\n"
        f"Total freqs:     {len(freqs)} ({n_imag} imaginary)\n"
    )
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    (OUTPUT_DIR / "sn2-ts-energy.txt").write_text(
        f"# SN2 TS energy at {METHOD}/{BASIS}\n{ts_energy:.12f}\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "ch3f-energy.txt").write_text(
        f"# CH3F energy at {METHOD}/{BASIS}\n{ch3f_energy:.12f}\n",
        encoding="utf-8",
    )
    refresh_manifest(OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print(f"RESULTS at {METHOD}/{BASIS}")
    print(f"{'=' * 60}")
    print(f"  C-F (TS):               {cf1:.4f} / {cf2:.4f} A  (lit: ~1.83-1.85)")
    print(f"  C-F (CH3F):             {cf_dist:.4f} A  (expt: 1.382)")
    print(f"  Imaginary freq:         {freqs[freqs < 0][0]:.1f} cm^-1")
    print(f"  Barrier (vs reactants): {barrier_vs_reactants:.2f} kcal/mol")
    print(f"  Barrier (vs complex):   {barrier_vs_complex:.2f} kcal/mol  (lit: ~13-15)")
    print(f"  Resources:              {OUTPUT_DIR}")
    print(f"  Psi4 log:               {LOG_PATH}")
    print("  Manifest hashes refreshed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
