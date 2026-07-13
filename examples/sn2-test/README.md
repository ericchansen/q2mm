# SN2 Transition State Example

End-to-end test of the Q2MM TSFF pipeline using the F⁻ + CH₃F → FCH₃ + F⁻
SN2 transition state — a small system ideal for validating force field
parameterisation.

## Prerequisites

| Software | Required for | Install |
|----------|-------------|---------|
| **Q2MM** | All scripts | `pip install q2mm` or `pip install -e .` from repo root |
| **Psi4** | QM data generation | `conda install psi4 -c conda-forge` |
| **Tinker** | MM reference data | [dasher.wustl.edu/tinker](https://dasher.wustl.edu/tinker/) |

Pre-computed QM reference data is installed with Q2MM under
`q2mm/data/sn2/`. MM reference outputs remain in this repository. Psi4 and
Tinker are only needed if you want to regenerate either dataset.

## Files

### Python scripts

| Script | Description |
|--------|-------------|
| `generate_qm_data.py` | Generate QM reference data with Psi4 (B3LYP/6-31+G(d)) |
| `generate_mm_data.py` | Run Tinker MM3 energy/frequency calculations on the TS geometry |
| `run_tsff_pipeline.py` | Full TSFF optimisation pipeline (QFUERZA init → scipy optimize) |
| `compare_implementations.py` | Compare SN2 QFUERZA bond projections against pinned fixtures |
| `compare_direct.py` | Wrapper that runs `compare_implementations.py` |
| `compare_rh_enamide.py` | Compare Rh-enamide bond projections against pinned fixtures |
| `compute_barrier.py` | Compute SN2 reaction barrier height for literature comparison |
| `demo_pipeline.py` | Demonstrate the Q2MM pipeline on the SN2 system |
| `demo_backends.py` | Demonstrate backend engine usage |

### Reference data

- **`../../q2mm/data/sn2/`** — Packaged Psi4 results with checksum/provenance metadata
- **`mm-reference/`** — Pre-computed Tinker MM3 results (energies, frequencies)
- **`sn2-ts-guess.xyz`** — Initial transition state geometry guess

## Suggested execution order

```bash
# 1. Generate QM data (requires Psi4; skip if using pre-computed)
python generate_qm_data.py

# 2. Generate MM data (requires Tinker; skip if using pre-computed)
python generate_mm_data.py

# 3. Run the TSFF pipeline (uses pre-computed data)
python run_tsff_pipeline.py

# 4. Validate against pinned fixtures
python compare_direct.py
```

`generate_qm_data.py` is repository-only: it always writes to this checkout's
canonical `q2mm/data/sn2/` directory, derives both `*-normal-modes.npz`
archives from the generated Hessians, normalizes text output, and refreshes
every size and SHA-256 in `manifest.json`. Its verbose Psi4 log is written to
the ignored `examples/sn2-test/psi4-output.dat`; it never writes into an
installed package.

## See also

- [Tutorial](https://ericchansen.github.io/q2mm/tutorial/) — full walkthrough
  using this example
- `test/integration/test_seminario_parity.py` — CI tests that validate
  QFUERZA results against these fixtures
