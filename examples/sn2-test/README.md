# SN2 Transition State Example

End-to-end test of the Q2MM TSFF pipeline using the F⁻ + CH₃F → FCH₃ + F⁻
SN2 transition state — a small system ideal for validating force field
parameterisation.

## Prerequisites

| Software | Required for | Install |
|----------|-------------|---------|
| **Q2MM** | All scripts | `pip install -e .` from repo root |
| **Psi4** | QM data generation | `conda install psi4 -c conda-forge` |
| **Tinker** | MM reference data | [dasher.wustl.edu/tinker](https://dasher.wustl.edu/tinker/) |

Pre-computed QM and MM reference data are included, so Psi4 and Tinker are
only needed if you want to regenerate from scratch.

## Files

### Python scripts

| Script | Description |
|--------|-------------|
| `generate_qm_data.py` | Generate QM reference data with Psi4 (B3LYP/6-31+G(d)) |
| `generate_mm_data.py` | Run Tinker MM3 energy/frequency calculations on the TS geometry |
| `test_pipeline.py` | Test Q2MM's model layer: molecule → Seminario → force field |
| `test_backends.py` | Quick smoke test of TinkerEngine and Psi4Engine backends |
| `run_tsff_pipeline.py` | Full TSFF optimisation pipeline (Seminario init → scipy optimize) |
| `compare_implementations.py` | Compare SN2 Seminario bond projections against pinned fixtures |
| `compare_direct.py` | Wrapper that runs `compare_implementations.py` |
| `compare_rh_enamide.py` | Compare Rh-enamide bond projections against pinned fixtures |
| `compute_barrier.py` | Compute SN2 reaction barrier height for literature comparison |

### Reference data

- **`qm-reference/`** — Pre-computed Psi4 results (geometries, Hessians, frequencies)
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

## See also

- [Tutorial](https://ericchansen.github.io/q2mm/tutorial/) — full walkthrough
  using this example
- `test/integration/test_seminario_parity.py` — CI tests that validate
  Seminario results against these fixtures
