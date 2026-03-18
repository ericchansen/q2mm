# Q2MM

**Quantum-guided molecular mechanics force field optimization.**

Q2MM optimizes molecular mechanics (MM) force field parameters by minimizing the difference between MM-calculated properties and quantum mechanics (QM) reference data. It supports multiple QM and MM backends through a plugin architecture.

[![CI](https://github.com/ericchansen/q2mm/actions/workflows/ci.yml/badge.svg)](https://github.com/ericchansen/q2mm/actions/workflows/ci.yml)

## Quick Start

```bash
pip install -e .
```

```python
from q2mm.io import GaussLog, Mol2
from q2mm.forcefields import MM3

# Parse QM reference data
log = GaussLog("ethane.log")
hessian = log.structures[0].hess

# Parse molecular structure
mol2 = Mol2("ethane.mol2")
print(f"Atoms: {len(mol2.structures[0].atoms)}")

# Load force field
ff = MM3("mm3.fld")
ff.import_ff()
print(f"Parameters: {len(ff.params)}")
```

## Installation

**Requirements:** Python 3.9+

```bash
pip install -e .          # Basic
pip install -e ".[dev]"   # With pytest + ruff
pip install -e ".[openmm]" # With OpenMM backend
pip install -e ".[amber]" # With parmed (AMBER support)
```

### QM/MM Backends

| Backend | Type | License | Install |
|---------|------|---------|---------|
| **OpenMM** | MM | MIT/LGPL-like open source | `pip install openmm` |
| **Psi4** | QM | BSD-3 (open source) | `conda install psi4 -c conda-forge` |
| **Tinker** | MM | Free (academic) | [download](https://dasher.wustl.edu/tinker/) |
| **Gaussian** | QM | Commercial | Site license |
| **Schrodinger** | QM/MM | Commercial | Site license |

## Package Structure

```
q2mm/
├── core/          # Optimization engine (gradient, simplex, objective function)
├── backends/      # QM/MM engine integrations (Psi4, Tinker, etc.)
├── io/            # File format parsers (Gaussian, Jaguar, MOL2, MAE)
├── forcefields/   # Force field types (MM3, AMBER, Tinker)
├── cli/           # Command-line interface
└── models/        # Clean molecule/force-field models + Seminario estimation

examples/          # Supported example workflows and bundled reference inputs
scripts/           # Utility scripts and screening tools
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest -v
ruff check q2mm/ test/ scripts/
ruff format --check q2mm test scripts examples
```

To regenerate the pinned upstream Seminario parity fixtures, create a sibling
worktree and run the fixture generator:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_upstream_worktree.ps1
python .\scripts\regenerate_parity_fixtures.py --worktree ..\q2mm-upstream-worktree
```

To validate current behavior against pinned fixtures or live upstream code:

```powershell
python .\scripts\validate_against_upstream.py --mode fixture
powershell -ExecutionPolicy Bypass -File .\scripts\validate_against_upstream.ps1 -Mode both
```

For overlapping old/new behavior, do not treat work as complete until the
relevant validation case passes.

The current OpenMM backend supports MM3-style bonded terms plus the primary
atom-type vdW table carried by MM3/Tinker force fields.

## Citations

If you use Q2MM in your research, please cite the relevant publications:

### Core Method

- Norrby, P.-O. Selectivity in Asymmetric Synthesis from QM-Guided Molecular Mechanics. *J. Mol. Struct. (THEOCHEM)* **2000**, *506*, 9–16. [DOI: 10.1016/S0166-1280(00)00398-5](https://doi.org/10.1016/S0166-1280(00)00398-5)

- Hansen, E.; Rosales, A. R.; Tutkowski, B.; Norrby, P.-O.; Wiest, O. Prediction of Stereochemistry using Q2MM. *Acc. Chem. Res.* **2016**, *49*, 996–1005. [DOI: 10.1021/acs.accounts.6b00037](https://doi.org/10.1021/acs.accounts.6b00037)

- Rosales, A. R.; Quinn, T. R.; Wahlers, J.; Tomberg, A.; Zhang, X.; Helquist, P.; Wiest, O.; Norrby, P.-O. Application of Q2MM to Predictions in Stereoselective Synthesis. *Chem. Commun.* **2018**, *54*, 8294–8301. [DOI: 10.1039/C8CC03695K](https://doi.org/10.1039/C8CC03695K)

### QFUERZA / Seminario Method

- Farrugia, L. M.; Helquist, P.; Norrby, P.-O.; Wiest, O. Rapid FF Generation via Hessian-Informed Initial Parameters and Automated Refinement. *J. Chem. Theory Comput.* **2026**, *22*, 469–476. [DOI: 10.1021/acs.jctc.4c01372](https://doi.org/10.1021/acs.jctc.4c01372)

### Applications

- Rosales, A. R.; Wahlers, J.; Limé, E.; Meadows, R. E.; Leslie, K. W.; Savin, R.; Bell, F.; Hansen, E.; Helquist, P.; Munday, R. H.; Wiest, O.; Norrby, P.-O. Rapid Virtual Screening of Enantioselective Catalysts using CatVS. *Nat. Catal.* **2019**, *2*, 41–45. [DOI: 10.1038/s41929-018-0193-3](https://doi.org/10.1038/s41929-018-0193-3)

- Burai Patrascu, M.; Pottel, J.; Pinus, S.; Bezanson, M.; Norrby, P.-O.; Moitessier, N. Virtual Chemist: Prediction of Enantioselectivity. *Nat. Catal.* **2020**, *3*, 574–584. [DOI: 10.1038/s41929-020-0467-0](https://doi.org/10.1038/s41929-020-0467-0)

- Rosales, A. R.; Ross, S. P.; Helquist, P.; Norrby, P.-O.; Sigman, M. S.; Wiest, O. Transition State Force Field for the Asymmetric Redox-Relay Heck Reaction. *J. Am. Chem. Soc.* **2020**, *142*, 9700–9707. [DOI: 10.1021/jacs.0c01979](https://doi.org/10.1021/jacs.0c01979)

- Wahlers, J.; Maloney, M.; Salahi, F.; Rosales, A. R.; Helquist, P.; Norrby, P.-O.; Wiest, O. Stereoselectivity Predictions for the Pd-Catalyzed 1,4-Conjugate Addition. *J. Org. Chem.* **2021**, *86*, 5660–5667. [DOI: 10.1021/acs.joc.0c02918](https://doi.org/10.1021/acs.joc.0c02918)

- Wahlers, J.; Margalef, J.; Hansen, E.; Bayesteh, A.; Helquist, P.; Diéguez, M.; Pàmies, O.; Wiest, O.; Norrby, P.-O. Proofreading Experimentally Assigned Stereochemistry through Q2MM Predictions. *Nat. Commun.* **2021**, *12*, 6508. [DOI: 10.1038/s41467-021-27065-2](https://doi.org/10.1038/s41467-021-27065-2)

- Quinn, T. R.; Patel, H. N.; Koh, K. H.; Haines, B. E.; Norrby, P.-O.; Helquist, P.; Wiest, O. Automated Fitting of Transition State Force Fields for Biomolecular Simulations. *PLOS ONE* **2022**, *17*, e0264960. [DOI: 10.1371/journal.pone.0264960](https://doi.org/10.1371/journal.pone.0264960)

- Wahlers, J.; Rosales, A. R.; Berkel, N.; Forbes, A.; Helquist, P.; Norrby, P.-O.; Wiest, O. MM3* Force Field for Ferrocenyl Ligands. *J. Org. Chem.* **2022**, *87*, 12334–12341. [DOI: 10.1021/acs.joc.2c01396](https://doi.org/10.1021/acs.joc.2c01396)

- Maloney, M. P.; Stenfors, B. A.; Helquist, P.; Norrby, P.-O.; Wiest, O. Interplay of Computation and Experiment in Enantioselective Catalysis. *ACS Catal.* **2023**, *13*, 14285–14299. [DOI: 10.1021/acscatal.3c03706](https://doi.org/10.1021/acscatal.3c03706)

## License

BSD-3-Clause. See [LICENSE](LICENSE).
