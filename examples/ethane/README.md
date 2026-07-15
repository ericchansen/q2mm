# Ethane Example

Simple example showing Q2MM file parsing with ethane (C₂H₆).

## Files
- `GS.mol2` — Ground state structure (MOL2 format)
- `GS.log` — Gaussian frequency calculation output
- `GS.fchk` — Gaussian formatted checkpoint
- `TS.mol2` — Transition state structure
- `TS.log` — Gaussian frequency calculation output
- `TS.fchk` — Gaussian formatted checkpoint

## Quick Start

```python
from q2mm.io import GaussLog, Mol2

# Parse QM data
log = GaussLog("GS.log")
molecule = log.molecules[-1]
hessian = molecule.hessian

# Parse structure
molecule = Mol2("GS.mol2").molecules[0]
print(f"Ethane: {molecule.n_atoms} atoms")
```
