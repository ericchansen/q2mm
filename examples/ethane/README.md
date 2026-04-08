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
structure = log.structures[0]
hessian = structure.hess

# Parse structure
mol2 = Mol2("GS.mol2")
atoms = mol2.structures[0].atoms
print(f"Ethane: {len(atoms)} atoms")
```
