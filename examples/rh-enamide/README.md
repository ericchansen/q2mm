# Rh-Enamide Hydrogenation Example

Training data for rhodium(I)-catalysed asymmetric hydrogenation of enamides —
a benchmark system from the
[Q2MM JCTC paper](https://doi.org/10.1021/acs.jctc.6b00654).  Nine diverse
Rh-diphosphine intermediates with QM-calculated Hessians are used to train
MM3 force field parameters via QFUERZA estimation.

## Files

| File | Description |
|------|-------------|
| `mm3.fld` | Base Allinger MM3 force field |
| `rh_hyd_enamide_start.fld` | Starting (untrained) force field for the Rh system |
| `rh_hyd_enamide_seminario.fld` | Force field with QFUERZA-estimated bond/angle constants |
| `rh-hydrogenation-enamide-template.mae` | Maestro template defining the Rh catalyst scaffold |
| `atom.typ` | Custom MMFF atom type definitions (Rh, P-ligands, H types) |
| `freq_runner.sh` | SLURM batch script for parallel Jaguar frequency jobs |

### `ff/` — Optimised force fields

| File | Description |
|------|-------------|
| `rh_hyd_enamide_start.fld` | Starting parameters |
| `rh_hyd_enamide_final.fld` | Final Q2MM-optimised parameters |

### `rh_enamide_training_set/` — 9-structure training set

| Directory | Contents |
|-----------|----------|
| `raw_xyz/` | 9 XYZ structures (ZDMP, DMPE, DuPHOS, BPE ligand variants) |
| `mae/` | 9 Maestro files from the JCTC Supplementary Information |
| `mol2/` | 18 MOL2 files (short + full JCTC names) |
| `jaguar_spe_freq_in_out/` | 18 Jaguar QM input/output files (Hessians, frequencies) |

Master files: `rh_enamide_training_set.mae`, `.mol2`, `.mmo`

## Quick start

```python
from q2mm.io.xyz import load_xyz
from q2mm.io.mm3 import load_mm3_fld
from q2mm.models.seminario import qfuerza_into

# Load a structure with its QM Hessian
mol = load_xyz("rh_enamide_training_set/raw_xyz/1_zdmp.xyz")

# Load the starting force field
ff = load_mm3_fld("mm3.fld")

# Overwrite active bond/angle params with QFUERZA-projected values;
# returns a NEW ForceField (ff itself is immutable and never mutated).
# Pass active_bonds=.../active_angles=... (0-based indices into
# ff.bonds/ff.angles) to restrict which rows QFUERZA may overwrite —
# see q2mm.models.parameters.opt_substructure_membership for deriving
# that set from a published-OPT-only force field, and
# q2mm.models.parameters.ActiveParameterSpace for the resulting
# active/frozen projection used by the optimizers.
estimated_ff = qfuerza_into(ff, mol, au_hessian=True)
```

## See also

- `examples/sn2-test/compare_rh_enamide.py` — validates QFUERZA results
  against pinned fixtures for this system
- [References](https://ericchansen.github.io/q2mm/references/) — full list of
  Q2MM publications including the Rh-enamide JCTC paper
