# Rh-enamide publication source and executable case

This directory is the canonical source-tree Rh-enamide example for the nine
transition-state structures described by Donoghue, Helquist, Norrby, and Wiest,
[*J. Chem. Theory Comput.* **2008**, 4, 1313–1323](https://doi.org/10.1021/ct800132a).

The scientific inputs are **tracked in the source repository and excluded from
q2mm wheel and sdist artifacts**. Redistribution/licensing is not established;
this page makes no broader licensing assertion and the release tooling does not
package these files.

Run the complete repository compatibility problem against an installed q2mm
wheel:

```bash
python run.py \
  --rh-enamide /path/to/q2mm/examples/publication/rh-enamide \
  --output-root /path/to/output \
  --bounded-ci
```

The bounded mode preserves all nine structures, the complete MM3 field and its
two OPT regions, `repository-geometry-eigenmatrix-v1`, and the active/frozen
partition. It evaluates the real problem and enters the optimizer once, but
makes no convergence claim. Omit `--bounded-ci` for the documented JAX
scientific workflow.

The current objective is a **partial repository reproduction**: it includes the
repository geometry and full-eigenmatrix targets, while the source ESP-charge
and relative-enthalpy terms are not part of this profile. Relative enthalpy
remains typed but blocked until a backend supplies thermochemical enthalpy.

## Source inventory

- `mm3.fld`: complete source force field parsed both as the full field and as
  the OPT-only active subset.
- `rh_enamide_training_set/rh_enamide_training_set.mmo`: authoritative
  MacroModel geometry/typing sequence.
- `rh_enamide_training_set/jaguar_spe_freq_in_out/`: nine paired Jaguar inputs
  carrying the Hessians used by the loader; associated outputs remain source
  evidence.
- `mae/`, `mol2/`, and `raw_xyz/`: source representations retained for
  provenance, not alternate executable example trees.
- `ff/`, template, atom typing, and source job files: original force-field and
  workflow records retained in this one canonical publication directory.
