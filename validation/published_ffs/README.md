# Published Force Fields

This directory documents published force fields available in this repository and
their provenance. These FFs are used for **Check 1** validation: proving that
the new q2mm engines can load published FFs and evaluate them against the
corresponding QM reference data.

## MM3 Base File

`mm3_base.fld` is the **standard, unmodified MM3 force field** (1,850 lines,
no OPT substructures). It provides the backbone parameters (bonds, angles,
torsions, vdW) that all Q2MM systems build on.

| Property | Value |
|----------|-------|
| **Origin** | Allinger, N. L.; Yuh, Y. H.; Lii, J. H. *JACS* **1989**, *111*, 8551 |
| **Copyright** | Columbia University 1990, "All rights reserved" |
| **Source** | [atlas-nano/ATLAS_toolkit](https://github.com/atlas-nano/ATLAS_toolkit) `ff/macromodel/mm3.fld` |
| **Parameters** | 178 bonds, 258 angles, 1,685 torsions, 32 vdW, 3 stretch-bends |

> ⚠️ **Not committed to this repo.** The file header says "Copyright Columbia
> University 1990 — All rights reserved", which does not grant redistribution
> rights. Download it from the ATLAS_toolkit repo linked above, or extract
> from a MacroModel installation. Place at `validation/published_ffs/mm3_base.fld`.

**Usage:** Systems with standalone OPT-substructure-only FFs (e.g., Wahlers
dissertation FFs) need this base file composed with their custom OPT blocks
to produce a complete, evaluable force field. **Alternatively**, the Rosales
full MM3 FFs already include the base section and do not need composition.

## Force Fields

### Rh-enamide hydrogenation (Donoghue 2008)

| Property | Value |
|----------|-------|
| **Paper** | Donoghue, P. J. et al. *J. Chem. Theory Comput.* **2008**, *4*, 1313–1323 |
| **DOI** | [10.1021/ct800132a](https://doi.org/10.1021/ct800132a) |
| **System** | Rh(I)-catalyzed asymmetric hydrogenation of enamides |
| **FF type** | MM3*/MacroModel transition-state force field |
| **Source repo** | [Q2MM/q2mm](https://github.com/Q2MM/q2mm) commit `b26404b8` |
| **Source file** | `forcefields/rh-hydrogenation-enamide.fld` (patch snippet) |
| **Full FF** | `rh-seminario/ff/rh_hyd_enamide_final.fld` (complete MM3 file) |
| **Training data** | 9 TS structures, B3LYP/LACVP** (Jaguar), located in `examples/rh-enamide/` |

**Files in this repo:**
- `examples/rh-enamide/ff/rh_hyd_enamide_final.fld` — The published optimized FF (full MM3 file, 156 KB)
- `examples/rh-enamide/ff/rh_hyd_enamide_start.fld` — Untrained starting FF (for Check 2 comparison)
- `examples/rh-enamide/ff/rh-hydrogenation-enamide-final.fld` — Patch snippet version (7 KB)

### Redox-relay Heck (Rosales 2020)

| Property | Value |
|----------|-------|
| **Paper** | Rosales, A. R. et al. *J. Am. Chem. Soc.* **2020**, *142*, 9700–9707 |
| **DOI** | [10.1021/jacs.0c01979](https://doi.org/10.1021/jacs.0c01979) |
| **System** | Pd-catalyzed asymmetric redox-relay Heck reaction |
| **FF type** | Full MM3 file with OPT substructures (2,030 lines) |
| **Source** | Rosales dissertation Ch 3 (supporting information) |
| **Training data** | 23 TS Gaussian logs with HPModes, in `validation/supporting-info/rosales/` |
| **Status** | 🔲 QM data available — Check 1 ready to implement |

### Pd-allyl amination (Wahlers 2021)

| Property | Value |
|----------|-------|
| **Paper** | Wahlers, J. et al. *Nat. Commun.* **2021**, *12*, 6508 |
| **DOI** | [10.1038/s41467-021-27065-2](https://doi.org/10.1038/s41467-021-27065-2) |
| **System** | Pd-catalyzed enantioselective allylic amination |
| **FF type** | OPT-only substructure file (238 lines, 8 OPT blocks) — needs `mm3_base.fld` |
| **Source** | Wahlers dissertation Ch 3 (supporting information) |
| **Training data** | 21 TS Gaussian logs (all with freq data), in `validation/supporting-info/wahlers/` |
| **Status** | 🔲 QM data available — Check 1 ready to implement |

### Pd 1,4-conjugate addition (Wahlers 2021)

| Property | Value |
|----------|-------|
| **Paper** | Wahlers, J. et al. *J. Org. Chem.* **2021**, *86*, 5660–5667 |
| **DOI** | [10.1021/acs.joc.0c02918](https://doi.org/10.1021/acs.joc.0c02918) |
| **System** | Pd-catalyzed 1,4-conjugate addition |
| **FF type** | OPT-only substructure file (157 lines, 6 OPT blocks) — needs `mm3_base.fld` |
| **Source** | Wahlers dissertation Ch 5 (supporting information) |
| **Training data** | 10 TS Gaussian logs (all with freq data), in `validation/supporting-info/wahlers/` |
| **Status** | 🔲 QM data available — Check 1 ready to implement |

### Ferrocene scaffold (Wahlers 2022)

| Property | Value |
|----------|-------|
| **Paper** | Wahlers, J. et al. *J. Org. Chem.* **2022**, *87*, 12334–12341 |
| **DOI** | [10.1021/acs.joc.2c01553](https://doi.org/10.1021/acs.joc.2c01553) |
| **System** | Ferrocene-based ligands for asymmetric catalysis |
| **FF type** | OPT-only (Wahlers, 71 lines) and full MM3 (Rosales, 3 variants) |
| **Source** | Wahlers Ch 4 / Rosales Ch 5 (supporting information) |
| **Training data** | 99–178 Gaussian logs (Wahlers), 28–67 (Rosales) |
| **Status** | 🔲 QM data available — Check 1 ready to implement |

### OsO₄ dihydroxylation (Norrby 2000)

| Property | Value |
|----------|-------|
| **Paper** | Norrby, P.-O. et al. *J. Am. Chem. Soc.* **2000**, *122*, 8295 |
| **DOI** | [10.1021/ja000854t](https://doi.org/10.1021/ja000854t) |
| **System** | OsO₄-catalyzed asymmetric dihydroxylation of alkenes |
| **Source repo** | [Q2MM/q2mm](https://github.com/Q2MM/q2mm) commit `b26404b8` |
| **Source file** | `forcefields/os-dihydroxylation-alkene.fld` |
| **Status** | ⚠️ FF available but **no QM training data** in repos — Check 1 blocked |

### Ru ketone hydrogenation (Hansen 2016)

| Property | Value |
|----------|-------|
| **Paper** | Hansen, E. et al. *J. Org. Chem.* **2016**, *81*, 10545 |
| **DOI** | [10.1021/acs.joc.6b01557](https://doi.org/10.1021/acs.joc.6b01557) |
| **System** | Ru-catalyzed asymmetric hydrogenation of ketones |
| **Source repo** | [Q2MM/q2mm](https://github.com/Q2MM/q2mm) commit `b26404b8` |
| **Source file** | `forcefields/ru-hydrogenation-ketone.fld` |
| **Status** | ⚠️ FF available but **no QM training data** in repos — Check 1 blocked |

### Sulfone (anomeric effect model)

| Property | Value |
|----------|-------|
| **Source repo** | [Q2MM/q2mm](https://github.com/Q2MM/q2mm) commit `b26404b8` |
| **Source file** | `forcefields/sulfone.fld` |
| **Status** | ⚠️ FF available but **no QM training data** in repos — Check 1 blocked |

## Validation Status

| System | Check 1 (FF eval) | Check 2 (re-derivation) | QM Data |
|--------|-------------------|-------------------------|---------|
| Rh-enamide | ✅ `test_published_ff_validation.py` | 🔲 Pending | In repo |
| Heck relay | 🔲 Ready to implement | 🔲 Pending | Rosales dissertation |
| Pd-allyl | 🔲 Ready to implement | 🔲 Pending | Wahlers dissertation |
| Pd 1,4-conj | 🔲 Ready to implement | 🔲 Pending | Wahlers dissertation |
| Ferrocene | 🔲 Ready to implement | 🔲 Pending | Both dissertations |
| OsO₄ | 🔲 Blocked (no QM data) | 🔲 Blocked | — |
| Ru ketone | 🔲 Blocked (no QM data) | 🔲 Blocked | — |
| Sulfone | 🔲 Blocked (no QM data) | 🔲 Blocked | — |
