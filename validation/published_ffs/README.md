# Published Force Fields

This directory contains copies of force fields published in peer-reviewed papers
and their provenance documentation. These FFs are used for **Check 1** validation:
proving that the new q2mm engines can evaluate published FFs and reproduce
the expected quality of fit to QM reference data.

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

| System | Check 1 (FF eval) | Check 2 (re-derivation) |
|--------|-------------------|-------------------------|
| Rh-enamide | ✅ `test_published_ff_validation.py` | 🔲 Pending |
| OsO₄ | 🔲 Blocked (no QM data) | 🔲 Blocked |
| Ru ketone | 🔲 Blocked (no QM data) | 🔲 Blocked |
| Sulfone | 🔲 Blocked (no QM data) | 🔲 Blocked |
