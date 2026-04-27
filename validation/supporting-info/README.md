# Validation Supporting Information

This directory contains external datasets used for validating q2mm against
published Q2MM force fields. The data is **not committed to git** (~1.9 GB)
due to its size and the presence of proprietary Gaussian output.

## How to Obtain

The data originates from two University of Notre Dame dissertations. Four zip
files are needed:

| Zip file | Size | Contents |
|----------|------|----------|
| `Wahlers_Jessica_Supporting_information.zip` | 87 MB | 5 reaction systems with published FFs and Gaussian DFT training sets |
| `Rosales_Anthony_Supporting_Information-*.zip` | 173 MB | 6 chapters covering Heck, ATH, ferrocene, VCP, BCH₃, HMGR |
| `version003_20190506-*.zip` | 117 MB | Rosales dissertation LaTeX source and figures |
| `drive-download-*.zip` | 588 MB | Rosales dissertation repository (PDFs, slides, LaTeX) |

Contact the repository maintainer for access to these files, or obtain them
from the original dissertation supporting information archives at the
University of Notre Dame.

### Setup

Extract into this directory so the structure looks like:

```
validation/supporting-info/
├── README.md           (this file, committed)
├── wahlers/            (gitignored)
│   └── Wahlers_Jessica_Supporting_information/
├── rosales/            (gitignored)
│   └── Rosales_Anthony_Supporting_Information/
├── version003/         (gitignored)
│   └── version003_20190506/
└── drive-download/     (gitignored)
```

## Data Inventory

### Wahlers Dissertation (Jessica Wahlers)

| Chapter | Reaction System | FF File | Gaussian Logs | Freq Data | Publication |
|---------|----------------|---------|---------------|-----------|-------------|
| Ch 3 | Pd-allyl amination | `mm3.Pd-allyl.fld` (OPT-only, 238 lines) | 21 | 21/21 | Wahlers et al. *Nat. Commun.* **2021**, 12, 6508. [DOI: 10.1038/s41467-021-27065-2](https://doi.org/10.1038/s41467-021-27065-2) |
| Ch 4 | Ferrocene scaffold | `mm3.ferrocene.fld` (OPT-only, 71 lines) | 178 | 99/178 | Wahlers et al. *J. Org. Chem.* **2022**, 87, 12334. [DOI: 10.1021/acs.joc.2c01553](https://doi.org/10.1021/acs.joc.2c01553) |
| Ch 5 | Pd 1,4-conjugate addition | `mm3.Pd-1,4.fld` (OPT-only, 157 lines) | 318 | 216/318 | Wahlers et al. *J. Org. Chem.* **2021**, 86, 5660. [DOI: 10.1021/acs.joc.0c02918](https://doi.org/10.1021/acs.joc.0c02918) |
| Ch 6 | Rh 1,4-conjugate addition | `mm3.Rh-1,4.fld` (OPT-only, 209 lines) | 10 | 10/10 | Thesis only |
| Ch 7 | Ir-imine hydrogenation | `mm3.Ir-imine.fld` (OPT-only, 174 lines) | 24 | 24/24 | Thesis only |

**DFT method**: M06/gen pseudo=read empiricaldispersion=GD3 freq=noraman

**Note**: Wahlers FF files are standalone OPT-substructure-only files. They
need to be composed with the standard MM3 base (`validation/published_ffs/mm3_base.fld`)
plus metal-specific vdW entries to create a complete force field.

### Rosales Dissertation (Anthony Rosales)

| Chapter | Reaction System | FF File(s) | Gaussian Logs | Freq Data | Publication |
|---------|----------------|------------|---------------|-----------|-------------|
| Ch 3 | Redox-relay Heck | `mm3.FF1.fld`, `mm3.FF2.fld` (full MM3) | 81 | 57/81 | Rosales et al. *JACS* **2020**, 142, 9700. [DOI: 10.1021/jacs.0c01979](https://doi.org/10.1021/jacs.0c01979) |
| Ch 4 | Asymmetric transfer hydrogenation | `mm3.gasphase.fld`, `mm3.solvent.fld` (full MM3) | 198 | 198/198 | Rosales thesis |
| Ch 5 | Ferrocene scaffold | 3× `mm3.fld` (full MM3, 3 contexts) | 67 | 28/67 | Overlaps with Wahlers Ch 4 |
| Ch 6 | Vinylcyclopropane | — | 38 | 28/38 | Rosales thesis |
| Ch 7 | Boronic acid (BCH₃) | — | 61 | 15/61 | Rosales thesis |
| Ch 8 | HMG-CoA reductase | — (no FF) | 0 | 0 | Biochemistry, not Q2MM |

**DFT method (Heck)**: freq=(noraman,HPModes) — HPModes provides high-precision
normal mode output, preferred for Hessian fitting.

**Key extra files**:
- `Chapter3_Heck/Predictions/SelectivityPredictions.xlsx` — 184 TSFF predictions
  vs experimental enantioselectivities
- `atom.typ` files in Ch 3, 4, 5 — MacroModel custom atom type dictionaries
- `readme` files in Ch 3, 4, 5 — data documentation

### Rosales Thesis Source (version003, drive-download)

The `version003/` and `drive-download/` directories contain Rosales'
dissertation LaTeX source, figures (.eps), final PDFs, and defense slides.
These provide context for interpreting the data (structure naming conventions,
methodology descriptions) but contain no computational data usable by q2mm.

## Totals

- **996** Gaussian DFT log files (696 with frequency/Hessian data)
- **12** published MM3* force field files
- **134** MacroModel structure files (.mae/.maegz)
- **12** spreadsheets (.xlsx) including selectivity predictions
- **~1.9 GB** total extracted size
