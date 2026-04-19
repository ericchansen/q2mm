# QFUERZA Zenodo Archive Data

Reference data from the QFUERZA paper's Zenodo archive for validation.

## Source

- **Paper**: Farrugia, M.; Helquist, P.; Norrby, P.-O.; Wiest, O.
  *J. Chem. Theory Comput.* **2025**, *22*, 469–476.
- **DOI**: [10.1021/acs.jctc.5c01751](https://doi.org/10.1021/acs.jctc.5c01751)
- **Zenodo**: [10.5281/zenodo.17386006](https://doi.org/10.5281/zenodo.17386006)

## Contents

### cisplatin/ (10 MB, fully extracted)

Ground-state cisplatin (cis-[Pt(NH₃)₂Cl₂]), M06-2X/genecp + GD3
optimized geometry and frequencies. Contains:

- `cisplatin_opt_freq_m06.log` — Gaussian frequency log (M06-2X/genecp + GD3)
- `fuerza/cisplatin_fuerza.fld` — FUERZA (Seminario) force field
- `qfuerza/qfuerza_H.fld` — QFUERZA force field (H-angle defaults applied)
- `gamma_fuerza/cisplatin_gamma_fuerza.fld` — γ-FUERZA variant
- Score files for each method variant

### Rh-enamide (1.8 GB, NOT downloaded)

The Rh-enamide TSFF data (1.8 GB) is too large to commit. Download
manually if needed:

```bash
curl -L -o /tmp/rh_enamide.zip \
  "https://zenodo.org/api/records/17386006/files/rh_hydrogn_enamides_TSFF.zip/content"
```

## Validation Results

### Cisplatin FUERZA Bond Force Constants (mdyn/Å, no DFT scaling)

| Bond  | Zenodo | Ours   | Δ%      | Notes |
|-------|--------|--------|---------|-------|
| N-Pt  | 1.1687 | 1.1687 | **0.0%** | ✅ exact match (no DFT scaling) |
| N-H   | 5.7649 | 6.7016 | +16.3%  | ⚠️ averaged over 6 N-H bonds; tracked in #236 |
| Pt-Cl | 1.3918 | 1.6276 | +16.9%  | ⚠️ tracked in #236 |

**Key finding**: N-Pt matches exactly *without* DFT frequency scaling
(`dft_scaling=1.0`), confirming the paper did not apply any scaling factor
to the M06-2X Hessian. The Pt-Cl and N-H divergences persist regardless
of scaling and likely stem from a different Hessian processing path in the
paper's unpublished code fork.

### Cisplatin QFUERZA Angle Force Constants (mdyn·Å/rad²)

| Angle   | Zenodo | Ours   | Status |
|---------|--------|--------|--------|
| H-N-Pt  | 0.5000 | 0.5000 | ✅ exact |
| H-N-H   | 0.5000 | 0.5000 | ✅ exact |

**Status**: QFUERZA H-angle substitution works correctly. Bond force
constant divergences (Pt-Cl, N-H) are tracked in issue #236 and likely
stem from a different Hessian processing path in the paper's code fork.
