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

### rh-enamide/ (extracted from 1.8 GB archive)

Rh-enamide transition state force field — the primary Q2MM benchmark
system (9 TS structures, Donoghue et al. 2008). Only `.fld` force field
files, `param_eig.txt`, `atom.typ`, and `rh-hydrogenation-enamide-template.mae`
are committed (the full archive includes ~1.8 GB of MacroModel `.in`/`.out`
files).

Subdirectories mirror the Zenodo archive structure:
- `fuerza/` — FUERZA (Seminario) starting point + gradient-optimized FF
- `qfuerza/` — QFUERZA initialization + gradient-optimized FF
- `approximation/` — approximation-based starting point + optimized FF

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

### Rh-enamide QFUERZA Pipeline (Zenodo archive)

The Zenodo archive contains the full FUERZA → QFUERZA → optimization
pipeline for Rh-enamide. Eigenmode fitting scores from the archive:

| Method | Score | vs FUERZA |
|--------|------:|----------:|
| FUERZA (Seminario) | 1.361 | baseline |
| QFUERZA (optimized) | 1.005 | −26% |

Key observations from the parameter comparison:
- Optimization changes most force constants dramatically (some by
  100–1000%) — Seminario/QFUERZA estimates are starting points, not
  final values
- 11 of 23 angles have QFUERZA H-angle substitution (0.5 mdyn·Å/rad²)
- Bond force constants shift substantially during optimization
  (e.g., Rh-P bond: 0.81 → 2.59 mdyn/Å, +218%)
- The C-H bond (reaction coordinate, atom 4-5) drops from 3.85 → 0.17
  mdyn/Å, consistent with TS character
