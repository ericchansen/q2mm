# Pd 1,4-conjugate addition

This ten-structure Pd system independently tests an OPT-only overlay composed
with a caller-supplied MM3 base.

## Source and membership

- Article: Wahlers et al.,
  [*J. Org. Chem.* **2021**, 86, 5660–5667](https://doi.org/10.1021/acs.joc.1c00136)
  (Zotero `R62E6EGV`).
- Derivation: Wahlers, *Ph.D. Dissertation*, University of Notre Dame,
  **2021**, Chapter 5,
  [10.7274/k930bv76q4n](https://doi.org/10.7274/k930bv76q4n)
  (Zotero `AAZ6I5V3`).
- Membership: exactly `TS1`–`TS10`, preserving the repository compatibility
  order.
- Composition: four conceptual source groups represented by six parser-visible
  OPT blocks plus an external MM3 base.

## Current claim

| Row | Status | Substantiation |
|---|---|---|
| `repository-geometry-eigenmatrix-v1`, published or QFUERZA start | `partial_repository_reproduction` | [Ten-case and six-block evidence](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) |

The source electrostatic, torsion, and equilibrium-tether terms are not
reconstructed by this compatibility row.

## Run

```bash
python examples/publication/pd-conjugate/run.py \
  --supporting-info /path/to/publication-data \
  --mm3-base /path/to/mm3_base.fld \
  --output-root /path/to/output \
  --bounded-ci
```
