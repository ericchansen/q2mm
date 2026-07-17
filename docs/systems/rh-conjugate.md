# Rh 1,4-conjugate addition

This is a **developmental SDK demonstration**, not a mature publication
validation target.

## Source and membership

The authoritative source is Wahlers, *Ph.D. Dissertation*, University of Notre
Dame, **2021**, Chapter 6,
[10.7274/k930bv76q4n](https://doi.org/10.7274/k930bv76q4n)
(Zotero `AAZ6I5V3`). There is no unrelated 2022 publication claim.

- Membership: eight bisphosphine cases followed in the source workflow by two
  diene cases; the compatibility row preserves all ten in its frozen order.
- Composition: five OPT-only blocks over a caller-supplied MM3 base.
- Source disposition: initial/developmental and missing training chemistry.

## Current claim

| Row | Status | Substantiation |
|---|---|---|
| Ten-case `repository-geometry-eigenmatrix-v1` path | `sdk_software_path_demonstration` | [Chapter 6 developmental evidence](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) |

The example proves loading, composition, preparation, active/frozen identity,
evaluation, optimizer entry, and persistence. It does not claim mature TSFF
validation or exact source-objective reproduction.

## Run

```bash
python examples/publication/rh-conjugate/run.py \
  --supporting-info /path/to/publication-data \
  --mm3-base /path/to/mm3_base.fld \
  --output-root /path/to/output \
  --bounded-ci
```
