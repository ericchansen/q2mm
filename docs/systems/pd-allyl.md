# Pd-allyl

Pd-allyl demonstrates composition of a caller-supplied MM3 base with an
OPT-only publication field and an exact frozen/active scalar partition.

## Source and membership

- Article: Wahlers et al.,
  [*Nature Communications* **2021**, 12, 6508](https://doi.org/10.1038/s41467-021-27065-2)
  (Zotero `QVKE99W3`).
- Derivation: Wahlers, *Ph.D. Dissertation*, University of Notre Dame,
  **2021**, Chapter 3,
  [10.7274/k930bv76q4n](https://doi.org/10.7274/k930bv76q4n)
  (Zotero `AAZ6I5V3`).
- Primary set: `TS1`–`TS21`.
- Auxiliary oxazole-fitting set: `TS22`–`TS25`.
- Force field: eight physical OPT blocks composed with an external MM3 base.

## Current claims

| Row | Status | Substantiation |
|---|---|---|
| 21 primary cases, `repository-geometry-eigenmatrix-v1` | `partial_repository_reproduction` | [Primary/auxiliary evidence](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) |
| Complete eight-block rederivation across 25 cases | `blocked_historical_record` | [Missing TS22–TS25 Hessian blocker](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) |

No charge, torsion, or equilibrium-tether source target is silently substituted
by the compatibility profile.

## Run

```bash
python examples/publication/pd-allyl/run.py \
  --supporting-info /path/to/publication-data \
  --mm3-base /path/to/mm3_base.fld \
  --output-root /path/to/output \
  --bounded-ci
```
