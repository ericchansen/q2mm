# Heck relay

The Heck relay case distinguishes the executable deposited archive from the
larger membership described by its source. Q2MM never manufactures the missing
structure.

## Source and membership

- Article: Rosales et al.,
  [*J. Am. Chem. Soc.* **2020**, 142, 9700–9707](https://doi.org/10.1021/jacs.0c01979)
  (Zotero `2NHVUNW5`).
- Membership evidence: Rosales, *Ph.D. Dissertation*, University of Notre Dame,
  **2019**, [10.7274/rj430290902](https://doi.org/10.7274/rj430290902)
  (Zotero `QCQ6Z5MR`).
- Deposited archive: 23 transition-state logs.
- Described set: 24; the sole absent patterned member is `prrts1`.
- Force field: complete MM3 field with four physical OPT blocks.

## Current claims

| Row | Status | Substantiation |
|---|---|---|
| 23 deposited cases, `repository-geometry-eigenmatrix-v1` | `executable_archive_reproduction` with a partial objective | [23/24 archive evidence](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) |
| 24 described cases | `blocked_historical_record` | [Missing `prrts1` blocker](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) |

The source fitted bond dipoles to direct CHELPG ESP error and used torsion and
equilibrium-tether terms. The compatibility objective does not reconstruct
those categories.

## Run

```bash
python examples/publication/heck-relay/run.py \
  --supporting-info /path/to/publication-data \
  --output-root /path/to/output \
  --bounded-ci
```

The scientific default is explicitly tighter than other TS examples:
`fc_fraction=0.05`; this is a Heck configuration, not a generic inference.
