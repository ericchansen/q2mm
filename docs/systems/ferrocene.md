# Ferrocene ground-state force field

The canonical Ferrocene profile is Wahlers Chapter 4: seven ground-state
training structures plus four scans. `TS1`–`TS7` are archive labels, not
transition-state semantics.

## Source and membership

- Article: Wahlers et al.,
  [*J. Org. Chem.* **2022**, 87, 12334–12341](https://doi.org/10.1021/acs.joc.2c01553)
  (Zotero `SXWNJTQ2`).
- Derivation: Wahlers, *Ph.D. Dissertation*, University of Notre Dame,
  **2021**, Chapter 4,
  [10.7274/k930bv76q4n](https://doi.org/10.7274/k930bv76q4n)
  (Zotero `AAZ6I5V3`).
- Training structures: exactly seven Gaussian logs, `TS1`–`TS7`, all
  `ground_state`.
- Additional fitting data: four numerical scans described by the source but
  absent from the recovered archive.
- Composition: four OPT blocks over a caller-supplied MM3 base.

## Current claims

| Row | Status | Substantiation |
|---|---|---|
| `wahlers-ferrocene-seven-structure-v1`, published start | `partial_repository_reproduction` | [Seven-case ground-state evidence](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) |
| Full scan reoptimization | `blocked_historical_record` | [Four absent numerical scans](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) |
| QFUERZA start | `blocked_historical_record` | [Unproven D1 dummy topology/frozen partition](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) |

The provisionable row explicitly excludes Fe from nonbonded pairs because the
source OPT file has no Fe vdW row. It does not invent D1 dummy topology.

## Run

```bash
python examples/publication/ferrocene/run.py \
  --supporting-info /path/to/publication-data \
  --mm3-base /path/to/mm3_base.fld \
  --output-root /path/to/output \
  --bounded-ci
```

The script permits only the published start and reports both scan and QFUERZA
blockers in its JSON output.
