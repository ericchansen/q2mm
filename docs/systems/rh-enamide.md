# Rh-enamide

Rh-enamide is the first full bring-your-own-system case because it combines
nine QM transition-state structures, a complete MM3 field, two custom OPT
regions, a frozen base, QFUERZA initialization, objective evaluation, optimizer
entry, and persistence.

## Source and membership

- Governing article: Donoghue, Helquist, Norrby, and Wiest,
  [*J. Chem. Theory Comput.* **2008**, 4, 1313–1323](https://doi.org/10.1021/ct800132a)
  (Zotero `JXH5HHS6`).
- Authoritative membership: nine transition-state structures in the source
  sequence preserved by the loader.
- Force field: one complete source MM3 file with the RhH3-E core and RH-PX OPT
  regions.

The inputs are tracked at `examples/publication/rh-enamide`, but wheel and sdist
artifacts exclude the complete examples tree. Redistribution/licensing is not
established; no broader rights statement is made.

## Current claim

| Row | Status | Substantiation |
|---|---|---|
| `repository-geometry-eigenmatrix-v1`, published or QFUERZA start | `partial_repository_reproduction` | [Canonical Rh-enamide evidence and objective gaps](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) |

The repository profile includes all-nine bond, angle, and full-eigenmatrix
targets. The article also used ESP charges and relative enthalpies; those are
not silently approximated. Relative enthalpy remains typed but blocked until an
MM backend exposes thermochemical enthalpy.

## Run

```bash
python examples/publication/rh-enamide/run.py \
  --rh-enamide /path/to/q2mm/examples/publication/rh-enamide \
  --output-root /path/to/output \
  --bounded-ci
```

The JSON result reports citation/source status, exact order, active/frozen
counts, QFUERZA audit, objectives/categories, execution policy, and saved paths.
Remove `--bounded-ci` only when you intend to run the documented scientific
optimizer. See the [tutorial](../tutorial.md#first-full-case-rh-enamide).
