# Publication Force-Field Coverage

This is the canonical inventory of publication-scale force-field problems in
q2mm. It tells a new user **which scientific problem can actually be built**
and **how strong a reproduction claim the current objective supports**.
Loading a published force field is not, by itself, an exact reproduction of
the optimization that produced it.

The machine-readable source of the same claim boundaries is
[`q2mm.benchmarks.publications`](../../q2mm/benchmarks/publications.py).
No recovered dissertation archive or licensed MM3 file is distributed by this
repository.

## Reproduction-status vocabulary

Every publication profile and saved run uses one of these exact values:

| Status | Meaning |
|---|---|
| `exact_publication_reproduction` | Membership, source values, objective categories, weights, units, force-field composition, tethers, and optimization protocol all match the governing source. |
| `executable_archive_reproduction` | The executable deposited archive is represented, while a documented conflict with the publication remains visible. |
| `partial_repository_reproduction` | The repository's established geometry/eigenmatrix problem is preserved, but source objective categories are omitted or blocked. |
| `sdk_software_path_demonstration` | Loading, preparation, active/frozen partitioning, evaluation, optimizer entry, and persistence are exercised without a mature publication-validation claim. |
| `blocked_historical_record` | A known row cannot be constructed from authoritative available inputs. |

No currently provisionable row is labeled
`exact_publication_reproduction`.

Full canonical smooth-gradient optimization is currently a separate
`blocked_methodology` proof for every relaxed-geometry publication row. The
loader, preparation, evaluation, bounded optimizer-entry, and save paths remain
provisionable. A future scientific-methodology change must define local-basin
semantics before convergence can be claimed; this PR does not invent a
restraint or continuation policy.

## Coverage at a glance

Each comparison row links to the system-specific evidence and blocker details
below.

| System | Authoritative membership | Provisionable claim | Exact row |
|---|---:|---|---|
| [Rh-enamide](#rh-enamide) | 9 TS structures | `partial_repository_reproduction` | Blocked by omitted ESP-charge and relative-enthalpy objective terms; installed-data distribution policy also remains unresolved. |
| [Heck relay](#heck-relay) | 23 deposited / 24 described | 23-case `executable_archive_reproduction` with a partial objective | 24-case row is blocked by the sole absent member, `prrts1`. |
| [Pd-allyl](#pd-allyl) | 21 primary + 4 auxiliary | 21-case `partial_repository_reproduction` | Exact eight-block rederivation is blocked by missing auxiliary `TS22`–`TS25` Hessian data. |
| [Pd 1,4-conjugate addition](#pd-14-conjugate-addition) | `TS1`–`TS10` | `partial_repository_reproduction` | Source electrostatic, torsion, and equilibrium-tether terms are not reconstructed. |
| [Rh 1,4-conjugate addition](#rh-14-conjugate-addition) | 8 bisphosphine + 2 diene cases | `sdk_software_path_demonstration` | The Chapter 6 force field is developmental, not a mature validation target. |
| [Ferrocene](#ferrocene) | 7 ground-state structures + 4 scans | Seven-structure `partial_repository_reproduction` | Exact reoptimization is blocked by four absent numerical scan data sets. |
| [Historical force-field-only rows](#blocked-historical-records) | No authoritative training problem | `blocked_historical_record` | OsO4, Ru ketone, and sulfone remain blocked. |

## Compatibility objective

`repository-geometry-eigenmatrix-v1` is the default for existing publication
loaders. Its observations and existing lexicographic/natural source order are
frozen by
[`publication_problem_compatibility.json`](../../test/fixtures/publication_problem_compatibility.json):

| Included target | Repository weight | Governing-source comparison |
|---|---:|---|
| Bond length | `10.0 Å⁻¹` | The general Q2MM source tolerance corresponds to `100 Å⁻¹`. |
| Bond angle | `5.0 degree⁻¹` | The general Q2MM source tolerance corresponds to `2 degree⁻¹`. |
| Eigenmatrix diagonal | `0.1` | Source value `0.1`; reaction/negative mode remains zero-weighted. |
| Eigenmatrix off-diagonal | `0.05` | Source value `0.05`. |

The source weights and fitting order are documented in the
[Wahlers dissertation](https://doi.org/10.7274/k930bv76q4n) and independently
in the [Rosales dissertation](https://doi.org/10.7274/rj430290902).
`ObservationSet.from_molecules()` retains its established defaults; selecting
`published` versus `qfuerza` changes starting parameters, never observations.

### Typed objective support

Publication targets use canonical immutable types rather than callbacks:

| Target | Representation | Executor status |
|---|---|---|
| Bond, angle, and torsion geometry | Scalar observations with explicit atom indices | Python and JAX executors |
| Grouped relative energy | Explicit group, zero/reference case, energy quantity, and unit | Python and JAX executors, including analytical gradients |
| Grouped relative enthalpy | Explicit group, zero/reference case, enthalpy quantity, and unit | Typed but blocked until a backend exposes thermochemical enthalpy rather than potential energy |
| Equilibrium-parameter tether | Stable `ParameterId`, target value, slot unit, and harmonic weight | Python and JAX executors; this is distinct from optimizer bounds |
| Atomic partial charge | Atom index, value, charge unit, and weight | Typed but blocked until a backend exposes calculated atomic charges |
| Direct electrostatic potential | Cartesian grid point, potential unit, value, and weight | Typed but blocked until a backend exposes direct ESP data |
| Constrained scan energy | Group, reference, scan coordinate, energy unit, and weight | Typed but blocked until explicit constrained-scan data/backend support exists |

Unsupported backend-dependent targets raise a typed
`UnsupportedObservationError`; they are never silently dropped or replaced
with a surrogate.

## System evidence

### Rh-enamide

The governing source is Donoghue, Helquist, Norrby, and Wiest,
[*J. Chem. Theory Comput.* **2008**, 4, 1313](https://doi.org/10.1021/ct800132a).
It defines nine transition-state structures and an objective that also uses
ESP charges and relative enthalpies. The repository profile currently includes
all-nine geometry and full-eigenmatrix targets only.

The canonical source tree at
[`examples/publication/rh-enamide`](../../examples/publication/rh-enamide)
contains a complete MM3 file with two OPT blocks. Those inputs are tracked in
the source repository and excluded from wheel and sdist artifacts.
Redistribution/licensing is not established; no broader rights statement is
made.

### Heck relay

The governing article is Rosales et al.,
[*J. Am. Chem. Soc.* **2020**, 142, 9700](https://doi.org/10.1021/jacs.0c01979);
the 24-member fitting description is in the
[Rosales 2019 dissertation](https://doi.org/10.7274/rj430290902).
The deposited archive has 23 Gaussian logs. Its only patterned omission is
`prrts1`; `prrts2`, `prsts1`, and `prsts2` are present.

The provisionable row therefore represents the 23 deposited cases and the
four physical OPT blocks (`Heck Palladium`, `Palladium pyridine`,
`Palladium oxazoline`, and `Sqr Plane`). The source fitted bond dipoles to
direct CHELPG ESP error, not to atomic-charge residuals. The exact 24-case row
is a separate blocked record; q2mm never manufactures `prrts1`.

### Pd-allyl

The primary source is Wahlers et al.,
[*Nature Communications* **2021**, 12, 6508](https://doi.org/10.1038/s41467-021-27065-2),
with derivation detail in
[Wahlers 2021, Chapter 3](https://doi.org/10.7274/k930bv76q4n).
The primary set is `TS1`–`TS21`. `TS22`–`TS25` are separate auxiliary
oxazole-fitting structures. The compatibility row intentionally retains the
existing 21-case lexicographic order.

The published OPT-only file has eight physical blocks and requires an
externally supplied licensed MM3 base. Complete eight-block rederivation is
blocked until authoritative auxiliary Hessians are available.

### Pd 1,4-conjugate addition

The governing article is Wahlers et al.,
[*J. Org. Chem.* **2021**, 86, 5660](https://doi.org/10.1021/acs.joc.1c00136),
with fitting detail in
[Wahlers 2021, Chapter 5](https://doi.org/10.7274/k930bv76q4n).
The training membership is exactly `TS1`–`TS10`. The source describes four
conceptual substructure groups; the parser-visible OPT file contains six
physical blocks. It is OPT-only and requires the external MM3 base.

### Rh 1,4-conjugate addition

The authoritative source is the
[Wahlers 2021 dissertation, Chapter 6](https://doi.org/10.7274/k930bv76q4n),
not a 2022 publication. It describes staged fitting of eight bisphosphine
cases followed by two diene cases. The five-block OPT-only force field requires
the external MM3 base.

The dissertation calls this force field initial/developmental and identifies
missing training chemistry. Its repository row is therefore an SDK path
demonstration, not a mature validation claim.

### Ferrocene

The canonical Ferrocene system is Wahlers Chapter 4 and Wahlers et al.,
[*J. Org. Chem.* **2022**, 87, 12334](https://doi.org/10.1021/acs.joc.2c01553).
The article explicitly defines a **ground-state force field** with seven
training structures and four scans. `TS1`–`TS7` are archive labels, not
transition-state semantics; every training case is
`StationaryPointKind.GROUND_STATE`.

`mm3.ferrocene.fld` is OPT-only and contains exactly four blocks:
`Ferrocene_2016`, `Ferrocene_C2_Ligands_2016`,
`Ferrocene_C3_Ligands_2016`, and `Ferrocene_PX_Ligands_2016`. It is composed
with the external MM3 base. The seven-case partial evaluator explicitly
excludes the Fe atom type from nonbonded pairs because the source OPT file
defines no Fe vdW row; it does not reconstruct the D1 dummy topology.

The recovered authoritative Chapter 4 training directory contains exactly
seven Gaussian logs. External crystal, diastereomer, and selectivity
directories are not training inputs. Numerical data for the Cp rotation,
Fe-centered bend, dummy-centered bend, and Fe–dummy stretch scans are absent.
The published seven-case evaluation profile is provisionable; exact
reoptimization remains blocked. QFUERZA initialization is separately blocked
until the D1 dummy-atom topology and its frozen partition are supported rather
than guessed.

The path-free preparation/evaluation identity is committed in
[`ferrocene_publication_profile.json`](../../test/fixtures/ferrocene_publication_profile.json).
It contains only counts and fingerprints, never geometries, Hessians,
parameters, or machine paths.

## Blocked historical records

These rows are coverage records, not executable successes:

| Row | Candidate governing source | Blocker |
|---|---|---|
| OsO4 asymmetric dihydroxylation | Norrby et al., *JACS* 1999, Zotero `BT3U4GKA`, [10.1021/ja992023n](https://doi.org/10.1021/ja992023n) | The legacy file mapping is unverified and no authoritative QM training files are available. |
| Ru ketone hydrogenation | Limé et al., *JCTC* 2014, Zotero `KF2F4U5E`, [10.1021/ct500178w](https://doi.org/10.1021/ct500178w) | No authoritative QM training files are available. |
| `sulfone.fld` | Hansen et al. 2016 candidate, Zotero `RPQ4XDL2`, [10.1021/acs.jpca.6b02757](https://doi.org/10.1021/acs.jpca.6b02757) | The file-to-paper mapping is unverified and no authoritative QM training or scan data are available. |

A related title is not sufficient evidence that a legacy file belongs to that
paper. The stale, non-resolving DOI mappings previously listed here are not
used.

## External data and licensing

Wahlers OPT-only force fields need a standard MM3 base supplied by the user.
The base is intentionally not committed because its header states
"Copyright Columbia University 1990 — All rights reserved." A user with
appropriate access can obtain an MM3 base from a licensed MacroModel
installation or inspect the
[ATLAS toolkit source](https://github.com/atlas-nano/ATLAS_toolkit/blob/master/ff/macromodel/mm3.fld).

Configure external roots explicitly:

```text
Q2MM_SUPPORTING_INFO=<extracted authoritative supporting archives>
Q2MM_MM3_BASE=<licensed mm3_base.fld>
Q2MM_RH_ENAMIDE=<source-tree examples/publication/rh-enamide root>
```

No loader downloads scientific data implicitly.

## What the committed proof establishes

- The ten existing system/start rows remain byte-for-byte compatible with
  [`repository-geometry-eigenmatrix-v1`](../../test/fixtures/publication_problem_compatibility.json).
- The new Ferrocene row has numeric `TS1`–`TS7` order, ground-state semantics,
  exact OPT active membership, a path-free identity, and a real seven-case
  evaluation gate in
  [`test_ferrocene_publication.py`](../../test/test_ferrocene_publication.py).
- Every provisionable publication row enters the generic SDK optimizer and
  saves a force field plus manifest with frozen slots and reproduction status
  intact in
  [`test_publication_sdk_matrix.py`](../../test/integration/test_publication_sdk_matrix.py).
- Missing exact-source rows are asserted as explicit blocked records in
  [`test_publication_models.py`](../../test/test_publication_models.py).

These are compatibility and software-path proofs, not convergence claims.
Canonical optimization artifacts and any numerical comparison intended for
publication belong in
[`ericchansen/q2mm-data`](https://github.com/ericchansen/q2mm-data), not this
code repository.

### Optimization-proof boundary

`publication_success_spec(...)` records whether a row is a bounded software
path or a blocked canonical optimization proof. The five QFUERZA-start TS rows
under `repository-geometry-eigenmatrix-v1` and the published-start Ferrocene
seven-structure row are `blocked_methodology`: the relaxed-geometry objective
can switch local minima as parameters move, so this PR makes no convergence
claim and does not run the downstream full matrix.

The multi-signal audit remains fail-closed for future methodology work. It
requires optimizer convergence, at least one-percent improvement in the Python
objective of record, initial and final JAX/Python executor agreement, bounded
weighted-category regression, and ordinary candidate acceptance. Iteration
count and a single favorable R² are not success criteria.

The committed proof instead executes every provisionable row through real
preparation/evaluation plus a deliberately bounded optimizer-entry/save path in
[`test_publication_sdk_matrix.py`](../../test/integration/test_publication_sdk_matrix.py)
and
[`check_installed_publication_sdk.py`](../../scripts/check_installed_publication_sdk.py).
