# Publication reproduction coverage

Publication-scale systems are the strongest software-path tests in q2mm because
they combine many QM Hessians, full + OPT force-field composition, frozen
parameters, QFUERZA initialization, evaluation, optimizer entry, and
persistence. They do **not** all reproduce the complete source objective.

The canonical machine-readable records are in
`q2mm.benchmarks.publications`; the source-linked evidence is
[Publication Force-Field Coverage](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md).

## Current matrix

Every comparison row links to a detailed system page and its evidence.

| System | Authoritative membership | Provisionable claim | Exact-row blocker |
|---|---:|---|---|
| [Rh-enamide](../systems/rh-enamide.md) | 9 TS structures | `partial_repository_reproduction` | Source ESP-charge and relative-enthalpy terms are omitted; source inputs are tracked but excluded from distributions and have no established redistribution/licensing statement. |
| [Heck relay](../systems/heck-relay.md) | 23 deposited / 24 described | 23-case `executable_archive_reproduction` with a partial objective | `prrts1` is absent. |
| [Pd-allyl](../systems/pd-allyl.md) | 21 primary + 4 auxiliary | 21-case `partial_repository_reproduction` | Auxiliary `TS22`–`TS25` Hessians needed for complete eight-block rederivation are unavailable. |
| [Pd 1,4-conjugate](../systems/pd-conjugate.md) | `TS1`–`TS10` | `partial_repository_reproduction` | Source electrostatic, torsion, and equilibrium-tether terms are not reconstructed. |
| [Rh 1,4-conjugate](../systems/rh-conjugate.md) | 8 bisphosphine + 2 diene | `sdk_software_path_demonstration` | Wahlers 2021 Chapter 6 calls the field developmental. |
| [Ferrocene](../systems/ferrocene.md) | 7 ground-state structures + 4 scans | Seven-case `partial_repository_reproduction` | Four numerical scans are absent; QFUERZA also lacks proven D1 topology/frozen membership. |

No provisionable row is labeled `exact_publication_reproduction`.

## Compatibility profile

The five TS systems use `repository-geometry-eigenmatrix-v1`. It preserves the
repository bond, angle, and full-eigenmatrix observations and weights exactly.
Those weights are not silently rewritten to the general source tolerances, and
selecting published versus QFUERZA values changes the starting vector—not the
observations.

Ferrocene additionally names
`wahlers-ferrocene-seven-structure-v1`, a published-start, seven-ground-state
partial profile. The exact scan and QFUERZA rows remain blocked.

## Typed but unsupported source terms

Grouped relative energy and equilibrium-parameter tethers execute in the Python
and JAX objective paths. Relative enthalpy, atomic charge, direct ESP, and
constrained scan observations have canonical identities and fingerprints, but
raise typed errors when a backend cannot calculate the required property. Q2MM
does not replace them with potential energy, charges, optimizer bounds, or
another surrogate.

## Historical force-field-only records

OsO4, Ru ketone hydrogenation, and `sulfone.fld` remain
`blocked_historical_record` rows. The candidate citations and unverified
file-to-paper mappings are recorded in the
[blocked inventory](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#blocked-historical-records);
no authoritative QM training problem is available.

## Executable proof

The source-only scripts under
[`examples/publication/`](https://github.com/ericchansen/q2mm/tree/master/examples/publication)
run each provisionable study through the root application functions. Their
bounded mode uses the real case membership and objective, enters the optimizer
once, saves a field and manifest, and explicitly makes no convergence claim.

Numerical optimization claims require a linked canonical record in
[`ericchansen/q2mm-data`](https://github.com/ericchansen/q2mm-data). Example
outputs are not committed to this repository.
