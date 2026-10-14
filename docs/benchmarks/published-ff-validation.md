# Publication reproduction coverage

Publication-scale systems are the strongest software-path tests in q2mm because
they combine many QM Hessians, full + OPT force-field composition, frozen
parameters, QFUERZA initialization, evaluation, optimizer entry, and
persistence. They do **not** all reproduce the complete source objective.

The canonical machine-readable records are in
[`q2mm.benchmarks.publications`](https://github.com/ericchansen/q2mm/blob/master/q2mm/benchmarks/publications.py);
the source-linked evidence is
[Publication Force-Field Coverage](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md).

## Current matrix

The following is a deterministic projection of **every canonical
system/profile/starting-point record**, including blocked and historical rows.
**Provisionable** means the recorded problem can be constructed when its
required inputs are supplied; it does not mean those inputs are bundled, that
optimization converges, or that the source study is exactly reproduced.

Read the [claim vocabulary](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#reproduction-status-vocabulary)
alongside each row. Target dispositions mean: `implemented` is populated in this
profile; `available` has source values but is not populated; `omitted` is not
populated; `blocked` has unresolved input or execution requirements. They are
not interchangeable. An empty membership means no authoritative training cases
are recorded, not a successful zero-case problem.

The committed Markdown and built site share the same projection. Refresh it
after changing canonical metadata with
`python scripts/gen_publication_pages.py --write`; run the command without
`--write` to check freshness. The docs build and core tests also reject stale
snapshots. This check verifies documentation consistency, **not scientific
validation**.

<!-- BEGIN GENERATED publication-profiles -->

<a id="rh-enamide"></a>

### Rh-enamide profiles

[Source evidence and limitations](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) · [System guide](../systems/rh-enamide.md)

#### `repository-geometry-eigenmatrix-v1` / `published`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `published` | 9 | `transition_state` | `partial_repository_reproduction` | Yes | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) |

**Authoritative case IDs (record order):** `1ZDMPfromJCTCSI_loner1.01`, `2DMPEConformation1fromJCTCSI_isomer1.01`, `3DMPEConformation2fromJCTCSI_isomer2.01`, `4RR-Me-DuPHOSpro-RfromJCTCSI_isomer3.01`, `5RR-Me-DuPHOSpro-SfromJCTCSI_isomer4.01`, `6RR-Me-BPEpro-RConformation1fromJCTCSI_isomer5.01`, `7RR-Me-BPEpro-RConformation2fromJCTCSI_isomer6.01`, `8RR-Me-BPEpro-SConformation1fromJCTCSI_isomer7.01`, `9RR-Me-BPEpro-SConformation2fromJCTCSI_isomer8.01`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `implemented` | Reaction/negative mode weight is 0. The source final fit used a restricted structure/Hessian selection, unlike all-nine compatibility rows. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `implemented` | The source final fit used a restricted structure/Hessian selection, unlike all-nine compatibility rows. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `blocked` | The paper used ESP charges with a 0.02 e tolerance; MM backends do not expose calculated atomic charges. |
| [`relative_enthalpy`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `available` | Source conformer/diastereomer enthalpies are tabulated; the compatibility profile intentionally omits them. |

- **Blocker:** Inputs are tracked in the source repository and excluded from distribution artifacts; redistribution/licensing is not established.
- **Note:** The published objective also used ESP charges and relative enthalpies.

#### `repository-geometry-eigenmatrix-v1` / `qfuerza`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `qfuerza` | 9 | `transition_state` | `partial_repository_reproduction` | Yes | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) |

**Authoritative case IDs (record order):** `1ZDMPfromJCTCSI_loner1.01`, `2DMPEConformation1fromJCTCSI_isomer1.01`, `3DMPEConformation2fromJCTCSI_isomer2.01`, `4RR-Me-DuPHOSpro-RfromJCTCSI_isomer3.01`, `5RR-Me-DuPHOSpro-SfromJCTCSI_isomer4.01`, `6RR-Me-BPEpro-RConformation1fromJCTCSI_isomer5.01`, `7RR-Me-BPEpro-RConformation2fromJCTCSI_isomer6.01`, `8RR-Me-BPEpro-SConformation1fromJCTCSI_isomer7.01`, `9RR-Me-BPEpro-SConformation2fromJCTCSI_isomer8.01`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `implemented` | Reaction/negative mode weight is 0. The source final fit used a restricted structure/Hessian selection, unlike all-nine compatibility rows. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `implemented` | The source final fit used a restricted structure/Hessian selection, unlike all-nine compatibility rows. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `blocked` | The paper used ESP charges with a 0.02 e tolerance; MM backends do not expose calculated atomic charges. |
| [`relative_enthalpy`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-enamide) | `available` | Source conformer/diastereomer enthalpies are tabulated; the compatibility profile intentionally omits them. |

- **Blocker:** Inputs are tracked in the source repository and excluded from distribution artifacts; redistribution/licensing is not established.
- **Note:** The published objective also used ESP charges and relative enthalpies.

<a id="heck-relay"></a>

### Heck relay profiles

[Source evidence and limitations](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) · [System guide](../systems/heck-relay.md)

#### `repository-geometry-eigenmatrix-v1` / `published`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `published` | 23 | `transition_state` | `executable_archive_reproduction` | Yes | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) |

**Authoritative case IDs (record order):** `aagts1`, `aagts2`, `bgts1`, `bgts2`, `cf3ets1`, `cf3ets2`, `dimeets1`, `dimeets2`, `epgts1`, `epgts2`, `ipets1`, `ipets2`, `meets1`, `meets2`, `prrts2`, `prsts1`, `prsts2`, `tbets1`, `tbets2`, `tmets1`, `tmets2`, `zpgts1`, `zpgts2`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `implemented` | Reaction/negative mode weight is 0. The deposited archive contains 23 of the 24 structures described in the dissertation. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `implemented` | The deposited archive contains 23 of the 24 structures described in the dissertation. |
| [`direct_electrostatic_potential`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `blocked` | Bond dipoles were fitted to direct CHELPG ESP error, not to atomic-charge residuals. |
| [`equilibrium_parameter_tether`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `blocked` | The source used equilibrium-value tethers; authoritative per-slot target values are not reconstructed. |

- **Note:** This row preserves the deposited 23-case archive and a partial repository objective.

#### `repository-geometry-eigenmatrix-v1` / `qfuerza`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `qfuerza` | 23 | `transition_state` | `executable_archive_reproduction` | Yes | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) |

**Authoritative case IDs (record order):** `aagts1`, `aagts2`, `bgts1`, `bgts2`, `cf3ets1`, `cf3ets2`, `dimeets1`, `dimeets2`, `epgts1`, `epgts2`, `ipets1`, `ipets2`, `meets1`, `meets2`, `prrts2`, `prsts1`, `prsts2`, `tbets1`, `tbets2`, `tmets1`, `tmets2`, `zpgts1`, `zpgts2`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `implemented` | Reaction/negative mode weight is 0. The deposited archive contains 23 of the 24 structures described in the dissertation. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `implemented` | The deposited archive contains 23 of the 24 structures described in the dissertation. |
| [`direct_electrostatic_potential`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `blocked` | Bond dipoles were fitted to direct CHELPG ESP error, not to atomic-charge residuals. |
| [`equilibrium_parameter_tether`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `blocked` | The source used equilibrium-value tethers; authoritative per-slot target values are not reconstructed. |

- **Note:** This row preserves the deposited 23-case archive and a partial repository objective.

#### `rosales-heck-publication-24-v1` / `published`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `published` | 24 | `transition_state` | `blocked_historical_record` | No | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) |

**Authoritative case IDs (record order):** `aagts1`, `aagts2`, `bgts1`, `bgts2`, `cf3ets1`, `cf3ets2`, `dimeets1`, `dimeets2`, `epgts1`, `epgts2`, `ipets1`, `ipets2`, `meets1`, `meets2`, `prrts1`, `prrts2`, `prsts1`, `prsts2`, `tbets1`, `tbets2`, `tmets1`, `tmets2`, `zpgts1`, `zpgts2`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `implemented` | Reaction/negative mode weight is 0. The deposited archive contains 23 of the 24 structures described in the dissertation. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `implemented` | The deposited archive contains 23 of the 24 structures described in the dissertation. |
| [`direct_electrostatic_potential`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `blocked` | Bond dipoles were fitted to direct CHELPG ESP error, not to atomic-charge residuals. |
| [`equilibrium_parameter_tether`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#heck-relay) | `blocked` | The source used equilibrium-value tethers; authoritative per-slot target values are not reconstructed. |

- **Blocker:** The sole missing member of the publication's 24-case set is prrts1.
- **Note:** No synthetic prrts1 structure is permitted.

<a id="pd-allyl"></a>

### Pd-allyl amination profiles

[Source evidence and limitations](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) · [System guide](../systems/pd-allyl.md)

#### `repository-geometry-eigenmatrix-v1` / `published`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `published` | 21 | `transition_state` | `partial_repository_reproduction` | Yes | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) |

**Authoritative case IDs (record order):** `TS1`, `TS10`, `TS11`, `TS12`, `TS13`, `TS14`, `TS15`, `TS16`, `TS17`, `TS18`, `TS19`, `TS2`, `TS20`, `TS21`, `TS3`, `TS4`, `TS5`, `TS6`, `TS7`, `TS8`, `TS9`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `implemented` | Reaction/negative mode weight is 0. The compatibility row contains the 21 primary cases only. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `implemented` | The compatibility row contains the 21 primary cases only. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `blocked` | The source electrostatic stage is not executable through current MM backend results. |
| [`equilibrium_parameter_tether`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `blocked` | The source used equilibrium-value tethers; exact per-slot targets are unavailable. |

- **Blocker:** Complete eight-block rederivation requires auxiliary TS22-TS25 oxazole Hessian data.
- **Note:** The executable compatibility row contains the 21 primary TS1-TS21 structures.

#### `repository-geometry-eigenmatrix-v1` / `qfuerza`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `qfuerza` | 21 | `transition_state` | `partial_repository_reproduction` | Yes | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) |

**Authoritative case IDs (record order):** `TS1`, `TS10`, `TS11`, `TS12`, `TS13`, `TS14`, `TS15`, `TS16`, `TS17`, `TS18`, `TS19`, `TS2`, `TS20`, `TS21`, `TS3`, `TS4`, `TS5`, `TS6`, `TS7`, `TS8`, `TS9`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `implemented` | Reaction/negative mode weight is 0. The compatibility row contains the 21 primary cases only. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `implemented` | The compatibility row contains the 21 primary cases only. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `blocked` | The source electrostatic stage is not executable through current MM backend results. |
| [`equilibrium_parameter_tether`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `blocked` | The source used equilibrium-value tethers; exact per-slot targets are unavailable. |

- **Blocker:** Complete eight-block rederivation requires auxiliary TS22-TS25 oxazole Hessian data.
- **Note:** The executable compatibility row contains the 21 primary TS1-TS21 structures.

#### `wahlers-pd-allyl-eight-block-v1` / `published`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `published` | 25 | `transition_state` | `blocked_historical_record` | No | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) |

**Authoritative case IDs (record order):** `TS1`, `TS2`, `TS3`, `TS4`, `TS5`, `TS6`, `TS7`, `TS8`, `TS9`, `TS10`, `TS11`, `TS12`, `TS13`, `TS14`, `TS15`, `TS16`, `TS17`, `TS18`, `TS19`, `TS20`, `TS21`, `TS22`, `TS23`, `TS24`, `TS25`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `implemented` | Reaction/negative mode weight is 0. The compatibility row contains the 21 primary cases only. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `implemented` | The compatibility row contains the 21 primary cases only. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `blocked` | The source electrostatic stage is not executable through current MM backend results. |
| [`equilibrium_parameter_tether`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-allyl) | `blocked` | The source used equilibrium-value tethers; exact per-slot targets are unavailable. |

- **Blocker:** Auxiliary oxazole cases TS22-TS25 lack the Hessian data required for exact eight-block rederivation.
- **Note:** The executable compatibility row contains the 21 primary TS1-TS21 structures.

<a id="pd-conjugate"></a>

### Pd 1,4-conjugate addition profiles

[Source evidence and limitations](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) · [System guide](../systems/pd-conjugate.md)

#### `repository-geometry-eigenmatrix-v1` / `published`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `published` | 10 | `transition_state` | `partial_repository_reproduction` | Yes | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) |

**Authoritative case IDs (record order):** `TS1`, `TS10`, `TS2`, `TS3`, `TS4`, `TS5`, `TS6`, `TS7`, `TS8`, `TS9`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `implemented` | Reaction/negative mode weight is 0. All ten primary structures are represented. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `implemented` | All ten primary structures are represented. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `blocked` | CHELPG electrostatics are not executable through current MM backend results. |
| [`equilibrium_parameter_tether`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `blocked` | The source used equilibrium-value tethers; exact per-slot targets are unavailable. |

- **Note:** The source describes four conceptual groups implemented as six physical OPT blocks.

#### `repository-geometry-eigenmatrix-v1` / `qfuerza`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `qfuerza` | 10 | `transition_state` | `partial_repository_reproduction` | Yes | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) |

**Authoritative case IDs (record order):** `TS1`, `TS10`, `TS2`, `TS3`, `TS4`, `TS5`, `TS6`, `TS7`, `TS8`, `TS9`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `implemented` | Reaction/negative mode weight is 0. All ten primary structures are represented. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `implemented` | All ten primary structures are represented. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `blocked` | CHELPG electrostatics are not executable through current MM backend results. |
| [`equilibrium_parameter_tether`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#pd-14-conjugate-addition) | `blocked` | The source used equilibrium-value tethers; exact per-slot targets are unavailable. |

- **Note:** The source describes four conceptual groups implemented as six physical OPT blocks.

<a id="rh-conjugate"></a>

### Rh 1,4-conjugate addition profiles

[Source evidence and limitations](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) · [System guide](../systems/rh-conjugate.md)

#### `repository-geometry-eigenmatrix-v1` / `published`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `published` | 10 | `transition_state` | `sdk_software_path_demonstration` | Yes | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) |

**Authoritative case IDs (record order):** `TS1`, `TS10`, `TS2`, `TS3`, `TS4`, `TS5`, `TS6`, `TS7`, `TS8`, `TS9`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) | `implemented` | Reaction/negative mode weight is 0. The compatibility row combines the source's staged eight-case and two-case fits. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) | `implemented` | The compatibility row combines the source's staged eight-case and two-case fits. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) | `blocked` | CHELPG electrostatics are not executable through current MM backend results. |

- **Note:** This 2021 dissertation force field is developmental, not a mature validation system.

#### `repository-geometry-eigenmatrix-v1` / `qfuerza`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `qfuerza` | 10 | `transition_state` | `sdk_software_path_demonstration` | Yes | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) |

**Authoritative case IDs (record order):** `TS1`, `TS10`, `TS2`, `TS3`, `TS4`, `TS5`, `TS6`, `TS7`, `TS8`, `TS9`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) | `implemented` | Reaction/negative mode weight is 0. The compatibility row combines the source's staged eight-case and two-case fits. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) | `implemented` | The compatibility row combines the source's staged eight-case and two-case fits. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#rh-14-conjugate-addition) | `blocked` | CHELPG electrostatics are not executable through current MM backend results. |

- **Note:** This 2021 dissertation force field is developmental, not a mature validation system.

<a id="ferrocene"></a>

### Ferrocene ground-state force field profiles

[Source evidence and limitations](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) · [System guide](../systems/ferrocene.md)

#### `repository-geometry-eigenmatrix-v1` / `published`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `published` | 7 | `ground_state` | `partial_repository_reproduction` | Yes | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) |

**Authoritative case IDs (record order):** `TS1`, `TS2`, `TS3`, `TS4`, `TS5`, `TS6`, `TS7`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Reaction/negative mode weight is 0. The source describes Hessian eigenvalues; the repository partial profile retains its full eigenmatrix recipe. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | The source describes Hessian eigenvalues; the repository partial profile retains its full eigenmatrix recipe. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `blocked` | Published point-charge/bond-dipole fitting is not executable through current MM backend results. |
| [`constrained_scan_energy`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `blocked` | Numerical data for all four source scans are absent from the recovered archive. |

- **Blocker:** Exact reoptimization requires four absent numerical constrained-scan data sets.
- **Note:** TS-prefixed archive labels do not change the governing ground-state semantics.
- **Note:** External crystal, diastereomer, and selectivity validation structures are not training cases.
- **Note:** The seven-case partial evaluator explicitly excludes the Fe atom type from nonbonded pairs because the source OPT file defines no Fe vdW row; it does not reconstruct the D1 dummy topology.

#### `wahlers-ferrocene-seven-structure-v1` / `published`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `published` | 7 | `ground_state` | `partial_repository_reproduction` | Yes | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) |

**Authoritative case IDs (record order):** `TS1`, `TS2`, `TS3`, `TS4`, `TS5`, `TS6`, `TS7`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Reaction/negative mode weight is 0. The source describes Hessian eigenvalues; the repository partial profile retains its full eigenmatrix recipe. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | The source describes Hessian eigenvalues; the repository partial profile retains its full eigenmatrix recipe. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `blocked` | Published point-charge/bond-dipole fitting is not executable through current MM backend results. |
| [`constrained_scan_energy`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `blocked` | Numerical data for all four source scans are absent from the recovered archive. |

- **Blocker:** Exact reoptimization requires four absent numerical constrained-scan data sets.
- **Note:** TS-prefixed archive labels do not change the governing ground-state semantics.
- **Note:** External crystal, diastereomer, and selectivity validation structures are not training cases.
- **Note:** The seven-case partial evaluator explicitly excludes the Fe atom type from nonbonded pairs because the source OPT file defines no Fe vdW row; it does not reconstruct the D1 dummy topology.

#### `wahlers-ferrocene-full-scan-v1` / `published`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `published` | 7 | `ground_state` | `blocked_historical_record` | No | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) |

**Authoritative case IDs (record order):** `TS1`, `TS2`, `TS3`, `TS4`, `TS5`, `TS6`, `TS7`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Reaction/negative mode weight is 0. The source describes Hessian eigenvalues; the repository partial profile retains its full eigenmatrix recipe. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | The source describes Hessian eigenvalues; the repository partial profile retains its full eigenmatrix recipe. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `blocked` | Published point-charge/bond-dipole fitting is not executable through current MM backend results. |
| [`constrained_scan_energy`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `blocked` | Numerical data for all four source scans are absent from the recovered archive. |

- **Blocker:** Numerical data are absent for all four constrained scans used by the publication.
- **Note:** TS-prefixed archive labels do not change the governing ground-state semantics.
- **Note:** External crystal, diastereomer, and selectivity validation structures are not training cases.
- **Note:** The seven-case partial evaluator explicitly excludes the Fe atom type from nonbonded pairs because the source OPT file defines no Fe vdW row; it does not reconstruct the D1 dummy topology.

#### `repository-geometry-eigenmatrix-v1` / `qfuerza`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `qfuerza` | 7 | `ground_state` | `blocked_historical_record` | No | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) |

**Authoritative case IDs (record order):** `TS1`, `TS2`, `TS3`, `TS4`, `TS5`, `TS6`, `TS7`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Reaction/negative mode weight is 0. The source describes Hessian eigenvalues; the repository partial profile retains its full eigenmatrix recipe. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | The source describes Hessian eigenvalues; the repository partial profile retains its full eigenmatrix recipe. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `blocked` | Published point-charge/bond-dipole fitting is not executable through current MM backend results. |
| [`constrained_scan_energy`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `blocked` | Numerical data for all four source scans are absent from the recovered archive. |

- **Blocker:** QFUERZA initialization is unsupported until the D1 dummy-atom topology is represented and its frozen partition is proven.
- **Note:** TS-prefixed archive labels do not change the governing ground-state semantics.
- **Note:** External crystal, diastereomer, and selectivity validation structures are not training cases.
- **Note:** The seven-case partial evaluator explicitly excludes the Fe atom type from nonbonded pairs because the source OPT file defines no Fe vdW row; it does not reconstruct the D1 dummy topology.

#### `wahlers-ferrocene-seven-structure-v1` / `qfuerza`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `qfuerza` | 7 | `ground_state` | `blocked_historical_record` | No | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) |

**Authoritative case IDs (record order):** `TS1`, `TS2`, `TS3`, `TS4`, `TS5`, `TS6`, `TS7`

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`bond_length`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`bond_angle`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Compatibility profile preserves the repository weight byte-for-byte. |
| [`torsion_geometry`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `omitted` | The canonical torsion observation exists, but this compatibility profile does not populate it. |
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | Reaction/negative mode weight is 0. The source describes Hessian eigenvalues; the repository partial profile retains its full eigenmatrix recipe. |
| [`eigenmatrix_offdiagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `implemented` | The source describes Hessian eigenvalues; the repository partial profile retains its full eigenmatrix recipe. |
| [`atomic_partial_charge`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `blocked` | Published point-charge/bond-dipole fitting is not executable through current MM backend results. |
| [`constrained_scan_energy`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#ferrocene) | `blocked` | Numerical data for all four source scans are absent from the recovered archive. |

- **Blocker:** QFUERZA initialization is unsupported until the D1 dummy-atom topology is represented and its frozen partition is proven.
- **Note:** TS-prefixed archive labels do not change the governing ground-state semantics.
- **Note:** External crystal, diastereomer, and selectivity validation structures are not training cases.
- **Note:** The seven-case partial evaluator explicitly excludes the Fe atom type from nonbonded pairs because the source OPT file defines no Fe vdW row; it does not reconstruct the D1 dummy topology.

<a id="osmium-dihydroxylation"></a>

### osmium-dihydroxylation profiles

[Source evidence and limitations](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#blocked-historical-records)

#### `historical-force-field-record-v1` / `not specified`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `not specified` | 0 | `transition_state` | `blocked_historical_record` | No | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#blocked-historical-records) |

**Authoritative case IDs (record order):** None recorded.

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#blocked-historical-records) | `blocked` |  |

- **Blocker:** No authoritative QM training files are available and the legacy force-field-to-publication mapping is unverified.

<a id="ru-ketone-hydrogenation"></a>

### ru-ketone-hydrogenation profiles

[Source evidence and limitations](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#blocked-historical-records)

#### `historical-force-field-record-v1` / `not specified`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `not specified` | 0 | `transition_state` | `blocked_historical_record` | No | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#blocked-historical-records) |

**Authoritative case IDs (record order):** None recorded.

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`eigenmatrix_diagonal`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#blocked-historical-records) | `blocked` |  |

- **Blocker:** No authoritative QM training files are available.

<a id="sulfone"></a>

### sulfone profiles

[Source evidence and limitations](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#blocked-historical-records)

#### `historical-force-field-record-v1` / `not specified`

| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |
|---|---:|---|---|---|---|
| `not specified` | 0 | `unknown` | `blocked_historical_record` | No | [Source/status details](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#blocked-historical-records) |

**Authoritative case IDs (record order):** None recorded.

| Target category | Disposition | Canonical detail |
|---|---|---|
| [`constrained_scan_energy`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md#blocked-historical-records) | `blocked` |  |

- **Blocker:** The legacy file-to-paper mapping is unverified and no authoritative QM training or scan data are available.

<!-- END GENERATED publication-profiles -->

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

Historical force-field-only records are included in the generated matrix, not
counted as executable successes. Candidate citations and unverified
file-to-paper mappings are explained in the
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
