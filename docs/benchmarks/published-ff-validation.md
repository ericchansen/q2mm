# Publication Reproduction Coverage

Publication systems are q2mm's strongest software-path tests because they
combine multiple organometallic structures, QM Hessians, a complete MM3
template, a smaller active OPT region, and explicit frozen parameters.
They are not all exact publication reproductions.

The canonical, source-linked coverage report is
[Publication Force-Field Coverage](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md).
It records:

- the five reproduction-status values used in saved provenance;
- authoritative case membership and order;
- the default `repository-geometry-eigenmatrix-v1` weights;
- source objective categories that are implemented, available, omitted, or
  blocked;
- separate published and QFUERZA starts;
- the 23-case Heck archive row and blocked 24-case publication row;
- the 21 primary and four auxiliary Pd-allyl structures;
- the developmental status of Wahlers 2021 Chapter 6 Rh-conjugate;
- the seven-ground-state-structure/four-scan Ferrocene profile; and
- the blocked OsO4, Ru ketone, and sulfone historical records.

## What is executable

The generic SDK can prepare, evaluate, enter a bounded optimizer, and save every
provisionable row when the caller supplies the required external roots.
Compatibility identities are committed as path-free fingerprints rather than
raw scientific data:

- [`publication_problem_compatibility.json`](https://github.com/ericchansen/q2mm/blob/master/test/fixtures/publication_problem_compatibility.json)
- [`ferrocene_publication_profile.json`](https://github.com/ericchansen/q2mm/blob/master/test/fixtures/ferrocene_publication_profile.json)

Grouped relative energies/enthalpies and equilibrium-parameter tethers are
executable in both Python and JAX objective executors. Atomic-charge, direct
ESP, and constrained-scan observations are typed and fingerprintable, but an
executor without the required calculated data raises a typed error rather than
dropping them.

## Claim boundary

Current geometry/eigenmatrix rows are partial repository reproductions unless
the canonical report says otherwise. Starting from published parameters does
not restore omitted electrostatic, torsional, energy, scan, or tether terms.
Likewise, QFUERZA is a starting-parameter method
([Farrugia et al.](https://doi.org/10.1021/acs.jctc.5c01751)); it does not
replace a publication's objective.

No benchmark result number is published on this page. Numerical claims must
link to a committed artifact in
[`ericchansen/q2mm-data`](https://github.com/ericchansen/q2mm-data).
