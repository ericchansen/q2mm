# Check 1: Published Force Field Evaluation

This page answers one question: can q2mm load a published force field and
evaluate it against the original QM reference data? It is Check 1 in the
published-validation program and is separate from the optimizer benchmark
matrices.

## Scope

- System: Rh-enamide hydrogenation from Donoghue et al. 2008
- Published force field: MM3* force field optimized with the original Q2MM code
  plus MacroModel
- Engine under test: OpenMM MM3 custom force implementation
- Current status: the evaluation harness works, but the parity gap is still
  unresolved

The current Check 1 result is intentionally simple to interpret: the published
force field performs far worse than the Seminario baseline under q2mm/OpenMM.
The published-force-field objective score is 139,910.7 versus 36.1 for the
Seminario baseline, and the overall RMSD is 13,576.9 cm⁻¹. That means the
validation path is operational, but historical MM3 parity has not yet been
reproduced.

## Per-molecule Check 1 result

| Molecule | Atoms | Freq refs | RMSD (cm⁻¹) | MAE (cm⁻¹) | R^2 |
|----------|------:|----------:|-------------:|------------:|----:|
| TS 1 | 36 | 54 | 13680.1 | 7919.9 | -2033.1 |
| TS 2 | 38 | 59 | 13771.9 | 8021.4 | -1753.1 |
| TS 3 | 38 | 59 | 13773.7 | 8023.5 | -1749.1 |
| TS 4 | 62 | 101 | 13348.1 | 7543.9 | -1297.4 |
| TS 5 | 62 | 100 | 13429.0 | 7627.2 | -1336.0 |
| TS 6 | 58 | 97 | 13567.5 | 7801.8 | -1204.4 |
| TS 7 | 58 | 98 | 13488.9 | 7730.0 | -1175.3 |
| TS 8 | 58 | 97 | 13565.8 | 7816.1 | -1206.3 |
| TS 9 | 58 | 97 | 13567.1 | 7817.2 | -1207.3 |
| **Average** |  | **762** | **13576.9** | **7811.2** | **-1440.2** |

## Interpretation

- The evaluation harness is not the problem. q2mm can load the published force
  field, evaluate it, and preserve the result as a regression fixture.
- The parity problem is substantive: the published force field should beat the
  Seminario baseline, but under q2mm/OpenMM it performs much worse.
- The leading explanation is still an MM3 implementation mismatch between the
  original MacroModel-based workflow and the current OpenMM custom-force path.
  Likely sources include functional-form differences, parameter-interpretation
  differences, and missing or differently handled interaction terms.
- This gap is tracked in [issue #197](https://github.com/ericchansen/q2mm/issues/197).
  Check 2 (re-derivation) remains blocked on resolving Check 1 first.

## Artifacts and provenance

- Paper:
  Donoghue et al. *J. Chem. Theory Comput.* **2008**, *4*, 1313-1323.
  [DOI: 10.1021/ct800132a](https://doi.org/10.1021/ct800132a)
- Force-field source:
  `examples/rh-enamide/ff/rh_hyd_enamide_final.fld`
  (provenance: [Q2MM/q2mm](https://github.com/Q2MM/q2mm) commit `b26404b8`,
  `forcefields/rh-hydrogenation-enamide.fld`)
- Golden fixture:
  `test/fixtures/published_ff/rh_enamide_donoghue2008.json`
- Test harness:
  `test/integration/test_published_ff_validation.py`
- Provenance notes:
  `validation/published_ffs/README.md`

## Reproducing

```bash
python3 -m pytest test/integration/test_published_ff_validation.py --run-slow -v
Q2MM_UPDATE_GOLDEN=1 python3 -m pytest test/integration/test_published_ff_validation.py --run-slow -v
```
