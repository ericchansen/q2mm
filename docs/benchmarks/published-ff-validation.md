# Published Force Field Validation

## What this is

Before trusting q2mm's optimizers to produce new force fields, we need to
prove it can correctly *evaluate* force fields that have already been published
and validated against experiment. This page documents that proof — or,
honestly, the current lack of it.

The validation program has two checks, run in order:

| Check | Question | Status |
|-------|----------|--------|
| **Check 1** | Can q2mm load a published force field and reproduce its fit quality against the original QM data? | ⚠️ Harness works; parity gap likely due to MM3 functional-form differences ([#197](https://github.com/ericchansen/q2mm/issues/197), closed) |
| **Check 2** | Can q2mm re-derive the published force field from scratch using its own optimizers? | ⏳ Blocked on Check 1 |

Check 1 must pass before Check 2 makes sense — if we can't even evaluate a
known-good force field correctly, there's no point trying to re-derive it.

The umbrella tracker for the full validation program is
[issue #198](https://github.com/ericchansen/q2mm/issues/198).

---

## Check 1: published force field evaluation

**System**: Rh-enamide hydrogenation (Donoghue et al. *J. Chem. Theory
Comput.* **2008**, *4*, 1313–1323;
[DOI](https://doi.org/10.1021/ct800132a))

**What we did**: loaded the published MM3* force field (originally optimized
with Q2MM + MacroModel) and evaluated it with q2mm's OpenMM MM3 custom-force
implementation against the same 9 transition-state structures and QM
frequencies.

**What we expected**: the published force field should beat the untrained
Seminario baseline.

**What we got**: it doesn't. The published force field scores 139,910.7
(objective) with an RMSD of 13,576.9 cm⁻¹, versus 36.1 for the Seminario
baseline. Every per-molecule R² is strongly negative:

| Molecule | Atoms | Freq refs | RMSD (cm⁻¹) | MAE (cm⁻¹) | R² |
|----------|------:|----------:|------------:|------------:|---:|
| TS 1 | 36 | 54 | 13,680 | 7,920 | −2,033 |
| TS 2 | 38 | 59 | 13,772 | 8,021 | −1,753 |
| TS 3 | 38 | 59 | 13,774 | 8,024 | −1,749 |
| TS 4 | 62 | 101 | 13,348 | 7,544 | −1,297 |
| TS 5 | 62 | 100 | 13,429 | 7,627 | −1,336 |
| TS 6 | 58 | 97 | 13,568 | 7,802 | −1,204 |
| TS 7 | 58 | 98 | 13,489 | 7,730 | −1,175 |
| TS 8 | 58 | 97 | 13,566 | 7,816 | −1,206 |
| TS 9 | 58 | 97 | 13,567 | 7,817 | −1,207 |
| **Average** |  | **762** | **13,577** | **7,811** | **−1,440** |

**What this means**: the evaluation harness works — q2mm can load the
published force field, run the evaluation, and save results as a regression
fixture. The problem is substantive: there is an MM3 implementation mismatch
between the original MacroModel workflow and the OpenMM custom-force path.
Likely sources include functional-form differences, parameter-interpretation
differences, and missing or differently handled interaction terms.

This gap was investigated in [issue #197](https://github.com/ericchansen/q2mm/issues/197) (now closed), which identified MM3 functional-form differences between MacroModel and OpenMM as a likely source of the mismatch.

---

## Check 2: force field re-derivation

Not started. Blocked on resolving Check 1 first — there is no point
re-deriving a force field if the evaluation engine doesn't match the
original.

---

## How to reproduce this

The Check 1 evaluation is an automated integration test. It loads the
published force field, evaluates it against the QM reference data, and
compares the results to a saved snapshot (a JSON file containing the
expected per-molecule metrics). If the results change — because of a code
fix, a bug, or a parameter reinterpretation — the test fails, which is
the point: it guards against silent regressions.

```bash
# Run the evaluation (requires OpenMM)
python3 -m pytest test/integration/test_published_ff_validation.py --run-slow -v

# Update the saved snapshot after a verified change
Q2MM_UPDATE_GOLDEN=1 python3 -m pytest test/integration/test_published_ff_validation.py --run-slow -v
```

**Where things live:**

- The published force field came from the
  [Q2MM/q2mm](https://github.com/Q2MM/q2mm/blob/b26404b/forcefields/rh-hydrogenation-enamide.fld)
  repository and is stored at
  [`examples/rh-enamide/ff/rh_hyd_enamide_final.fld`](https://github.com/ericchansen/q2mm/blob/master/examples/rh-enamide/ff/rh_hyd_enamide_final.fld).
- The saved snapshot is at
  [`test/fixtures/published_ff/rh_enamide_donoghue2008.json`](https://github.com/ericchansen/q2mm/blob/master/test/fixtures/published_ff/rh_enamide_donoghue2008.json).
- The test itself is
  [`test/integration/test_published_ff_validation.py`](https://github.com/ericchansen/q2mm/blob/master/test/integration/test_published_ff_validation.py).
- Provenance notes (where the force field came from and how it was
  extracted) are in
  [`validation/published_ffs/README.md`](https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md).
