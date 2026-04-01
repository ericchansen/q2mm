# Check 1: Published Force Field Evaluation

## What is Check 1?

Check 1 answers: **can q2mm correctly load a published force field and evaluate it
against the original QM reference data?**

This is a prerequisite for Check 2 (re-deriving the FF with our optimizers). If we
can't evaluate a known FF correctly, optimization results are meaningless.

## System: Rh-enamide hydrogenation (Donoghue 2008)

- **Paper:** Donoghue et al. *J. Chem. Theory Comput.* **2008**, *4*, 1313–1323.
  [DOI: 10.1021/ct800132a](https://doi.org/10.1021/ct800132a)
- **System:** 9 Rh-diphosphine transition state structures for asymmetric
  hydrogenation of enamides
- **QM level:** B3LYP/LACVP\*\* (Jaguar)
- **Published FF:** MM3\* force field optimized with the original Q2MM code +
  MacroModel as the MM engine
- **FF source:** `examples/rh-enamide/ff/rh_hyd_enamide_final.fld` (provenance:
  [Q2MM/q2mm](https://github.com/Q2MM/q2mm) commit `b26404b8`,
  `forcefields/rh-hydrogenation-enamide.fld`)

## Results

**Engine:** OpenMM (MM3 custom force implementation)
**Parameters:** 182
**Frequency reference points:** 762 (across 9 molecules)

### Per-molecule QM vs MM comparison

| Molecule | Atoms | Freq refs | RMSD (cm⁻¹) | MAE (cm⁻¹) | R² |
|----------|------:|----------:|------------:|----------:|-------:|
| TS 1 (36 atoms)  |    36 |        54 |    13680.1  |    7919.9 | −2033.1 |
| TS 2 (38 atoms)  |    38 |        59 |    13771.9  |    8021.4 | −1753.1 |
| TS 3 (38 atoms)  |    38 |        59 |    13773.7  |    8023.5 | −1749.1 |
| TS 4 (62 atoms)  |    62 |       101 |    13348.1  |    7543.9 | −1297.4 |
| TS 5 (62 atoms)  |    62 |       100 |    13429.0  |    7627.2 | −1336.0 |
| TS 6 (58 atoms)  |    58 |        97 |    13567.5  |    7801.8 | −1204.4 |
| TS 7 (58 atoms)  |    58 |        98 |    13488.9  |    7730.0 | −1175.3 |
| TS 8 (58 atoms)  |    58 |        97 |    13565.8  |    7816.1 | −1206.3 |
| TS 9 (58 atoms)  |    58 |        97 |    13567.1  |    7817.2 | −1207.3 |
| **Average** | | **762** | **13576.9** | **7811.2** | **−1440.2** |

### Summary metrics

| Metric | Value |
|--------|-------|
| Published FF objective score | 139,910.7 |
| Seminario baseline objective score | 36.1 |
| Published FF overall RMSD | 13,576.9 cm⁻¹ |
| Published FF overall R² | −1440.2 |

## Known gap

The published FF produces **dramatically worse** results than the Seminario
baseline when evaluated under OpenMM. This is unexpected — the published FF was
optimized specifically for these molecules and should outperform an automated
initial estimate.

**Root cause hypothesis:** The published FF was optimized using MacroModel as the
MM engine, which implements MM3\* differently from our OpenMM custom-force
implementation. Likely sources of discrepancy include:

- Functional form differences (MM3 cubic/quartic stretch vs OpenMM implementation)
- Parameter interpretation differences (force constant conventions, angle
  definitions)
- Missing or differently handled interaction terms (cross-terms, Urey-Bradley,
  1-4 scaling)

This gap is tracked as [issue #197](https://github.com/ericchansen/q2mm/issues/197).
Three test assertions are maintained as strict `xfail` promotion gates in
`test/integration/test_published_ff_validation.py` — they will pass only when the
gap is resolved.

## Artifacts

- **Golden fixture:** `test/fixtures/published_ff/rh_enamide_donoghue2008.json`
  — full per-molecule metrics, QM/MM frequencies, and parameter vector
- **Test harness:** `test/integration/test_published_ff_validation.py`
  — run with `--run-slow` to execute Check 1
- **Provenance docs:** `validation/published_ffs/README.md`

## Reproducing

```bash
# Run Check 1 evaluation (requires OpenMM)
python -m pytest test/integration/test_published_ff_validation.py --run-slow -v

# Regenerate golden fixture
Q2MM_UPDATE_GOLDEN=1 python -m pytest test/integration/test_published_ff_validation.py --run-slow -v
```

## Next steps

1. Investigate and resolve the OpenMM parity gap ([#197](https://github.com/ericchansen/q2mm/issues/197))
2. Complete Check 2: re-derive the FF from scratch and compare against published parameters
3. Expand to additional published systems (see [#198](https://github.com/ericchansen/q2mm/issues/198))
