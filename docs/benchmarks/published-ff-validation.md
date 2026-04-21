# Published Force Field Validation

## What this is

Before trusting q2mm's optimizers to produce new force fields, we need to
prove it can correctly *evaluate* force fields that have already been published
and validated against experiment. This page documents that proof.

The validation program has two checks, run in order:

| Check | Question | Status |
|-------|----------|--------|
| **Check 1** | Can q2mm load a published force field and reproduce its fit quality against the original QM data? | ✅ R² = 0.60 (JAX engine + near-linear torsion damping + RC exclusion). Published FF < Seminario baseline due to engine gap ([#255](https://github.com/ericchansen/q2mm/issues/255)). |
| **Check 2** | Can q2mm re-derive the published force field from scratch using its own optimizers? | ✅ JaxOpt L-BFGS on Rh-enamide: RMSD 260 → 153 cm⁻¹ (50 iters, GPU) |

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
with Q2MM + MacroModel) and evaluated it with q2mm's JAX MM3 engine against
the same 9 transition-state structures and QM frequencies.

**Root cause found and fixed**: all 9 TS structures have near-linear C-C-N
angles (~179°).  The dihedral gradient diverges as 1/sin²(θ) at 180°,
amplifying even reasonable torsion force constants into 10,000+ kcal/(mol·Å)
forces.  Production MM3 codes (TINKER, MacroModel) universally handle this
by skipping/damping near-linear torsions.  We added a Hermite smoothstep
that smoothly suppresses torsion terms when either central angle exceeds 170°.

Additionally, the reaction-coordinate MM frequency (>4000 cm⁻¹ from the
deliberately stiffened TS bond) was excluded from comparison, matching the
Q2MM convention of weight 0.00 for the first eigenvalue (`eig_i`).

**Results after fix** (JAX engine, near-linear damping, RC exclusion):

| Molecule | Atoms | Freq refs | RMSD (cm⁻¹) | MAE (cm⁻¹) | R² |
|----------|------:|----------:|------------:|------------:|---:|
| TS 1 | 36 | 88 | 583 | 306 | 0.428 |
| TS 2 | 38 | 95 | 527 | 277 | 0.597 |
| TS 3 | 38 | 95 | 526 | 276 | 0.598 |
| TS 4 | 62 | 157 | 566 | 312 | 0.438 |
| TS 5 | 62 | 156 | 603 | 347 | 0.311 |
| TS 6 | 58 | 153 | 422 | 201 | 0.767 |
| TS 7 | 58 | 153 | 401 | 182 | 0.794 |
| TS 8 | 58 | 151 | 445 | 217 | 0.732 |
| TS 9 | 58 | 151 | 445 | 217 | 0.733 |
| **Average** |  | **1199** | **502** | **262** | **0.600** |

**Status of Check 1 promotion gates:**

| Gate | Issue | Status |
|------|-------|--------|
| All per-molecule R² > 0 | [#256](https://github.com/ericchansen/q2mm/issues/256) | ✅ Passes |
| Average R² > 0.40 | [#257](https://github.com/ericchansen/q2mm/issues/257) | ✅ Passes |
| Published FF beats Seminario | [#255](https://github.com/ericchansen/q2mm/issues/255) | ⚠️ xfail — engine gap |

The published FF does not beat the Seminario (QFUERZA) baseline because it
was optimized for MacroModel's MM3* engine, which includes functional-form
features (metal-center torsion rules, possibly stretch-bend cross terms) that
our engine doesn't replicate.  The Seminario method projects QM Hessian
eigenvalues directly (engine-independent), so it naturally outperforms a
cross-engine evaluation.  This is documented as an inherent limitation, not
a bug.

---

## Check 2: force field re-derivation

JaxOpt L-BFGS (analytical gradients, 50 iterations) on the Rh-enamide
9-molecule training set:

- **RMSD**: 260 → 153 cm⁻¹
- **Score**: 91.5 → 77.0
- **Time**: 341 s (GPU, RTX 5090)
- **Optimizer**: `jaxopt:lbfgs` via topology-grouped vmap Hessian batching

Results archived in `benchmarks/rh-enamide/results/`. The Zenodo
QFUERZA-optimized FF (same system, different objective function) scores
9120 on our frequency objective — expected, since it was optimized for
eigenmode fitting rather than frequency RMSD.

---

## How to reproduce this

The Check 1 evaluation is an automated integration test. It loads the
published force field, evaluates it against the QM reference data, and
compares the results to a saved snapshot (a JSON file containing the
expected per-molecule metrics). If the results change — because of a code
fix, a bug, or a parameter reinterpretation — the test fails, which is
the point: it guards against silent regressions.

```bash
# Run the evaluation (requires JAX or OpenMM)
python3 -m pytest test/integration/test_published_ff_validation.py --run-validation -v

# Update the saved snapshot after a verified change
Q2MM_UPDATE_GOLDEN=1 python3 -m pytest test/integration/test_published_ff_validation.py --run-validation -v
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
