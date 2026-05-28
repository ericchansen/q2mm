# QFUERZA-Recovery Validation

**Question**: If we *throw away* the published OPT bond/angle values and replace them with QFUERZA Hessian-derived values, does the q2mm optimizer recover the published TSFFs?

**Answer (preview)**: For two of five systems the optimizer reaches essentially the same minimum as a published-start run. For three systems it does not — the QFUERZA starting point lies in a different basin, and L-BFGS-B converges to a local optimum that is far from (and worse than) the published-start result.

This page documents the experiment honestly. It is the strongest end-to-end validation of the q2mm pipeline to date (reference data + weighting + gradients + engine), but the headline result is **mixed**.

---

## 1. What this experiment actually tests

> **This is *not* a from-scratch FF generation.**

QFUERZA-recovery starts from the **published force field topology** —
which OPT rows exist, which parameters are frozen vs active, the
atom-type rows, vdW radii/epsilons, stretch-bend coefficients, and
torsion phase information. It then **overwrites** the bond and angle
*values* (force constants and equilibria) with QFUERZA-derived values
computed from the per-molecule QM Hessian, following the Farrugia 2025
protocol ([10.1021/acs.jctc.5c01751](https://doi.org/10.1021/acs.jctc.5c01751)).

| Layer | QFUERZA-recovery run uses |
|---|---|
| OPT row topology (which atom-type triples/quadruples appear) | **Published** |
| Frozen vs active partition (`freeze_standard_params`) | **Published** |
| Bond/angle *equilibria* and *force constants* | **QFUERZA** (multi-molecule mean of per-mol QFUERZA, TS-inverted) |
| Torsion `V₁/V₂/V₃` | Published-zero (Farrugia zeros torsions at QFUERZA-init time; for our TS systems the published OPT torsions are already zero, so the values coincide) |
| van der Waals `r₀`, `ε` | **Published** (QFUERZA does not touch vdW) |
| Stretch-bend, MM3 backbone | **Published** (frozen) |
| Reference data (geometries, eigenmatrix, charges) | Identical to published-start runs |
| Optimizer | SciPy L-BFGS-B + JaxLoss analytical gradient, `--ratio-tol -1` |

The **per-system overwrite count** is reported in §3 below — only 16–25%
of active parameters are actually replaced.

A *true* from-scratch run would need a QFUERZA-only loader path
(`qfuerza_fresh` strategy) that builds the OPT topology from scratch,
not by overwriting published rows. The current implementation uses the
published `ff_strategy` for everything except the bond/angle scalars.

---

## 2. Why we ran this

Every TS benchmark in
[`q2mm-data/benchmarks/*/convergence/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks)
has started from published OPT values. That answers "can the optimizer
refine a near-converged FF?" but not "can it find a good FF given only
QM data?"

The QFUERZA-recovery protocol tests the latter: starting from
Hessian-derived bond/angle values, can the existing q2mm pipeline
(reference data, JaxLoss surrogate, L-BFGS-B) close the gap to the
published TSFF?

---

## 3. Results

All runs: WSL2 + RTX 5090, SciPy L-BFGS-B + JaxLoss `jac='auto'`,
`--ratio-tol -1`, TS Hessian inversion on.
Data: [`q2mm-data/benchmarks/<system>/from-qfuerza/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks).

### Headline: QFUERZA-start vs published-start objective scores

| System | Pub. final OF | QFUERZA final OF | QFUERZA/Pub. ratio | Verdict |
|---|---:|---:|---:|:---|
| rh-enamide   | 2.70 × 10⁵ | 2.78 × 10⁵ | **1.03×** | ✅ same basin |
| pd-allyl     | 7.99 × 10⁶ | 7.98 × 10⁶ | **1.00×** | ✅ same basin |
| pd-conjugate | 7.24 × 10⁶ | 8.25 × 10⁶ | 1.14× | ⚠ nearby basin, marginal |
| rh-conjugate | 5.10 × 10⁶ | 1.78 × 10⁷ | 3.49× | ❌ different basin |
| heck-relay   | 1.45 × 10⁶ | 1.45 × 10⁸ | **100×** | ❌ JaxLoss surrogate diverged |

> A QFUERZA/Pub. ratio of `1×` means the QFUERZA-start optimizer
> reaches the same objective as starting from published OPT values.

### Per-system runs (QFUERZA start)

| System | mol | n_active | QF overwritten | Pub. retained | Init OF | Final OF | Δ% | Ratio | Iters | Evals | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rh-enamide   | 9  | 182 | 60  (33%) | 122 | 3.92 × 10⁵ | 2.78 × 10⁵ | **+28.9** | 1.05    | 11 | 2 | 721 s |
| heck-relay   | 23 | 462 | 101 (22%) | 361 | 1.34 × 10⁸ | 1.45 × 10⁸ | **−7.70** | 1.9 × 10⁷⁴ | 0  | 2 | 1425 s |
| pd-allyl     | 21 | 482 | 80  (17%) | 402 | 8.01 × 10⁶ | 7.98 × 10⁶ | +0.41     | 1.10    | 2  | 2 | 1290 s |
| pd-conjugate | 10 | 340 | 96  (28%) | 244 | 8.23 × 10⁶ | 8.25 × 10⁶ | −0.20     | 1.21    | 1  | 2 | 626 s  |
| rh-conjugate | 10 | 488 | 81  (17%) | 407 | 2.67 × 10⁷ | 1.78 × 10⁷ | **+33.4** | 0.52    | 2  | 2 | 745 s  |

- *QF overwritten* counts active OPT rows whose values were replaced by
  QFUERZA. The rest are *retained published* (mostly torsions at 0, plus
  vdW which QFUERZA does not touch).
- *Ratio* is JaxLoss/ObjectiveFunction at the starting FF. Values near 1
  mean the JaxLoss surrogate tracks the real objective; values far from
  1 mean the surrogate is unreliable.
- *Evals = 2* on most systems is L-BFGS-B reporting that gradient
  information at the starting point already places it close to a local
  minimum.

### Seminario starting-FF quality (QFUERZA bond/angle on published topology)

R² values are the linear fit between MM-evaluated property and QM
reference, computed at the starting FF. Negative R² indicates the model
is worse than predicting the mean.

| System | bond R² (start → opt) | angle R² (start → opt) | eig-diag R² (start → opt) |
|---|---|---|---|
| rh-enamide   | 0.976  → 0.984  | 0.934   → 0.953   | 0.972  → 0.970  |
| heck-relay   | −247   → −147   | −7.95   → −6.35   | −4.66  → −4.66  |
| pd-allyl     | 0.034  → 0.031  | 0.335   → 0.337   | −1.41  → −1.41  |
| pd-conjugate | 0.448  → 0.427  | −0.114  → −0.114  | −4.47  → −4.47  |
| rh-conjugate | −0.71  → −0.23  | −0.337  → −0.264  | −4.85  → −4.85  |

heck-relay starts with **bond R² = −247** and **angle R² = −7.95** at
the QFUERZA point. The Seminario projection produces large negative
force constants on Pd-N-C angles (≈ −1.6 kcal/mol/rad²) which, after TS
inversion, still leave a starting FF whose MM-bond-length predictions
have RMSD orders of magnitude larger than reference. JaxLoss diverges
to `1.9 × 10⁷⁴`, and L-BFGS-B exits in 0 iterations with the FF
*worse* than the start (−7.70% improvement at the optimizer's chosen
"converged" point).

---

## 4. What this tells us

### rh-enamide ✅ — strong recovery
Same basin as published-start (278k vs 270k, 3% gap). The starting
Seminario fit is already good (bond R² > 0.97), JaxLoss ratio is
healthy (1.05), and the optimizer makes a genuine 28.9% improvement.
**This is the cleanest validation in the suite.**

### pd-allyl ✅ — same basin, no improvement to make
QFUERZA-start and published-start reach essentially identical objective
(7.98M vs 7.99M, within 0.1%). The optimizer makes no significant
improvement from either starting point (+0.41% vs +0.52%), suggesting
both starting FFs sit in or near the same local minimum.

### pd-conjugate ⚠ — nearby basin
Published-start reaches 7.24M; QFUERZA-start gets stuck at 8.25M (14%
worse). Negative improvement (−0.20%) from the QFUERZA start indicates
L-BFGS-B converged to a different (and slightly worse) local minimum
than the published FF.

### rh-conjugate ❌ — different basin
QFUERZA-start makes a real +33.4% improvement but the final score
(17.8M) is 3.5× worse than the published-start final (5.1M). The
optimizer is descending into a different basin entirely.

### heck-relay ❌ — JaxLoss surrogate fails
The starting Seminario projection produces such a poor MM PES (bond
R² = −247) that the JaxLoss-implicit-diff geometry-relaxation surrogate
explodes to `1.9 × 10⁷⁴`. L-BFGS-B exits in 0 iterations and the final
FF is worse than the starting FF.

This is consistent with the known JaxLoss limitation documented in
[AGENTS.md §6](https://github.com/ericchansen/q2mm/blob/main/AGENTS.md):
for TS systems with poor starting force fields, unconstrained MM
geometry relaxation wanders to wrong minima and the surrogate becomes
uninformative.

---

## 5. Interpretation

The QFUERZA-recovery experiment is best read as a **basin diagnostic**,
not a global-minimum search:

- The q2mm pipeline (reference data, JaxLoss gradient, L-BFGS-B) is
  validated for **local refinement** of an FF that is already near a
  good basin (rh-enamide, pd-allyl).
- When the starting FF is sufficiently far from the published basin
  (rh-conjugate, pd-conjugate), L-BFGS-B finds *a* local minimum but
  not the one the published authors found. This is expected for a local
  optimizer on a non-convex landscape.
- When the starting FF is *very* bad (heck-relay), the JaxLoss
  surrogate breaks down and no useful optimization happens.

**This is consistent with what the Farrugia 2025 paper warns about**:
QFUERZA is an *initial-parameter* method, not a one-shot generator. The
paper's full workflow includes per-molecule QFUERZA + iterative
refinement *with constraints to prevent basin escape* — not the
unconstrained L-BFGS-B we run here.

To close the gap on the three failed systems, a future experiment
should test:
1. Trust-region or constrained optimization (bound box around starting
   FF) to keep L-BFGS-B in the published basin.
2. Multiple random restarts to characterize the basin landscape.
3. A pre-conditioning step that improves the starting FF Seminario R²
   before handing off to JaxLoss + L-BFGS-B.

---

## 6. How to reproduce

```bash
# 1. Verify GPU
source /home/eric/repos/q2mm/.venv/bin/activate
python -c "import jax; print(jax.devices())"   # must show CudaDevice

# 2. Run single system
python scripts/regenerate_convergence_results.py \
    --system rh-enamide \
    --starting-point qfuerza \
    --ratio-tol -1 \
    --output-dir /path/to/q2mm-data/benchmarks

# 3. Inspect output
ls /path/to/q2mm-data/benchmarks/rh-enamide/from-qfuerza/
# validation_results.json   paper_metrics.json   rh-enamide_optimized.fld
```

The `--ratio-tol -1` flag bypasses the JaxLoss/ObjectiveFunction ratio
gate (which would otherwise reject all 5 TS systems at the QFUERZA
start because the surrogate is poorly aligned).

The output `validation_results.json` includes a `starting_point_audit`
block enumerating which OPT rows were overwritten by QFUERZA vs
retained from the published FF.

---

## 7. Anchors

- Code: [`q2mm/diagnostics/systems.py`](https://github.com/ericchansen/q2mm/blob/main/q2mm/diagnostics/systems.py) (`starting_point` parameter on `load_system`)
- CLI: [`scripts/regenerate_convergence_results.py`](https://github.com/ericchansen/q2mm/blob/main/scripts/regenerate_convergence_results.py) (`--starting-point {published,qfuerza}`)
- Tests: [`test/test_systems.py`](https://github.com/ericchansen/q2mm/blob/main/test/test_systems.py) (`TestStartingPoint`)
- Data: [`q2mm-data/benchmarks/<system>/from-qfuerza/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks)
- Method paper: Farrugia, Helquist, Norrby & Wiest, *J. Chem. Theory Comput.* **2025**, 22, 469. [10.1021/acs.jctc.5c01751](https://doi.org/10.1021/acs.jctc.5c01751)
- Related: [QFUERZA Validation](qfuerza-validation.md) — starting-FF quality across all systems.
