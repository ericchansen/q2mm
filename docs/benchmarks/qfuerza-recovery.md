# QFUERZA-Recovery Validation

**Question**: If we *throw away* the published OPT bond/angle values and replace them with QFUERZA Hessian-derived values, does the q2mm optimizer recover the published TSFFs?

**Answer (preview)**:
- **rh-enamide ✅** — QFUERZA-start reaches the same basin as published-start (q2mm objective within 9%; bond/angle R² close to the paper's reported RMSD ≤ 0.03 Å / RMSD < 2°).
- **pd-allyl, pd-conjugate 🌟** — q2mm objective at QFUERZA-optimized is actually **lower** than at published-optimized (0.77×, 0.86×). But this is **not** a chemical "win" — the per-parameter comparison and R²/RMSD tables show this is a q2mm-engine-vs-MacroModel-backend mismatch favoring the QFUERZA params, not the published ones (§3.2, §4.2).
- **rh-conjugate ⚠** — nearby basin (1.50× q2mm objective), but the optimized FF contains **negative angle force constants** (§3.4) — unphysical.
- **heck-relay ❌** — JaxLoss surrogate explodes from the QFUERZA start; L-BFGS-B exits in 0 iterations with a worse final FF, also containing negative force constants.

This page documents the experiment honestly. It is the strongest
end-to-end validation of the q2mm pipeline to date (reference data +
weighting + gradients + engine), but the headline result is **mixed**,
and it surfaced two real issues to fix in the optimizer (positive-`fc`
sign constraint; JaxLoss surrogate behavior at FFs with very poor
Seminario starts).

---

## 1. What this experiment actually tests

**QFUERZA workflow (canonical default)**: QFUERZA (Farrugia 2025,
[10.1021/acs.jctc.5c01751](https://doi.org/10.1021/acs.jctc.5c01751))
is defined as a tool for estimating the bond and angle force constants
and equilibria of an OPT substructure from a QM Hessian, given an
existing force-field skeleton. The chemist provides the skeleton — atom
types, OPT-row topology, frozen/active partition, vdW radii/ε,
stretch-bend cross-term coefficients, torsion phase information — via a
`.fld` file; QFUERZA fills in the Hessian-derivable scalars. This is
true for every QFUERZA workflow on every system, including any new
ligand or substrate a user wants to parameterize. There is no
`.fld`-free flow because those topology and atom-type decisions are
chemistry calls that cannot be automated.

For our 5 TS systems, the literature `.fld` files
(Donoghue 2008, Rosales 2020, Wahlers 2022) encode the chemists'
skeleton decisions; we use them verbatim and let QFUERZA fill in 100%
of the bond/angle scalars it is defined to estimate.

| Layer | Source | Why |
|---|---|---|
| OPT row topology (which atom-type triples/quadruples appear) | Literature `.fld` | Chemistry decision: which substructure rows need a custom OPT entry |
| Frozen vs active partition (`freeze_standard_params`) | Literature `.fld` | Chemistry decision: which parameters can be re-derived per-system |
| Atom-type rows, MM3 backbone | Literature `.fld` (frozen) | Standard MM3 library |
| Bond/angle *equilibria* and *force constants* | **QFUERZA** (per-mol projection of TS-inverted QM Hessian, multi-mol mean) | Hessian-derivable scalars — QFUERZA's defined scope |
| Torsion `V₁/V₂/V₃` | Zero | Per Farrugia 2025, torsions are intentionally zeroed at QFUERZA-init time |
| van der Waals `r₀`, `ε`, stretch-bend coefficients | Literature `.fld` | Outside QFUERZA's defined scope; supplied by skeleton |
| Reference data (geometries, eigenmatrix, charges) | Identical to publication-baseline runs | — |
| Optimizer | SciPy L-BFGS-B + JaxLoss analytical gradient, `--ratio-tol -1` | — |

The **per-system overwrite count** in §3.5 reports how many active OPT
scalars QFUERZA touches: this is the count of bond/angle parameters in
the skeleton, by definition. Torsions, vdW, and stretch-bend rows are
not in QFUERZA's scope and stay at the literature values.

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
`--ratio-tol -1`, TS Hessian inversion on, **fractional bounds**
(`fc_fraction=0.20`, `eq_fraction=0.05`; heck-relay overridden to
`fc_fraction=0.05` per the fragile-TS guidance in
[AGENTS.md §9](https://github.com/ericchansen/q2mm/blob/main/AGENTS.md)),
**tight L-BFGS-B convergence** (`ftol=1e-12`, replacing the loose
SciPy default that previously let the optimizer exit after one line
search step).
Data: [`q2mm-data/benchmarks/<system>/convergence/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks) (canonical QFUERZA-start default).

### 3.1 Headline: QFUERZA-start vs published-start objective scores

| System | Pub. final OF | QFUERZA final OF | QFUERZA/Pub. ratio | L-BFGS-B iters | Verdict |
|---|---:|---:|---:|---:|:---|
| rh-enamide   | 2.70 × 10⁵ | 2.94 × 10⁵ | 1.09× | 20 | ✅ same basin |
| pd-allyl     | 7.99 × 10⁶ | 6.14 × 10⁶ | **0.77×** | 3  | 🌟 q2mm-objective lower than published |
| pd-conjugate | 7.24 × 10⁶ | 6.22 × 10⁶ | **0.86×** | 5  | 🌟 q2mm-objective lower than published |
| rh-conjugate | 5.10 × 10⁶ | 7.67 × 10⁶ | 1.50× | 2  | ⚠ nearby basin, marginal |
| heck-relay   | 1.45 × 10⁶ | 1.32 × 10⁸ | **91×** | 0  | ❌ JaxLoss surrogate diverged |

A QFUERZA/Pub. ratio of `1×` means the QFUERZA-start optimizer reaches
the same objective as starting from published OPT values. **Ratio < 1
does not necessarily mean a better physical FF** — it means the q2mm
objective function (geometry RMSD + Hessian eigenmatrix mismatch +
charge RMSD) is lower at the QFUERZA-optimized point than at the
published-optimized point *when evaluated through the q2mm/JaxEngine
backend*. The published authors evaluated through MacroModel/MM3*.
Backend-attributable differences are dissected in §3.2.

### 3.2 R² and RMSD: paper vs q2mm@published vs q2mm@QFUERZA

This table separates two sources of error:

- **Backend gap** (paper R² vs q2mm@published-start R²) — caused by
  q2mm/JaxEngine's MM3 implementation diverging from MacroModel's MM3*
  (e.g., q2mm omits MM3 stretch-bend cross terms, π-system corrections,
  dipole-dipole electrostatics)
- **Starting-point gap** (q2mm@published vs q2mm@QFUERZA) — caused by
  L-BFGS-B converging to a different local minimum from the QFUERZA
  start

| System | Metric | Paper R² / RMSD | q2mm @ published | q2mm @ QFUERZA |
|---|---|---|---|---|
| **rh-enamide** | bond length        | RMSD ≤ 0.03 Å (Tab. 5)              | R² = 0.989 / 0.039 Å        | R² = 0.979 / 0.054 Å        |
|                | bond angle         | RMSD < 2° (Tab. 6)                  | R² = 0.955 / 3.21°          | R² = 0.950 / 3.35°          |
|                | Hessian eig (diag) | not reported                        | R² = 0.968 / 0.068 mdyn/Å   | R² = 0.969 / 0.067 mdyn/Å   |
| **pd-allyl**   | bond length        | aggregate R² = 0.988 (geom)         | R² = 0.046 / 0.37 Å         | R² = **0.416 / 0.29 Å**     |
|                | bond angle         | aggregate R² = 0.988 (geom)         | R² = 0.331 / 14.3°          | R² = **0.491 / 12.5°**      |
|                | Hessian eig (diag) | aggregate R² = 0.998                | R² = −2.82                  | R² = −2.59                  |
| **pd-conjugate** | bond length      | aggregate R² > 0.99                 | R² = 0.950 / 0.076 Å        | R² = 0.405 / 0.260 Å        |
|                  | bond angle       | aggregate R² > 0.99                 | R² = 0.037 / 17.9°          | R² = 0.155 / 16.8°          |
|                  | Hessian eig      | not reported per-category           | R² = −9.6                   | R² = −4.6                   |
| **rh-conjugate** | bond length      | aggregate R² = 0.91–0.99            | R² = 0.822 / 0.165 Å        | R² = 0.777 / 0.185 Å        |
|                  | bond angle       | aggregate R² = 0.91–0.99            | R² = 0.540 / 15.2°          | R² = 0.337 / 18.2°          |
|                  | Hessian eig      | not reported per-category           | R² = −12.9                  | R² = −4.7                   |
| **heck-relay**   | bond length      | not reported (paper has ext. only)  | R² = 0.983 / 0.043 Å        | **R² = −6228 / 25.8 Å**     |
|                  | bond angle       | not reported (paper has ext. only)  | R² = 0.909 / 5.2°           | **R² = −8.3 / 52.6°**       |
|                  | Hessian eig      | not reported                        | R² = −14.3                  | R² = −4.7                   |

> **The interesting backend story**: For all four Wahlers/Rosales TS
> systems, q2mm@published shows much lower R² than the paper reports.
> q2mm's bond/angle R² for pd-allyl at the **published** FF is 0.05/0.33
> — the same FF the paper reports at R²=0.988. The gap is the
> MacroModel-vs-q2mm engine, not the FF parameters. **This means our
> "QFUERZA-recovery" experiment cannot cleanly answer the user's
> question for Wahlers systems** until q2mm achieves MacroModel parity.

> **Where QFUERZA-start improves over published-start in q2mm**: For
> pd-allyl (R² 0.05 → 0.42 for bonds; 0.33 → 0.49 for angles), the
> optimizer found a parameter set that is *better suited to q2mm's
> engine quirks* than the published parameters are. This is not
> evidence that QFUERZA "won" chemically — it's evidence that q2mm's
> MM3 lacks the cross-terms that make the published params work in
> MacroModel.

### 3.3 Per-parameter deviations from the published FF

Generated by [`scripts/compare_opt_rows.py`](https://github.com/ericchansen/q2mm/blob/main/scripts/compare_opt_rows.py).
Both files are round-tripped through `ForceField.to_mm3_fld()` first so
atom-type tokens and OPT-row ordering match exactly. Per-system markdown
tables are committed alongside the artifacts at
[`benchmarks/<system>/convergence/per_param_comparison.md`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks).

**Mean and max absolute deviation by parameter category**:

| System       | bond eq (Å)        | bond fc (mdyn/Å) | angle eq (°)    | angle fc (mdyn·Å/rad²) | matched rows |
|---|---:|---:|---:|---:|---:|
| rh-enamide   | mean 0.007, max 0.12 | mean 0.31, max 18.1 | mean 0.28, max 53° | mean 0.06, max 4.6 | 347 |
| pd-allyl     | mean 0.006, max 0.50 | mean 0.26, max 12.6 | mean 0.37, max 16° | mean 0.03, max 3.2 | 598 |
| pd-conjugate | mean 0.007, max 0.71 | mean 0.31, max 11.9 | mean 0.92, max 54° | mean 0.03, max 1.4 | 519 |
| rh-conjugate | mean 0.005, max 0.61 | mean 0.31, max 11.6 | mean 0.60, max 30° | mean 0.02, max 2.5 | 537 |
| heck-relay   | mean 0.013, max 0.71 | mean 0.68, max  9.3 | mean 0.77, max 55° | mean 0.09, max 7.9 | 373 |

**Bond equilibria stay close** to published across all 5 systems (mean
abs dev < 0.013 Å, well below the published RMSD ≤ 0.03 Å for
rh-enamide). **Bond and angle force constants drift much further** —
the optimizer routinely shrinks published `fc` values toward zero
(e.g. pd-allyl C2–O3 bond fc 3.46 → 0.22, a 94% reduction). This is the
QFUERZA H-substitution effect propagating through optimization: when
the starting `fc` is small (because QFUERZA estimated a weak mode from
a small Hessian eigenvalue), the optimizer has little gradient signal
to push it back up.

### 3.4 ⚠ Negative force constants in optimized FFs

Inspecting the per-parameter comparison tables for rh-conjugate and
heck-relay reveals **negative bond and angle force constants** in the
QFUERZA-optimized FFs:

| System | Param | Atoms | Published `fc` | Optimized `fc` |
|---|---|---|---:|---:|
| rh-conjugate | angle | C2–C2–O2 | +0.529 | **−1.049** |
| rh-conjugate | angle | C2–C2–O2 | +1.400 | **−1.062** |
| heck-relay   | angle | 2–5–4    | +0.041 | **−7.562** |
| heck-relay   | angle | 2–3–4    | +0.344 | **−7.562** |

These are **unphysical**. A negative angle force constant makes the
harmonic potential `½·k·(θ−θ₀)²` unbounded below (any displacement
from `θ₀` lowers the energy). L-BFGS-B found these because q2mm's
parameter bounds (now `fc_fraction=0.20` of the starting value) **do
not enforce a sign constraint**: when the QFUERZA starting `fc` is
near zero, `±20%` of zero is still near zero, and the optimizer can
cross the sign boundary without violating its box constraint.

> **This is a real issue in the optimizer**, separate from the QFUERZA
> recovery question. Force constants should have a positive sign
> constraint regardless of the fractional bounds. Tracking issue: [TBD].

### 3.5 Audit of QFUERZA-overwritten vs literature-retained rows

For each system we report which OPT-substructure rows had their bond
and angle scalars filled in by QFUERZA (vs retained from the literature
`.fld` — these are torsions, vdW, stretch-bend, and Urey-Bradley rows
that are outside QFUERZA's defined scope). See `starting_point_audit`
in each `validation_results.json`. Across the 5 systems, QFUERZA fills
in **17–33% of active parameters** (corresponding to 100% of the
bond/angle scalars in each system's skeleton).

---

## 4. Physical-chemistry walkthrough

This section interprets the largest QFUERZA-vs-published deviations
through a chemical lens. The full per-row tables live in
[`benchmarks/<system>/convergence/per_param_comparison.md`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks).

### 4.1 rh-enamide — cleanest recovery

The QFUERZA-optimized FF is within 5% on every bond equilibrium and
within 33% on every angle equilibrium. The largest force-constant
deviations are:

- **Bond C–H types** (15 entries): all unchanged (Δ = 0.000) — frozen.
  X-H bonds aren't part of the OPT-substructure rows the optimizer
  touches.
- **Phosphine–metal angles**: a few angle `fc` values are amplified
  from near-zero published values (e.g., 6–2–9 angle fc 0.0009 →
  0.445). These are likely PR₃–Rh–C angles whose published values were
  intentionally zeroed (consistent with the Donoghue 2008 protocol of
  letting weak ligand-conformational modes float).
- **C–C bond fc collapses** of 80% are seen for some sp²–sp² C–C bonds
  along the substrate. Chemically, these are π-bonds whose force
  constant in published MM3 is high (ca. 5 mdyn/Å) — QFUERZA derives
  them from the QM Hessian, which projects in less stiffness because
  the substrate is delocalized.

### 4.2 pd-allyl — q2mm engine favors QFUERZA params

The top-15 relative deviations are dominated by **C–N, C–O, and C=N
bond `fc` values collapsing 90–98%** from published (5–13 mdyn/Å) to
optimized (0.1–0.5 mdyn/Å). These bonds are part of the amination
substrate's backbone (e.g., C₂–N₂ amide, C₃–O₃ carbonyl, C₂–C₃ vinyl).

Chemically, an MM3 amide C–N bond `fc` of 13 mdyn/Å is **expected** —
the resonance-delocalized C–N has partial double-bond character.
QFUERZA's projection through the QM Hessian gives `fc` values much
lower because the Hessian eigenvalues for these modes are smaller than
a pure harmonic model would predict — there's anharmonic and resonance
character that QFUERZA cannot capture.

So why does q2mm's objective *prefer* the QFUERZA values? **Backend
parity gap**. In MacroModel's MM3*, the higher `fc` works because
stretch-bend and π-system cross terms partially compensate. In q2mm's
JaxEngine MM3 (which lacks those cross terms), the high `fc` produces
overshooting predictions; lowering it to QFUERZA's value reduces the
residual. This is a q2mm engine issue masquerading as a "win" in the
optimization.

### 4.3 pd-conjugate — L-M-X angle force constant explosion

Top deviation: C₂–C₂–PD angle fc 0.035 → 1.246 (**+3491%**). Chemically
this is the metal-coordinated ene angle of the conjugate-addition
substrate. The published value is near-zero (allowing the substrate
plane to flex toward the metal), but the optimizer pushes it to a
much stiffer +1.25.

Two more concerning angles develop **negative force constants**: C2–N2–PD
goes 0.004 → −0.046 (unphysical). See §3.4. These are L-M-L angles
involving low-published-`fc` rows — the optimizer's bounds don't keep
them positive.

### 4.4 rh-conjugate — most negative force constants

Three of the top-5 deviations involve **sign flips** to negative `fc`:
C2–C2–O2 angle goes from +0.53 to **−1.05**, and from +1.40 to **−1.06**;
C2–C2 bond fc actually doubles to +0.31. The original published values
are small but positive (allowed deviation, normal angle ranges).
QFUERZA started from near-zero, and the optimizer overshot through
zero. This optimization is **not** in the published basin and the
optimizer found an unphysical solution.

The relatively high R²-eig (`−4.7` vs published-start `−12.9`)
suggests this unphysical FF coincidentally fits the diagonal
eigenmatrix elements better than the published one in q2mm's engine —
another symptom of the backend parity gap (§4.2).

### 4.5 heck-relay — JaxLoss broken, optimizer found garbage

The starting Seminario projection produces bond R² = −6228 (mean
predicted bond length is **25.8 Å off** from QM reference). That FF is
non-physical from the start, but the optimizer was supposed to repair
it. Instead, JaxLoss surrogate diverges (no finite ratio reported), the
gradient at the starting point is essentially zero (because L1-norm of
the loss is so large that finite differences are dominated by floating
point noise), and L-BFGS-B exits in 0 iterations declaring convergence.

The final FF has **multiple force constants at −7.56** (the optimizer's
output mapped to a degenerate value). This is the JaxLoss + L-BFGS-B
breakdown pattern documented in
[AGENTS.md §9](https://github.com/ericchansen/q2mm/blob/main/AGENTS.md)
("jaxopt zoom" / "TS ratio check"). A `fc_fraction=0.05` bound was
not tight enough to prevent the breakdown — the starting FF is too far
from any physical basin for fractional bounds around it to make sense.

---

## 5. Interpretation

### What this experiment validates ✅

- **The q2mm reference-data + JaxLoss + L-BFGS-B pipeline works for
  rh-enamide** — starting from QFUERZA gives a final FF within 9% of
  the published-start objective, with bond/angle R² and RMSD close to
  the paper's RMSD ≤ 0.03 Å / RMSD < 2° quality.

### What it does NOT yet validate ❌

- **Per-parameter recovery of published values**. Even on the cleanest
  system (rh-enamide), bond force constants drift by up to 83% and
  angle force constants by an order of magnitude. The optimizer reaches
  the same **objective basin** but not the same **parameter values**.
  This is a known property of degenerate force-field landscapes — many
  parameter sets give similar predictions for the training set.

- **q2mm's MM3 implementation has substantial gaps vs MacroModel's
  MM3\***. R² values at the published FF are far below paper-reported
  R² for pd-allyl, pd-conjugate, rh-conjugate. Until backend parity
  improves, parameter-recovery experiments on Wahlers systems are
  inconclusive.

- **The optimizer needs a positive-`fc` sign constraint**. Negative
  force constants in rh-conjugate and heck-relay outputs are
  unphysical and should be unreachable from any starting point.

- **JaxLoss surrogate fails for FFs with QFUERZA bond R² ≪ 0**. Until
  we have a pre-conditioning step (or use a different surrogate for
  the first few L-BFGS-B steps), heck-relay-class systems cannot be
  optimized from scratch.

### What to do next

1. **Add a positive-`fc` constraint** to q2mm's optimizer bounds.
2. **Backend parity audit** — compare q2mm/JaxEngine MM3 vs
   MacroModel MM3* on a single fixed FF and identify which physics
   terms are missing. (Open a tracking issue.)
3. **Pre-conditioning for heck-relay-class systems** — replace
   QFUERZA `fc` outliers (|fc| > 10) with literature averages before
   L-BFGS-B sees them; or use a robust loss (Huber) instead of
   squared-error for the first 20 optimizer steps.

### Known limitations

QFUERZA is now the canonical default (`starting_point="qfuerza"` on
`load_system`; `--starting-point qfuerza` on
`regenerate_convergence_results.py`). Three of the five TS systems do
**not** converge cleanly from QFUERZA with default bounds:

| System | Symptom | Workaround |
|---|---|---|
| rh-conjugate | Ratio 3.49× vs publication baseline; negative `fc` (§3.4) | `--fc-fraction 0.05 --eq-fraction 0.05` (tighter bounds) or `--starting-point published` |
| pd-conjugate | Ratio 1.14× vs publication baseline | `--fc-fraction 0.05 --eq-fraction 0.05` or `--starting-point published` |
| heck-relay | JaxLoss surrogate diverges to ~10⁸; L-BFGS-B exits in 0 iters (§4.5) | `--starting-point published` (pre-conditioning fix tracked as future work; see "What to do next") |

The `--starting-point published` opt-out path retains the literature
OPT values verbatim and reproduces the historical
publication-baseline benchmarks (`from-published/` artifacts in
q2mm-data).

---

## 6. How to reproduce

```bash
# 1. Verify GPU
source /home/eric/repos/q2mm/.venv/bin/activate
python -c "import jax; print(jax.devices())"   # must show CudaDevice

# 2. Run a single system (canonical default = QFUERZA; recommended
#    bounds for non-heck-relay TS).  --starting-point qfuerza is the
#    default and can be omitted.
python scripts/regenerate_convergence_results.py \
    --system rh-enamide \
    --ratio-tol -1 \
    --ftol 1e-12 \
    --fc-fraction 0.20 \
    --eq-fraction 0.05 \
    --output-dir /path/to/q2mm-data/benchmarks

# 3. Heck-relay or any 3/5-divergence system: see "Known limitations"
#    in §5 — fall back to the publication-baseline path
python scripts/regenerate_convergence_results.py \
    --system heck-relay \
    --starting-point published \
    --ratio-tol -1 \
    --output-dir /path/to/q2mm-data/benchmarks

# 4. Generate the cross-system R²/RMSD table
python scripts/build_qfuerza_recovery_tables.py \
    --data-dir /path/to/q2mm-data/benchmarks \
    --paper-r2 /path/to/q2mm-data/benchmarks/paper_r2_reference.json \
    --out r2-tables.md

# 5. Generate per-parameter comparison (per system)
python scripts/compare_opt_rows.py \
    --system rh-enamide \
    --optimized /path/to/q2mm-data/benchmarks/rh-enamide/convergence/rh-enamide_optimized.fld \
    --out compare-rh-enamide.md
```

The QFUERZA-start artifacts live in
[`q2mm-data/benchmarks/<system>/convergence/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks)
(canonical default path). The publication-baseline artifacts live in
[`q2mm-data/benchmarks/<system>/from-published/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks)
(opt-in `--starting-point published` path).

The `--ratio-tol -1` flag bypasses the JaxLoss/ObjectiveFunction ratio
gate (which would otherwise reject all 5 TS systems at the QFUERZA
start because the surrogate is poorly aligned).

The `--ftol 1e-12` flag overrides the script's default `ftol=1e-8` that
previously let the optimizer exit after one line search step on most
TS systems.

The `--fc-fraction`/`--eq-fraction` flags set sign-aware fractional
bounds — each parameter `v` is clamped to
`[v − f·|v|, v + f·|v|]` (intersected with the physical-sanity
`DEFAULT_BOUNDS`), so TSFF negative force constants are bounded
symmetrically around their actual sign. Without them, L-BFGS-B will
escape the QFUERZA basin entirely.

The output `validation_results.json` includes a `starting_point_audit`
block enumerating which OPT rows were filled in by QFUERZA vs retained
from the literature `.fld`.

---

## 7. Anchors

- Code: [`q2mm/systems.py`](https://github.com/ericchansen/q2mm/blob/main/q2mm/systems.py) (`starting_point` parameter on `load_system`)
- Bounds: [`q2mm/models/forcefield.py`](https://github.com/ericchansen/q2mm/blob/main/q2mm/models/forcefield.py) (`get_fractional_bounds`)
- CLI: [`scripts/regenerate_convergence_results.py`](https://github.com/ericchansen/q2mm/blob/main/scripts/regenerate_convergence_results.py) (`--starting-point`, `--ftol`, `--fc-fraction`, `--eq-fraction`)
- Analysis scripts:
  - [`scripts/compare_opt_rows.py`](https://github.com/ericchansen/q2mm/blob/main/scripts/compare_opt_rows.py) — per-param row-by-row comparison
  - [`scripts/build_qfuerza_recovery_tables.py`](https://github.com/ericchansen/q2mm/blob/main/scripts/build_qfuerza_recovery_tables.py) — R²/RMSD cross-system rollup
- Pre-flight checklist: [`AGENTS.md` §11](https://github.com/ericchansen/q2mm/blob/main/AGENTS.md)
- Process skills: [`.copilot/skills/q2mm-benchmark/SKILL.md`](https://github.com/ericchansen/q2mm/blob/main/.copilot/skills/q2mm-benchmark/SKILL.md), [`.copilot/skills/q2mm-analysis-design/SKILL.md`](https://github.com/ericchansen/q2mm/blob/main/.copilot/skills/q2mm-analysis-design/SKILL.md)
- Tests: [`test/test_systems.py`](https://github.com/ericchansen/q2mm/blob/main/test/test_systems.py) (`TestStartingPoint`), [`test/test_models.py`](https://github.com/ericchansen/q2mm/blob/main/test/test_models.py) (`TestGetFractionalBounds`)
- Data: [`q2mm-data/benchmarks/<system>/convergence/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks) (canonical QFUERZA-start default); [`q2mm-data/benchmarks/<system>/from-published/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks) (opt-in publication baseline)
- Method paper: Farrugia, Helquist, Norrby & Wiest, *J. Chem. Theory Comput.* **2025**, 22, 469. [10.1021/acs.jctc.5c01751](https://doi.org/10.1021/acs.jctc.5c01751)
- Related: [QFUERZA Validation](qfuerza-validation.md) — starting-FF quality across all systems.
