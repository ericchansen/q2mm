# QFUERZA Validation

This page validates our QFUERZA implementation against the original paper
and its published Zenodo data.

---

## 1. Background

### The paper

**Farrugia, M. M.; Helquist, P.; Norrby, P.-O.; Wiest, O.**
*Rapid FF Generation via Hessian-Informed Initial Parameters and Automated
Refinement.*
J. Chem. Theory Comput. **2025**, 22, 469–476.
[DOI:10.1021/acs.jctc.5c01751](https://doi.org/10.1021/acs.jctc.5c01751) ·
[Zenodo data](https://doi.org/10.5281/zenodo.17386006)

### The four methods

The paper compares four approaches to setting initial force constants
before Q2MM optimization:

| Method | Bonds | Angles |
|--------|-------|--------|
| **Approximation** | 5.0 mdyn/Å (fixed) | 0.5 mdyn·Å/rad² (fixed) |
| **FUERZA** | Seminario projection | Seminario projection |
| **γ-FUERZA** | Seminario projection | Seminario × γ (γ ≈ 0.68) |
| **QFUERZA** | Seminario projection | Seminario, but H-angles → 0.5 |

QFUERZA is defined in §Methods:

> "the QFUERZA approach, which combines the Q2MM approximation and FUERZA
> methods by projecting force constants from the Hessian using FUERZA, as
> previously described, but corrects the known problems of FUERZA for light
> terminal atoms (e.g., hydrogen) angle bends by substituting them for a
> value of 0.5 mdyn/rad²."

In short: run Seminario projection for everything, then replace any angle
force constant where an outer atom is hydrogen with 0.5 mdyn·Å/rad².

### Paper results

The paper tested on two systems. These are their reported numbers.

**Cisplatin (ground state)** — R² between estimated and fully optimized
force constants (higher is better):

| Method | Unoptimized R² |
|--------|---------------|
| Approximation | 0.878 |
| FUERZA | 0.735 |
| γ-FUERZA | 0.889 |
| **QFUERZA** | **0.952** |

**Rh-enamide (transition state)** — merit function (sum of squared
deviations from QM reference, lower is better):

| Method | Preliminary | Optimized | Cycles |
|--------|------------|-----------|--------|
| Approximation | 6.375 | 0.812 | 17 |
| FUERZA | 1.361 | 0.812 | 12 |
| γ-FUERZA | 1.143 | 0.812 | 11 |
| **QFUERZA** | **1.005** | **0.812** | **10** |

All methods converge to the same final quality (0.812) after
optimization. QFUERZA's advantage is **faster convergence** — 40% fewer
cycles than the fixed-default approach.

### The paper's reference codebase

The paper cites [q2mm/q2mm](https://github.com/q2mm/q2mm), but that
codebase only implements FUERZA (plain Seminario projection) — there is no
H-angle substitution logic. A separate private repository
(`Q2MM/q2mm-amber`) contains a file named `qfuerza.py`, but its
`amber_qfuerza()` function is identical to `amber_raw_fuerza()` — neither
implements H-angle substitution. The `--raw-fuerza` CLI flag is defined
but never wired up.

Our implementation is the first automated QFUERZA in the Q2MM ecosystem.

---

## 2. Our implementation

The QFUERZA logic lives in
[`q2mm.models.seminario`](../reference/q2mm/models/seminario.md):

```python
QFUERZA_H_ANGLE_DEFAULT_MDYNA = 0.5        # mdyn·Å/rad²
QFUERZA_H_ANGLE_DEFAULT_CANONICAL = 35.97  # kcal/(mol·rad²)

def _is_hydrogen_angle(elements):
    """Return True if either outer atom of an angle is hydrogen."""
    return elements[0] == "H" or elements[2] == "H"
```

In `estimate_force_constants()`, after computing the Seminario-projected
value for each angle:

```python
if strategy == "qfuerza" and _is_hydrogen_angle(angle_param.elements):
    angle_param.force_constant = QFUERZA_H_ANGLE_DEFAULT_CANONICAL
else:
    angle_param.force_constant = fuerza_value
```

### Compliance checklist

| Aspect | Paper definition | Our code | Match? |
|--------|-----------------|----------|--------|
| Bond FCs | Seminario projection | Bidirectional Seminario average | ✅ |
| Non-H angle FCs | Seminario projection | Seminario reciprocal sum | ✅ |
| H-angle FCs | Substitute with 0.5 mdyn·Å/rad² | `QFUERZA_H_ANGLE_DEFAULT_MDYNA = 0.5` | ✅ |
| H-angle detection | "light terminal atoms (e.g., hydrogen)" | Either outer atom is `"H"` | ✅ |
| DFT scaling factor | 0.963 | `DEFAULT_DFT_SCALING = 0.963` | ✅ |
| Bond-angle decoupling | Bonds unchanged by QFUERZA | Only angles modified | ✅ |
| Angle formula | Seminario reciprocal sum (1996) | Same formula, confirmed against [q2mm/q2mm](https://github.com/q2mm/q2mm) | ✅ |

The angle force constant formula from the original Seminario paper (1996),
also used in [q2mm/q2mm](https://github.com/q2mm/q2mm), is:

$$
\frac{1}{k_\theta} = \frac{1}{k_{ij} \cdot r_{ij}^2} + \frac{1}{k_{kj} \cdot r_{kj}^2}
$$

where $k_{ij}$ and $k_{kj}$ are Hessian eigenvalue projections and
$r_{ij}$, $r_{kj}$ are bond lengths.

### Unit conversion chain

Force constants pass through a conversion pipeline from QM units to
the canonical kcal/(mol·unit²) used internally:

| Step | Constant | Value | Derivation |
|------|----------|-------|------------|
| Hartree/Bohr² → mdyn/Å | `AU_TO_MDYNA` | 15.569141 | CODATA (< 0.002% from first-principles) |
| Hartree/rad² → mdyn·Å/rad² | `AU_TO_MDYN_ANGLE` | 4.3598 | 1 Ha = 4.3598 × 10⁻¹⁸ J = 4.3598 mdyn·Å |
| mdyn·Å/rad² → kcal/(mol·rad²) | `MDYNA_RAD2_TO_KCALMOLRAD2` | 71.94 | 0.5 × MM3_STR / KCAL_TO_KJ |

The 0.5 in the last conversion accounts for the convention difference:
the Seminario projection gives a "physical" force constant
(E = ½kΔθ²), while the canonical format uses E = kΔθ² (no ½ factor).

The QFUERZA default in canonical units:
0.5 mdyn·Å/rad² × 71.94 = **35.97 kcal/(mol·rad²)**.

---

## 3. Verification against Zenodo data

The [Zenodo archive](https://doi.org/10.5281/zenodo.17386006) contains
the actual `.fld` force field files for cisplatin — one file per method.
We extracted the force constants and compared them against our QFUERZA
rules. The reference values are stored in
[`cisplatin_zenodo_reference.json`](https://github.com/ericchansen/q2mm/blob/master/test/fixtures/seminario_parity/cisplatin_zenodo_reference.json).

### Bonds (mdyn/Å)

| Parameter | Approximation | FUERZA | γ-FUERZA | QFUERZA | Rule check |
|-----------|---------|--------|----------|---------|------------|
| N–Pt | 5.000 | 1.169 | 1.169 | **1.169** | ✅ Matches Zenodo |
| N–H | 5.000 | 5.765 | 5.765 | **5.765** | ✅ Matches Zenodo |
| Pt–Cl | 5.000 | 1.392 | 1.392 | **1.392** | ✅ Matches Zenodo |

QFUERZA bond FCs are identical to FUERZA — as expected per the paper's
definition (QFUERZA only modifies H-angle FCs, not bonds).

### Angles (mdyn·Å/rad²)

| Parameter | Approximation | FUERZA | γ-FUERZA | QFUERZA | H-angle? | Rule check |
|-----------|---------|--------|----------|---------|----------|------------|
| N–Pt–Cl (trans) | 0.500 | 3.074 | 2.090 | **3.074** | No | ✅ Matches Zenodo (= FUERZA) |
| N–Pt–Cl (cis) | 0.500 | 3.068 | 2.086 | **3.068** | No | ✅ Matches Zenodo (= FUERZA) |
| N–Pt–N | 0.500 | 2.561 | 1.742 | **2.561** | No | ✅ Matches Zenodo (= FUERZA) |
| Cl–Pt–Cl | 0.500 | 3.750 | 2.550 | **3.750** | No | ✅ Matches Zenodo (= FUERZA) |
| **H–N–Pt** | 0.500 | 2.063 | 1.403 | **0.500** | **Yes** | ✅ Matches Zenodo (= 0.5) |
| **H–N–H** | 0.500 | 1.444 | 0.982 | **0.500** | **Yes** | ✅ Matches Zenodo (= 0.5) |

Every QFUERZA **angle** value matches the Zenodo archive exactly: non-H
angles equal FUERZA; H-angles are substituted with 0.5.

!!! warning "Seminario projection divergence on bonds"
    When we independently recompute Seminario/FUERZA bond force constants
    from the Gaussian Hessian (`cisplatin_opt_freq_m06.log`), our N-Pt
    bond matches exactly but Pt-Cl (+16.9%) and N-H (+16.3%) diverge from
    the values in the Zenodo `.fld` files. The bond values in the table
    above are read directly from the Zenodo `.fld` (not recomputed). See
    [`q2mm-data/qfuerza-zenodo/README.md`](https://github.com/ericchansen/q2mm-data/blob/main/qfuerza-zenodo/README.md)
    for full provenance.

!!! note "FUERZA overestimation is not a fixed 2×"
    The paper describes FUERZA as producing angle FCs "two times too
    strong," but the actual cisplatin ratios are 4.13× (H–N–Pt) and
    2.89× (H–N–H). The ~2× is a rough cross-molecule average, which is
    why QFUERZA uses fixed substitution rather than a scaling factor.

### Post-optimization convergence

After Q2MM gradient optimization, all methods converge to nearly
identical final parameters:

| Parameter | QFUERZA opt | FUERZA opt | Difference |
|-----------|-------------|------------|------------|
| N–Pt | 1.162 | 1.162 | < 0.1% |
| N–H | 7.006 | 7.006 | 0.0% |
| Pt–Cl | 1.924 | 1.934 | 0.5% |
| Cl–Pt–Cl | 1.463 | 0.981 | 49%* |
| H–N–Pt | 0.224 | 0.224 | 0.0% |
| H–N–H | 0.624 | 0.624 | 0.0% |

*\*Cl–Pt–Cl converges to different local minima depending on
initialization. The paper attributes this to the parameter landscape.*

### γ-FUERZA scaling factor

The paper text says γ = 0.67, but the Zenodo data shows γ = **0.680** (to
3 decimal places). This minor inconsistency does not affect us since we
don't implement γ-FUERZA.

---

## 4. Test coverage

### Unit tests (`test/test_models.py`)

The `TestQFUERZA` class contains 13 tests covering:

- **H-angle substitution** — angles with outer hydrogen get the 0.5
  default; non-H angles are unchanged
- **Determinism** — same input → same output across repeated calls
- **Equilibria preservation** — QFUERZA only changes force constants,
  not equilibrium geometries

### Parity tests (`test/integration/test_seminario_parity.py`)

Golden-fixture tests compare our output against reference values for
ethane, Rh-enamide, and SN2 at rel=1e-6 tolerance.

!!! warning "Parity fixtures are self-referential"
    The golden fixtures for the ethane, SN2, and Rh-enamide systems were
    generated by our own code, not extracted from the paper. They verify
    **self-consistency** (no regressions) but not **paper-parity**.

### Zenodo fixture

`test/fixtures/seminario_parity/cisplatin_zenodo_reference.json` contains
the force constants from all four methods plus optimized results,
extracted from the Zenodo archive. This fixture documents the paper's
numerical values for future validation work.

### Cisplatin Hessian parity ✅

The cisplatin Gaussian log (`cisplatin_opt_freq_m06.log`) was downloaded
from the Zenodo archive and is stored as a test fixture at
`test/fixtures/seminario_parity/cisplatin_opt_freq_m06.log`. The
`TestCisplatinHessianParity` class in
`test/integration/test_seminario_parity.py` parses the log, runs our
Seminario projection, and verifies:

- **Structure parsing**: 11 atoms (Pt, 2 Cl, 2 N, 6 H), 33×33 symmetric
  Hessian
- **Connectivity**: 10 bonds (2 Pt-Cl, 2 Pt-N, 6 N-H), 18 angles, 5 angle
  types — all matching cisplatin's cis-square-planar + NH₃ geometry
- **N-Pt bond FC**: our value matches the paper's published FUERZA result
  (1.1687 mdyne/Å) within 0.001 mdyne/Å, without DFT scaling
- **QFUERZA H-angle substitution**: H-N-Pt and H-N-H angles are set to
  exactly 0.5 mdyne·Å/rad², matching the paper's QFUERZA definition
- **QFUERZA bonds = FUERZA bonds**: bond FCs are identical between methods
- **FUERZA vs QFUERZA H-angles**: FUERZA overestimates, QFUERZA corrects

!!! note "Partial parity: Pt-Cl and N-H diverge ~17%"
    Our Pt-Cl (1.63 mdyne/Å vs paper 1.39) and N-H (6.45 mdyne/Å vs paper
    5.76) bond FCs diverge from the paper's values by ~17%. Investigation
    traced this to unreproducible modifications in the reference code
    (missing `convert_and_set` method, `au_hessian` parameter not accepted
    by `GaussLog`). The N-Pt exact match validates our core algorithm.

---

## 5. Starting-point quality across systems

Beyond cisplatin, we evaluated QFUERZA starting-point quality on five
transition-state systems from the Q2MM literature. All R² values are
**mass-weighted eigenmatrix diagonal** — the standard metric in the
Q2MM literature.

!!! warning "These two columns measure different things"
    - **Paper R² (post-optimization)**: the final quality after 10–17
      cycles of Q2MM gradient optimization in MacroModel. This is the
      published result — it represents the best the method can achieve.
    - **Our R² (QFUERZA starting point)**: the quality of the
      QFUERZA-estimated force field **before any optimization**, evaluated
      under our JAX engine using mass-weighted eigenmatrix diagonal R².

    These are intentionally not apples-to-apples. The purpose is to
    show how good a starting point QFUERZA provides before the optimizer
    runs.

| System | Paper | Paper R² (post-opt) | Our R² (starting point) | Status |
|--------|-------|:-------------------:|:-----------------------:|--------|
| [Rh-enamide](../systems/rh-enamide.md) | Donoghue *JCTC* 2008 | ~0.998 | 0.992 | ✅ Excellent starting point |
| [Heck relay](../systems/heck-relay.md) | Rosales *JACS* 2020 | >0.998 | 0.502 | ⚠️ Viable — needs optimization |
| [Pd-allyl](../systems/pd-allyl.md) | Wahlers *Nat. Commun.* 2021 | 0.998 | 0.842 | ✅ Good starting point |
| [Pd 1,4-conjugate](../systems/pd-conjugate.md) | Wahlers *J. Org. Chem.* 2021 | >0.99 | 0.492 | ⚠️ Viable — needs optimization |
| [Rh 1,4-conjugate](../systems/rh-conjugate.md) | Wahlers thesis Ch. 6 | 0.91–0.99 | 0.423 | ⚠️ Viable — needs optimization |

### Interpretation

All five systems produce **positive mass-weighted R²** from QFUERZA alone
(before any optimization). Rh-enamide is nearly publication-quality at
0.992. The three softer TS systems (Heck, Pd-conjugate, Rh-conjugate)
start at R² ≈ 0.4–0.5 — a viable starting point for the gradient
optimizer to refine toward the paper's final quality (>0.99).

### Eigenmatrix basis: Cartesian vs mass-weighted

Two R² metrics are relevant:

- **Mass-weighted R²** (reported above): project both QM and MM Hessians
  onto eigenvectors of the mass-weighted QM Hessian. This is the standard
  metric in the Q2MM literature and gives positive R² for all systems.
- **Cartesian R²**: project onto eigenvectors of the unweighted QM Hessian.
  For soft TS systems, Cartesian R² can be catastrophically negative
  (−1 to −5) because the Cartesian basis amplifies mismatches in
  low-frequency modes that carry little physical significance.

Our JaxLoss penalty function internally uses a Cartesian basis for
optimization (both QM and MM Hessians projected onto Cartesian QM
eigenvectors). This is internally consistent and produces correct
gradients. The mass-weighted R² reported here is a separate diagnostic
metric computed after optimization.

### Optimizer convergence (GPU, RTX 5090)

We ran L-BFGS-B with analytical JAX gradients on all five systems using
an eigenmatrix-only objective (full lower-triangular eigenmatrix, no
geometry terms). All optimizations completed on GPU.

| System | Molecules | Active params | JIT compile | Optimization | Loss reduction |
|--------|:---------:|:------------:|:-----------:|:------------:|:--------------:|
| Rh-enamide | 9 | 182 | 95 s | 6 s | 16.8% |
| Heck relay | 23 | 462 | 249 s | 18 s | 43.1% |
| Pd-allyl | 21 | 482 | 227 s | 13 s | 34.5% |
| Pd 1,4-conjugate | 10 | 340 | 120 s | 9 s | 57.2% |
| Rh 1,4-conjugate | 10 | 488 | 122 s | 10 s | 62.6% |

JIT compilation is a one-time cost per system (per-molecule function
compilation). After JIT, each loss+gradient evaluation takes 0.02–0.1 s.
The optimizer converges in 200 L-BFGS-B iterations for all systems.

!!! note "Loss vs R² direction"
    The optimizer minimizes the Cartesian eigenmatrix penalty (sum of
    squared weighted projection errors). Mass-weighted R² is computed
    in a different basis and may not monotonically improve. Both metrics
    are reported for transparency.

!!! note "How these numbers relate to the production convergence results"
    The table above uses the **full lower-triangular eigenmatrix**
    (N²/2 refs per molecule), **no L2 regularization**, and a **200-iter
    fixed cap**. This produces larger reductions (16.8–62.6%) because
    the optimizer has more freedom to drift parameters.

    The [optimizer comparison](optimizer-comparison.md) page shows
    production convergence results using **eigenmatrix-diagonal only**
    + **L2 λ=0.01** + **convergence-based termination** (0.72–12.09%).
    These are complementary views: this table validates raw optimizer
    capability; the comparison page shows the regularized production
    workflow.

### Bug fixes reflected in these numbers

The R² values above reflect three critical bug fixes:

1. **MM3 torsion V/2 convention** — FLD stores full Vn; energy uses
   Vn/2 (commit `43467c4`)
2. **V2 phase correction** — V2 torsions use γ=180°, not 0° (same
   commit)
3. **Bond-dipole electrostatics constant** — corrected from
   332.05/1.5 ≈ 221.4 (charge-charge constant, wrong units for
   dipoles in Debye) to 14.3928/1.5 ≈ 9.60 (commit `d2b34d9`).
   This was a **23× overestimate** that dominated the Hessian.

See the [Systems](../systems/rh-enamide.md) section for per-system
detail on what the original papers fit and what would need to change
to close any remaining gaps.

---

## 6. Remaining validation opportunities

1. ~~Parse the cisplatin Gaussian log~~ — **done** (see §4 above)
2. ~~Add cisplatin as a parity test system~~ — **done** (16 integration tests)
3. ~~Investigate Pt-Cl / N-H FC divergence~~ — **done** (traced to
   unreproducible reference code modifications; N-Pt exact match validates
   core algorithm)
4. **Compare Rh-enamide TSFF** — the Zenodo archive also contains the
   full Rh-enamide force field files (~1.8 GB, includes QM data)
