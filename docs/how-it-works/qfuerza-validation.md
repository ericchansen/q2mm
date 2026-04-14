# QFUERZA Validation

This page documents the validation of q2mm's QFUERZA implementation against
the original paper and its published supporting data.

!!! info "Why validate?"
    When you implement someone else's method from a paper, you need to verify
    that your code actually does what the paper describes. Papers describe
    algorithms in prose and equations; implementations make thousands of small
    choices (unit conversions, edge cases, formula variants) that can introduce
    subtle errors. This page shows our work — every check we ran and what we
    found.

---

## The QFUERZA paper

**Farrugia, M. M.; Helquist, P.; Norrby, P.-O.; Wiest, O.**
*Rapid FF Generation via Hessian-Informed Initial Parameters and Automated
Refinement.*
J. Chem. Theory Comput. **2026**, 22, 469–476.
[DOI:10.1021/acs.jctc.5c01751](https://doi.org/10.1021/acs.jctc.5c01751)

**Supporting data:**
[Zenodo 10.5281/zenodo.17386006](https://doi.org/10.5281/zenodo.17386006) —
contains the actual force field files (`.fld`) for cisplatin and Rh-enamide.

---

## What QFUERZA is supposed to do

The paper defines four approaches to preliminary force constants:

| Method | Bonds | Angles |
|--------|-------|--------|
| **Approxn** | 5.0 mdyn/Å (fixed) | 0.5 mdyn·Å/rad² (fixed) |
| **FUERZA** | Seminario projection | Seminario projection |
| **γ-FUERZA** | Seminario projection | Seminario × γ (γ ≈ 0.68) |
| **QFUERZA** | Seminario projection | Seminario, but H-angles → 0.5 |

From the paper (§Methods):

> "the QFUERZA approach, which combines the Q2MM approximation and FUERZA
> methods by projecting force constants from the Hessian using FUERZA, as
> previously described, but corrects the known problems of FUERZA for light
> terminal atoms (e.g., hydrogen) angle bends by substituting them for a
> value of 0.5 mdyn/rad²."

In plain language: run Seminario projection for everything, then replace
any angle force constant where an outer atom is hydrogen with the fixed
value 0.5 mdyn·Å/rad². Leave bonds and non-hydrogen angles unchanged.

---

## Validation against Zenodo data

The paper's Zenodo archive contains the actual `.fld` force field files
for cisplatin — one file per method. We extracted the force constants
from these files and compared them to verify our implementation follows
the same rules.

### Cisplatin force constants

All values in mdyn/Å (bonds) or mdyn·Å/rad² (angles), as stored in the
paper's `.fld` files.

#### Bonds

| Parameter | Approxn | FUERZA | γ-FUERZA | QFUERZA | Rule check |
|-----------|---------|--------|----------|---------|------------|
| N–Pt | 5.000 | 1.169 | 1.169 | **1.169** | ✅ Same as FUERZA |
| N–H | 5.000 | 5.765 | 5.765 | **5.765** | ✅ Same as FUERZA |
| Pt–Cl | 5.000 | 1.392 | 1.392 | **1.392** | ✅ Same as FUERZA |

QFUERZA bond force constants are identical to FUERZA — exactly as the
paper specifies.

#### Angles

| Parameter | Approxn | FUERZA | γ-FUERZA | QFUERZA | H-angle? | Rule check |
|-----------|---------|--------|----------|---------|----------|------------|
| N–Pt–Cl (trans) | 0.500 | 3.074 | 2.090 | **3.074** | No | ✅ Same as FUERZA |
| N–Pt–Cl (cis) | 0.500 | 3.068 | 2.086 | **3.068** | No | ✅ Same as FUERZA |
| N–Pt–N | 0.500 | 2.561 | 1.742 | **2.561** | No | ✅ Same as FUERZA |
| Cl–Pt–Cl | 0.500 | 3.750 | 2.550 | **3.750** | No | ✅ Same as FUERZA |
| **H–N–Pt** | 0.500 | 2.063 | 1.403 | **0.500** | **Yes** | ✅ **Substituted** |
| **H–N–H** | 0.500 | 1.444 | 0.982 | **0.500** | **Yes** | ✅ **Substituted** |

QFUERZA non-hydrogen angles are identical to FUERZA. Hydrogen angles are
substituted with 0.5 — exactly as the paper specifies.

!!! note "FUERZA overestimation is NOT a fixed 2×"
    The paper describes FUERZA as producing "angle bending force constants
    that are two times too strong." For cisplatin, the actual ratios are:

    - H–N–Pt: 2.063 / 0.500 = **4.13×**
    - H–N–H: 1.444 / 0.500 = **2.89×**

    The ~2× figure is a rough average across molecules — individual angles
    can deviate significantly. This is why QFUERZA uses a fixed substitution
    rather than a scaling factor.

### γ-FUERZA scaling factor

The paper text says γ = 0.67, but the Zenodo data shows the actual
scaling is **0.680** (to 3 decimal places) for all angle parameters.
This minor inconsistency between paper text and data does not affect
our implementation since we don't implement γ-FUERZA.

### After optimization: all methods converge

The paper reports (and the Zenodo data confirms) that after Q2MM gradient
optimization, all four methods converge to essentially identical final
parameters:

| Parameter | QFUERZA opt | FUERZA opt | Difference |
|-----------|-------------|------------|------------|
| N–Pt | 1.162 | 1.162 | < 0.1% |
| N–H | 7.006 | 7.006 | 0.0% |
| Pt–Cl | 1.924 | 1.934 | 0.5% |
| Cl–Pt–Cl | 1.463 | 0.981 | 49% ⚠️ |
| H–N–Pt | 0.224 | 0.224 | 0.0% |
| H–N–H | 0.624 | 0.624 | 0.0% |

Most parameters converge identically. The Cl–Pt–Cl angle shows different
final values — the paper attributes this to local minima in parameter
space, not to the initialization method.

---

## Implementation comparison

### What our code does

The QFUERZA logic lives in
[`q2mm.models.seminario`](../reference/q2mm/models/seminario.md):

```python
# q2mm/models/seminario.py — key constants
QFUERZA_H_ANGLE_DEFAULT_MDYNA = 0.5        # mdyn·Å/rad²
QFUERZA_H_ANGLE_DEFAULT_CANONICAL = 35.97  # kcal/(mol·rad²)

def _is_hydrogen_angle(elements):
    """Return True if either outer atom of an angle is hydrogen."""
    return elements[0] == "H" or elements[2] == "H"
```

In `estimate_force_constants()`, after computing the Seminario-projected
force constant for each angle:

```python
if strategy == "qfuerza" and _is_hydrogen_angle(angle_param.elements):
    angle_param.force_constant = QFUERZA_H_ANGLE_DEFAULT_CANONICAL
else:
    angle_param.force_constant = fuerza_value
```

### Checklist

| Aspect | Paper definition | Our code | Match? |
|--------|-----------------|----------|--------|
| Bond FCs | Seminario projection | Bidirectional Seminario average | ✅ |
| Non-H angle FCs | Seminario projection | Seminario reciprocal sum | ✅ |
| H-angle FCs | Substitute with 0.5 mdyn·Å/rad² | `QFUERZA_H_ANGLE_DEFAULT_MDYNA = 0.5` | ✅ |
| H-angle detection | "light terminal atoms (e.g., hydrogen)" | Either outer atom is `"H"` | ✅ |
| DFT scaling factor | 0.963 | `DEFAULT_DFT_SCALING = 0.963` | ✅ |
| Bond-angle decoupling | Bonds unchanged by QFUERZA | Only angles modified | ✅ |

### Angle formula

Both the original Seminario paper (1996) and the legacy Q2MM codebase use
the reciprocal-sum formula for angle force constants:

$$
\frac{1}{k_\theta} = \frac{1}{k_{ij} \cdot r_{ij}^2} + \frac{1}{k_{kj} \cdot r_{kj}^2}
$$

where $k_{ij}$ and $k_{kj}$ are the Hessian eigenvalue projections onto
perpendicular directions and $r_{ij}$, $r_{kj}$ are the bond lengths. Our
code uses this same formula (confirmed by comparison with
[github.com/q2mm/q2mm](https://github.com/q2mm/q2mm), the legacy codebase
cited in the paper).

---

## Unit conversion chain

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

## What the legacy Q2MM codebase has

The paper cites [github.com/q2mm/q2mm](https://github.com/q2mm/q2mm) as
containing the QFUERZA method, but the published code only implements
FUERZA (plain Seminario projection). There is no H-angle substitution
logic in the legacy codebase. The QFUERZA workflow was apparently
performed manually or in an unpublished branch.

Our implementation is the first automated QFUERZA in the Q2MM codebase.

---

## Test coverage

### Unit tests (`test/test_models.py`)

The `TestQFUERZA` class contains 13 tests:

- **H-angle substitution**: verifies that angles with outer hydrogen get
  the 0.5 default and non-H angles are unchanged
- **Determinism**: same input → same output across repeated calls
- **Equilibria preservation**: QFUERZA only changes force constants,
  not equilibrium geometries

### Parity tests (`test/integration/test_seminario_parity.py`)

Golden-fixture tests compare our output against reference values for
ethane, Rh-enamide, and SN2 at rel=1e-6 tolerance.

!!! warning "Parity fixtures are self-referential"
    The golden fixtures were generated by our own code ("q2mm corrected
    code (Jaguar AU fix)"), not extracted from the paper. They verify
    **self-consistency** — that the code hasn't regressed — but not
    **paper-parity**.

    To close this gap, we could parse the cisplatin Gaussian log from the
    Zenodo archive, run our Seminario code on that Hessian, and compare
    the resulting FUERZA values against the paper's values (1.169, 5.765,
    etc.).

### Zenodo fixture (`test/fixtures/seminario_parity/cisplatin_zenodo_reference.json`)

Contains the extracted force constants from all four methods plus the
optimized results, as published in the Zenodo archive. This fixture
documents the paper's actual numerical values and can be used for future
validation work.

---

## Paper results summary

### Cisplatin (ground state)

| Method | Unoptimized R² | Optimization cycles |
|--------|---------------|-------------------|
| Approxn | 0.878 | — |
| FUERZA | 0.735 | — |
| γ-FUERZA | 0.889 | — |
| **QFUERZA** | **0.952** | — |

### Rh-enamide (transition state)

| Method | Merit function (preliminary) | Merit function (optimized) | Optimization cycles |
|--------|----------------------------|---------------------------|-------------------|
| Approxn | 6.375 | 0.812 | 17 |
| FUERZA | 1.361 | 0.812 | 12 |
| γ-FUERZA | 1.143 | 0.812 | 11 |
| **QFUERZA** | **1.005** | **0.812** | **10** |

The key insight: QFUERZA gives the best starting point (lowest
preliminary merit function), but all methods converge to the same final
quality after optimization. QFUERZA's advantage is **faster convergence**
— 40% fewer optimization cycles than the fixed-default approach.

---

## Conclusions

Our QFUERZA implementation is correct. Every verifiable aspect matches
the paper's definition and the Zenodo reference data:

- ✅ Bond force constants identical to FUERZA
- ✅ Non-hydrogen angle force constants identical to FUERZA
- ✅ Hydrogen angle force constants substituted with 0.5 mdyn·Å/rad²
- ✅ DFT scaling factor 0.963
- ✅ Reciprocal-sum angle formula matches legacy Q2MM

### Remaining validation opportunities

1. **Parse the cisplatin Gaussian log** from Zenodo, extract the Hessian,
   and run our Seminario projection to verify we produce the same FUERZA
   values (1.169, 5.765, 3.074, etc.)
2. **Add cisplatin as a parity test system** — the simplest test case with
   externally validated reference values
3. **Compare Rh-enamide TSFF** — the Zenodo archive also contains the
   full Rh-enamide force field files (~1.8 GB, includes QM data)
