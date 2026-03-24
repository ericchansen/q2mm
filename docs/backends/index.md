# Backend Engines

Q2MM supports multiple MM backend engines for energy evaluation, frequency
calculation, and geometry optimization.  This page compares their capabilities
and documents when optimized parameters can be transferred between engines.

For detailed information on each backend, see the individual pages:

- [OpenMM](openmm.md) — mature, dual functional-form support (Harmonic + MM3)
- [Tinker](tinker.md) — subprocess-based MM3 engine
- [JAX](jax.md) — differentiable engines with analytical gradients (JaxEngine + JaxMDEngine)

---

## Engine Overview

| Feature | OpenMM | Tinker | JAX | JAX-MD |
|---------|--------|--------|-----|--------|
| **Functional forms** | Harmonic, MM3 | MM3 | Harmonic | Harmonic |
| **Bond/angle terms** | ✅ | ✅ | ✅ | ✅ |
| **Torsions** | ⚠️ awaiting [#127](https://github.com/ericchansen/q2mm/issues/127) | ⚠️ awaiting [#127](https://github.com/ericchansen/q2mm/issues/127) | ⚠️ awaiting [#127](https://github.com/ericchansen/q2mm/issues/127) | ⚠️ engine ready, awaiting [#127](https://github.com/ericchansen/q2mm/issues/127) |
| **Improper torsions** | ❌ | ❌ | ❌ | ❌ |
| **vdW (LJ 12-6)** | ✅ Harmonic mode | ✅ via MM3 | ✅ | ✅ |
| **vdW (Buckingham exp-6)** | ✅ MM3 mode | ✅ | ❌ | ❌ |
| **Electrostatics** | ❌ | Tinker default | ❌ | Infrastructure only (charges zeroed) |
| **1-4 scaling** | ✅ AMBER (ε/2) in Harmonic | MM3 default | ❌ Not implemented | ✅ Configurable (default AMBER) |
| **Periodic boundaries** | ❌ | ❌ | ❌ | ✅ |
| **Neighbor lists** | ❌ | ❌ | ❌ | ✅ (jax-md native) |
| **Runtime param updates** | ✅ | ❌ (subprocess per call) | ✅ | ✅ |
| **Analytical gradients** | ❌ | ❌ | ✅ via `jax.grad` | ✅ via `jax.grad` |
| **JIT compilation** | N/A | N/A | ✅ | ✅ |
| **Platform** | Linux, macOS, Windows | Linux, macOS | Linux, macOS, WSL2 | Linux, macOS, WSL2 |

---

## Functional Forms

Each engine only accepts force fields whose `functional_form` is in its
supported set.  Attempting to use an unsupported form raises an error.

### Harmonic

Standard AMBER/OPLSAA-style potential:

- **Bonds:** `E = k·(r − r₀)²`
- **Angles:** `E = k·(θ − θ₀)²`
- **vdW:** `E = 4ε·[(σ/r)¹² − (σ/r)⁶]`

Supported by: **OpenMM**, **JAX**, **JAX-MD**

### MM3

Allinger's MM3 potential with higher-order anharmonic corrections:

- **Bonds:** `E = k·(10·Δr)²·(1 − 2.55·(10·Δr) + 4.7266·(10·Δr)²)`
- **Angles:** `E = k·Δθ²·(1 − 0.014·Δθ° + 5.6×10⁻⁵·Δθ°² − …)`
- **vdW:** Buckingham exp-6: `E = ε·[184000·exp(−12r/rᵥ) − 2.25·(rᵥ/r)⁶]`

Supported by: **OpenMM**, **Tinker**

!!! info "Adding MM3 to JAX engines"
    Issue [#91](https://github.com/ericchansen/q2mm/issues/91) tracks adding
    MM3 support to the JAX engine.  The harmonic form is a subset of MM3
    (set anharmonic coefficients to zero), so MM3 parameters cannot be used
    on a harmonic-only engine without losing the higher-order terms.

---

## Non-Bonded Treatment

Non-bonded interactions (van der Waals, electrostatics) are computed
between all atom pairs not excluded by bonding topology.  The details
differ between engines.

### Exclusions

All engines exclude **1-2** (bonded) and **1-3** (angle endpoint) pairs
from non-bonded calculations.

### 1-4 Scaling

Atoms separated by exactly 3 bonds ("1-4 pairs") often receive scaled-down
non-bonded interactions.  **This is a key compatibility difference:**

| Engine | 1-4 LJ Scaling | 1-4 Coulomb Scaling |
|--------|----------------|---------------------|
| **OpenMM** (Harmonic) | ε/2 (AMBER `scnb=2.0`) | N/A (no charges) |
| **OpenMM** (MM3) | None (MM3 convention) | N/A |
| **Tinker** | MM3 default | MM3 default |
| **JAX** | **Not implemented** | N/A |
| **JAX-MD** | Configurable (default: 0.5) | Configurable |

!!! warning "JAX Engine lacks 1-4 scaling"
    The JAX engine does not implement 1-4 pair scaling.  For molecules with
    1-4 non-bonded interactions (anything with 4+ atoms in a chain), JAX
    will compute slightly different non-bonded energies than OpenMM or
    JAX-MD.  For small molecules where the bonded energy dominates (bonds +
    angles only), this difference is negligible.

### Combining Rules

All engines use **geometric** combining rules for cross-term vdW
parameters:

- `σ_ij = √(σ_i · σ_j)`
- `ε_ij = √(ε_i · ε_j)`

### Cutoffs

| Engine | Default | Notes |
|--------|---------|-------|
| **OpenMM** | No cutoff | All pairs computed |
| **Tinker** | Tinker config | Depends on .key file |
| **JAX** | No cutoff | All pairs computed |
| **JAX-MD** | ~12 Å | Configurable; uses neighbor lists |

---

## Parameter Transferability

Can parameters optimized on one engine be used on another?  This depends
on whether the engines compute the same energy for the same force field.

### Compatibility Matrix

| From ↓ / To → | OpenMM (Harmonic) | OpenMM (MM3) | Tinker | JAX | JAX-MD |
|----------------|:-:|:-:|:-:|:-:|:-:|
| **OpenMM (Harmonic)** | ✅ | ❌ | ❌ | ⚠️ | ✅ |
| **OpenMM (MM3)** | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Tinker** | ❌ | ✅ | ✅ | ❌ | ❌ |
| **JAX** | ⚠️ | ❌ | ❌ | ✅ | ⚠️ |
| **JAX-MD** | ✅ | ❌ | ❌ | ⚠️ | ✅ |

**Legend:**

- ✅ **Identical** — same energy to machine precision
- ⚠️ **Bonded terms match, non-bonded may differ** — see notes below
- ❌ **Incompatible** — different functional form

### When ⚠️ Becomes ✅

The ⚠️ entries (OpenMM Harmonic ↔ JAX, JAX ↔ JAX-MD) produce identical
energies when:

1. **The molecule has no 1-4 non-bonded pairs** (e.g., water, CH₃F with
   only 3–5 atoms) — then the missing 1-4 scaling in JAX doesn't matter.
2. **vdW parameters are zero** (only optimizing bonded terms) — then
   non-bonded differences vanish entirely.

For molecules with significant 1-4 interactions (longer chains, rings),
the JAX engine will give different non-bonded energies than OpenMM or
JAX-MD.

### Verified Parity

Cross-engine energy and frequency agreement has been measured on CH₃F
(see [benchmarks](../benchmarks/small-molecules.md#cross-engine-parity)):

- **JAX ↔ JAX-MD:** < 10⁻²⁰ kcal/mol energy difference (machine precision)
- **JAX ↔ OpenMM:** < 10⁻¹⁸ kcal/mol energy difference
- **Frequencies:** < 0.001 cm⁻¹ max deviation across all engines

CH₃F has no 1-4 pairs, so all three harmonic engines agree exactly.

---

## Choosing an Engine

| Use Case | Recommended Engine | Why |
|----------|-------------------|-----|
| **Fast optimization** | JAX or JAX-MD | 5–10× faster than OpenMM, analytical gradients |
| **MM3 force fields** | OpenMM or Tinker | Only engines supporting MM3 |
| **Periodic systems** | JAX-MD | Only engine with periodic boundary support |
| **Torsion optimization** | JAX-MD (once [#127](https://github.com/ericchansen/q2mm/issues/127) lands) | Only engine with full torsion support |
| **Widest compatibility** | OpenMM | Supports both Harmonic and MM3, mature ecosystem |
| **Gradient-based optimizers** | JAX or JAX-MD | Analytical `jax.grad` eliminates finite-difference overhead |

---

## Unit Conventions

All engines accept parameters in **canonical units** (defined in
`q2mm.models.units`).  Each engine converts internally as needed:

| Quantity | Canonical Unit | Convention |
|----------|---------------|------------|
| Bond force constant | kcal/(mol·Å²) | `E = k·(r − r₀)²` (no ½ factor) |
| Bond equilibrium | Å | — |
| Angle force constant | kcal/(mol·rad²) | `E = k·(θ − θ₀)²` (no ½ factor) |
| Angle equilibrium | degrees | Converted to radians internally |
| vdW epsilon | kcal/mol | — |
| vdW radius | Å (Rmin/2) | Converted to LJ σ where needed |

!!! note "The ½ factor"
    Q2MM uses `E = k·(x − x₀)²` **without** the ½ factor.  This matches
    AMBER and MM3 conventions.  OpenMM's `HarmonicBondForce` uses
    `E = ½·k·(r − r₀)²`, so the engine doubles the force constant during
    conversion.
