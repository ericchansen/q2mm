# Backends

Q2MM supports multiple MM backends for energy evaluation, frequency
calculation, and geometry optimization.  This page compares their capabilities
and documents when optimized parameters can be transferred between backends.

For detailed information on each backend, see the individual pages:

- [OpenMM](openmm.md) — mature, dual functional-form support (Harmonic + MM3)
- [Tinker](tinker.md) — subprocess-based MM3 backend
- [JAX](jax-engine.md) — pure-JAX differentiable backend with analytical gradients
- [JAX-MD](jax-md.md) — JAX-MD backend with periodic boundaries and neighbor lists
- [Psi4](psi4.md) — quantum mechanics backend for generating reference data
- QCEngine — reference adapter for QCSchema-compatible programs
- ASE — reference adapter for lightweight ASE calculators
- [Authoring a plugin](authoring.md) — public backend API-v1 contract and conformance

---

## Backend overview

Psi4 is a reference backend used for quantum-mechanical calculations and is not
included in the MM comparison tables below.

| Feature | OpenMM | Tinker | JAX | JAX-MD |
|---------|--------|--------|-----|--------|
| **Functional forms** | Harmonic, MM3 | MM3 | Harmonic, MM3 | Harmonic |
| **Bond/angle terms** | ✅ | ✅ | ✅ | ✅ |
| **Torsions** | ✅ | ✅ | ✅ | ✅ |
| **Improper torsions** | [Cosine model](openmm.md#supported-energy-terms) | ❌ | [Cosine model](jax-engine.md#supported-energy-terms) | [Rejected](jax-md.md#preparation-gates) |
| **vdW (LJ 12-6)** | ✅ Harmonic mode | ❌ | ✅ | ✅ |
| **vdW (Buckingham exp-6)** | ✅ MM3 mode | ✅ | ✅ MM3 mode | ❌ |
| **Electrostatics** | ❌ | Tinker default | [MM3 bond dipoles](jax-engine.md#preparation-gates) | Infrastructure only (charges zeroed) |
| **1-4 scaling** | ✅ AMBER (ε/2) in Harmonic | MM3 default | ❌ Not implemented | ✅ Configurable (default AMBER) |
| **Periodic boundaries** | ❌ | ❌ | ❌ | ✅ |
| **Neighbor lists** | ❌ | ❌ | ❌ | ✅ (jax-md native) |
| **Runtime param updates** | ✅ | ❌ (subprocess per call) | ✅ | ✅ |
| **Analytical gradients** | ⚠️ bond/angle only | ❌ | ✅ via `jax.grad` | ✅ via `jax.grad` |
| **JIT compilation** | N/A | N/A | ✅ | ✅ |
| **Platform** | Linux, macOS, Windows | Linux, macOS | Linux, macOS, WSL2 | Linux, macOS, WSL2 |

---

## Functional forms

Each backend only accepts force fields whose `functional_form` is in its
supported set.  Attempting to use an unsupported form raises an error.

### Populated-term preparation gates

A supported functional-form name does not imply that every supplied term is
implemented. The following unsupported content raises `PreparationError`
before parameter-layout or native-state construction:

| Backend | Rejected populated content |
|---------|----------------------------|
| [JAX](jax-engine.md#preparation-gates) | CMAP, nondefault vdW reduction, and [known wildcard torsions](#wildcard-torsion-boundary) in both forms; stretch-bend and bond dipoles in harmonic mode |
| [OpenMM](openmm.md#preparation-gates) | Bond dipoles, nondefault vdW reduction, and [known wildcard torsions](#wildcard-torsion-boundary) in both forms |
| [JAX-MD](jax-md.md#preparation-gates) | Urey-Bradley, CMAP, improper torsions, bond dipoles, and nondefault vdW reduction |

Populated zero-valued grids, stretch-bend/improper records, and Urey-Bradley
fields still declare terms. Either Urey-Bradley field being supplied is
enough to require support. Bond dipoles are populated when their moment is
nonzero; vdW reduction is nondefault when it differs from `0.0`, even if
the current epsilon is zero.

These gates inspect canonical force-field parameters, not source labels,
opaque template records, or molecular reference partial charges. Reference
charges are not automatically a request for point-charge MM energy.
Existing supported terms and native functional models are unchanged.

`test/test_backend_term_coverage.py` separates dependency-light preparation
checks from backend-marked runtime cases. In particular, its mocked JAX-MD
preparation checks are not JAX-MD runtime evidence; the `jax_md` cases require
an available supported-platform installation.

#### Wildcard torsion boundary

JAX and OpenMM do not expand native wildcard torsions. Their preparation
gates reject proper or improper records containing the known tokens below,
even when their amplitude is zero or other valid terms make the system
nonempty. Exact typed and ordinary element-based matching is unchanged;
full wildcard matching and its precedence rules remain deferred.

| Token | Format evidence |
|-------|-----------------|
| `X` (case-sensitive, whole token) | [AMBER parameter card 6 and general improper types](https://ambermd.org/FileFormats.php); retained by the frcmod loader |
| `00` | Existing MM3 loader representation and [documented MM3 transfer boundary](../benchmarks/optimizer-comparison.md#macromodel-mm3-transfer-boundary) |
| Integer zero (`0`, including padded or signed zero spellings) | [Tinker native torsion assignment](https://github.com/TinkerTools/tinker/blob/c9698d2101c5f66ce1d413f4aa2d5f62e4c22df2/source/ktors.f) uses zero atom classes for terminal wildcard matching; the loader retains their labels |

A complete dash-separated type quadruplet in `TorsionParam.env_id` takes
precedence over inferred elements. If no complete quadruplet is available,
the canonical element labels are inspected instead. This avoids turning
ordinary type names such as `X1` or `Xe` into wildcards merely because an
element inference produced `X`. Substrings, lowercase `x`, asterisks, and
other unknown names are not guessed to be wildcard syntax.

Names, comments, source paths, row numbers, and reference partial charges
do not trigger this gate. `test/test_wildcard_term_rejection.py` includes
small CPU omission reproductions, exact-binding controls, and preflight
checks. No loader, Tinker backend, or native AMBER/Tinker engine is changed.

### Harmonic

Standard AMBER/OPLSAA-style potential:

- **Bonds:** `E = k·(r − r₀)²`
- **Angles:** `E = k·(θ − θ₀)²`
- **vdW:** `E = 4ε·[(σ/r)¹² − (σ/r)⁶]`

Supported by: **OpenMM** (Harmonic mode), **JAX** (Harmonic mode), **JAX-MD**

### MM3

Allinger's MM3 potential with higher-order anharmonic corrections:

- **Bonds:** `E = k·(10·Δr)²·(1 − 2.55·(10·Δr) + 4.7266·(10·Δr)²)`
- **Angles:** `E = k·Δθ²·(1 − 0.014·Δθ° + 5.6×10⁻⁵·Δθ°² − …)`
- **vdW:** Buckingham exp-6: `E = ε·[184000·exp(−12r/rᵥ) − 2.25·(rᵥ/r)⁶]`

Supported by: **OpenMM**, **JAX**, **Tinker**

!!! info "JAX MM3 support"
    The JAX backend supports both harmonic and MM3 functional forms, including
    cubic bond stretch, sextic angle bend, and Buckingham exp-6 vdW terms.

---

## Non-bonded treatment

Non-bonded interactions (van der Waals, electrostatics) are computed
between all atom pairs not excluded by bonding topology.  The details
differ between backends.

### Exclusions

All backends exclude **1-2** (bonded) and **1-3** (angle endpoint) pairs
from non-bonded calculations.

### 1-4 Scaling

Atoms separated by exactly 3 bonds ("1-4 pairs") often receive scaled-down
non-bonded interactions.  **This is a key compatibility difference:**

| Backend | 1-4 LJ Scaling | 1-4 Coulomb Scaling |
|--------|----------------|---------------------|
| **OpenMM** (Harmonic) | ε/2 (AMBER `scnb=2.0`) | N/A (no charges) |
| **OpenMM** (MM3) | None (MM3 convention) | N/A |
| **Tinker** | MM3 default | MM3 default |
| **JAX** | **Not implemented** | N/A |
| **JAX-MD** | Configurable (default: 0.5) | Configurable |

!!! warning "JAX backend lacks 1-4 scaling"
    The JAX backend does not implement 1-4 pair scaling.  For molecules with
    1-4 non-bonded interactions (anything with 4+ atoms in a chain), JAX
    will compute slightly different non-bonded energies than OpenMM or
    JAX-MD.  For small molecules where the bonded energy dominates (bonds +
    angles only), this difference is negligible.

### Combining rules

All backends use **geometric** combining rules for cross-term vdW
parameters:

- `σ_ij = √(σ_i · σ_j)`
- `ε_ij = √(ε_i · ε_j)`

### Cutoffs

| Backend | Default | Notes |
|--------|---------|-------|
| **OpenMM** | No cutoff | All pairs computed |
| **Tinker** | Tinker config | Depends on .key file |
| **JAX** | No cutoff | All pairs computed |
| **JAX-MD** | ~12 Å | Configurable; uses neighbor lists |

---

## Parameter transferability

Can parameters optimized on one backend be used on another?  This depends
on whether the backends compute the same energy for the same force field.

### Compatibility matrix

| From ↓ / To → | OpenMM (Harmonic) | OpenMM (MM3) | Tinker | JAX | JAX-MD |
|----------------|:-:|:-:|:-:|:-:|:-:|
| **OpenMM (Harmonic)** | ✅ | ❌ | ❌ | ⚠️ | ✅ |
| **OpenMM (MM3)** | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Tinker** | ❌ | ✅ | ✅ | ✅ | ❌ |
| **JAX** | ⚠️ | ✅ | ✅ | ✅ | ⚠️ |
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
the JAX backend will give different non-bonded energies than OpenMM or
JAX-MD.

### Verified parity

Cross-backend energy and frequency agreement has been measured on CH₃F
(see [benchmarks](../systems/small-molecules.md#interpretation)):

- **JAX ↔ JAX-MD:** < 10⁻²⁰ kcal/mol energy difference (machine precision)
- **JAX ↔ OpenMM:** < 10⁻¹⁸ kcal/mol energy difference
- **Frequencies:** < 0.001 cm⁻¹ max deviation across all backends

CH₃F has no 1-4 pairs, so all three harmonic backends agree exactly.

---

## Choosing a backend

| Use Case | Recommended Backend | Why |
|----------|-------------------|-----|
| **Fast optimization** | JAX or JAX-MD | Fastest harmonic / analytical-gradient options in the current benchmark set; see [benchmarks](../benchmarks/index.md) for workload-specific comparisons |
| **MM3 force fields** | OpenMM, Tinker, or JAX | Backends supporting MM3 functional forms |
| **Periodic systems** | JAX-MD | Only backend with periodic boundary support |
| **Torsion optimization** | OpenMM, Tinker, JAX, or JAX-MD | All backends support torsions |
| **Widest compatibility** | OpenMM | Supports both Harmonic and MM3, mature ecosystem |
| **Gradient-based optimizers** | JAX or JAX-MD | Analytical `jax.grad` eliminates finite-difference overhead |

---

## Unit conventions

All backends accept parameters in **canonical units** (defined in
`q2mm.models.units`).  Each backend converts internally as needed:

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
    `E = ½·k·(r − r₀)²`, so the backend doubles the force constant during
    conversion.

---

## Backend plugins

Backend API version 1 is the stable public authoring contract. External
distributions advertise a lightweight JSON-safe manifest in the exact
`q2mm.backends` entry-point group; Q2MM discovers descriptors lazily and imports
implementations only on explicit load. See
[Authoring a backend plugin](authoring.md) for the exact manifest schema,
runtime contract, failure-isolation rules, and public conformance runner.
