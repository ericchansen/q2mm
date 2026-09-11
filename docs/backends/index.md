# Backends

Q2MM supports multiple MM backends for energy evaluation, frequency
calculation, and geometry optimization.  This page compares their capabilities
and documents when optimized parameters can be transferred between backends.
Shared interfaces and canonical units do not imply the same physical model.
Term coverage, mixing rules, exclusions, and native conventions must also match.

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
| **Electrostatics** | [Canonical bond dipoles rejected](openmm.md#preparation-gates) | [Native template terms are not canonical support](tinker.md#limitations) | [MM3 bond dipoles](jax-engine.md#preparation-gates) | [No general requested point-charge model](#populated-term-preparation-gates) |
| **1-4 scaling** | ✅ AMBER (ε/2) in Harmonic | MM3 default | ❌ Not implemented | ✅ Configurable (default AMBER) |
| **Periodic boundaries** | ❌ | ❌ | ❌ | ✅ |
| **Neighbor lists** | ❌ | ❌ | ❌ | ✅ (jax-md native) |
| **Runtime param updates** | ✅ | ❌ (subprocess per call) | ✅ | ✅ |
| **Parameter gradients** | [Analytical bond/angle/torsion; numerical vdW](openmm.md#capabilities) | [Not exposed](tinker.md#capabilities) | [AD on supported paths](jax-engine.md) | [AD on supported paths](jax-md.md) |
| **JIT compilation** | N/A | N/A | ✅ | ✅ |
| **Platform** | [Linux, macOS, Windows](openmm.md#installation) | [Linux, macOS, Windows with compatible binaries](tinker.md#installation) | [Linux/macOS and native Windows CPU](jax-engine.md#installation) | [Supported Linux/macOS environments, including WSL2](jax-md.md#installation) |

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
precedence over inferred elements. Empty delimiter fields are discarded as
in existing identifier cleaning, so signed zero cannot hide a wildcard.
If no complete quadruplet is available,
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

MM3-style models include higher-order anharmonic corrections. In canonical
coordinate units, their schematic terms are:

- **Bonds:** `E = k·Δr²·(1 − c3·Δr + c4·Δr²)`, with `Δr` in Angstrom.
- **Angles:** `E = k·Δθ_rad²·(1 + a3·Δθ_deg + a4·Δθ_deg² + …)`.
- **vdW:** Buckingham exp-6: `E = ε·[184000·exp(−12r/rᵥ) − 2.25·(rᵥ/r)⁶]`

Supported by: **OpenMM**, **JAX**, **Tinker**

The exact coefficients, damping, short-range modifications, and native
prefactors are part of each implementation's model. The schematic form
does not assert that all three implementations use identical policies;
see the [compatibility qualifications](#compatibility-matrix).

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
    JAX-MD. If a system has no such pairs, this particular scaling difference
    does not apply. That does not eliminate other differences or establish
    cross-backend equivalence.

### Combining rules

Combining rules are backend/form policies, not a consequence of the
canonical length and energy units:

| Backend/form | LJ sigma mixing | LJ epsilon mixing | Source |
|--------------|-----------------|-------------------|--------|
| OpenMM harmonic | Arithmetic: `(sigma_i + sigma_j) / 2` | Geometric | [Native NonbondedForce and 1-4 construction](https://github.com/ericchansen/q2mm/blob/a03e518463bf1b7f8a37333927e9cd13fb3f3e17/q2mm/backends/mm/openmm.py) |
| JAX harmonic | Geometric: `sqrt(sigma_i * sigma_j)` | Geometric | [`_lj_12_6_energy`](https://github.com/ericchansen/q2mm/blob/a03e518463bf1b7f8a37333927e9cd13fb3f3e17/q2mm/backends/mm/jax_engine.py) |
| JAX-MD harmonic | Geometric | Geometric | [Explicit pair-energy construction](https://github.com/ericchansen/q2mm/blob/a03e518463bf1b7f8a37333927e9cd13fb3f3e17/q2mm/backends/mm/jax_md_engine.py) |

Different sigma rules can produce different mixed-species energies even
without any 1-4 pairs. Neither rule is silently substituted for the other.
MM3 and native Tinker conventions must be considered separately, including
their radius definitions, template settings, and functional coefficients.

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

There is no blanket machine-precision transfer guarantee between backend
names, even when both accept the same `FunctionalForm`.

| Comparison | Required qualification |
|------------|------------------------|
| Harmonic OpenMM, JAX, and JAX-MD | Match [mixing rules](#combining-rules), [exclusions](#exclusions), [1-4 scaling](#1-4-scaling), [cutoff/nonbonded settings](#cutoffs), [charge models](#backend-overview), and supported terms for the actual system; validate relevant objective/derivative agreement, not identical differentiation implementations. |
| MM3 OpenMM, JAX, and Tinker | Check [populated-term coverage](#populated-term-preparation-gates), [cutoffs](#cutoffs), [exclusions](#exclusions), [electrostatic models](#backend-overview), native/template coefficients, damping, short-range behavior, and signed interactions; see [Tinker limitations](tinker.md#limitations). |
| Any backend or configuration change | Re-establish the intended energy, gradient, Hessian, and objective agreement with the recorded settings; [API conformance](authoring.md) alone is not physical conformance. |

Removing vdW terms or 1-4 pairs only removes those specific contributions.
It does not prove equality of torsion conventions, damping, other populated
terms, or the complete objective. In particular, the Tinker standalone
writer retains its own angle-sextic coefficient rather than silently
adopting the JAX/OpenMM value; see the
[native writer](https://github.com/ericchansen/q2mm/blob/a03e518463bf1b7f8a37333927e9cd13fb3f3e17/q2mm/backends/mm/tinker.py).

### Case-specific parity evidence

Historical small-system comparisons are described in the
[benchmark results](../systems/small-molecules.md#interpretation).
Such results apply to their recorded inputs, parameter assignments,
software versions, and settings. They do not establish arbitrary
cross-backend transfer, full native-model support, or publication
reproduction.

Record cutoff, periodic-boundary, charge-model, and nonbonded settings
alongside the force field. If historical artifacts omit a setting, do not
assume that the current backend default was used or that two runs shared it.

---

## Choosing a backend

| Use Case | Recommended Backend | Why |
|----------|-------------------|-----|
| **Fast optimization** | JAX or JAX-MD | Fastest harmonic / analytical-gradient options in the current benchmark set; see [benchmarks](../benchmarks/index.md) for workload-specific comparisons |
| **MM3 force fields** | OpenMM, Tinker, or JAX | Check [populated-term coverage](#populated-term-preparation-gates) and native conventions, not just the form name |
| **Periodic systems** | JAX-MD | Only backend with periodic boundary support |
| **Torsion optimization** | Select for the required torsion model | Check [wildcards](#wildcard-torsion-boundary), proper/improper coverage, and signed conventions before choosing |
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
