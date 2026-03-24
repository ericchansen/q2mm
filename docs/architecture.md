# Architecture

This page describes Q2MM's internal architecture, the motivation behind
recent design decisions, and how the pieces fit together.

---

## Background

Q2MM originated in the Norrby and Wiest research groups as a tool for
deriving transition-state force fields using quantum mechanical reference data.
Early versions of the code were tightly integrated with the **MM3 force field**
and Schrödinger's MacroModel as the primary MM engine. This coupling was a
natural fit for the research context — MM3 was the target force field and
MacroModel was the available computational engine.

However, it introduced architectural limitations:

- **MM3 units leaked into the optimizer.** Bond force constants were stored in
  mdyn/Å, angle force constants in mdyn·Å/rad² — both MM3 conventions.
  Step sizes, bounds, and convergence thresholds were all calibrated to these
  units, making it difficult to optimize parameters for other force fields.
- **Single-backend assumption.** The optimization loop assumed MacroModel-style
  I/O (write parameter file → run calculation → parse output), with no clean
  abstraction for swapping in a different MM engine.
- **Format-coupled data flow.** Parsing, model construction, and optimization
  were interleaved, so adding support for a new file format or force field
  required changes across multiple subsystems.

---

## Design Goals of the Refactor

The current architecture addresses those limitations with three principles:

### 1. Format-Agnostic Data Models

All scientific algorithms operate on **format-neutral data structures**:

| Structure | Purpose |
|-----------|---------|
| `ForceField` | Bond, angle, torsion, and vdW parameters with metadata |
| `Q2MMMolecule` | Cartesian geometry, topology, and optional Hessian |
| `ReferenceData` | QM reference values (energies, frequencies, geometries) |

These models have no knowledge of MM3, AMBER, CHARMM, or any file format.
Parsers and savers translate between external formats and internal models at
the boundary.

### 2. Canonical Internal Units

To decouple the optimizer from any particular force field convention, Q2MM
uses a **canonical unit system** internally:

| Quantity | Canonical Unit | Convention |
|----------|----------------|------------|
| Bond force constant | kcal/(mol·Å²) | E = k(r − r₀)² (no ½ factor) |
| Angle force constant | kcal/(mol·rad²) | E = k(θ − θ₀)² (no ½ factor) |
| Torsion barrier | kcal/mol | Standard Fourier form |
| vdW epsilon | kcal/mol | — |
| Bond equilibrium | Å | — |
| Angle equilibrium | degrees | — |
| vdW radius | Å | — |

This is an AMBER-like convention and the most common in computational
chemistry. The key insight is that the **optimization pipeline** — step sizes,
bounds, convergence criteria, and objective function weights — is calibrated
once in canonical units and works for any force field.

**Conversion happens at the boundary:**

```
┌──────────────────────────────────────────────────────────────┐
│                   Canonical Unit Space                       │
│                                                              │
│   Seminario ──→ ForceField ──→ Objective ──→ Optimizer      │
│                                    ↕                        │
│                               MM Engine                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
        ↑                                           ↓
   Loaders convert                            Savers convert
   format → canonical                     canonical → format

   MM3 .fld  ──→  ×71.94  ──→ canonical  ──→  ÷71.94  ──→ MM3 .fld
   AMBER .frcmod ──→  (identity)  ──→ canonical  ──→  (identity)  ──→ AMBER .frcmod
   Tinker .prm  ──→  ×71.94  ──→ canonical  ──→  ÷71.94  ──→ Tinker .prm
```

Each loader (e.g., `load_mm3_fld`) multiplies by the appropriate conversion
factor on read; each saver divides on write. The optimizer never sees
format-specific values.

### 3. Pluggable Backends

MM engines implement the `MMEngine` abstract base class:

```python
class MMEngine(ABC):
    @abstractmethod
    def energy(self, structure, forcefield) -> float: ...

    @abstractmethod
    def minimize(self, structure, forcefield) -> tuple: ...

    @abstractmethod
    def hessian(self, structure, forcefield) -> np.ndarray: ...

    @abstractmethod
    def frequencies(self, structure, forcefield) -> list[float]: ...

    def supported_functional_forms(self) -> set[str]: ...
```

Each engine handles its own unit conversions between canonical and whatever
the underlying library expects (e.g., OpenMM uses kJ/mol internally, Tinker
uses kcal/mol).

---

## Module Organization

```
q2mm/
├── models/               # Format-neutral data structures
│   ├── forcefield.py     # ForceField, BondParam, AngleParam, TorsionParam, FunctionalForm
│   ├── molecule.py       # Q2MMMolecule, DetectedBond, DetectedTorsion
│   ├── ff_io.py          # Loaders/savers (MM3, AMBER, Tinker, OpenMM XML)
│   ├── seminario.py      # Hessian → initial force constants
│   ├── hessian.py        # Hessian manipulation, eigenvalue analysis
│   ├── units.py          # Conversion constants and helpers
│   └── identifiers.py    # Atom type matching utilities
│
├── backends/             # MM and QM engine integrations
│   ├── base.py           # MMEngine and QMEngine ABCs
│   ├── mm/
│   │   ├── openmm.py     # OpenMM engine (harmonic + MM3 dual-mode)
│   │   ├── tinker.py        # Tinker engine (subprocess-based)
│   │   ├── jax_engine.py   # JAX engine (differentiable, analytical gradients)
│   │   └── jax_md_engine.py # JAX-MD engine (periodic, neighbor lists)
│   └── qm/
│       └── psi4.py       # Psi4 engine (QM single-points, Hessians)
│
├── optimizers/           # Parameter fitting machinery
│   ├── objective.py      # ObjectiveFunction, ReferenceData
│   ├── scipy_opt.py      # ScipyOptimizer (L-BFGS-B, Nelder-Mead, etc.)
│   ├── cycling.py        # GRAD→SIMP parameter cycling
│   ├── scoring.py        # Legacy scoring functions
│   └── defaults.py       # Default step sizes and bounds
│
├── parsers/              # File format I/O
│   ├── gaussian.py       # Gaussian log/fchk parsing
│   ├── jaguar.py         # Jaguar input/output parsing
│   ├── macromodel.py     # MacroModel log parsing
│   ├── mm3.py            # MM3 .fld file parsing
│   ├── tinker_ff.py      # Tinker parameter file parsing
│   ├── amber_ff.py       # AMBER frcmod parsing
│   ├── mol2.py           # MOL2 structure parsing
│   └── ...               # Supporting utilities
│
├── diagnostics/          # Analysis and reporting
│   ├── benchmark.py      # Timing and accuracy benchmarks
│   ├── pes_distortion.py # PES distortion analysis
│   ├── report.py         # Summary report generation
│   └── tables.py         # Formatted table output
│
├── constants.py          # Physical constants
└── elements.py           # Periodic table data
```

### Dependency Flow

```mermaid
flowchart TD
    subgraph Core["Core (no external deps)"]
        constants[constants.py]
        elements[elements.py]
        units[models/units.py]
    end

    subgraph Models["Model Layer"]
        ff[ForceField]
        mol[Q2MMMolecule]
        hess[Hessian]
        ffio[ff_io]
        sem[Seminario]
    end

    subgraph Opt["Optimizer Layer"]
        obj[ObjectiveFunction]
        ref[ReferenceData]
        scipy[ScipyOptimizer]
        cycling[OptimizationLoop]
    end

    subgraph Engines["Backend Layer"]
        omm[OpenMMEngine]
        tk[TinkerEngine]
        jax[JaxEngine]
        psi4[Psi4Engine]
    end

    subgraph IO["I/O Layer"]
        parsers[Parsers]
        savers[Savers]
    end

    constants --> units
    units --> ffio
    units --> sem
    ff --> ffio
    ff --> sem
    mol --> sem
    hess --> sem

    ff --> obj
    ref --> obj
    mol --> obj
    obj --> scipy
    obj --> cycling

    obj --> omm
    obj --> tk
    obj --> jax

    parsers --> mol
    parsers --> ref
    savers --> ff
```

---

## Functional Forms

The `FunctionalForm` enum tracks which energy expression a `ForceField` uses:

| Form | Bond Energy | Angle Energy | vdW |
|------|-------------|--------------|-----|
| `HARMONIC` | k(r − r₀)² | k(θ − θ₀)² | Lennard-Jones 12-6 |
| `MM3` | k(r − r₀)²[1 − 2.55Δ + …] | k(θ − θ₀)²[1 − 0.014Δ + …] | Buckingham exp-6 |

The functional form is set when loading a force field and validated by savers
and engines:

- **Loaders** set `forcefield.functional_form` based on the source format.
- **Savers** check `_FORMAT_COMPATIBLE_FORMS` before writing — e.g., an AMBER
  frcmod saver rejects an MM3-form force field.
- **Engines** validate compatibility in `_validate_forcefield()` before
  creating a simulation context.

This prevents silent mismatches where, for example, an MM3 force field's
cubic/sextic constants would be interpreted as harmonic spring constants.

---

## Data Flow: End-to-End Pipeline

A typical optimization workflow follows this path:

```mermaid
sequenceDiagram
    participant User
    participant Parser
    participant Seminario
    participant Objective
    participant Optimizer
    participant Engine

    User->>Parser: Load QM data (.fchk, .log)
    Parser->>Parser: Extract geometry, Hessian, energies
    Parser-->>User: Q2MMMolecule + ReferenceData

    User->>Seminario: estimate_force_constants(molecule)
    Seminario->>Seminario: Project Hessian → internal coords
    Seminario-->>User: ForceField (canonical units)

    User->>Objective: ObjectiveFunction(ff, engine, mols, ref)

    User->>Optimizer: optimizer.optimize(objective)
    loop Until converged
        Optimizer->>Objective: evaluate(param_vector)
        Objective->>Engine: energy(mol, ff) / frequencies(mol, ff)
        Engine->>Engine: Convert canonical → engine units
        Engine->>Engine: Run MM calculation
        Engine-->>Objective: MM values
        Objective->>Objective: Compute weighted residuals
        Objective-->>Optimizer: penalty score
    end
    Optimizer-->>User: OptimizationResult

    User->>User: ff.to_amber_frcmod() / ff.to_mm3_fld()
```

### Key Design Invariants

1. **Canonical units everywhere inside the pipeline.** No format-specific unit
   ever reaches the optimizer. Loaders convert on input; savers convert on
   output; engines convert at their own boundary.

2. **Immutable topology.** `Q2MMMolecule` topology (bonds, angles) is fixed at
   construction. Only `ForceField` parameter *values* change during
   optimization.

3. **Stateless engines.** `energy()` and `frequencies()` are pure functions of
   (molecule, forcefield). Engines may cache OpenMM `Context` objects for
   performance, but the cache is invalidated when topology changes.

4. **Functional form consistency.** A `ForceField` carries its `functional_form`
   from load to save. Engines and savers validate compatibility — you cannot
   accidentally evaluate an MM3 force field with a harmonic engine.

---

## Adding a New Backend

To support a new MM engine:

1. **Subclass `MMEngine`** in `q2mm/backends/mm/`.
2. **Implement** `energy()`, `frequencies()`, and optionally `gradient()`.
3. **Handle unit conversion** between canonical (kcal/mol/Å²) and whatever your
   engine expects internally.
4. **Declare supported forms** via `supported_functional_forms()`.
5. **Add tests** with the appropriate backend marker (e.g., `@pytest.mark.myengine`).

The optimizer, objective function, and all other pipeline components work
unchanged.

## Adding a New Force Field Format

To support a new file format:

1. **Add a loader** in `ff_io.py` (e.g., `load_charmm_rtf()`) that reads the
   file and returns a `ForceField` with parameters in canonical units.
2. **Add a saver** (e.g., `save_charmm_rtf()`) that converts canonical → format
   units on write.
3. **Register compatible forms** in `_FORMAT_COMPATIBLE_FORMS`.
4. **Add convenience methods** on `ForceField` (e.g., `from_charmm_rtf()`,
   `to_charmm_rtf()`).
