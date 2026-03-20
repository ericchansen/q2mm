# API Overview

This page documents the public API of Q2MM — the key classes, functions, and
modules you'll use to build force field optimization workflows.

---

## Models (`q2mm.models`)

Core data structures for force fields and molecules.

### `ForceField`

[source](https://github.com/ericchansen/q2mm/blob/master/q2mm/models/forcefield.py)

Central data structure holding all force field parameters.

```python
from q2mm.models.forcefield import ForceField

ff = ForceField(
    bonds=[BondParam(...)],
    angles=[AngleParam(...)],
    torsions=[TorsionParam(...)],
    vdws=[VdwParam(...)],
)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `bonds` | `list[BondParam]` | Bond stretch parameters |
| `angles` | `list[AngleParam]` | Angle bend parameters |
| `torsions` | `list[TorsionParam]` | Torsion (dihedral) parameters |
| `vdws` | `list[VdwParam]` | Van der Waals parameters |

**Key methods:**

```python
vec = ff.get_param_vector()      # flat array of all tunable parameters
ff.set_param_vector(new_vec)     # update parameters from flat array
bounds = ff.get_bounds()         # (lower, upper) bounds for each parameter
n = ff.n_params                  # total number of tunable parameters
ff2 = ff.copy()                  # deep copy
```

**Factory methods:**

```python
ff = ForceField.from_mm3_fld("mol.fld")      # load from MM3 .fld file
ff = ForceField.from_tinker_prm("mol.prm")   # load from Tinker .prm file
```

**Export methods:**

```python
ff.to_mm3_fld("optimized.fld")               # export to MM3 .fld
ff.to_tinker_prm("optimized.prm")            # export to Tinker .prm
ff.to_openmm_xml("forcefield.xml")            # export to OpenMM XML
ff.to_openmm_xml("forcefield.xml", mol)       # with AtomTypes/Residues
```

---

### `Q2MMMolecule`

[source](https://github.com/ericchansen/q2mm/blob/master/q2mm/models/molecule.py)

Molecular structure with coordinates, optional Hessian, and auto-detected
bonds/angles. Bonds are inferred from covalent radii — see the
[tutorial](tutorial.md#step-2-build-a-q2mmmolecule) for details on
`bond_tolerance` and when to increase it for transition states.

**Constructors:**

| Method | Input | Notes |
|--------|-------|-------|
| `Q2MMMolecule(...)` | Raw symbols, geometry arrays | Full control, any data source |
| `Q2MMMolecule.from_xyz(path)` | XYZ file | Simplest option |
| `Q2MMMolecule.from_structure(s)` | Legacy `Structure` (from Gaussian, MacroModel) | Preserves atom types and bond tables |
| `Q2MMMolecule.from_qcel(mol)` | QCElemental `Molecule` | For MolSSI ecosystem workflows |

```python
from q2mm.models.molecule import Q2MMMolecule

# From XYZ file (most common)
mol = Q2MMMolecule.from_xyz("ts.xyz", charge=-1, bond_tolerance=1.4)

# From raw arrays
mol = Q2MMMolecule(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], ...],
    hessian=hessian_matrix,       # optional, ndarray
    atom_types=["1", "5", "5"],
    bond_tolerance=1.3,
)

# From a Gaussian log (via parser)
from q2mm.parsers.gaussian import GaussLog
log = GaussLog("opt-freq.log", au_hessian=True)
mol = Q2MMMolecule.from_structure(log.structures[-1], charge=-1)
```

**Parameters (raw constructor):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `symbols` | `list[str]` | Element symbols (e.g. `["O", "H", "H"]`) |
| `geometry` | `array-like` | Cartesian coordinates (Å), shape `(N, 3)` |
| `hessian` | `ndarray`, optional | Hessian matrix, shape `(3N, 3N)` |
| `atom_types` | `list[str]` | MM atom type labels (default: element symbols) |
| `bond_tolerance` | `float` | Multiplier on covalent radii sum for bond detection (default: 1.3; use 1.4+ for TS) |

!!! tip
    Bonds and angles are auto-detected from atomic coordinates using covalent
    radii — you don't need to specify them manually. Formats with explicit
    bond tables (MOL2, MacroModel `.mmo`) preserve connectivity via
    `from_structure()`.

---

### `estimate_force_constants()`

[source](https://github.com/ericchansen/q2mm/blob/master/q2mm/models/seminario.py)

Extract bond and angle force constants from a QM Hessian using the
[Seminario method](https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6<616::AID-JCC5>3.0.CO;2-X).

```python
from q2mm.models.seminario import estimate_force_constants

ff = estimate_force_constants(molecules)
```

**Input:** `Q2MMMolecule` or `list[Q2MMMolecule]` — each molecule must have a Hessian.

**Output:** `ForceField` with estimated bond/angle force constants and
equilibrium values.

!!! note
    The Seminario method is pure linear algebra (eigenvalue decomposition of
    3×3 Hessian sub-blocks). It runs in < 1 ms for small molecules.

---

## Force Field I/O (`q2mm.models.ff_io`)

Read and write force field files in different formats.

```python
from q2mm.models.ff_io import (
    load_mm3_fld, save_mm3_fld,
    load_tinker_prm, save_tinker_prm,
    save_openmm_xml,
)
```

| Function | Description |
|----------|-------------|
| `load_mm3_fld(path)` | Load a ForceField from an MM3 `.fld` file |
| `save_mm3_fld(ff, path)` | Write a ForceField to an MM3 `.fld` file |
| `load_tinker_prm(path)` | Load a ForceField from a Tinker `.prm` file |
| `save_tinker_prm(ff, path)` | Write a ForceField to a Tinker `.prm` file |
| `save_openmm_xml(ff, path, molecule=None)` | Write a ForceField to an OpenMM ForceField XML file |

### OpenMM XML Export

Two export modes are available for OpenMM:

**ForceField XML** — standalone format loadable by `openmm.app.ForceField()`:

```python
from q2mm.models.ff_io import save_openmm_xml

# Without topology (force definitions only)
save_openmm_xml(ff, "forcefield.xml")

# With topology (includes AtomTypes and Residues)
save_openmm_xml(ff, "forcefield.xml", molecule=mol)
```

**System XML** — serialize an exact OpenMM System (topology-specific):

```python
from q2mm.backends.mm.openmm import OpenMMEngine

engine = OpenMMEngine()
engine.export_system_xml("system.xml", molecule, ff)

# Load back
system = OpenMMEngine.load_system_xml("system.xml")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ff` | `ForceField` | *(required)* | Force field to export |
| `path` | `str` or `Path` | *(required)* | Output file path |
| `molecule` | `Q2MMMolecule`, optional | `None` | Molecule(s) for `<AtomTypes>` and `<Residues>` sections |

!!! note "Custom MM3 force definitions"
    The exported XML uses `CustomBondForce`, `CustomAngleForce`, and
    `CustomNonbondedForce` with MM3 functional forms (cubic bond stretch,
    sextic angle bend, buffered 14-7 vdW). This preserves the exact
    physics of the Q2MM force field — standard harmonic approximations
    are **not** used.

---

## Optimizers (`q2mm.optimizers`)

Classes for defining objectives and running parameter optimization.

### `ReferenceData` / `ReferenceValue`

[source](https://github.com/ericchansen/q2mm/blob/master/q2mm/optimizers/objective.py)

Container for QM or experimental reference observations used as optimization
targets.

**Factory methods** — auto-populate from QM outputs:

```python
from q2mm.optimizers.objective import ReferenceData

# From a Gaussian formatted checkpoint (returns ref data + molecule)
ref, mol = ReferenceData.from_fchk("opt-freq.fchk", bond_tolerance=1.4)

# From a Gaussian log file (with optional frequencies)
ref, mol = ReferenceData.from_gaussian("opt-freq.log", include_frequencies=True)

# From an already-constructed molecule
ref = ReferenceData.from_molecule(mol, frequencies=freqs)

# Multi-molecule training set
ref = ReferenceData.from_molecules([mol1, mol2], frequencies_list=[f1, f2])
```

| Factory | Input | Returns | What it extracts |
|---------|-------|---------|-----------------|
| `from_fchk(path)` | `.fchk` file | `(ReferenceData, Q2MMMolecule)` | Bond lengths, angles, Hessian |
| `from_gaussian(path)` | `.log` file | `(ReferenceData, Q2MMMolecule)` | Bond lengths, angles, frequencies, Hessian |
| `from_molecule(mol)` | `Q2MMMolecule` | `ReferenceData` | Bond lengths, angles; optionally frequencies and eigenmatrix |
| `from_molecules(mols)` | `list[Q2MMMolecule]` | `ReferenceData` | Same as `from_molecule` for each, with sequential `molecule_idx` |

**`from_molecule()` parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mol` | `Q2MMMolecule` | *(required)* | Molecule with geometry |
| `weights` | `dict`, optional | `{"bond_length": 10.0, "bond_angle": 5.0, "frequency": 1.0}` | Weight overrides by data type |
| `molecule_idx` | `int` | `0` | Index for multi-molecule fits |
| `frequencies` | `array-like`, optional | `None` | Vibrational frequencies (cm⁻¹) |
| `skip_imaginary` | `bool` | `False` | Skip negative frequencies |
| `include_eigenmatrix` | `bool` | `False` | Add Hessian eigenmatrix training data |
| `eigenmatrix_diagonal_only` | `bool` | `False` | Only diagonal eigenmatrix elements |

**`from_gaussian()` parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` or `Path` | *(required)* | Path to Gaussian `.log` file |
| `weights` | `dict`, optional | see above | Weight overrides |
| `bond_tolerance` | `float` | `1.3` | Covalent-radii multiplier for bond detection (use 1.4+ for TS) |
| `charge` | `int` | `0` | Molecular charge |
| `multiplicity` | `int` | `1` | Spin multiplicity |
| `include_frequencies` | `bool` | `True` | Add frequency data from the log |
| `skip_imaginary` | `bool` | `False` | Skip imaginary frequencies |
| `au_hessian` | `bool` | `True` | Keep Hessian in atomic units (Hartree/Bohr²) |

**`from_fchk()` parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` or `Path` | *(required)* | Path to the Gaussian `.fchk` file |
| `weights` | `dict`, optional | see above | Weight overrides |
| `bond_tolerance` | `float` | `1.3` | Covalent-radii multiplier for bond detection |
| `charge` | `int` | `0` | Molecular charge (overridden by file value if present) |
| `multiplicity` | `int` | `1` | Spin multiplicity (overridden by file value if present) |

**Bulk loaders** — add data in batch to an existing `ReferenceData`:

```python
ref = ReferenceData()

# Add all frequencies from an array
n = ref.add_frequencies_from_array(freqs, weight=1.0, skip_imaginary=True)

# Add eigenmatrix elements from a QM Hessian
n = ref.add_eigenmatrix_from_hessian(mol.hessian, diagonal_only=False)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `add_frequencies_from_array(freqs)` | `int` (count added) | Bulk-add all vibrational frequencies from a 1-D array |
| `add_eigenmatrix_from_hessian(hessian)` | `int` (count added) | Decompose Hessian, extract eigenmatrix, add diagonal + off-diagonal elements with legacy weight scheme |

**Manual entry** — add individual observations:

```python
ref = ReferenceData()
ref.add_energy(value=0.0, weight=1.0)
ref.add_frequency(value=1648.5, data_idx=0, weight=0.1)
ref.add_bond_length(value=0.9572, atom_indices=(0, 1), weight=10.0)
ref.add_bond_angle(value=104.52, atom_indices=(1, 0, 2), weight=5.0)
ref.add_torsion_angle(value=180.0, atom_indices=(0, 1, 2, 3), weight=2.0)
```

Each `ReferenceValue` has:

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `str` | Type of observation (energy, frequency, bond_length, bond_angle, torsion_angle, eig_diagonal, eig_offdiagonal) |
| `value` | `float` | Target value |
| `weight` | `float` | Relative importance in the objective |
| `atom_indices` | `tuple[int, ...]`, optional | Atoms involved (for geometric properties) |
| `data_idx` | `int` | Index into the raw data array (for matching frequencies/eigenvalues when `atom_indices` is not applicable) |
| `label` | `str` | Human-readable label (used in error messages and debugging) |
| `molecule_idx` | `int` | Index into the molecules list (default: 0) |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `n_observations` | `int` | Total number of reference entries |
| `values` | `list[ReferenceValue]` | All reference observations |

---

### `ObjectiveFunction`

[source](https://github.com/ericchansen/q2mm/blob/master/q2mm/optimizers/objective.py)

Callable objective for `scipy.optimize`: maps a parameter vector to a scalar
score.

```python
from q2mm.optimizers.objective import ObjectiveFunction

obj = ObjectiveFunction(
    forcefield=ff,
    engine=engine,
    molecules=[mol],
    reference=ref,
)

score = obj(param_vector)  # f(param_vector) -> float
```

- Connects **ForceField** ↔ **MM engine** ↔ **reference data**
- Computes weighted sum-of-squares of residuals
- Caches MM engine handles for performance

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__(param_vector)` | `ndarray → float` | Evaluate objective (sum of squared residuals). This is what `scipy.optimize.minimize` calls. |
| `residuals(param_vector)` | `ndarray → ndarray` | Compute weighted residual vector. Used by `least_squares` method. |
| `reset()` | `→ None` | Reset evaluation counter, history, and cached engine handles. Called automatically by `ScipyOptimizer.optimize()`. |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `forcefield` | `ForceField` | The force field being optimized |
| `engine` | `MMEngine` | MM backend (OpenMM, Tinker, etc.) |
| `molecules` | `list[Q2MMMolecule]` | Training set structures |
| `reference` | `ReferenceData` | QM/experimental reference observations |
| `n_eval` | `int` | Number of objective evaluations so far |
| `history` | `list[float]` | Score at each evaluation (for convergence tracking) |

---

### `ScipyOptimizer`

[source](https://github.com/ericchansen/q2mm/blob/master/q2mm/optimizers/scipy_opt.py)

Wraps `scipy.optimize` with sensible defaults for force field optimization.

```python
from q2mm.optimizers.scipy_opt import ScipyOptimizer

optimizer = ScipyOptimizer(method="L-BFGS-B", eps=1e-3)
result = optimizer.optimize(objective)

print(result.summary())
print(f"Improvement: {result.improvement:.1%}")
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"L-BFGS-B"` | Optimization algorithm (see table below) |
| `maxiter` | `int` | `500` | Maximum iterations |
| `ftol` | `float` | `1e-8` | Function tolerance for convergence |
| `eps` | `float` | `1e-3` | Finite-difference step size (scipy default ~1e-8 is too small for FF parameters) |
| `use_bounds` | `bool` | `True` | Use parameter bounds from `ForceField.get_bounds()` |
| `verbose` | `bool` | `True` | Log progress during optimization |

### Optimization Methods

| Method | Type | Best for |
|--------|------|----------|
| `L-BFGS-B` | Quasi-Newton | Smooth problems (default) |
| `Nelder-Mead` | Simplex | Derivative-free, robust |
| `trust-constr` | Trust-region | Constrained optimization |
| `Powell` | Direction-set | Derivative-free |
| `least_squares` | Levenberg-Marquardt | Residual-based fitting |

!!! info "Choosing a method"
    **L-BFGS-B** is the best starting point for most problems. Switch to
    **Nelder-Mead** if the objective is noisy or non-smooth. Use
    **least_squares** when you have many residuals and few parameters.

### `OptimizationResult`

Returned by `ScipyOptimizer.optimize()`.

```python
result = optimizer.optimize(objective)
print(result.summary())          # human-readable summary
print(result.improvement)        # fractional improvement (0–1)
print(result.history)            # score at each evaluation
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `success` | `bool` | Whether the optimizer converged |
| `message` | `str` | Convergence status message from scipy |
| `initial_score` | `float` | Objective value before optimization |
| `final_score` | `float` | Objective value after optimization |
| `n_iterations` | `int` | Number of optimizer iterations |
| `n_evaluations` | `int` | Total objective function evaluations |
| `initial_params` | `ndarray` | Starting parameter vector |
| `final_params` | `ndarray` | Optimized parameter vector |
| `history` | `list[float]` | Score at each evaluation |
| `method` | `str` | Optimization method used |

**Properties and methods:**

| Member | Returns | Description |
|--------|---------|-------------|
| `improvement` | `float` | Fractional improvement: `(initial - final) / initial`. 0 = no change, 1 = perfect. |
| `summary()` | `str` | Human-readable summary (method, scores, improvement, eval count) |

---

## Backends (`q2mm.backends`)

Abstract interfaces and concrete implementations for MM and QM engines.

### `MMEngine` (abstract base)

[source](https://github.com/ericchansen/q2mm/blob/master/q2mm/backends/base.py)

```python
from q2mm.backends.base import MMEngine

class MMEngine(ABC):
    def energy(self, structure, forcefield) -> float: ...
    def minimize(self, molecule, forcefield) -> tuple[float, list, ndarray]: ...
    def frequencies(self, structure, forcefield) -> list[float]: ...
    def hessian(self, structure, forcefield) -> ndarray: ...
```

**Implementations:** `OpenMMEngine`, `TinkerEngine`

---

### `QMEngine` (abstract base)

[source](https://github.com/ericchansen/q2mm/blob/master/q2mm/backends/base.py)

```python
from q2mm.backends.base import QMEngine

class QMEngine(ABC):
    def energy(self, molecule) -> float: ...
    def hessian(self, molecule) -> ndarray: ...
    def optimize(self, molecule) -> Q2MMMolecule: ...
    def frequencies(self, molecule) -> list[float]: ...
```

**Implementations:** `Psi4Engine`

!!! warning
    QM engines are used for **reference data generation**, not during the
    optimization loop. A single Hessian calculation can take minutes to hours
    depending on molecule size and level of theory.

---

## Parsers (`q2mm.parsers`)

File format parsers for reading computational chemistry output.

| Parser | Module | Description |
|--------|--------|-------------|
| `GaussLog` | `q2mm.parsers` | Gaussian `.log` output files |
| `Mol2` | `q2mm.parsers` | MOL2 structure files |
| `JaguarIn` / `JaguarOut` | `q2mm.parsers` | Jaguar input/output |
| `MacroModel` / `MacroModelLog` | `q2mm.parsers` | MacroModel files |
| `MM3` | `q2mm.parsers` | MM3 force field files |
| `TinkerFF` | `q2mm.parsers` | Tinker parameter files |

```python
from q2mm.parsers import GaussLog

log = GaussLog("optimization.log")
structures = log.structures  # list[Structure] — coordinates, atoms
eigenvalues = log.evals      # frequency eigenvalues
eigenvectors = log.evecs     # frequency eigenvectors
```

---

## Software Compatibility

Feature support across force field formats and compute backends.

### Force Field Formats

| Capability | MM3 `.fld` | Tinker `.prm` | OpenMM XML | AMBER `.frcmod` |
|------------|:----------:|:-------------:|:----------:|:---------------:|
| **Read** (load parameters) | ✅ | ✅ | — | ⚠️ legacy only |
| **Write** (standalone) | ✅ | ✅ | ✅ | ❌ |
| **Write** (template-based) | ✅ | ✅ | — | ❌ |
| Bond stretch | ✅ | ✅ | ✅ | ⚠️ |
| Angle bend | ✅ | ✅ | ✅ | ⚠️ |
| Torsion (dihedral) | ✅ template | ✅ template | ✅ | ❌ |
| van der Waals | ✅ | ✅ | ✅ | ❌ |
| MM3 functional forms | ✅ native | ✅ native | ✅ custom forces | ❌ |

!!! info "Template-based vs standalone export"
    **Template-based** export updates parameters in an existing file, preserving
    headers, comments, and parameters that weren't optimised. Use this for
    round-trip compatibility with the original software.

    **Standalone** export writes a minimal file from scratch — useful when no
    template exists (e.g., Seminario-estimated parameters).

### MM Backends

| Capability | OpenMM | Tinker |
|------------|:------:|:------:|
| Single-point energy | ✅ | ✅ |
| Energy minimisation | ✅ | ✅ |
| Numerical Hessian | ✅ | ✅ |
| Vibrational frequencies | ✅ | ✅ |
| Runtime parameter update | ✅ `update_forcefield()` | ❌ (re-writes files) |
| System XML export | ✅ `export_system_xml()` | — |
| Install | `pip install openmm` | System binary |

### QM Backends

| Capability | Gaussian | Psi4 |
|------------|:--------:|:----:|
| Parse optimised geometry | ✅ `.log` / `.fchk` | ✅ via QCElemental |
| Parse Hessian | ✅ | ✅ |
| Parse frequencies | ✅ | ✅ |
| Live QM engine | ❌ (file-based) | ✅ `Psi4Engine` |

### Seminario Method

| Feature | Supported |
|---------|:---------:|
| Bond force constants | ✅ |
| Angle force constants | ✅ |
| Transition states (imaginary mode handling) | ✅ Methods C, D, E |
| Multiple molecules (ensemble averaging) | ✅ |
| Eigenmatrix training data | ✅ |

!!! tip "Quick reference"
    For the best out-of-the-box experience, use **OpenMM** as your MM backend
    and **Gaussian `.fchk`** or **Psi4** for QM reference data. This combination
    gives you fast runtime parameter updates, full Hessian access, and
    export to both System XML and ForceField XML.

---

*See [Performance](performance.md) for benchmarks across backends and
optimization methods.*
