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

---

### `Q2MMMolecule`

[source](https://github.com/ericchansen/q2mm/blob/master/q2mm/models/molecule.py)

Molecular structure with coordinates, optional Hessian, and auto-detected
bonds/angles.

```python
from q2mm.models.molecule import Q2MMMolecule

mol = Q2MMMolecule(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], ...],
    hessian=hessian_matrix,       # optional, ndarray
    atom_types=[1, 5, 5],
    bond_tolerance=1.2,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `symbols` | `list[str]` | Element symbols (e.g. `["O", "H", "H"]`) |
| `geometry` | `array-like` | Cartesian coordinates (Å), shape `(N, 3)` |
| `hessian` | `ndarray`, optional | Hessian matrix, shape `(3N, 3N)` |
| `atom_types` | `list[int]` | MM atom type indices |
| `bond_tolerance` | `float` | Scaling factor for covalent radii bond detection |

!!! tip
    Bonds and angles are auto-detected from atomic coordinates using covalent
    radii — you don't need to specify them manually.

---

### `estimate_force_constants()`

[source](https://github.com/ericchansen/q2mm/blob/master/q2mm/models/seminario.py)

Extract bond and angle force constants from a QM Hessian using the
[Seminario method](https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6<616::AID-JCC5>3.0.CO;2-X).

```python
from q2mm.models.seminario import estimate_force_constants

ff = estimate_force_constants(molecules)
```

**Input:** `list[Q2MMMolecule]` — each molecule must have a Hessian.

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
)
```

| Function | Description |
|----------|-------------|
| `load_mm3_fld(path)` | Load a ForceField from an MM3 `.fld` file |
| `save_mm3_fld(ff, path)` | Write a ForceField to an MM3 `.fld` file |
| `load_tinker_prm(path)` | Load a ForceField from a Tinker `.prm` file |
| `save_tinker_prm(ff, path)` | Write a ForceField to a Tinker `.prm` file |

---

## Optimizers (`q2mm.optimizers`)

Classes for defining objectives and running parameter optimization.

### `ReferenceData` / `ReferenceValue`

[source](https://github.com/ericchansen/q2mm/blob/master/q2mm/optimizers/objective.py)

Container for QM or experimental reference observations used as optimization
targets.

```python
from q2mm.optimizers.objective import ReferenceData, ReferenceValue

ref = ReferenceData()
ref.add_energy(value=0.0, weight=1.0)
ref.add_frequency(value=1648.5, weight=0.1)
ref.add_bond_length(value=0.9572, atom_indices=(0, 1), weight=10.0)
ref.add_bond_angle(value=104.52, atom_indices=(1, 0, 2), weight=5.0)
ref.add_torsion_angle(value=180.0, atom_indices=(0, 1, 2, 3), weight=2.0)
```

Each `ReferenceValue` has:

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `str` | Type of observation (energy, frequency, etc.) |
| `value` | `float` | Target value |
| `weight` | `float` | Relative importance in the objective |
| `atom_indices` | `tuple[int, ...]`, optional | Atoms involved (for geometric properties) |

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
    reference_data=ref,
)

score = obj(param_vector)  # f(param_vector) -> float
```

- Connects **ForceField** ↔ **MM engine** ↔ **reference data**
- Computes weighted sum-of-squares of residuals
- Caches MM engine handles for performance

---

### `ScipyOptimizer`

[source](https://github.com/ericchansen/q2mm/blob/master/q2mm/optimizers/scipy_opt.py)

Wraps `scipy.optimize` with sensible defaults for force field optimization.

```python
from q2mm.optimizers.scipy_opt import ScipyOptimizer

optimizer = ScipyOptimizer(method="L-BFGS-B", eps=1e-3)
result = optimizer.optimize(objective)

print(result.x)       # optimized parameter vector
print(result.score)   # final objective value
```

**Default:** L-BFGS-B with `eps=1e-3` and bounds from `ForceField.get_bounds()`.

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
energies = log.energies
geometries = log.geometries
```

---

*See [Performance](performance.md) for benchmarks across backends and
optimization methods.*
