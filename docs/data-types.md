# Reference Data Types

This page documents all reference data types supported by Q2MM's
`ReferenceData` container. Each type represents a QM-computed observable that
the objective function compares against the corresponding MM prediction during
force field optimization.

---

## Overview

| Data Type | Kind String | Typical Weight | Best For |
|-----------|-------------|----------------|----------|
| [Bond Length](#bond-length) | `bond_length` | 10.0 | Equilibrium geometry accuracy |
| [Bond Angle](#bond-angle) | `bond_angle` | 5.0 | Equilibrium geometry accuracy |
| [Torsion Angle](#torsion-angle) | `torsion_angle` | 2.0 | Rotational barriers, conformational preferences |
| [Energy](#energy) | `energy` | 1.0 | Relative energetics, conformer ranking |
| [Frequency](#frequency) | `frequency` | 1.0 | Vibrational spectrum, force constant quality |
| [Eigenvalue (diagonal)](#eigenvalue-diagonal) | `eig_diagonal` | 0.1 | Per-mode force constant accuracy |
| [Eigenmatrix (off-diagonal)](#eigenmatrix-off-diagonal) | `eig_offdiagonal` | 0.05 | Cross-coupling between modes |

All data types are stored in `ReferenceData` and can be combined freely.
The objective function computes weighted squared residuals:

$$\text{Score} = \sum_i w_i \cdot (x_i^\text{QM} - x_i^\text{MM})^2$$

---

## Geometry Data

### Bond Length

**Definition:** Interatomic distance between two bonded atoms.

**Units:** Ångströms (Å)

**When to use:** Always include bond lengths as baseline training data. They
anchor equilibrium bond distances and are inexpensive to evaluate (no Hessian
or frequency calculation needed from the MM engine).

**Weight guidance:** Default weight of 10.0 reflects that bond lengths are
typically well-determined by QM and should be reproduced accurately. For
transition states, consider lower weights for partially-formed/broken bonds
where the force field may not be expected to reproduce the QM value exactly.

**API:**

```python
ref.add_bond_length(
    value=1.384,                  # QM bond length in Å
    atom_indices=(0, 1),          # 0-indexed atom pair
    weight=10.0,
    label="C-F bond",
)
```

**Literature:** Bond lengths are the most fundamental geometric observable
and have been used in force field parameterization since the earliest MM
methods. See [Allinger, *J. Am. Chem. Soc.* **1977**, 99, 8127](https://doi.org/10.1021/ja00467a001)
for the original MM2 parameterization approach.

---

### Bond Angle

**Definition:** Angle formed by three atoms A–B–C where B is the central
(vertex) atom.

**Units:** Degrees (°)

**When to use:** Always include alongside bond lengths. Bond angles are
particularly important for transition states where unusual geometries
(e.g., near-linear angles at a forming bond) must be reproduced.

**Weight guidance:** Default weight of 5.0. Angles are less precisely
determined than bond lengths and have a larger natural range of variation,
so a lower weight avoids over-fitting angular parameters at the expense of
other observables.

**API:**

```python
ref.add_bond_angle(
    value=104.52,                   # QM angle in degrees
    atom_indices=(1, 0, 2),         # vertex atom is index 1 (middle)
    weight=5.0,
    label="F-C-F angle",
)
```

---

### Torsion Angle

**Definition:** Dihedral angle formed by four atoms A–B–C–D, measuring
the rotation about the B–C bond.

**Units:** Degrees (°)

**When to use:** Include when torsional barriers or conformational preferences
are important — for example, when training a force field that must reproduce
a QM torsion scan. For transition-state force fields, torsion barriers are
often set to zero (`zero_torsions=True` in the Seminario method) and torsion
data may not be needed.

**Weight guidance:** Default weight of 2.0. Torsion angles are soft degrees
of freedom with shallow potential energy surfaces, so lower weights prevent
them from dominating the fit.

**API:**

```python
ref.add_torsion_angle(
    value=180.0,                      # QM dihedral in degrees
    atom_indices=(0, 1, 2, 3),        # four atoms defining the dihedral
    weight=2.0,
    label="F-C-C-H torsion",
)
```

**Literature:** The importance of torsion training data for conformational
accuracy is discussed in [Halgren, *J. Comput. Chem.* **1996**, 17, 490](https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6%3C490::AID-JCC1%3E3.0.CO;2-P).

---

## Energy Data

### Energy

**Definition:** Electronic energy from a QM calculation. Typically used as
**relative** energies between conformers or along a reaction path.

**Units:** kcal/mol. `MMEngine.energy()` returns energies in kcal/mol and
`ObjectiveFunction` compares QM and MM values directly with no unit conversion,
so you must convert QM energies (often reported in Hartree or kJ/mol) to
kcal/mol before adding them as reference data.

**When to use:** Include when you have multiple conformers or structures
and want the force field to reproduce their relative energetics. Energy
data is particularly valuable for torsion scan fitting, where the energy
profile along a dihedral is compared point-by-point.

**Weight guidance:** Default weight of 1.0. Energy differences are often
on a different scale than geometric observables, so the weight may need
adjustment depending on the magnitude of the energy differences relative
to geometric residuals.

**API:**

```python
ref.add_energy(
    value=0.0,                        # QM energy (usually relative)
    weight=1.0,
    label="TS energy",
)
```

---

## Vibrational Data

### Frequency

**Definition:** Vibrational frequency from normal mode analysis. The MM
engine computes its own frequencies from the MM Hessian, and each
MM frequency is compared to the corresponding QM frequency.

**Units:** cm⁻¹

**When to use:** Frequencies are the primary training observable for
transition-state force fields. They encode the curvature of the potential
energy surface in every direction and directly reflect the quality of
force constants. The Nelder-Mead optimizer is particularly effective for
frequency-based objectives
([Quinn et al., *PLOS ONE* **2022**](https://doi.org/10.1371/journal.pone.0264960)).

**Weight guidance:** Default weight of 1.0. For transition states, imaginary
frequencies (the reaction coordinate mode) are typically excluded via
`skip_imaginary=True`. Low-frequency modes (< 100 cm⁻¹) are often
poorly described by harmonic force fields and may warrant lower weights.

**API:**

```python
# Bulk-add from an array (most common)
ts_freqs = np.loadtxt("qm-frequencies.txt")
n = ref.add_frequencies_from_array(ts_freqs, weight=1.0, skip_imaginary=True)

# Or add individually
ref.add_frequency(value=1648.5, data_idx=0, weight=1.0)
```

**Literature:** The use of vibrational frequencies as training data for
Q2MM is central to the method — see
[Norrby, *J. Mol. Struct.* **2000**](https://doi.org/10.1016/S0166-1280(00)00398-5)
and [Hansen et al., *Acc. Chem. Res.* **2016**](https://doi.org/10.1021/acs.accounts.6b00037).

---

## Hessian-Derived Data

The Hessian matrix (second derivatives of the energy with respect to nuclear
coordinates) contains complete information about the curvature of the
potential energy surface at a given geometry. Q2MM supports two ways to use
Hessian information as training data, both derived from eigendecomposition
of the Hessian.

!!! note "Why not use raw Hessian elements directly?"
    The raw Hessian matrix has 3N×3N elements (most redundant due to
    symmetry), and individual Cartesian second derivatives are
    frame-dependent and difficult to interpret physically. The
    eigendecomposition rotates the Hessian into normal-mode space where
    each element has a clear physical meaning: a diagonal element is the
    force constant for that mode, and off-diagonal elements measure
    coupling between modes.

### Eigenvalue (diagonal)

**Definition:** Diagonal elements of the Hessian after eigendecomposition —
i.e., the eigenvalues. Each eigenvalue is the force constant for one normal
mode in Hartree/Bohr².

**Kind string:** `eig_diagonal`

**When to use:** Eigenvalues provide more information than frequencies because
they preserve the sign (negative eigenvalues indicate saddle-point directions)
and magnitude without the mass-weighting that converts eigenvalues to
frequencies. They are particularly useful for transition states where the
negative eigenvalue (reaction coordinate) carries information about barrier
curvature.

**Weight guidance:** The default weight scheme separates low-frequency and
high-frequency modes:

- `eig_i = 0.0` — the first (imaginary/reaction coordinate) mode gets zero weight
- `eig_d_low = 0.1` — eigenvalues below 0.1173 Hartree/Bohr² (≈ 1100 kJ/(mol·Å²))
- `eig_d_high = 0.1` — eigenvalues above that threshold

**API:**

```python
ref.add_hessian_eigenvalue(
    value=0.0543,            # in Hartree/Bohr²
    mode_idx=1,              # which eigenvalue (0 = most negative)
    weight=0.1,
)
```

**Literature:**
[Limé & Norrby, *J. Comput. Chem.* **2015**, 36, 1130](https://doi.org/10.1002/jcc.23797)
introduced Methods C, D, and E for handling the negative eigenvalue at
transition states during Seminario estimation.

---

### Eigenmatrix (off-diagonal)

**Definition:** Off-diagonal elements of the Hessian projected into the
eigenvector basis: `V^T · H · V`, where `V` is the matrix of eigenvectors.
These elements measure coupling between normal modes.

**Kind string:** `eig_offdiagonal`

**When to use:** Off-diagonal eigenmatrix elements capture **cross-coupling**
between modes that frequencies and eigenvalues alone miss. For example, if
stretching one bond simultaneously affects the force constant of an adjacent
angle, this coupling appears in the off-diagonal elements. Including them
provides a tighter constraint on the force field and is recommended for
transition states where modes are strongly coupled.

**Weight guidance:** Default weight of 0.05 (lower than diagonal elements
because off-diagonal values are typically smaller in magnitude and less
critical to reproduce exactly).

**API:**

```python
# Add all eigenmatrix data from a Hessian (recommended)
n = ref.add_eigenmatrix_from_hessian(
    hessian,                       # (3N, 3N) array in Hartree/Bohr²
    diagonal_only=False,           # include off-diagonal elements
    skip_first=True,               # zero-weight the reaction coordinate
)

# Or add individual off-diagonal elements
ref.add_hessian_offdiagonal(
    value=0.00234,
    row=1, col=3,                     # mode indices
    weight=0.05,
)
```

---

## Choosing Data Types for Your Problem

### Transition-State Force Fields

For TSFF parameterization, a typical combination is:

- **Bond lengths + bond angles** — anchor the TS geometry
- **Frequencies** (skip imaginary) — reproduce the vibrational spectrum
- **Eigenmatrix** (diagonal + off-diagonal) — capture mode coupling

This is the approach used in
[Rosales et al., *Chem. Commun.* **2018**](https://doi.org/10.1039/C8CC03695K)
and subsequent Q2MM publications.

### Ground-State Force Fields

For ground-state parameterization:

- **Bond lengths + bond angles** — equilibrium geometry
- **Frequencies** — force constant quality
- **Energies** — relative conformer energetics (if multiple structures)
- **Torsion angles** — if torsion barriers matter

### Minimal vs Comprehensive Training Sets

| Training Data | Parameters Constrained | Typical Use Case |
|---------------|----------------------|------------------|
| Geometry only (bonds + angles) | Equilibrium values | Quick initial check |
| Geometry + frequencies | Equilibrium + force constants | Standard optimization |
| Geometry + eigenmatrix | Equilibrium + force constants + coupling | Transition states |
| All of the above + energies | Everything | Multi-conformer fits |

!!! tip "Start simple, add complexity"
    Begin with geometry + frequencies. If the resulting force field has
    problems with mode coupling or energy ordering, add eigenmatrix or
    energy data. Adding more data types always increases the number of
    reference observations, which improves the conditioning of the
    optimization but also increases evaluation cost.

---

## Loading Reference Data

### From files (recommended)

```python
from q2mm.optimizers.objective import ReferenceData

# Auto-extract from a molecule (bonds, angles, and optionally frequencies/eigenmatrix)
ref = ReferenceData.from_molecule(
    mol,
    frequencies=ts_freqs,
    skip_imaginary=True,
    include_eigenmatrix=True,
    eigenmatrix_diagonal_only=False,
)

# From a Gaussian formatted checkpoint file
ref, mol = ReferenceData.from_fchk("calculation.fchk")

# From a Gaussian log file
ref, mol = ReferenceData.from_gaussian("calculation.log")

# From multiple molecules
ref = ReferenceData.from_molecules(
    [mol1, mol2, mol3],
    frequencies_list=[freqs1, freqs2, freqs3],
)
```

### Manual construction

For fine-grained control, build `ReferenceData` manually:

```python
ref = ReferenceData()
ref.add_bond_length(value=1.384, atom_indices=(0, 1), weight=10.0)
ref.add_bond_angle(value=104.5, atom_indices=(1, 0, 2), weight=5.0)
ref.add_frequency(value=1648.5, data_idx=0, weight=1.0)
ref.add_eigenmatrix_from_hessian(hessian, diagonal_only=False)
```

See the [API docs](api.md) for the full method signatures.
