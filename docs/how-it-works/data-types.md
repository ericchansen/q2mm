# Reference Data Types

This page documents all reference data types supported by Q2MM's
`ReferenceData` container. Each type represents a QM-computed observable that
the objective function compares against the corresponding MM prediction during
force field optimization.

---

## Overview

| Data type | Kind string | Typical weight | Best for |
|-----------|-------------|----------------|----------|
| [Bond length](#bond-length) | `bond_length` | 10.0 | Equilibrium geometry accuracy |
| [Bond angle](#bond-angle) | `bond_angle` | 5.0 | Equilibrium geometry accuracy |
| [Torsion angle](#torsion-angle) | `torsion_angle` | 2.0 | Equilibrium dihedral geometry |
| [Energy](#energy) | `energy` | 1.0 | Relative energetics, conformer ranking |
| [Frequency](#frequency) | `frequency` | 1.0 | Vibrational spectrum, force constant quality |
| [Eigenvalue (diagonal)](#eigenvalue-diagonal) | `eig_diagonal` | 0.1 | Per-mode force constant accuracy |
| [Eigenmatrix (off-diagonal)](#eigenmatrix-off-diagonal) | `eig_offdiagonal` | 0.05 | Cross-coupling between modes |

All data types are stored in `ReferenceData` and can be combined freely.
The objective function computes weighted squared residuals:

$$\text{Score} = \sum_i w_i \cdot (x_i^\text{QM} - x_i^\text{MM})^2$$

---

## Geometry data

Bonds, angles, and torsions have been standard training
observables since the earliest molecular mechanics force fields
([Allinger, *J. Am. Chem. Soc.* **1977**, 99, 8127](https://doi.org/10.1021/ja00467a001)).
Torsion profiles became especially important for conformational accuracy
with the development of transferable force fields
([Halgren, *J. Comput. Chem.* **1996**, 17, 490](https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6%3C490::AID-JCC1%3E3.0.CO;2-P)).

### Bond length

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

---

### Bond angle

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

### Torsion angle

**Definition:** Dihedral angle formed by four atoms A–B–C–D, measuring
the rotation about the B–C bond.

**Units:** Degrees (°)

**When to use:** Include when you need the force field to reproduce specific
dihedral values — for example, ensuring that a particular conformation
(anti, gauche, syn) is the equilibrium geometry. Note that torsion *angles*
constrain where the dihedral sits; torsion *barriers* (the energy cost of
rotating) are constrained by energy data from torsion scans. For
transition-state force fields, torsion terms are often zeroed
(`zero_torsions=True` in QFUERZA estimation) and torsion angle data may
not be needed.

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

---

## Energy data

Reproducing relative energies between conformers or along a reaction path
tests whether the force field captures the overall shape of the potential
energy surface — not just the curvature at a single point. Gundertofte et al.
showed that force fields based on the MM2/MM3 functional form achieve average
conformational energy errors around 0.5 kcal/mol, while simpler harmonic
force fields typically exceed 1 kcal/mol
([Gundertofte et al., *J. Comput. Chem.* **1996**, 17, 429](https://doi.org/10.1002/(SICI)1096-987X(199603)17:4%3C429::AID-JCC5%3E3.0.CO;2-W)).

### Energy

**Definition:** Electronic energy from a QM calculation. Typically used as
**relative** energies between conformers or along a reaction path.

**Units:** kcal/mol. `MMEngine.energy()` returns energies in kcal/mol and
`ObjectiveFunction` compares QM and MM values directly with no unit conversion,
so you must convert QM energies (often reported in Hartree or kJ/mol) to
kcal/mol before adding them as reference data.

**When to use:** Include when you have multiple structures and want the
force field to reproduce their relative energetics. Common scenarios:

- **Conformer energies** — a set of local minima that must be ranked
  correctly.
- **Torsion scans** — a series of constrained geometries along a
  dihedral angle. Each scan point is simply another energy data point;
  together they define the torsional energy profile the force field must
  reproduce. Hansen et al. used this approach to implicitly parameterize
  coupled anomeric effects in sulfamides
  ([Hansen et al., *J. Phys. Chem. A* **2016**, 120, 3677](https://doi.org/10.1021/acs.jpca.6b02757)).

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

## Vibrational data

Vibrational frequencies encode the curvature of the potential energy surface
in every direction, making them one of the richest single observables for
force field training. Lii and Allinger demonstrated their value in the
original MM3 parameterization, fitting 213 experimental frequencies across
eight hydrocarbons to an RMS error of 35 cm⁻¹
([Lii & Allinger, *J. Am. Chem. Soc.* **1989**, 111, 8566](https://doi.org/10.1021/ja00205a002)).
In Q2MM, frequency fitting was extended to transition states by comparing
MM-computed frequencies against QM reference values
([Norrby, *J. Mol. Struct.* **2000**](https://doi.org/10.1016/S0166-1280(00)00398-5);
[Hansen et al., *Acc. Chem. Res.* **2016**, 49, 996](https://doi.org/10.1021/acs.accounts.6b00037)).

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

---

## Hessian-derived data

The Hessian matrix (second derivatives of the energy with respect to nuclear
coordinates) contains complete information about the curvature of the
potential energy surface at a given geometry.
QFUERZA estimation uses Seminario's FUERZA procedure to extract internal force constants
directly from the Cartesian Hessian via eigenanalysis of atom-pair
interaction submatrices
([Seminario, *Int. J. Quantum Chem.* **1996**, 60, 1271](https://doi.org/10.1002/(SICI)1097-461X(1996)60:7%3C1271::AID-QUA8%3E3.0.CO;2-W)),
with improved handling of hydrogen angle bends
([Farrugia et al., *J. Chem. Theory Comput.* **2025**, 22, 469](https://doi.org/10.1021/acs.jctc.5c01751)).
Limé and Norrby later extended this to transition states, introducing
methods for handling the negative eigenvalue along the reaction
coordinate
([Limé & Norrby, *J. Comput. Chem.* **2015**, 36, 244](https://doi.org/10.1002/jcc.23797)).
Q2MM supports two ways to use Hessian information as training data, both
derived from eigendecomposition of the Hessian.

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

## Choosing data types for your problem

### Transition-state force fields

For TSFF parameterization, the standard Q2MM combination is:

- **Geometry** (bond lengths, bond angles, and optionally torsion
  angles) — anchor the TS structure. Torsion terms are often zeroed
  in QFUERZA estimation, so torsion angle data may not be needed.
- **Eigenmatrix** (diagonal + off-diagonal) — capture per-mode force
  constants *and* cross-coupling between modes

This is the default in `ReferenceData.from_molecule()` and the approach
used in
[Rosales et al., *Chem. Commun.* **2018**](https://doi.org/10.1039/C8CC03695K)
and subsequent Q2MM publications.

!!! note "Why eigenmatrix instead of frequencies?"
    Frequencies are derived from eigenvalues — they contain the same
    per-mode force constant information, just mass-weighted. Eigenmatrix
    data is strictly richer: it also includes off-diagonal elements
    (mode coupling) that frequencies cannot capture. Using both
    simultaneously would double-count the diagonal information, so the
    standard workflow uses eigenmatrix alone.

!!! note "What about the reaction coordinate?"
    The reaction coordinate (imaginary frequency mode) is included in
    the eigenmatrix as mode 0. By default its weight is zero (`eig_i =
    0.0`), effectively excluding it from the fit. When
    `invert_ts_curvature=True`, the negative eigenvalue is replaced with
    a large positive value, making the TS appear as a minimum in the MM
    energy surface. The force field *does* encode the reaction
    coordinate — it just isn't trained against the QM value. Setting
    `eig_i` to a non-zero weight would train against the flipped
    eigenvalue; see
    [Limé & Norrby, *J. Comput. Chem.* **2015**, 36, 244](https://doi.org/10.1002/jcc.23797)
    for the tradeoffs.

### Ground-state force fields

For ground-state (minimum-energy) parameterization, the data combination
depends on what you need the force field to reproduce:

- **Geometry** (bond lengths, bond angles, torsion angles) — equilibrium
  structure. Always include these as the baseline.
- **Eigenmatrix or frequencies** — force constant quality. Eigenmatrix
  is preferred when a full Hessian is available (e.g., from a Gaussian
  `.fchk` file); frequencies are sufficient when only normal-mode output
  is available. Unlike TSFF, there is no imaginary mode to worry about.
- **Energies** — relative conformer energies or torsion scan profiles.
  See the [Energy section](#energy) above for details and the
  distinction between conformer ranking and torsion scans.

The Q2MM ferrocene force field
([Wahlers et al., *J. Org. Chem.* **2022**, 87, 12334](https://doi.org/10.1021/acs.joc.2c01553))
is an example of a ground-state parameterization that uses geometry +
eigenmatrix data.

!!! note "Ground-state vs transition-state differences"
    Ground-state fitting is simpler in one key way: all eigenvalues are
    positive (no reaction coordinate to handle). On the other hand,
    ground-state fitting often requires multiple conformers, making
    energy data more important than in single-structure TSFF fits.

---

## Loading reference data

### From files (recommended)

```python
from q2mm.optimizers.objective import ReferenceData

# Standard TSFF: geometry + eigenmatrix (the default)
ref = ReferenceData.from_molecule(mol)

# Explicitly include off-diagonal eigenmatrix elements (also the default)
ref = ReferenceData.from_molecule(
    mol,
    include_eigenmatrix=True,
    eigenmatrix_diagonal_only=False,
)

# Ground-state with frequency data instead of eigenmatrix
ref = ReferenceData.from_molecule(
    mol,
    frequencies=gs_freqs,
    include_eigenmatrix=False,
)

# From a Gaussian formatted checkpoint file
ref, mol = ReferenceData.from_fchk("calculation.fchk")

# From a Gaussian log file
ref, mol = ReferenceData.from_gaussian("calculation.log")

# Multi-conformer fit
ref = ReferenceData.from_molecules(
    [mol1, mol2, mol3],
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

See the [API Reference](../reference/q2mm/index.md) for the full method signatures.
