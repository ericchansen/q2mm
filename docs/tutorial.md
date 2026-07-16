# Tutorial: Full Workflow

A complete end-to-end guide for optimizing a **Transition State Force Field
(TSFF)** using Q2MM's clean model layer. We walk through the SN2 reaction
F⁻ + CH₃F → FCH₃ + F⁻ — a textbook nucleophilic substitution with a
well-defined D₃ₕ-like transition state.

### Why build a transition state force field?

A **transition state (TS)** is the highest-energy point along a reaction
pathway — the molecular geometry at the instant bonds are breaking and
forming. Standard force fields are designed for stable molecules, not
these fleeting arrangements. A **transition state force field (TSFF)**
captures the unusual bonding at the TS so you can run fast molecular
mechanics calculations that reproduce quantum mechanical accuracy for
reaction barriers and selectivity predictions.

---

## Prerequisites

!!! note "What you need before starting"
    - **Python 3.10+** with Q2MM installed (`pip install q2mm`)
    - **[NumPy](https://numpy.org/)** and **[SciPy](https://scipy.org/)** (installed automatically with Q2MM)
    - An **MM backend** — [OpenMM](https://openmm.org/) (`pip install openmm`), [JAX](https://jax.readthedocs.io/) (`pip install "q2mm[jax]"`), [JAX-MD](https://github.com/jax-md/jax-md) (`pip install "q2mm[jax-md]"`), or [Tinker](https://dasher.wustl.edu/tinker/) (free for academic use)
    - The installed SN2 reference files from `q2mm.resources`

    !!! note "Regeneration scripts require a git clone"
        The pre-computed SN2 data is included in the PyPI package. Clone the
        repository only if you want the example and regeneration scripts:
        ```bash
        git clone https://github.com/ericchansen/q2mm.git
        cd q2mm
        ```

    **QM backend optional:** Q2MM includes pre-computed QM reference data as
    installed package resources, so you can complete the full
    workflow without a QM backend. If you want to generate your own QM data,
    you'll need [Psi4](https://psicode.org/) or [Gaussian](https://gaussian.com/).

!!! tip "Quick install"
    ```bash
    pip install "q2mm[openmm]"              # Q2MM + OpenMM backend (from PyPI)
    ```
    Add `--pre` to `pip install` if a stable release hasn't been published yet.

**Atom numbering for this tutorial** (0-indexed):

```
Index   Element   Role
  0       C       Central carbon
  1       F       Leaving / attacking fluorine
  2       F       Leaving / attacking fluorine
  3       H       Methyl hydrogen
  4       H       Methyl hydrogen
  5       H       Methyl hydrogen
```

---

## Step 1: Obtain QM Reference Data

Every TSFF parameterisation starts with quantum-mechanical reference data for
the transition state: an **optimized geometry** and the **Hessian matrix**
(second derivatives of the energy with respect to nuclear coordinates).

???+ example "Using pre-computed data (fastest)"

    The installed `q2mm/data/sn2/` resource contains ready-to-use QM data for
    the SN2 tutorial. **No QM backend needed:**

    ```python
    import numpy as np
    from q2mm.resources import sn2_reference_dir

    QM_REF = sn2_reference_dir()

    hessian     = np.load(str(QM_REF / "sn2-ts-hessian.npy"))       # (18, 18)
    frequencies = np.loadtxt(QM_REF / "sn2-ts-frequencies.txt")     # cm⁻¹
    # Geometry is loaded in Step 2 via load_xyz()
    ```

    Skip to [Step 2](#step-2-build-a-molecule) if using these files.

### Generating your own QM data

If you want to run the QM calculation yourself (or adapt this for your own
molecule), expand the section for your QM backend:

??? example "Psi4 (recommended, open-source)"

    Psi4 runs **inside Python** — you extract the Hessian and frequencies
    directly from the wavefunction object (`wfn`), then save them as NumPy
    arrays.

    ```python
    import numpy as np
    import psi4
    from pathlib import Path

    QM_OUTPUT = Path("my-sn2-reference")
    QM_OUTPUT.mkdir(exist_ok=True)
    psi4.set_memory("2 GB")
    psi4.set_num_threads(4)
    psi4.core.set_output_file(str(QM_OUTPUT / "psi4-output.dat"), False)

    # Define the SN2 transition-state geometry (charge −1, singlet)
    ts_mol = psi4.geometry("""
        -1 1
        C     0.000000    0.000000    0.000000
        F     0.000000    0.000000    1.800000
        F     0.000000    0.000000   -1.800000
        H     1.026720    0.000000    0.000000
        H    -0.513360    0.889165    0.000000
        H    -0.513360   -0.889165    0.000000
    """)

    # Saddle-point optimisation at B3LYP/6-31G*
    psi4.set_options({
        "basis": "6-31G*",
        "reference": "rhf",
        "opt_type": "ts",          # ← saddle-point search
        "geom_maxiter": 100,
    })
    ts_energy = psi4.optimize("b3lyp", molecule=ts_mol)

    # Frequency calculation → Hessian
    ts_energy_freq, ts_wfn = psi4.frequency(
        "b3lyp", molecule=ts_mol, return_wfn=True
    )
    hessian = np.array(ts_wfn.hessian())          # shape (3N, 3N), Hartree/Bohr²
    frequencies = np.array(ts_wfn.frequencies())   # cm⁻¹

    # Verify: exactly 1 imaginary frequency (negative value) = valid TS
    n_imaginary = np.sum(frequencies < 0)
    assert n_imaginary == 1, f"Expected 1 imaginary freq, got {n_imaginary}"

    # Save for later steps
    ts_mol.save_xyz_file(str(QM_OUTPUT / "sn2-ts-optimized.xyz"), True)
    np.save(QM_OUTPUT / "sn2-ts-hessian.npy", hessian)
    np.savetxt(QM_OUTPUT / "sn2-ts-frequencies.txt", frequencies)
    ```

    Install: `conda install psi4 -c conda-forge`

??? example "Gaussian (commercial license)"

    Run a `opt=(ts,calcfc) freq` job, then parse the log file with Q2MM's
    `GaussLog` parser:

    ```python
    import numpy as np
    from q2mm.io.gaussian import GaussLog

    log = GaussLog("sn2-ts.log", au_hessian=True)

    # Geometry + Hessian are already packaged as Molecule objects
    mol = log.molecules[-1]
    hessian = mol.hessian               # (3N, 3N), Hartree/Bohr²
    frequencies = np.array(log.frequencies)
    ```

    Pass `au_hessian=True` to keep the Hessian in atomic units
    (Hartree/Bohr²) — QFUERZA estimation expects this. Use
    `log.molecules[-1].hessian` directly; do **not** reconstruct the
    Cartesian Hessian from Gaussian's mass-weighted eigenvectors.

??? example "Jaguar (Schrödinger)"

    Parse the `.in` file (Hessian) and `.out` file (frequencies, eigenvectors):

    ```python
    from q2mm.io.jaguar import JaguarIn, JaguarOut

    jag_out = JaguarOut("sn2-ts.out")
    eigenvalues = jag_out.eigenvalues
    eigenvectors = jag_out.eigenvectors
    molecules = jag_out.molecules
    frequencies = jag_out.frequencies

    num_atoms = molecules[0].n_atoms
    jag_in = JaguarIn("sn2-ts.in")
    hessian = jag_in.get_hessian(num_atoms)   # (3N, 3N), Hartree/Bohr²
    ```

    Jaguar is commonly used for organometallic transition states where
    pseudopotentials like LACVP** are needed. See the
    [Rh-enamide benchmark](systems/rh-enamide.md) for a worked case study.

!!! warning "Transition-state validation"
    A valid transition state has **exactly one** imaginary (negative)
    vibrational frequency — the reaction coordinate.  If you see zero or
    more than one, the geometry has not converged to a first-order saddle
    point.

---

## Step 2: Build a Molecule

`Molecule` is Q2MM's format-agnostic molecular structure. Loader functions
construct it from XYZ, MOL2, MacroModel, Gaussian, Jaguar, or QCElemental
inputs; when needed, it auto-detects bonds and angles from covalent radii and
stores the QM Hessian alongside the geometry.

!!! info "Bond detection and `bond_tolerance`"
    Not all file formats include bond information — XYZ files, for instance,
    only store atom symbols and Cartesian coordinates. When connectivity is
    missing, Q2MM infers bonds by comparing every atom–atom distance to the
    sum of their covalent radii scaled by `bond_tolerance`:

    **bonded if** `distance < bond_tolerance × (r_cov_A + r_cov_B)`

    The default `bond_tolerance=1.3` works for ground-state molecules. For
    **transition states** — where bonds are partially formed or broken — you
    typically need `1.4` or higher. For example, the C–F distance in the SN2
    TS (~1.84 Å) is much longer than a typical C–F bond (~1.38 Å). If bonds
    are missing from your molecule, increase this value.

    Formats that **do** include explicit bond tables (MOL2, MacroModel `.mmo`)
    skip detection entirely — their dedicated loaders preserve the explicit bond
    tables from the file, with no recalculation.

???+ example "From an XYZ file (simplest)"

    The XYZ and Hessian files here were saved by Psi4 in Step 1
    (`ts_mol.save_xyz_file(...)` and `np.save(..., hessian)`). If you
    skipped that step, the packaged files returned by
    `sn2_reference_dir()` are identical. If you generated your own data, set
    `QM_REF = Path("my-sn2-reference")` instead.

    ```python
    import numpy as np
    from q2mm.io.xyz import load_xyz
    from q2mm.resources import sn2_reference_dir

    QM_REF = sn2_reference_dir()

    # Load the optimised TS geometry saved by Psi4
    mol = load_xyz(
        QM_REF / "sn2-ts-optimized.xyz",
        charge=-1,
        name="SN2_TS",
        bond_tolerance=1.4,   # ← 1.4× covalent radii to catch long TS bonds
    )

    # Attach the QM Hessian (also saved from Psi4's wfn object)
    hessian = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
    mol = mol.with_hessian(hessian)

    # Inspect auto-detected connectivity
    print(f"Atoms:  {mol.n_atoms}")
    print(f"Bonds:  {len(mol.bonds)}")
    print(f"Angles: {len(mol.angles)}")

    for bond in mol.bonds:
        print(f"  {bond.element_pair}: {bond.length:.4f} Å")
    ```

    Expected output:

    ```
    Atoms:  6
    Bonds:  5
    Angles: 7
      ('C', 'F'): 1.8427 Å
      ('C', 'F'): 1.8427 Å
      ('C', 'H'): 1.0767 Å
      ('C', 'H'): 1.0767 Å
      ('C', 'H'): 1.0767 Å
    ```

??? example "From a Gaussian log file"

    If you already have a Gaussian `opt freq` log file, you can build the
    molecule directly from the parsed structures — no separate XYZ file needed:

    ```python
    from q2mm.io.gaussian import GaussLog

    log = GaussLog("sn2-ts.log", au_hessian=True)

    # Build the molecule from the last (optimised) geometry in the log
    mol = log.molecules[-1].with_overrides(
        charge=-1,
        bond_tolerance=1.4,
    )
    ```

    Gaussian and MacroModel loaders preserve the atom typing / connectivity
    information they already know about, which is useful when matching to
    existing force-field parameters.

??? example "From a QCElemental Molecule"

    If you use [QCElemental](https://github.com/MolSSI/QCElemental) in your
    QM workflow:

    ```python
    import qcelemental as qcel
    from q2mm.io.qcelemental import molecule_from_qcel

    qcel_mol = qcel.models.Molecule(...)
    mol = molecule_from_qcel(qcel_mol, name="my-molecule")
    ```

??? example "From raw arrays (manual construction)"

    If your data comes from a custom source rather than an XYZ file:

    ```python
    import numpy as np
    from q2mm.models.molecule import Molecule

    coordinates = np.array([
        [ 0.000000,  0.000000,  0.000000],   # C
        [ 0.000000,  0.000000,  1.800000],   # F
        [ 0.000000,  0.000000, -1.800000],   # F
        [ 1.026720,  0.000000,  0.000000],   # H
        [-0.513360,  0.889165,  0.000000],   # H
        [-0.513360, -0.889165,  0.000000],   # H
    ])

    mol = Molecule(
        symbols=("C", "F", "F", "H", "H", "H"),
        atom_types=("C", "F", "F", "H", "H", "H"),
        geometry=coordinates,
        charge=-1,
        name="sn2-ts",
        bond_tolerance=1.4,
        hessian=hessian,   # (18×18) array in Hartree/Bohr²
    )

    print(f"Bonds: {len(mol.bonds)}, Angles: {len(mol.angles)}")
    ```

---

## Step 3: Initialise the Force Field with QFUERZA

**QFUERZA**
([Farrugia et al., *J. Chem. Theory Comput.* **2025**, 22, 469–476](https://doi.org/10.1021/acs.jctc.5c01751))
extracts harmonic force constants directly from the QM Hessian matrix using
Seminario projection. For each bond or angle, it projects the Hessian onto the
internal coordinate's subspace and takes the eigenvalue along that direction.
For hydrogen angle bends — where plain projection overestimates by ~2× — QFUERZA
substitutes a reliable empirical default. This produces excellent initial
parameter estimates — often within 10–20% of the final optimised values —
without running a single MM calculation.

???+ example "Quick start — auto-create and estimate"

    ```python
    from q2mm.models.forcefield import FunctionalForm
    from q2mm.models.seminario import qfuerza_fresh

    # qfuerza_fresh accepts a single molecule and builds a fresh FF from
    # its QM Hessian.  For multi-molecule averaging (with a template FF
    # whose frozen partition is preserved), use qfuerza_into instead.
    # functional_form is required — no default is chosen for you; pick
    # HARMONIC for JAX/JAX-MD or MM3 for OpenMM/Tinker.
    ff = qfuerza_fresh(
        mol,
        functional_form=FunctionalForm.HARMONIC,
        zero_torsions=True,    # set torsion barriers to zero (common for TS)
        au_hessian=True,       # Hessian is in Hartree/Bohr²
        invalid_policy="skip", # skip negative force constants (TS artefacts)
    )

    print(f"Bond params:    {len(ff.bonds)}")
    print(f"Angle params:   {len(ff.angles)}")
    print(f"Torsion params: {len(ff.torsions)}")

    for b in ff.bonds:
        print(f"  {b.elements}: k = {b.force_constant:.3f} kcal/(mol·Å²), "
              f"r₀ = {b.equilibrium:.4f} Å")
    for a in ff.angles:
        print(f"  {a.elements}: k = {a.force_constant:.6f} kcal/(mol·rad²), "
              f"θ₀ = {a.equilibrium:.1f}°")
    ```

???+ example "With an existing force field template"

    If you already have an MM3 `.fld` file with initial guesses (or placeholder
    values), use ``qfuerza_into`` to return a new force field whose *selected*
    parameter rows are overwritten while preserving atom types and row numbers.
    Active/frozen state now lives outside `ForceField`: benchmark loaders build
    OPT-only subsets with `opt_substructure_membership(...)` and
    `ActiveParameterSpace`, while standalone scripts can simply overwrite every
    compatible bond/angle/torsion row.

    ```python
    from q2mm.io import load_mm3_fld
    from q2mm.models.seminario import qfuerza_into

    # Load template with initial guesses (replace with your .fld path)
    initial_ff = load_mm3_fld("my-system.fld")

    # Return a new ForceField with selected rows overwritten by QFUERZA.
    estimated_ff = qfuerza_into(
        initial_ff,
        mol,
        zero_torsions=True,
        au_hessian=True,
        invalid_policy="skip",
    )

    # Compare before / after
    for old, new in zip(initial_ff.bonds, estimated_ff.bonds):
        delta = new.force_constant - old.force_constant
        print(f"  Bond {old.elements}: {old.force_constant:.3f} → "
              f"{new.force_constant:.3f} kcal/(mol·Å²)  (Δ = {delta:+.3f})")
    ```

    !!! note "What `invalid_policy='skip'` does"
        At a transition state the reaction-coordinate mode has **negative**
        curvature.  The Seminario projection used by QFUERZA can produce negative or complex
        force constants for bonds along this coordinate.  `invalid_policy="skip"`
        leaves those parameters unchanged rather than inserting unphysical
        values.

---

## Step 4: Set Up Reference Data

The `ObservationSet` container holds the QM target values that the objective
function will try to reproduce. Each entry has a **kind** (energy, frequency,
bond length, bond angle, torsion angle, eigenmatrix term, ...), a **value**,
and a **weight** that controls its importance in the fit.

???+ example "Quick start — auto-populate from a molecule"

    The simplest approach auto-extracts bond lengths, angles, and
    Hessian-derived eigenmatrix terms from the molecule we already built, and
    optionally adds frequencies from the QM calculation:

    ```python
    import numpy as np
    from q2mm.models.observations import ObservationSet
    from q2mm.resources import sn2_reference_dir

    # Load frequencies from QM output
    ts_freqs = np.loadtxt(sn2_reference_dir() / "sn2-ts-frequencies.txt")

    # One call populates geometry + eigenmatrix targets, plus the real frequencies
    ref = ObservationSet.from_molecule(
        mol,
        frequencies=ts_freqs,
        skip_imaginary=True,  # skip the imaginary TS mode
    )

    print(f"Reference observations: {ref.n_observations}")
    # → bonds + angles + eigenmatrix terms + real frequencies
    ```

    Default weights are `bond_length=10.0`, `bond_angle=5.0`,
    `frequency=1.0`, with separate defaults for eigenmatrix terms. Override
    them with the `weights` parameter:

    ```python
    ref = ObservationSet.from_molecule(
        mol,
        frequencies=ts_freqs,
        weights={"bond_length": 50.0, "bond_angle": 25.0, "frequency": 2.0},
    )
    ```

??? example "Auto-populate from a Gaussian .fchk file"

    If you have a Gaussian formatted checkpoint file (`.fchk`), you can
    build both the molecule and reference data in one step:

    ```python
    from q2mm.io.fchk import load_fchk_reference

    ref, mol = load_fchk_reference(
        "examples/ethane/GS.fchk",
        bond_tolerance=1.3,
    )
    print(f"Molecule: {mol}")
    print(f"Observations: {ref.n_observations}")
    # The molecule has the Hessian attached automatically
    print(f"Hessian shape: {mol.hessian.shape}")
    ```

??? example "Auto-populate from a Gaussian .log file"

    For Gaussian log files from `opt freq` jobs:

    ```python
    from q2mm.io.gaussian import load_gaussian_reference

    ref, mol = load_gaussian_reference(
        "sn2-ts.log",
        bond_tolerance=1.4,
        charge=-1,
        include_frequencies=True,
        skip_imaginary=True,
    )
    ```

??? example "Multi-molecule training sets"

    For optimising against multiple conformers or molecules:

    ```python
    ref = ObservationSet.from_molecules(
        [mol_gs, mol_ts],
        case_ids=["gs", "ts"],
        frequencies_list=[freqs_gs, freqs_ts],
        skip_imaginary=True,
    )
    # Each molecule is bound to its case_id ("gs", "ts", ...) rather than
    # a positional index — every observation carries that stable ID.
    ```

??? example "Manual construction (full control)"

    You can still build `ObservationSet` entry by entry when you need
    complete control over what goes in. Every `with_*` method returns a
    **new** `ObservationSet`, so reassign `ref` each time:

    ```python
    from q2mm.models.observations import ObservationSet

    ref = ObservationSet()

    for bond in mol.bonds:
        ref = ref.with_bond_length(
            bond.length,
            atom_indices=(bond.atom_i, bond.atom_j),
            weight=10.0,
            case_id="0",
            label=f"{bond.element_pair} bond",
        )

    for angle in mol.angles:
        ref = ref.with_bond_angle(
            angle.value,
            atom_indices=(angle.atom_i, angle.atom_j, angle.atom_k),
            weight=5.0,
            case_id="0",
            label=f"{angle.elements} angle",
        )

    # Bulk-add frequencies
    ref = ref.with_frequencies_from_array(ts_freqs, weight=1.0, case_id="0", skip_imaginary=True)

    # Add an energy target
    ref = ref.with_energy(-239.12345, weight=1.0, case_id="0", label="TS energy")
    ```

!!! tip "Choosing weights"
    Weights balance the influence of different data types:

    - **Bond lengths** are in Ångströms (small numbers); give them higher
      weight (~10) so a 0.01 Å error matters as much as a 1 kcal/mol energy
      error.
    - **Angles** are in degrees; weight ~5 is typical.
    - **Frequencies** can have large absolute values but small relative
      errors; weight ~1 is usually fine.

    There is no single "correct" weighting — iterate and compare results.

---

## Step 5: Compile the Objective Plan

The `ObjectivePlan` is the immutable, backend-neutral description of *what* to
fit: the training molecules with their stable case IDs and stationary-point
kinds, the observation set, the `ParameterLayout`, and the
`ActiveParameterSpace` projection. A concrete executor
(`PythonObjectiveExecutor` or `JaxObjectiveExecutor`) attaches the MM backend
and turns the plan into a callable that `scipy.optimize.minimize` can drive.

At each evaluation the executor:

1. Materializes a candidate `ForceField` from the current parameter vector via `ParameterLayout.replace()`
2. Runs the MM backend (energy, geometry, frequencies) for each case
3. Computes weighted residuals against the observation set
4. Returns the sum of squared residuals

```python
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.python import PythonObjectiveExecutor

layout = ParameterLayout.from_force_field(ff)
space = ActiveParameterSpace.all_active(layout, ff)

plan = ObjectivePlan(
    case_ids=("0",),
    molecules=(mol,),
    stationary_points=(StationaryPointKind.TRANSITION_STATE,),
    observations=ref,
    layout=layout,
    active_space=space,
)

# Attach a backend to get an evaluator scipy.optimize can drive.
objective = PythonObjectiveExecutor(plan, backend, ff)

# Evaluate at the initial (QFUERZA) parameters
initial_score = objective.value(space.baseline)
print(f"Initial score: {initial_score:.6f}")
print(f"Parameters:    {len(layout)} total / {space.n_active} active")
```

!!! note "Setting up an MM backend"
    The `backend` argument is any object implementing the prepared-session
    backend contract from `q2mm.backends.contracts`.  Q2MM ships with backends
    for OpenMM, JAX, JAX-MD, and Tinker:

    ```python
    from q2mm.backends.mm.openmm import OpenMMBackend
    backend = OpenMMBackend()
    ```

    ```python
    from q2mm.backends.mm import JaxBackend
    backend = JaxBackend()
    ```

    ```python
    from q2mm.backends.mm import JaxMdBackend
    backend = JaxMdBackend()
    ```

    ```python
    from q2mm.backends.mm.tinker import TinkerBackend
    backend = TinkerBackend(tinker_dir="/usr/local/bin")
    ```

---

## Step 6: Optimise the Force Field

`ScipyOptimizer` wraps `scipy.optimize.minimize` with sensible defaults for
force-field fitting.

???+ example "Single-shot optimisation with L-BFGS-B"

    | Setting | Value | Rationale |
    |---------|-------|-----------|
    | `method` | `L-BFGS-B` | Bounded quasi-Newton — fast convergence for smooth, differentiable objectives |
    | `eps` | `1e-3` | Finite-difference step for gradient estimation. FF parameters have magnitudes ~0.5–10, so scipy's default (~1e-8) is far too small and produces noisy gradients |
    | `maxiter` | `500` | Generous iteration budget; most runs converge in 50–200 |
    | `use_bounds` | `True` | Prevents parameters from drifting to unphysical values (e.g., negative bond lengths) |

    ```python
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    optimizer = ScipyOptimizer(
        method="L-BFGS-B",
        maxiter=500,
        ftol=1e-8,
        eps=1e-3,
        use_bounds=True,
        verbose=True,
    )

    result = optimizer.optimize(objective, space)
    print(result.summary())
    ```

    Expected output:

    ```
    Method: L-BFGS-B
    Success: True — CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
    Score: 0.045321 → 0.001234 (97.3% improvement)
    Iterations: 87, Evaluations: 1043
    ```

??? example "Alternative optimisers"
    For noisy or discontinuous landscapes, derivative-free methods can be
    more robust:

    ```python
    optimizer = ScipyOptimizer(method="Nelder-Mead", maxiter=2000)

    optimizer = ScipyOptimizer(method="Powell", maxiter=1000)

    optimizer = ScipyOptimizer(method="least_squares", maxiter=500)
    ```

???+ example "Inspecting the result"

    ```python
    # Fractional improvement (0 = no change, 1 = perfect)
    print(f"Improvement: {result.improvement:.1%}")

    # Materialize the optimised immutable ForceField from result.final_params
    optimised_ff = layout.replace(ff, result.final_params)
    for b in optimised_ff.bonds:
        print(f"  {b.elements}: k = {b.force_constant:.4f} kcal/(mol·Å²), "
              f"r₀ = {b.equilibrium:.4f} Å")

    # Convergence history (score at each evaluation)
    # Requires: pip install matplotlib
    import matplotlib.pyplot as plt
    plt.semilogy(result.history)
    plt.xlabel("Evaluation")
    plt.ylabel("Objective")
    plt.title("Convergence")
    plt.savefig("convergence.png")
    ```

---

## Step 6b: Grad-Simp Cycling (Recommended for Large Systems)

For systems with more than ~10 parameters, a single optimizer often leaves
residual error.  The `OptimizationLoop` alternates between a gradient-based
pass on all parameters and a simplex pass on the least gradient-suitable
parameters, combining the strengths of both approaches.

???+ example "Grad-simp cycling"

    ```python
    from q2mm.optimizers.cycling import OptimizationLoop

    loop = OptimizationLoop(
        objective,
        space,
        max_params=3,         # simplex on bottom 3 by simp_var per cycle
        max_cycles=10,        # up to 10 grad-simp cycles
        convergence=0.01,     # stop when <1% improvement per cycle
        full_method="L-BFGS-B",
        simp_method="Nelder-Mead",
        full_maxiter=200,
        simp_maxiter=200,
        verbose=True,
    )

    result = loop.run()
    print(result.summary())
    ```

    Each cycle:

    1. **Full-space gradient pass** — L-BFGS-B on all N parameters
    2. **Sensitivity analysis** — rank parameters by simplex suitability
       (lowest `simp_var`)
    3. **Subspace simplex** — Nelder-Mead on only the 3 least gradient-suitable
       parameters
    4. **Convergence check** — stop when improvement drops below threshold

!!! tip "When to use cycling vs single-shot"
    For ≤ 10 parameters, a single `ScipyOptimizer` call (Step 6) is usually
    sufficient. For larger systems — especially transition-state force fields
    with coupled parameters — the cycling loop typically produces better
    results. See the [Optimization Guide](how-it-works/optimization-guide.md) for a
    detailed comparison.

---

## Step 6c: Optax Optimizers (JAX only)

If you're using the **JAX backend**, you can use [optax](https://optax.readthedocs.io/)
optimizers — Adam, AdaGrad, SGD, and AdamW — as an alternative to SciPy's
L-BFGS-B. These are JAX-native adaptive optimizers that use analytical
gradients automatically.

**Why would you use optax?** On rugged potential energy surfaces like MM3,
Adam's per-parameter adaptive learning rates and momentum can dramatically
outperform L-BFGS-B. On CH₃F with MM3, Adam achieves **56.3 cm⁻¹ RMSD** —
10× better than L-BFGS-B's 579.0 (see
[Small Molecules](systems/small-molecules.md)).

???+ example "Optax Adam optimization"

    ```python
    from q2mm.optimizers.optax import OptaxOptimizer

    optimizer = OptaxOptimizer(
        optimizer="adam",
        learning_rate=0.01,
        max_steps=2000,
    )

    result = optimizer.optimize(objective, space)
    print(result.summary())
    ```

    Expected output:

    ```
    Method: optax:adam
    Success: False — max_steps reached
    Score: 192.000000 → 56.300000 (70.7% improvement)
    Iterations: 2000, Evaluations: 2000
    ```

??? example "Learning rate schedules"
    Optax supports learning rate schedules that decay the LR during
    optimisation. Cosine annealing starts at your LR and decays smoothly
    to zero:

    ```python
    optimizer = OptaxOptimizer(
        optimizer="adam",
        learning_rate=0.01,
        max_steps=2000,
        schedule="cosine",
    )
    ```

!!! warning "JAX backend required"
    `OptaxOptimizer` works best with the JAX backend because it uses
    analytical gradients via `jax.grad`. It will fall back to finite-difference
    gradients on other backends (OpenMM, Tinker), but the FD overhead negates
    the advantage of adaptive optimizers. Use `ScipyOptimizer` for non-JAX
    backends.

!!! tip "When to use optax vs SciPy vs cycling"
    - **Smooth landscape (harmonic form):** Use `ScipyOptimizer("L-BFGS-B")` —
      curvature information matters here and L-BFGS-B excels.
    - **Rugged landscape (MM3 form):** Use `OptaxOptimizer("adam")` — momentum
      helps escape local minima.
    - **Large systems (10+ params):** Use `OptimizationLoop` (Step 6b) — the
      grad-simp cycling loop combines multiple strategies.
    - See the [Optimization Guide](how-it-works/optimization-guide.md) for a
      full comparison with benchmark data.

---

## Step 6d: JaxOpt Optimizers (JAX only — Analytical-Gradient Solvers)

If you're using the **JAX backend**, you can also use
[JAXopt](https://jaxopt.github.io/) for analytical-gradient optimisation.
Unlike optax's adaptive first-order updates, JaxOpt gives you solvers such as
L-BFGS and L-BFGS-B while still sourcing exact gradients from the
`JaxObjectiveExecutor`'s per-case compiled loss functions. q2mm deliberately
keeps the outer JaxOpt solver loop in Python (`jit=False`) so multi-molecule
jobs do **not** get re-inlined into one giant XLA program.

???+ example "JaxOpt L-BFGS-B optimization"

    ```python
    from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

    optimizer = JaxOptOptimizer(
        method="lbfgsb",   # or "lbfgs", "gradient_descent"
        maxiter=500,
    )

    result = optimizer.optimize(objective, space)
    print(result.summary())
    ```

    On CH₃F (harmonic), JaxOpt L-BFGS-B achieves **528 cm⁻¹ RMSD** — matching
    SciPy L-BFGS-B's 529 cm⁻¹ while using exact gradients throughout.

!!! warning "JAX backend required"
    `JaxOptOptimizer` only works with `JaxBackend`. It builds a
    `JaxObjectiveExecutor` (per-case JIT, Python aggregation) and evaluates
    analytical gradients through it; non-JAX backends are not supported.

!!! tip "When to use JaxOpt"
    JaxOpt is most useful when you want the same algorithms as SciPy (L-BFGS-B)
    but with **exact analytical gradients** instead of finite differences. The
    gradient quality is identical to optax, but the optimiser itself is
    second-order. In production, the SciPy + `JaxObjectiveExecutor` route
    remains the default for literature-scale TS benchmarks; JaxOpt is most
    useful on smaller JAX problems. See [Workflow D](how-it-works/optimization-guide.md#workflow-d-end-to-end-differentiable-jax)
    in the Optimization Guide for details.

---

## Step 7: Export the Optimised Force Field

Q2MM can write the optimised parameters to **MM3 `.fld`**, **Tinker `.prm`**,
**AMBER `.frcmod`**, or **OpenMM `.xml`** format via free I/O functions. For
**JAX** and **JAX-MD** backends, save the parameter vector directly as a NumPy
array using the same `ParameterLayout` that defined the optimization:

```python
from q2mm.io import (
    save_amber_frcmod,
    save_mm3_fld,
    save_openmm_xml,
    save_tinker_prm,
)

save_mm3_fld(optimised_ff, "optimized_mm3.fld")
save_tinker_prm(optimised_ff, "optimized.prm")
save_amber_frcmod(optimised_ff, "optimized.frcmod")
save_openmm_xml(optimised_ff, "forcefield.xml", molecule=mol)
np.save("optimized_params.npy", layout.vector(optimised_ff))
```

Expand each format below for details and template-based export options:

??? example "MM3 `.fld` (Schrödinger MacroModel)"

    ```python
    from q2mm.io import save_mm3_fld

    output_path = save_mm3_fld(
        optimised_ff,
        "optimized_mm3.fld",
        template_path="my-system.fld",         # preserves header / metadata
        substructure_name="SN2 TS Optimized",
    )
    print(f"Saved: {output_path}")
    ```

    When you pass `template_path`, Q2MM reads the original `.fld` file,
    updates only the bond and angle parameters that were optimised, and
    writes everything else (headers, VdW parameters, comments) unchanged.
    This is essential for round-trip compatibility with MacroModel.

??? example "Tinker `.prm`"

    ```python
    from q2mm.io import save_tinker_prm

    save_tinker_prm(
        optimised_ff,
        "optimized.prm",
        template_path="template.prm",
    )
    ```

??? example "AMBER `.frcmod`"

    ```python
    from q2mm.io import save_amber_frcmod

    save_amber_frcmod(
        optimised_ff,
        "optimized.frcmod",
        template_path="template.frcmod",   # preserves headers and unmodified sections
    )
    ```

??? example "OpenMM `.xml`"

    ```python
    from q2mm.io import save_openmm_xml

    # Standalone ForceField XML (with AtomTypes and Residues)
    save_openmm_xml(optimised_ff, "forcefield.xml", molecule=mol)
    ```

    The prepared-session backends are evaluation surfaces, not file exporters;
    use the standalone ForceField XML writer for portable OpenMM archival.
    **ForceField XML** is loadable by `openmm.app.ForceField()` and can be
    applied to compatible topologies.

??? example "JAX / JAX-MD (parameter vector)"

    JAX and JAX-MD backends work with the `ForceField` parameter vector
    directly — there's no separate file format. Save and reload with NumPy:

    ```python
    import numpy as np

    np.save("optimized_params.npy", layout.vector(optimised_ff))
    ```

    To reload into a new session:

    ```python
    params = np.load("optimized_params.npy")
    reloaded_ff = layout.replace(ff, params)
    ```

---

## Complete script

Here is the full pipeline in one script. The installed `q2mm/data/sn2/`
resource contains the pre-computed QM data, so you can run the QFUERZA +
analysis steps immediately without a source checkout.

```python
"""Full TSFF pipeline — SN2 F⁻ + CH₃F transition state."""

import numpy as np

from q2mm.io import save_mm3_fld
from q2mm.io.xyz import load_xyz
from q2mm.models.forcefield import FunctionalForm
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.models.seminario import qfuerza_fresh
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.python import PythonObjectiveExecutor
from q2mm.optimizers.scipy_opt import ScipyOptimizer
from q2mm.resources import sn2_reference_dir

QM_REF = sn2_reference_dir()

# ── Step 1: Load QM data ──────────────────────────────────────────
mol = load_xyz(
    QM_REF / "sn2-ts-optimized.xyz",
    charge=-1,
    name="SN2_TS",
    bond_tolerance=1.4,
)
hessian = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
mol = mol.with_hessian(hessian)

print(f"Loaded: {mol.n_atoms} atoms, {len(mol.bonds)} bonds, "
      f"{len(mol.angles)} angles")

# ── Step 2: QFUERZA estimation ──────────────────────────────────
ff = qfuerza_fresh(
    mol,
    functional_form=FunctionalForm.MM3,
    zero_torsions=True,
    au_hessian=True,
    invalid_policy="skip",
)

print("\nQFUERZA estimates:")
for b in ff.bonds:
    print(f"  Bond {b.elements}: k={b.force_constant:.4f} kcal/(mol·Å²), "
          f"r₀={b.equilibrium:.4f} Å  {b.label}")
for a in ff.angles:
    print(f"  Angle {a.elements}: k={a.force_constant:.6f} kcal/(mol·rad²), "
          f"θ₀={a.equilibrium:.1f}°  {a.label}")

# ── Step 3: Reference data ────────────────────────────────────────
ts_freqs = np.loadtxt(str(QM_REF / "sn2-ts-frequencies.txt"))

ref = ObservationSet.from_molecule(
    mol,
    frequencies=ts_freqs,
    skip_imaginary=True,
)
print(f"\nReference observations: {ref.n_observations}")

# ── Step 4: Optimise (requires an MM backend) ─────────────────────
from q2mm.backends.mm.openmm import OpenMMBackend

backend = OpenMMBackend()

layout = ParameterLayout.from_force_field(ff)
space = ActiveParameterSpace.all_active(layout, ff)

plan = ObjectivePlan(
    case_ids=("0",),
    molecules=(mol,),
    stationary_points=(StationaryPointKind.TRANSITION_STATE,),
    observations=ref,
    layout=layout,
    active_space=space,
)
objective = PythonObjectiveExecutor(plan, backend, ff)

optimizer = ScipyOptimizer(
    method="L-BFGS-B", maxiter=500, eps=1e-3
)
result = optimizer.optimize(objective, space)
print(result.summary())

# ── Step 5: Export ────────────────────────────────────────────────
final_ff = layout.replace(ff, result.final_params)
save_mm3_fld(final_ff, "sn2-ts-qfuerza.fld")
```

---

## Next steps

Once you have completed this tutorial, consider:

- **Multiple conformers** — add ground-state CH₃F alongside the TS to train
  a force field that reproduces both minima and the saddle point. Load
  `sn2_reference_dir() / "ch3f-optimized.xyz"` and
  `sn2_reference_dir() / "ch3f-hessian.npy"` as a second molecule.

- **Frequency matching** — add QM vibrational frequencies to the reference
  data (Step 4) for a tighter fit of force constants.

- **Torsion scanning** — for systems with soft torsions, run a QM torsion
  scan and add the energy profile to `ObservationSet` for proper barrier
  heights.

- **Custom weighting** — experiment with different weights to balance
  geometry accuracy against energy/frequency reproduction.

- **Larger systems** — the Rh-enamide example in `examples/rh-enamide/`
  demonstrates TSFF fitting for a transition-metal catalysed reaction with
  significantly more parameters.

- **Alternative optimisers** — try `Nelder-Mead` for noisy landscapes, or
  `least_squares` (Levenberg-Marquardt) when you have more observations
  than parameters.

- **Consult the API reference** — see the [API Reference](reference/q2mm/index.md) for the
  complete interface of `ForceField`, `Molecule`, `ObservationSet`,
  `ObjectivePlan`, and all I/O functions.
