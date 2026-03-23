"""Seminario/QFUERZA force constant estimation using Q2MM's clean data models.

Estimates bond and angle force constants directly from a QM Hessian matrix
using the Seminario (FUERZA) projection method. This implementation uses
Q2MM's internal models (Q2MMMolecule, ForceField) instead of the legacy
MM3-specific data structures.

Reference:
    Farrugia et al., J. Chem. Theory Comput. 2026, 22, 469-476.
    Seminario, Int. J. Quantum Chem. 1996, 60, 1271-1277.
"""

import copy
from collections.abc import Iterable
import logging
import numpy as np
from typing import Literal

from q2mm.constants import AU_TO_MDYNA as _AU_TO_MDYNA
from q2mm.constants import AU_TO_MDYN_ANGLE as _AU_TO_MDYN_ANGLE
from q2mm.constants import BOHR_TO_ANG
from q2mm.models.units import MDYNA_TO_KCALMOLA2, MDYNA_RAD2_TO_KCALMOLRAD2
from q2mm.models.molecule import Q2MMMolecule, DetectedBond, DetectedAngle
from q2mm.models.forcefield import ForceField, BondParam, AngleParam

# AU → canonical: Hartree/Bohr² → kcal/(mol·Å²) for bonds,
#                  Hartree/rad² → kcal/(mol·rad²) for angles.
HARTREE_BOHR2_TO_KCALMOLA2 = _AU_TO_MDYNA * MDYNA_TO_KCALMOLA2
HARTREE_RAD2_TO_KCALMOLRAD2 = _AU_TO_MDYN_ANGLE * MDYNA_RAD2_TO_KCALMOLRAD2
from q2mm.models.hessian import (
    detect_problematic_params,
    invert_ts_curvature,
    lock_params,
)

logger = logging.getLogger(__name__)

# Default DFT Hessian scaling factor (B3LYP/6-31G* level).
# See: Scott & Radom, J. Phys. Chem. 1996, 100, 16502-16513.
DEFAULT_DFT_SCALING = 0.963


def _coerce_molecules(
    molecule: Q2MMMolecule | Iterable[Q2MMMolecule],
) -> list[Q2MMMolecule]:
    """Normalize a single molecule or iterable of molecules into a list."""
    if isinstance(molecule, Q2MMMolecule):
        return [molecule]

    molecules = list(molecule)
    if not molecules:
        raise ValueError("At least one molecule is required")
    if not all(isinstance(item, Q2MMMolecule) for item in molecules):
        raise TypeError("estimate_force_constants expects Q2MMMolecule instances")
    return molecules


def _match_mode(param, items: list) -> str:
    """Choose the most specific available matching strategy for parameters."""
    if param.ff_row is not None and any(item.ff_row is not None for item in items):
        return "ff_row"
    if param.env_id and any(item.env_id for item in items):
        return "env_id"
    return "elements"


def _collect_matching(
    molecules: list[Q2MMMolecule],
    param,
    items_attr: str,
    element_key_attr: str,
) -> list[tuple[Q2MMMolecule, object]]:
    """Collect all items (bonds or angles) across molecules that match a parameter."""
    all_items = [(mol, item) for mol in molecules for item in getattr(mol, items_attr)]
    match_mode = _match_mode(param, [item for _, item in all_items])
    if match_mode == "ff_row":
        return [(mol, item) for mol, item in all_items if item.ff_row == param.ff_row]
    if match_mode == "env_id":
        return [(mol, item) for mol, item in all_items if item.env_id == param.env_id]
    return [(mol, item) for mol, item in all_items if getattr(item, element_key_attr) == param.key]


def _should_keep_force_constant(value: float, invalid_policy: Literal["keep", "skip"]) -> bool:
    """Decide whether a projected force constant should contribute to an average."""
    if np.iscomplexobj(value):
        return False
    if invalid_policy == "skip":
        return value > 0.0
    return True


def _project_hessian_block(hessian: np.ndarray, atom_i: int, atom_j: int, coords: np.ndarray, au_units: bool) -> float:
    """Project a single Hessian sub-block onto the bond vector.

    Returns the projected force constant in atomic units (Hartree/Bohr^2)
    or input units if au_units=False.
    """
    if au_units:
        coords_work = coords / BOHR_TO_ANG
    else:
        coords_work = coords.copy()

    r_vec = coords_work[atom_j] - coords_work[atom_i]
    r_len = np.linalg.norm(r_vec)
    if r_len < 1e-10:
        return 0.0
    r_hat = r_vec / r_len

    i3, j3 = 3 * atom_i, 3 * atom_j
    h_sub = -hessian[i3 : i3 + 3, j3 : j3 + 3]

    # General eigenvalue decomposition (NOT eigh — sub-block is NOT symmetric)
    eigenvalues, eigenvectors = np.linalg.eig(h_sub)
    # Keep complex eigenpairs through projection — only take real at the end
    # (upstream seminario_sum uses np.abs on complex dot products)

    # Seminario projection: k = sum_n lambda_n * |e_n · r_hat|
    # np.abs handles complex dot products correctly (returns magnitude)
    k = 0.0
    for n in range(3):
        k += eigenvalues[n] * np.abs(np.dot(eigenvectors[:, n], r_hat))
    # Result should be real (imaginary parts cancel in conjugate pairs)
    return k.real


def seminario_bond_fc(
    atom_i: int,
    atom_j: int,
    coords: np.ndarray,
    hessian: np.ndarray,
    au_units: bool = True,
    dft_scaling: float = DEFAULT_DFT_SCALING,
) -> float:
    """Estimate bond stretching force constant via Seminario method.

    Averages the i->j and j->i projections (bidirectional) to match
    the original Seminario method and upstream Q2MM implementation.

    Args:
        atom_i (int): 0-based index of the first atom.
        atom_j (int): 0-based index of the second atom.
        coords (np.ndarray): Atomic coordinates, shape (N, 3) in Angstrom.
        hessian (np.ndarray): Full Cartesian Hessian, shape (3N, 3N).
        au_units (bool): If True, Hessian is in Hartree/Bohr^2 (Gaussian/Psi4 default).
        dft_scaling (float): Scaling factor for DFT Hessians (default 0.963).

    Returns:
        Force constant in kcal/(mol·Å²) (scaled)
    """
    # Bidirectional: compute i->j and j->i, then average
    f_ij = _project_hessian_block(hessian, atom_i, atom_j, coords, au_units)
    f_ji = _project_hessian_block(hessian, atom_j, atom_i, coords, au_units)
    k_bond = 0.5 * (f_ij + f_ji) * dft_scaling

    # Convert to canonical kcal/(mol·Å²)
    if au_units:
        k_bond *= HARTREE_BOHR2_TO_KCALMOLA2

    return k_bond


def seminario_angle_fc(
    atom_i: int,
    atom_j: int,
    atom_k: int,
    coords: np.ndarray,
    hessian: np.ndarray,
    au_units: bool = True,
    dft_scaling: float = DEFAULT_DFT_SCALING,
) -> float:
    """Estimate angle bending force constant via modified Seminario method.

    Uses the Q2MM approximation for angles (FUERZA overestimates by ~2x).
    Uses |dot| projection and DFT scaling to match upstream Q2MM.

    Args:
        atom_i: outer atom (0-based)
        atom_j: center atom (0-based)
        atom_k: outer atom (0-based)
        coords: Atomic coordinates, shape (N, 3) in Angstrom
        hessian: Full Cartesian Hessian, shape (3N, 3N)
        au_units: If True, Hessian is in Hartree/Bohr^2
        dft_scaling: Scaling factor for DFT Hessians (default 0.963)

    Returns:
        Force constant in kcal/(mol·rad²) (scaled)
    """
    if au_units:
        coords_work = coords / BOHR_TO_ANG
    else:
        coords_work = coords.copy()

    # Vectors from center to outer atoms
    r_ij = coords_work[atom_i] - coords_work[atom_j]
    r_kj = coords_work[atom_k] - coords_work[atom_j]
    r_ij_len = np.linalg.norm(r_ij)
    r_kj_len = np.linalg.norm(r_kj)

    if r_ij_len < 1e-10 or r_kj_len < 1e-10:
        return 0.0

    r_ij_hat = r_ij / r_ij_len
    r_kj_hat = r_kj / r_kj_len

    # Normal to the angle plane
    cross = np.cross(r_ij_hat, r_kj_hat)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-10:
        # Linear angle — use a perpendicular direction
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(r_ij_hat, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        cross = np.cross(r_ij_hat, perp)
        cross_norm = np.linalg.norm(cross)

    n_hat = cross / cross_norm

    # Perpendicular unit vectors in the angle plane
    u_ij = np.cross(n_hat, r_ij_hat)
    u_ij /= np.linalg.norm(u_ij)
    u_kj = np.cross(n_hat, r_kj_hat)
    u_kj /= np.linalg.norm(u_kj)

    # Sub-block Hessians — keep complex eigenpairs through projection
    i3, j3, k3 = 3 * atom_i, 3 * atom_j, 3 * atom_k

    # For i-j interaction
    h_ij = -hessian[i3 : i3 + 3, j3 : j3 + 3]
    evals_ij, evecs_ij = np.linalg.eig(h_ij)
    k_ij = 0.0
    for n in range(3):
        k_ij += evals_ij[n] * np.abs(np.dot(evecs_ij[:, n], u_ij))
    k_ij = k_ij.real

    # For k-j interaction
    h_kj = -hessian[k3 : k3 + 3, j3 : j3 + 3]
    evals_kj, evecs_kj = np.linalg.eig(h_kj)
    k_kj = 0.0
    for n in range(3):
        k_kj += evals_kj[n] * np.abs(np.dot(evecs_kj[:, n], u_kj))
    k_kj = k_kj.real

    # Combine: 1/k_angle = 1/(k_ij * r_ij^2) + 1/(k_kj * r_kj^2)
    # Q2MM approximation (avoids FUERZA's 2x overestimate for angles)
    denom_ij = k_ij * r_ij_len**2
    denom_kj = k_kj * r_kj_len**2

    if abs(denom_ij) < 1e-10 or abs(denom_kj) < 1e-10:
        return 0.0

    k_angle = 1.0 / (1.0 / denom_ij + 1.0 / denom_kj)

    # Apply DFT scaling
    k_angle *= dft_scaling

    # Convert: Hartree/rad^2 -> canonical kcal/(mol·rad²)
    if au_units:
        k_angle *= HARTREE_RAD2_TO_KCALMOLRAD2

    return k_angle


def estimate_force_constants(
    molecule: Q2MMMolecule | Iterable[Q2MMMolecule],
    forcefield: ForceField | None = None,
    zero_torsions: bool = True,
    au_hessian: bool = True,
    invalid_policy: Literal["keep", "skip"] = "keep",
    ts_method: Literal["C", "D"] | None = None,
) -> ForceField:
    """Estimate force constants from one or more QM Hessians using Seminario.

    This is the main entry point for QFUERZA force constant estimation.

    Args:
        molecule: Molecule with Hessian attached, or an iterable of molecules.
        forcefield: Starting force field (if None, auto-creates from one molecule)
        zero_torsions: Whether to zero out torsional parameters
        au_hessian: Whether Hessian is in atomic units (Hartree/Bohr^2)
        invalid_policy: Whether to keep negative force constants ("keep") or
            mimic legacy MM3 Seminario averaging by skipping non-positive
            estimates ("skip")
        ts_method: Eigenvalue treatment for transition-state Hessians.
            ``"C"`` replaces the reaction-coordinate eigenvalue with a large
            positive value before Seminario projection (Method C from Limé &
            Norrby 2015).  ``"D"`` keeps the natural eigenvalue (Method D).
            ``None`` (default) uses the Hessian as-is, which is correct for
            ground-state molecules.

    Returns:
        ForceField with estimated parameters
    """
    molecules = _coerce_molecules(molecule)
    if any(item.hessian is None for item in molecules):
        raise ValueError("Molecule must have a Hessian attached. Use molecule.with_hessian(hess)")

    # Pre-process Hessians for TS eigenvalue treatment (Method C or D)
    if ts_method == "C":
        # Method C: replace the reaction-coordinate eigenvalue before projection
        processed_hessians: dict[int, np.ndarray] = {}
        for mol in molecules:
            processed_hessians[id(mol)] = invert_ts_curvature(mol.hessian, method="C")
    elif ts_method == "D":
        # Method D: keep the natural eigenvalue; use the raw Hessian unchanged
        processed_hessians = {id(mol): mol.hessian for mol in molecules}
    elif ts_method is None:
        processed_hessians = None
    else:
        raise ValueError(f"Unsupported ts_method {ts_method!r}; expected 'C', 'D', or None")

    # Create or copy force field
    if forcefield is None:
        if len(molecules) != 1:
            raise ValueError("An explicit force field is required when averaging across multiple molecules")
        ff = ForceField.create_for_molecule(
            molecules[0],
            name=f"Seminario FF for {molecules[0].name}",
        )
    else:
        ff = forcefield.copy()

    # Estimate bond force constants
    for bond_param in ff.bonds:
        matching_bonds = _collect_matching(molecules, bond_param, "bonds", "element_pair")

        if not matching_bonds:
            logger.warning(f"No bonds match {bond_param.key} in molecule")
            continue

        force_constants = []
        equilibria = [bond.length for _, bond in matching_bonds]
        for molecule_item, bond in matching_bonds:
            hess = processed_hessians[id(molecule_item)] if processed_hessians else molecule_item.hessian
            k = seminario_bond_fc(
                bond.atom_i,
                bond.atom_j,
                molecule_item.geometry,
                hess,
                au_units=au_hessian,
            )
            if _should_keep_force_constant(k, invalid_policy):
                force_constants.append(float(np.real(k)))
                if k < 0:
                    logger.warning(
                        f"  Bond {bond.elements} ({bond.atom_i}-{bond.atom_j}): "
                        f"negative FC = {k:.4f} (TS reaction coordinate?)"
                    )
            else:
                logger.warning(f"  Bond {bond.elements} ({bond.atom_i}-{bond.atom_j}): invalid FC = {k} — skipped")

        if equilibria:
            bond_param.equilibrium = float(np.mean(equilibria))
        if force_constants:
            bond_param.force_constant = float(np.mean(force_constants))
            logger.info(
                f"  Bond {bond_param.key}: k={bond_param.force_constant:.4f} kcal/(mol·Å²), r0={bond_param.equilibrium:.4f} Å"
            )
        else:
            logger.warning(f"  Bond {bond_param.key}: no valid force constants found, keeping existing force constant")

    # Estimate angle force constants
    for angle_param in ff.angles:
        matching_angles = _collect_matching(molecules, angle_param, "angles", "element_triple")

        if not matching_angles:
            logger.warning(f"No angles match {angle_param.key} in molecule")
            continue

        force_constants = []
        equilibria = [angle.value for _, angle in matching_angles]
        for molecule_item, angle in matching_angles:
            hess = processed_hessians[id(molecule_item)] if processed_hessians else molecule_item.hessian
            k = seminario_angle_fc(
                angle.atom_i,
                angle.atom_j,
                angle.atom_k,
                molecule_item.geometry,
                hess,
                au_units=au_hessian,
            )
            if _should_keep_force_constant(k, invalid_policy):
                force_constants.append(float(np.real(k)))
                if k < 0:
                    logger.warning(f"  Angle {angle.elements}: negative FC = {k:.4f}")
            else:
                logger.warning(f"  Angle {angle.elements}: invalid FC = {k} — skipped")

        if equilibria:
            angle_param.equilibrium = float(np.mean(equilibria))
        if force_constants:
            angle_param.force_constant = float(np.mean(force_constants))
            logger.info(
                f"  Angle {angle_param.key}: k={angle_param.force_constant:.4f}, "
                f"theta0={angle_param.equilibrium:.1f} deg"
            )
        else:
            logger.warning(
                f"  Angle {angle_param.key}: no valid force constants found, keeping existing force constant"
            )

    # Zero torsions if requested
    if zero_torsions:
        for t in ff.torsions:
            t.force_constant = 0.0

    return ff


# ---------------------------------------------------------------------------
# Method E: Hybrid D/C pipeline
# ---------------------------------------------------------------------------


def estimate_force_constants_method_e(
    molecule: Q2MMMolecule | Iterable[Q2MMMolecule],
    forcefield: ForceField | None = None,
    zero_torsions: bool = True,
    au_hessian: bool = True,
    invalid_policy: Literal["keep", "skip"] = "keep",
    fc_threshold: float = 0.0,
) -> tuple[ForceField, dict]:
    """Estimate force constants using Method E (hybrid D/C).

    Implements "Method E" from Limé & Norrby (J. Comput. Chem. 2015, 36,
    1130): run Method D (natural eigenvalues) first, identify parameters
    that converge to physically unreasonable values, then replace those
    with Method C estimates.

    This gives the accuracy benefits of Method D (~13× lower RMS error)
    where possible, while falling back to Method C for parameters that
    become unstable.

    Args:
        molecule: Molecule(s) with Hessian attached.
        forcefield: Starting force field.  When ``None``, a force field is
            auto-created from the molecule — but this only works for a
            **single** molecule.  Multiple molecules require an explicit
            force field (a ``ValueError`` is raised otherwise).
        zero_torsions: Whether to zero out torsional parameters.
        au_hessian: Whether Hessian is in atomic units.
        invalid_policy: How to handle negative projected force constants.
        fc_threshold: Force constants at or below this value are
            considered problematic and replaced with Method C values.

    Returns:
        Tuple of:
            - ForceField with Method E hybrid parameters.
            - Diagnostics dict with keys:
                ``"method_d"`` — ForceField from Method D,
                ``"method_c"`` — ForceField from Method C,
                ``"problematic"`` — dict from ``detect_problematic_params``.
    """
    common_kwargs = dict(
        forcefield=forcefield,
        zero_torsions=zero_torsions,
        au_hessian=au_hessian,
        invalid_policy=invalid_policy,
    )

    ff_d = estimate_force_constants(molecule, ts_method="D", **common_kwargs)
    ff_c = estimate_force_constants(molecule, ts_method="C", **common_kwargs)

    problematic = detect_problematic_params(ff_d, fc_threshold=fc_threshold)
    n_problematic = sum(len(v) for v in problematic.values())

    # Start from Method D result, override problematic params with C values
    ff_e = ff_d.copy()
    if n_problematic > 0:
        logger.info(f"Method E: {n_problematic} problematic param(s) detected, replacing with Method C values")
        lock_params(ff_e, problematic, ff_c)
    else:
        logger.info("Method E: no problematic params — using pure Method D result")

    return ff_e, {
        "method_d": ff_d,
        "method_c": ff_c,
        "problematic": problematic,
    }
