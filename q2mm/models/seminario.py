"""Seminario/QFUERZA force constant estimation using Q2MM's clean data models.

Estimates bond and angle force constants directly from a QM Hessian matrix
using the Seminario (FUERZA) projection method. This implementation uses
Q2MM's internal models (Q2MMMolecule, ForceField) instead of the legacy
MM3-specific data structures.

Reference:
    Farrugia et al., J. Chem. Theory Comput. 2026, 22, 469-476.
    Seminario, Int. J. Quantum Chem. 1996, 60, 1271-1277.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
import logging
import numpy as np
from typing import Literal

from q2mm.constants import AU_TO_MDYNA as HARTREE_BOHR2_TO_MDYNE_A
from q2mm.constants import AU_TO_MDYN_ANGLE
from q2mm.constants import BOHR_TO_ANG
from q2mm.models.molecule import Q2MMMolecule, DetectedBond, DetectedAngle
from q2mm.models.forcefield import ForceField, BondParam, AngleParam

logger = logging.getLogger(__name__)


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


def _match_mode_for_bonds(param: BondParam, bonds: list[DetectedBond]) -> str:
    """Choose the most specific available matching strategy for bond parameters."""
    if param.ff_row is not None and any(bond.ff_row is not None for bond in bonds):
        return "ff_row"
    if param.env_id and any(bond.env_id for bond in bonds):
        return "env_id"
    return "elements"


def _match_mode_for_angles(param: AngleParam, angles: list[DetectedAngle]) -> str:
    """Choose the most specific available matching strategy for angle parameters."""
    if param.ff_row is not None and any(angle.ff_row is not None for angle in angles):
        return "ff_row"
    if param.env_id and any(angle.env_id for angle in angles):
        return "env_id"
    return "elements"


def _collect_matching_bonds(
    molecules: list[Q2MMMolecule],
    param: BondParam,
) -> list[tuple[Q2MMMolecule, DetectedBond]]:
    """Collect all bonds across molecules that match a bond parameter."""
    all_bonds = [(molecule, bond) for molecule in molecules for bond in molecule.bonds]
    match_mode = _match_mode_for_bonds(param, [bond for _, bond in all_bonds])
    if match_mode == "ff_row":
        return [(molecule, bond) for molecule, bond in all_bonds if bond.ff_row == param.ff_row]
    if match_mode == "env_id":
        return [(molecule, bond) for molecule, bond in all_bonds if bond.env_id == param.env_id]
    return [(molecule, bond) for molecule, bond in all_bonds if bond.element_pair == param.key]


def _collect_matching_angles(
    molecules: list[Q2MMMolecule],
    param: AngleParam,
) -> list[tuple[Q2MMMolecule, DetectedAngle]]:
    """Collect all angles across molecules that match an angle parameter."""
    all_angles = [(molecule, angle) for molecule in molecules for angle in molecule.angles]
    match_mode = _match_mode_for_angles(param, [angle for _, angle in all_angles])
    if match_mode == "ff_row":
        return [(molecule, angle) for molecule, angle in all_angles if angle.ff_row == param.ff_row]
    if match_mode == "env_id":
        return [(molecule, angle) for molecule, angle in all_angles if angle.env_id == param.env_id]
    return [(molecule, angle) for molecule, angle in all_angles if angle.element_triple == param.key]


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
    atom_i: int, atom_j: int, coords: np.ndarray, hessian: np.ndarray, au_units: bool = True, dft_scaling: float = 0.963
) -> float:
    """Estimate bond stretching force constant via Seminario method.

    Averages the i->j and j->i projections (bidirectional) to match
    the original Seminario method and upstream Q2MM implementation.

    Args:
        atom_i, atom_j: 0-based atom indices
        coords: Atomic coordinates, shape (N, 3) in Angstrom
        hessian: Full Cartesian Hessian, shape (3N, 3N)
        au_units: If True, Hessian is in Hartree/Bohr^2 (Gaussian/Psi4 default)
        dft_scaling: Scaling factor for DFT Hessians (default 0.963)

    Returns:
        Force constant in mdyn/A (scaled)
    """
    # Bidirectional: compute i->j and j->i, then average
    f_ij = _project_hessian_block(hessian, atom_i, atom_j, coords, au_units)
    f_ji = _project_hessian_block(hessian, atom_j, atom_i, coords, au_units)
    k_bond = 0.5 * (f_ij + f_ji) * dft_scaling

    # Convert to mdyn/A
    if au_units:
        k_bond *= HARTREE_BOHR2_TO_MDYNE_A

    return k_bond


def seminario_angle_fc(
    atom_i: int,
    atom_j: int,
    atom_k: int,
    coords: np.ndarray,
    hessian: np.ndarray,
    au_units: bool = True,
    dft_scaling: float = 0.963,
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
        Force constant in mdyn*A/rad^2 (scaled)
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

    # Convert: Hartree/rad^2 -> mdyn*A/rad^2
    if au_units:
        k_angle *= AU_TO_MDYN_ANGLE

    return k_angle


def estimate_force_constants(
    molecule: Q2MMMolecule | Iterable[Q2MMMolecule],
    forcefield: ForceField | None = None,
    zero_torsions: bool = True,
    au_hessian: bool = True,
    invalid_policy: Literal["keep", "skip"] = "keep",
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

    Returns:
        ForceField with estimated parameters
    """
    molecules = _coerce_molecules(molecule)
    if any(item.hessian is None for item in molecules):
        raise ValueError("Molecule must have a Hessian attached. Use molecule.with_hessian(hess)")

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
        matching_bonds = _collect_matching_bonds(molecules, bond_param)

        if not matching_bonds:
            logger.warning(f"No bonds match {bond_param.key} in molecule")
            continue

        force_constants = []
        equilibria = [bond.length for _, bond in matching_bonds]
        for molecule_item, bond in matching_bonds:
            k = seminario_bond_fc(
                bond.atom_i,
                bond.atom_j,
                molecule_item.geometry,
                molecule_item.hessian,
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
                f"  Bond {bond_param.key}: k={bond_param.force_constant:.4f} mdyn/A, r0={bond_param.equilibrium:.4f} A"
            )
        else:
            logger.warning(f"  Bond {bond_param.key}: no valid force constants found, keeping existing force constant")

    # Estimate angle force constants
    for angle_param in ff.angles:
        matching_angles = _collect_matching_angles(molecules, angle_param)

        if not matching_angles:
            logger.warning(f"No angles match {angle_param.key} in molecule")
            continue

        force_constants = []
        equilibria = [angle.value for _, angle in matching_angles]
        for molecule_item, angle in matching_angles:
            k = seminario_angle_fc(
                angle.atom_i,
                angle.atom_j,
                angle.atom_k,
                molecule_item.geometry,
                molecule_item.hessian,
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
