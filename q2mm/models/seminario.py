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
import logging
import numpy as np

from q2mm.constants import AU_TO_MDYNA as HARTREE_BOHR2_TO_MDYNE_A
from q2mm.constants import BOHR_TO_ANG
from q2mm.models.molecule import Q2MMMolecule, DetectedBond, DetectedAngle
from q2mm.models.forcefield import ForceField, BondParam, AngleParam

logger = logging.getLogger(__name__)


def seminario_bond_fc(atom_i: int, atom_j: int,
                      coords: np.ndarray, hessian: np.ndarray,
                      au_units: bool = True) -> float:
    """Estimate bond stretching force constant via Seminario method.

    Projects the Hessian onto the bond vector using eigenvalue decomposition
    of the sub-block Hessian.

    Args:
        atom_i, atom_j: 0-based atom indices
        coords: Atomic coordinates, shape (N, 3) in Angstrom
        hessian: Full Cartesian Hessian, shape (3N, 3N)
        au_units: If True, Hessian is in Hartree/Bohr^2 (Gaussian/Psi4 default)

    Returns:
        Force constant in mdyn/A
    """
    # Convert coordinates to Bohr if Hessian is in atomic units
    if au_units:
        coords_work = coords / BOHR_TO_ANG
    else:
        coords_work = coords.copy()

    # Bond unit vector
    r_vec = coords_work[atom_j] - coords_work[atom_i]
    r_len = np.linalg.norm(r_vec)
    if r_len < 1e-10:
        return 0.0
    r_hat = r_vec / r_len

    # Extract the 3x3 sub-block Hessian for the atom pair
    # H_ij = -d²E/(dr_i dr_j)  (off-diagonal block, negated)
    i3, j3 = 3 * atom_i, 3 * atom_j
    h_sub = -hessian[i3:i3 + 3, j3:j3 + 3]

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(h_sub)

    # Project eigenvalues onto bond vector
    # k_bond = sum_n (lambda_n * (e_n · r_hat)^2)
    k_bond = 0.0
    for n in range(3):
        projection = np.dot(eigenvectors[:, n], r_hat) ** 2
        k_bond += eigenvalues[n] * projection

    # Convert to mdyn/A
    if au_units:
        k_bond *= HARTREE_BOHR2_TO_MDYNE_A

    return k_bond


def seminario_angle_fc(atom_i: int, atom_j: int, atom_k: int,
                       coords: np.ndarray, hessian: np.ndarray,
                       au_units: bool = True) -> float:
    """Estimate angle bending force constant via modified Seminario method.

    Uses the Q2MM approximation for angles (FUERZA overestimates by ~2x).

    Args:
        atom_i: outer atom (0-based)
        atom_j: center atom (0-based)
        atom_k: outer atom (0-based)
        coords: Atomic coordinates, shape (N, 3) in Angstrom
        hessian: Full Cartesian Hessian, shape (3N, 3N)
        au_units: If True, Hessian is in Hartree/Bohr^2

    Returns:
        Force constant in mdyn*A/rad^2
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

    # Sub-block Hessians
    i3, j3, k3 = 3 * atom_i, 3 * atom_j, 3 * atom_k

    # For i-j interaction
    h_ij = -hessian[i3:i3 + 3, j3:j3 + 3]
    evals_ij, evecs_ij = np.linalg.eigh(h_ij)
    k_ij = 0.0
    for n in range(3):
        proj = np.dot(evecs_ij[:, n], u_ij) ** 2
        k_ij += evals_ij[n] * proj

    # For k-j interaction
    h_kj = -hessian[k3:k3 + 3, j3:j3 + 3]
    evals_kj, evecs_kj = np.linalg.eigh(h_kj)
    k_kj = 0.0
    for n in range(3):
        proj = np.dot(evecs_kj[:, n], u_kj) ** 2
        k_kj += evals_kj[n] * proj

    # Combine: 1/k_angle = 1/(k_ij * r_ij^2) + 1/(k_kj * r_kj^2)
    # Q2MM approximation (avoids FUERZA's 2x overestimate for angles)
    denom_ij = k_ij * r_ij_len ** 2
    denom_kj = k_kj * r_kj_len ** 2

    if abs(denom_ij) < 1e-10 or abs(denom_kj) < 1e-10:
        return 0.0

    k_angle = 1.0 / (1.0 / denom_ij + 1.0 / denom_kj)

    # Convert: Hartree/rad^2 -> mdyn*A/rad^2
    if au_units:
        k_angle *= HARTREE_BOHR2_TO_MDYNE_A * BOHR_TO_ANG ** 2

    return k_angle


def estimate_force_constants(
    molecule: Q2MMMolecule,
    forcefield: ForceField | None = None,
    zero_torsions: bool = True,
    au_hessian: bool = True,
) -> ForceField:
    """Estimate force constants from a QM Hessian using the Seminario method.

    This is the main entry point for QFUERZA force constant estimation.

    Args:
        molecule: Molecule with Hessian attached (molecule.hessian must be set)
        forcefield: Starting force field (if None, auto-creates from molecule)
        zero_torsions: Whether to zero out torsional parameters
        au_hessian: Whether Hessian is in atomic units (Hartree/Bohr^2)

    Returns:
        ForceField with estimated parameters
    """
    if molecule.hessian is None:
        raise ValueError("Molecule must have a Hessian attached. Use molecule.with_hessian(hess)")

    # Create or copy force field
    if forcefield is None:
        ff = ForceField.create_for_molecule(molecule, name=f"Seminario FF for {molecule.name}")
    else:
        ff = forcefield.copy()

    coords = molecule.geometry
    hessian = molecule.hessian

    # Estimate bond force constants
    for bond_param in ff.bonds:
        matching_bonds = [b for b in molecule.bonds if b.element_pair == bond_param.key]

        if not matching_bonds:
            logger.warning(f"No bonds match {bond_param.key} in molecule")
            continue

        force_constants = []
        equilibria = []
        for bond in matching_bonds:
            k = seminario_bond_fc(
                bond.atom_i, bond.atom_j, coords, hessian, au_units=au_hessian
            )
            if not np.iscomplex(k):
                force_constants.append(k)
                equilibria.append(bond.length)
                if k < 0:
                    logger.warning(
                        f"  Bond {bond.elements} ({bond.atom_i}-{bond.atom_j}): "
                        f"negative FC = {k:.4f} (TS reaction coordinate?)"
                    )
            else:
                logger.warning(
                    f"  Bond {bond.elements} ({bond.atom_i}-{bond.atom_j}): "
                    f"complex FC = {k} — skipped"
                )

        if force_constants:
            bond_param.force_constant = float(np.mean(force_constants))
            bond_param.equilibrium = float(np.mean(equilibria))
            logger.info(
                f"  Bond {bond_param.key}: k={bond_param.force_constant:.4f} mdyn/A, "
                f"r0={bond_param.equilibrium:.4f} A"
            )

    # Estimate angle force constants
    for angle_param in ff.angles:
        matching_angles = [a for a in molecule.angles if a.element_triple == angle_param.key]

        if not matching_angles:
            logger.warning(f"No angles match {angle_param.key} in molecule")
            continue

        force_constants = []
        equilibria = []
        for angle in matching_angles:
            k = seminario_angle_fc(
                angle.atom_i, angle.atom_j, angle.atom_k,
                coords, hessian, au_units=au_hessian
            )
            if not np.iscomplex(k):
                force_constants.append(k)
                equilibria.append(angle.value)
                if k < 0:
                    logger.warning(
                        f"  Angle {angle.elements}: negative FC = {k:.4f}"
                    )
            else:
                logger.warning(
                    f"  Angle {angle.elements}: complex FC = {k} — skipped"
                )

        if force_constants:
            angle_param.force_constant = float(np.mean(force_constants))
            angle_param.equilibrium = float(np.mean(equilibria))
            logger.info(
                f"  Angle {angle_param.key}: k={angle_param.force_constant:.4f}, "
                f"theta0={angle_param.equilibrium:.1f} deg"
            )

    # Zero torsions if requested
    if zero_torsions:
        for t in ff.torsions:
            t.force_constant = 0.0

    return ff
