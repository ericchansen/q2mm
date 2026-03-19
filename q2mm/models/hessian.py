from __future__ import annotations
import copy
import logging
from typing import List, Tuple
import numpy as np
from q2mm import constants as co
from q2mm.parsers.structures import Atom

logger = logging.getLogger(__name__)


# Functions originally from q2mm.parsers


def mass_weight_hessian(hess, atoms, reverse=False):
    """Mass weights Hessian by multiplying my 1/sqrt(mass1 * mass2). If reverse is True,
     it un-mass weights the Hessian. Note that this does not return a new object but rather
     modifies the one passed as hess.

    Args:
        hess (_type_): Hessian matrix to mass-weight, modifies the variable itself.
        atoms (_type_): Atom objects related to the Hessian (must be in correct order).
        reverse (bool, optional): Whether to reverse mass-weight (* sqrt(mass1 * mass2)). Defaults to False.
    """
    masses = [co.MASSES[x.element] for x in atoms if not x.is_dummy]
    changes = []
    for mass in masses:
        changes.extend([1 / np.sqrt(mass)] * 3)
    x, y = hess.shape
    for i in range(0, x):
        for j in range(0, y):
            if reverse:
                hess[i, j] = hess[i, j] / changes[i] / changes[j]
            else:
                hess[i, j] = hess[i, j] * changes[i] * changes[j]


def mass_weight_force_constant(force_const: float, atoms: List[Atom], reverse: bool = False, rm: bool = False) -> float:
    """Mass weights force constant. If reverse is True, it un-mass weights
    the force constant.

    Args:
        force_const (float): force constant value to mass-weight or un-mass-weight.
        atoms (List[Atom]): Atoms associated with the force constant.
        reverse (bool, optional): Whether to un-mass-weight the force constant instead. Defaults to False.
        rm (bool, optional): Whether to instead convert the force constant to reduced mass representation. Defaults to False.

    Returns:
        float: mass-weighted or un-mass-weighted value of force constant.
    """
    force_constant = force_const
    masses = [co.MASSES[x.element] for x in atoms]
    changes = []
    if rm:
        return force_constant * np.sqrt(masses[0] + masses[1])
    for mass in masses:
        change = 1 / np.sqrt(mass)
        if reverse:
            force_constant = force_constant / change
        else:
            force_constant = force_constant * change
    return force_constant


def mass_weight_eigenvectors(evecs, atoms, reverse=False):
    """
    Mass weights eigenvectors. If reverse is True, it un-mass weights
    the eigenvectors. TODO
    """
    changes = []
    for atom in atoms:
        if not atom.is_dummy:
            changes.extend([np.sqrt(atom.exact_mass)] * 3)
    x, y = evecs.shape
    for i in range(0, x):
        for j in range(0, y):
            if reverse:
                evecs[i, j] /= changes[j]
            else:
                evecs[i, j] *= changes[j]


# Functions from linear_algebra.py


def decompose(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decomposes matrix into its eigenvalues and eigenvectors.

    Args:
        matrix (np.ndarray): Matrix to decompose, matrix must be square.

    Returns:
        (np.ndarray, np.ndarray): (eigenvalues, eigenvectors) where eigenvalues
         is of shape (1,n) and eigenvectors is of shape (n,n) with n rows of
         eigenvectors of length n.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvalues, eigenvectors


def replace_neg_eigenvalue(
    eigenvalues: np.ndarray,
    replace_with=1.0,
    zer_out_neg=False,
    units=co.KJMOLA,
    strict=True,
) -> np.ndarray:
    """Replaces the most negative eigenvalue with a strong positive value to invert the curvature of the Potential Energy Surface.

    Args:
        eigenvalues (np.ndarray): Eigenvalues
        replace_with (float, optional): Value which should replace the most negative eigenvalue. Defaults to 1.0.
        zer_out_neg (bool, optional): If True, will zero out remaining negative eigenvalues. Defaults to False.
        units (_type_, optional): Units in which replaced eigenvalue should be returned. Defaults to co.KJMOLA.
        strict (bool, optional): If True, raise ValueError when more than one
            negative eigenvalue is found (indicates a higher-order saddle point
            or corrupted Hessian).  If False, proceed with a warning. Defaults
            to True.

    Returns:
        np.ndarray: Eigenvalues with most negative eigenvalue replaced and, if requested, remaining negative values zeroed out.

    Raises:
        ValueError: When *strict* is True and more than one negative eigenvalue
            is present.
    """
    import warnings

    neg_indices = np.argwhere([eval < 0 for eval in eigenvalues])

    if len(neg_indices) == 0:
        return eigenvalues

    if len(neg_indices) > 1:
        msg = (
            f"Hessian has {len(neg_indices)} negative eigenvalues "
            f"{[float(eigenvalues[i]) for i in neg_indices.ravel()]}, "
            "indicating a higher-order saddle point or corrupted data."
        )
        if strict:
            raise ValueError(msg + " Pass strict=False to override.")
        warnings.warn(msg, stacklevel=2)
        index_to_replace = np.argmin(eigenvalues)
    else:
        index_to_replace = neg_indices[0][0]
    replaced_eigenvalues = copy.deepcopy(eigenvalues)

    if zer_out_neg:
        for neg_index in neg_indices:
            replaced_eigenvalues[neg_index[0]] = 0.00
    replaced_eigenvalues[index_to_replace] = (
        replace_with * co.HESSIAN_CONVERSION if units == co.KJMOLA else replace_with
    )

    return replaced_eigenvalues


def reform_hessian(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """Forms the Hessian matrix by multiplying the eigenvalues and eigenvectors

    Args:
        eigenvalues (np.ndarray[float]): eigenvalues
        eigenvectors (np.ndarray[float]): eigenvectors

    Returns:
        np.ndarray: Hessian matrix
    """
    reformed_hessian = eigenvectors.dot(np.diag(eigenvalues).dot(eigenvectors.T))
    return reformed_hessian


def invert_ts_curvature(hessian_matrix: np.ndarray) -> np.ndarray:
    """Inverts the curvature of the Hessian matrix

    Args:
        hessian_matrix (np.ndarray): hessian matrix whose curvature to invert

    Returns:
        np.ndarray: inverted hessian matrix
    """
    eigenvalues, eigenvectors = decompose(hessian_matrix)
    inv_curv_hessian = reform_hessian(
        replace_neg_eigenvalue(eigenvalues, zer_out_neg=True, strict=False),
        eigenvectors,
    )

    if not (inv_curv_hessian >= 0.0).all():
        n_neg = int(np.sum(inv_curv_hessian < 0))
        import warnings

        warnings.warn(f"Inverted Hessian has {n_neg} negative values.", stacklevel=2)

    return inv_curv_hessian
