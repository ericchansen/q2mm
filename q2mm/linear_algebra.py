"""Contains methods which perform linear algebraic operations."""

import copy

import numpy as np
from q2mm import constants as co
from typing import Tuple

# region Generalized


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


# endregion Generalized

# region Hessian-specific (Hermitian)


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

    # check_evals = np.diag()

    if not inv_curv_hessian.all() >= 0.0:
        import warnings

        n_neg = int(np.sum(inv_curv_hessian < 0))
        warnings.warn(f"Inverted Hessian has {n_neg} negative values.", stacklevel=2)

    return inv_curv_hessian


# endregion Hessian-specific
