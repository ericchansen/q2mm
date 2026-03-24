"""Canonical location for Hessian and eigenvalue operations.

Consolidated from the former ``q2mm.linear_algebra`` module.

Implements eigenvalue manipulation methods from Limé & Norrby
(J. Comput. Chem. 2015, 36, 1130, DOI:10.1002/jcc.23797):

- **Method C** (``replace_neg_eigenvalue``): Force reaction coordinate
  eigenvalue to a large positive value. Simple but distorts the eigenspectrum.
- **Method D** (``keep_natural_eigenvalue``): Keep the natural (negative)
  eigenvalue — gives ~13× lower RMS error but may produce unstable FFs.
- **Method E** (``hybrid_eigenvalue_pipeline``): Run D first, detect
  problematic parameters (zero/negative force constants), lock those, and
  reoptimize with C.

Also provides the **eigenmatrix training data** pipeline:

- ``transform_to_eigenmatrix``: project a Hessian into an eigenvector basis
- ``extract_eigenmatrix_data``: extract diagonal/off-diagonal training data
"""

from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np

from q2mm import constants as co

logger = logging.getLogger(__name__)


def _resolve_symbols(atoms_or_symbols: Sequence[str] | object) -> list[str]:
    """Normalise the *atoms* argument to a plain list of element symbols.

    Accepts:
      - ``list[str]`` — element symbols directly (new API)
      - Any object with a ``.symbols`` attribute (``Q2MMMolecule``)
      - Legacy ``list[Atom]`` — reads ``.element`` and filters ``.is_dummy``

    Returns:
        list[str]: Non-dummy element symbols.
    """
    # Q2MMMolecule (has .symbols attribute)
    if hasattr(atoms_or_symbols, "symbols"):
        return list(atoms_or_symbols.symbols)

    items = list(atoms_or_symbols)
    if not items:
        return []

    # Plain strings
    if isinstance(items[0], str):
        return items

    # Legacy Atom objects (duck-typed: .element, .is_dummy)
    warnings.warn(
        "Passing Atom objects to mass-weighting functions is deprecated. "
        "Pass element symbols (list[str]) or a Q2MMMolecule instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return [a.element for a in items if not getattr(a, "is_dummy", False)]


# ---- Mass-weighting functions ----


def mass_weight_hessian(
    hess: np.ndarray,
    atoms: Sequence[str] | object,
    reverse: bool = False,
) -> None:
    """Mass-weight (or un-weight) a Hessian matrix **in place**.

    Multiplies each element ``H[i,j]`` by ``1 / sqrt(m_i * m_j)`` where
    ``m_i`` is the atomic mass of the atom owning Cartesian coordinate ``i``.
    When *reverse* is ``True``, the operation is inverted.

    Args:
        hess: ``(3N, 3N)`` Hessian matrix — modified in place.
        atoms: Element symbols (``list[str]``), a ``Q2MMMolecule``, or
            (deprecated) legacy ``Atom`` objects.
        reverse: If ``True``, un-mass-weight instead.
    """
    symbols = _resolve_symbols(atoms)
    masses = [co.MASSES[s] for s in symbols]
    inv_sqrt = []
    for m in masses:
        inv_sqrt.extend([1.0 / np.sqrt(m)] * 3)
    rows, cols = hess.shape
    for i in range(rows):
        for j in range(cols):
            if reverse:
                hess[i, j] /= inv_sqrt[i] * inv_sqrt[j]
            else:
                hess[i, j] *= inv_sqrt[i] * inv_sqrt[j]


def mass_weight_force_constant(
    force_const: float,
    atoms: Sequence[str] | object,
    reverse: bool = False,
    rm: bool = False,
) -> float:
    """Mass-weight a force constant.

    Args:
        force_const: Force constant value.
        atoms: Element symbols (``list[str]``), a ``Q2MMMolecule``, or
            (deprecated) legacy ``Atom`` objects.
        reverse: If ``True``, un-mass-weight instead.
        rm: If ``True``, convert to reduced-mass representation instead.

    Returns:
        Mass-weighted (or un-weighted) force constant.
    """
    symbols = _resolve_symbols(atoms)
    masses = [co.MASSES[s] for s in symbols]
    result = force_const
    if rm:
        return result * np.sqrt(masses[0] + masses[1])
    for m in masses:
        factor = 1.0 / np.sqrt(m)
        if reverse:
            result /= factor
        else:
            result *= factor
    return result


def mass_weight_eigenvectors(
    evecs: np.ndarray,
    atoms: Sequence[str] | object,
    reverse: bool = False,
) -> None:
    """Mass-weight (or un-weight) eigenvectors **in place**.

    Args:
        evecs: ``(3N, 3N)`` eigenvector matrix — modified in place.
        atoms: Element symbols (``list[str]``), a ``Q2MMMolecule``, or
            (deprecated) legacy ``Atom`` objects.
        reverse: If ``True``, un-mass-weight instead.
    """
    symbols = _resolve_symbols(atoms)
    sqrt_mass = []
    for s in symbols:
        sqrt_mass.extend([np.sqrt(co.MASSES[s])] * 3)
    rows, cols = evecs.shape
    for i in range(rows):
        for j in range(cols):
            if reverse:
                evecs[i, j] /= sqrt_mass[j]
            else:
                evecs[i, j] *= sqrt_mass[j]


# ---- Linear algebra operations ----


def decompose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decomposes matrix into its eigenvalues and eigenvectors.

    Args:
        matrix (np.ndarray): Matrix to decompose, matrix must be square.

    Returns:
        (np.ndarray, np.ndarray): (eigenvalues, eigenvectors) where eigenvalues
                is of shape ``(n,)`` and eigenvectors is of shape ``(n, n)`` with
                eigenvectors stored as **columns** (the ``np.linalg.eigh`` convention).
                That is, ``eigenvectors[:, i]`` is the eigenvector for ``eigenvalues[i]``.
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
    """Replace the most negative eigenvalue to invert TS curvature (Method C).

    Implements "Method C" from Limé & Norrby (J. Comput. Chem. 2015, 36, 1130,
    DOI:10.1002/jcc.23797): the reaction coordinate eigenvalue is forced to a
    large positive value so that the TS is treated as an energy minimum by
    the MM force field.

    The default replacement is 1.0 Hartree/Bohr², converted to
    kJ/mol/Å² via ``co.HESSIAN_CONVERSION`` (≈ 9376).  This operates
    on **Cartesian** Hessian eigenvalues — not mass-weighted ones.

    Note: Limé & Norrby showed that "Method D" (fitting the natural eigenvalue
    without forced replacement) gives ~13× lower RMS error, but can produce
    unstable force fields. Their recommended "Method E" is a hybrid: use D
    first, lock problematic parameters, then reoptimize with C. See the paper
    for details and issue #75 for implementation status.

    Args:
        eigenvalues (np.ndarray): Eigenvalues from Cartesian Hessian decomposition.
        replace_with (float): Replacement value in Hartree/Bohr². Defaults to 1.0.
        zer_out_neg (bool): If True, zero out remaining negative eigenvalues after
            replacing the most negative. Defaults to False.
        units (int): Target units for the replacement. If ``co.KJMOLA`` (default),
            *replace_with* is converted via ``constants.HESSIAN_CONVERSION``.
        strict (bool): If True, raise ValueError when more than one negative
            eigenvalue is found (indicates a higher-order saddle point or
            corrupted Hessian).  If False, proceed with a warning. Defaults
            to True.

    Returns:
        Eigenvalues with most negative eigenvalue replaced and, if requested,
        remaining negative values zeroed out.

    Raises:
        ValueError: When *strict* is True and more than one negative eigenvalue
            is present.
    """
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
    """Forms the Hessian matrix by multiplying the eigenvalues and eigenvectors.

    Args:
        eigenvalues (np.ndarray[float]): eigenvalues
        eigenvectors (np.ndarray[float]): eigenvectors

    Returns:
        np.ndarray: Hessian matrix
    """
    reformed_hessian = eigenvectors.dot(np.diag(eigenvalues).dot(eigenvectors.T))
    return reformed_hessian


# ---- Eigenmatrix operations ----


def transform_to_eigenmatrix(
    hessian: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Project a Hessian into an eigenvector basis.

    Computes ``eigenvectors.T @ hessian @ eigenvectors`` (using the
    ``np.linalg.eigh`` convention where eigenvectors are **columns**).
    When the eigenvectors come from the *same* Hessian the result is
    diagonal (the eigenvalues).  When they come from a *different*
    Hessian (e.g. projecting an MM Hessian onto QM eigenvectors) the
    off-diagonal elements measure how well the second Hessian reproduces
    the first's mode structure.

    This is the core operation behind the eigenmatrix training data
    approach in Q2MM — see the ``-jeigz`` / ``-mjeig`` commands in
    upstream ``calculate.py``.

    Args:
        hessian: ``(3N, 3N)`` Hessian matrix.
        eigenvectors: ``(3N, 3N)`` matrix whose **columns** are
            eigenvectors (the convention returned by ``np.linalg.eigh``).

    Returns:
        ``(3N, 3N)`` eigenmatrix.

    Note:
        The legacy code used ``evec @ hess @ evec.T`` because Jaguar
        stored eigenvectors as **rows**.  With numpy's column convention
        the equivalent is ``evec.T @ hess @ evec``.

        Both the Hessian and eigenvectors should be in the same unit
        system (typically mass-weighted Hartree/Bohr² after calling
        :func:`mass_weight_hessian` and :func:`mass_weight_eigenvectors`).
    """
    return eigenvectors.T @ hessian @ eigenvectors


def extract_eigenmatrix_data(
    eigenmatrix: np.ndarray,
    *,
    diagonal_only: bool = False,
) -> list[tuple[int, int, float]]:
    """Extract elements from an eigenmatrix as ``(row, col, value)`` tuples.

    Returns the lower-triangular elements (including the diagonal) by
    default, matching the legacy ``-mjeig`` command.  Set
    ``diagonal_only=True`` to return only diagonal elements (matching
    ``-jeigz``).

    Args:
        eigenmatrix: Square eigenmatrix from :func:`transform_to_eigenmatrix`.
        diagonal_only: If True, return only diagonal elements.

    Returns:
        List of ``(row, col, value)`` tuples with 0-based indices.
    """
    n = eigenmatrix.shape[0]
    data = []
    if diagonal_only:
        for i in range(n):
            data.append((i, i, float(eigenmatrix[i, i])))
    else:
        for i in range(n):
            for j in range(i + 1):
                data.append((i, j, float(eigenmatrix[i, j])))
    return data


def invert_ts_curvature(
    hessian_matrix: np.ndarray,
    method: Literal["C", "D"] = "C",
) -> np.ndarray:
    """Invert the curvature of a TS Hessian matrix.

    Args:
        hessian_matrix: Hessian matrix whose curvature to invert.
        method: Eigenvalue treatment method.

            - ``"C"`` (default): Replace reaction coordinate eigenvalue with a
              large positive value (Method C from Limé & Norrby 2015).
            - ``"D"``: Keep the natural eigenvalue without replacement
              (Method D). Produces better fits but may yield unstable FFs.

    Returns:
        Modified Hessian matrix.

    Raises:
        ValueError: If *method* is not ``"C"`` or ``"D"``.
    """
    if method not in ("C", "D"):
        raise ValueError(f"Unknown method {method!r}. Supported: 'C', 'D'.")

    eigenvalues, eigenvectors = decompose(hessian_matrix)

    if method == "D":
        modified_evals = keep_natural_eigenvalue(eigenvalues)
    else:
        modified_evals = replace_neg_eigenvalue(eigenvalues, zer_out_neg=True, strict=False)

    return reform_hessian(modified_evals, eigenvectors)


# ---- Method D: Natural eigenvalue fitting ----


def keep_natural_eigenvalue(eigenvalues: np.ndarray) -> np.ndarray:
    """Return eigenvalues unchanged (Method D).

    Method D from Limé & Norrby (J. Comput. Chem. 2015, 36, 1130): keeps
    the natural (negative) reaction coordinate eigenvalue during fitting.
    This avoids the large distortion introduced by Method C and produces
    ~13× lower RMS error, but the resulting force field may have zero or
    negative bending constants that lead to unphysical MM minima.

    Args:
        eigenvalues: Eigenvalues from Hessian decomposition.

    Returns:
        Unmodified eigenvalues (a copy for safety).
    """
    return eigenvalues.copy()


def detect_problematic_params(
    forcefield,
    *,
    fc_threshold: float = 0.0,
) -> dict[str, set[tuple]]:
    """Detect force field parameters with zero or negative force constants.

    After fitting with Method D, some parameters may converge to physically
    unreasonable values. This function identifies them by their canonical key
    so they can be re-set for Method E re-optimization.

    Args:
        forcefield (ForceField): ForceField with estimated parameters.
        fc_threshold (float): Force constants at or below this value are flagged.

    Returns:
        Dict with ``"bonds"`` and ``"angles"`` keys, each containing a set
        of canonical parameter keys (element tuples from ``param.key``).
    """
    problematic: dict[str, set[tuple]] = {"bonds": set(), "angles": set()}

    for bond in forcefield.bonds:
        if bond.force_constant <= fc_threshold:
            logger.info(f"Problematic bond {bond.key}: fc={bond.force_constant:.4f} (<= {fc_threshold})")
            problematic["bonds"].add(bond.key)

    for angle in forcefield.angles:
        if angle.force_constant <= fc_threshold:
            logger.info(f"Problematic angle {angle.key}: fc={angle.force_constant:.4f} (<= {fc_threshold})")
            problematic["angles"].add(angle.key)

    return problematic


def lock_params(
    forcefield,
    lock_keys: dict[str, set[tuple]],
    source_ff,
) -> None:
    """Reset problematic parameters to values from a reference force field.

    Copies force constant and equilibrium values from *source_ff* to
    *forcefield* for parameters whose keys appear in *lock_keys*.

    Note: this only copies values — it does **not** prevent a subsequent
    optimizer from overwriting them.  To truly freeze parameters during
    optimization, exclude them from the parameter vector (see
    ``ForceField.get_param_vector`` / ``set_param_vector``).

    Args:
        forcefield (ForceField): ForceField to modify in-place.
        lock_keys (dict[str, set[tuple]]): Dict from ``detect_problematic_params()``, keyed by
            ``"bonds"`` and ``"angles"`` with sets of canonical keys.
        source_ff (ForceField): Reference ForceField to copy values from (typically
            the Method D result or a standard force field).
    """
    bond_keys = lock_keys.get("bonds", set())
    angle_keys = lock_keys.get("angles", set())

    source_bonds = {b.key: b for b in source_ff.bonds}
    source_angles = {a.key: a for a in source_ff.angles}

    missing_bonds = bond_keys - source_bonds.keys()
    missing_angles = angle_keys - source_angles.keys()
    if missing_bonds:
        logger.warning(f"Source FF missing bond keys requested for locking: {missing_bonds}")
    if missing_angles:
        logger.warning(f"Source FF missing angle keys requested for locking: {missing_angles}")

    for bond in forcefield.bonds:
        if bond.key in bond_keys:
            src = source_bonds.get(bond.key)
            if src is not None:
                bond.force_constant = src.force_constant
                bond.equilibrium = src.equilibrium
                logger.info(f"Reset bond {bond.key} to fc={src.force_constant:.4f}")

    for angle in forcefield.angles:
        if angle.key in angle_keys:
            src = source_angles.get(angle.key)
            if src is not None:
                angle.force_constant = src.force_constant
                angle.equilibrium = src.equilibrium
                logger.info(f"Reset angle {angle.key} to fc={src.force_constant:.4f}")
