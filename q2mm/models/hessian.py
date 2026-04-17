"""Canonical location for Hessian and eigenvalue operations.

Canonical location: :mod:`q2mm.models.hessian` (formerly ``q2mm.linear_algebra``).

Implements eigenvalue manipulation from Limé & Norrby
(J. Comput. Chem. 2015, 36, 244–250, DOI:10.1002/jcc.23797):

- ``replace_neg_eigenvalue``: Force reaction coordinate eigenvalue to a
  large positive value.  Used by ``invert_ts_curvature`` to convert a
  transition-state Hessian into one suitable for Seminario projection.

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
from q2mm.models.units import hessian_au_to_kjmola2

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

    # Plain strings — validate they are known elements
    if isinstance(items[0], str):
        unknown = [s for s in items if s not in co.MASSES]
        if unknown:
            raise ValueError(
                f"Unknown element symbol(s): {unknown}. "
                "Dummy atoms ('X', 'Du') should be excluded before "
                "calling mass-weighting functions."
            )
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
    inv_sqrt = np.array([1.0 / np.sqrt(co.MASSES[s]) for s in symbols for _ in range(3)])
    scale = np.outer(inv_sqrt, inv_sqrt)
    if reverse:
        hess /= scale
    else:
        hess *= scale


# ---- Frequency pipeline ----


#: Penalty frequency (cm⁻¹) returned when eigendecomposition fails and
#: ``on_error="penalty"`` is set.  Large enough to dominate any residual
#: so the optimizer retreats from the pathological parameter region.
PENALTY_FREQUENCY: float = 1e5


def hessian_to_frequencies(
    hessian_au: np.ndarray,
    symbols: list[str] | Sequence[str],
    *,
    sort: bool = True,
    on_error: Literal["raise", "penalty"] = "raise",
) -> list[float]:
    """Convert a Hessian matrix (Hartree/Bohr²) to vibrational frequencies (cm⁻¹).

    Pipeline:
    1. Mass-weight the Hessian using atomic masses
    2. Symmetrise (guards against slight asymmetry from autodiff Hessians)
    3. Diagonalize (eigenvalues only)
    4. Convert eigenvalues to frequencies in cm⁻¹
    5. Handle imaginary frequencies (negative eigenvalues → negative cm⁻¹)

    The conversion factor from mass-weighted atomic-unit eigenvalues
    (Hartree / (amu · Bohr²)) to angular-frequency² (s⁻²) is::

        factor = HARTREE_TO_J / (AMU_TO_KG * bohr_to_m²)

    Angular frequencies (rad/s) are then divided by ``2π · c`` (with *c*
    in cm/s) to obtain wavenumbers in cm⁻¹.

    Args:
        hessian_au: ``(3N, 3N)`` Hessian in Hartree/Bohr².
        symbols: Element symbols for mass lookup (length *N*).
        sort: If ``True`` (default), return frequencies sorted ascending.
        on_error: Error handling strategy for eigendecomposition failures.
            ``"raise"`` (default) re-raises the exception.
            ``"penalty"`` returns ``3N`` copies of :data:`PENALTY_FREQUENCY`,
            signalling to the optimizer that this parameter region is bad.

    Returns:
        List of ``3N`` frequencies in cm⁻¹.  Negative values represent
        imaginary frequencies (from negative eigenvalues, e.g. at
        transition states).

    Raises:
        np.linalg.LinAlgError: If eigendecomposition fails and
            ``on_error="raise"``.

    """
    symbols_list = _resolve_symbols(symbols)
    n_freqs = 3 * len(symbols_list)

    # 1. Mass-weight (copy to avoid mutating the caller's array)
    hess = hessian_au.copy()
    mass_weight_hessian(hess, symbols_list)

    # 2. Symmetrise — autodiff Hessians (e.g. from jax.hessian) can have
    #    slight asymmetry due to floating-point accumulation.  OpenMM's FD
    #    Hessian does this explicitly; we do it unconditionally since the
    #    cost is negligible and it makes eigvalsh more robust.
    hess = 0.5 * (hess + hess.T)

    # 3. Check for non-finite entries before attempting eigendecomposition
    if not np.isfinite(hess).all():
        if on_error == "penalty":
            logger.warning("Hessian contains non-finite entries; returning penalty frequencies.")
            return [PENALTY_FREQUENCY] * n_freqs
        raise np.linalg.LinAlgError("Hessian contains non-finite entries (NaN or Inf)")

    # 4. Diagonalize — eigenvalues only (sorted ascending by numpy)
    try:
        eigenvalues = np.linalg.eigvalsh(hess)
    except np.linalg.LinAlgError:
        if on_error == "penalty":
            logger.warning("Eigenvalue decomposition failed; returning penalty frequencies.")
            return [PENALTY_FREQUENCY] * n_freqs
        raise

    # 5. Convert eigenvalues [Hartree / (amu · Bohr²)] → s⁻²
    bohr_to_m = co.BOHR_TO_ANG * 1e-10
    factor = co.HARTREE_TO_J / (co.AMU_TO_KG * bohr_to_m**2)
    vals_si = eigenvalues * factor

    # 6. eigenvalue → angular frequency → cm⁻¹
    #    Negative eigenvalue → imaginary frequency (negative cm⁻¹)
    freqs = np.sign(vals_si) * np.sqrt(np.abs(vals_si))
    freqs /= 2.0 * np.pi * co.SPEED_OF_LIGHT_MS * 100.0

    result = freqs.tolist()
    if sort:
        result.sort()
    return result


def frequency_param_jacobian(
    hessian_au: np.ndarray,
    dH_dp_au: np.ndarray,
    symbols: list[str] | Sequence[str],
    *,
    sort: bool = True,
    epsilon: float = 1e-20,
) -> tuple[list[float], np.ndarray]:
    """Compute frequencies and their derivatives w.r.t. FF parameters.

    Uses the eigenvalue sensitivity formula to avoid differentiating
    through the eigendecomposition:

    ``dλ_k/dp_j = v_k^T · (d(mw_H)/dp_j) · v_k``

    Then chains through the eigenvalue-to-frequency conversion.

    Args:
        hessian_au: ``(3N, 3N)`` Hessian in Hartree/Bohr².
        dH_dp_au: ``(3N, 3N, n_params)`` Hessian parameter Jacobian
            in Hartree/Bohr².
        symbols: Element symbols for mass lookup (length *N*).
        sort: If ``True`` (default), return frequencies sorted ascending
            and reorder the Jacobian rows to match.
        epsilon: Regularisation floor for near-zero eigenvalues to
            prevent division-by-zero in the ``1/√|λ|`` chain rule.

    Returns:
        ``(frequencies, d_freq_d_params)`` where ``frequencies`` is a
        list of ``3N`` frequencies in cm⁻¹ and ``d_freq_d_params`` has
        shape ``(3N, n_params)`` — row *k* is ``d(freq_k)/d(params)``.

    """
    symbols_list = _resolve_symbols(symbols)
    n3 = 3 * len(symbols_list)

    if dH_dp_au.ndim != 3 or dH_dp_au.shape[:2] != (n3, n3):
        raise ValueError(f"dH_dp_au must have shape (3N, 3N, n_params) = ({n3}, {n3}, ?), got {dH_dp_au.shape}")
    if hessian_au.shape != (n3, n3):
        raise ValueError(f"hessian_au must have shape ({n3}, {n3}), got {hessian_au.shape}")

    n_params = dH_dp_au.shape[2]

    # Mass-weighting scale: 1/sqrt(m_i * m_j)
    inv_sqrt = np.array([1.0 / np.sqrt(co.MASSES[s]) for s in symbols_list for _ in range(3)])
    scale = np.outer(inv_sqrt, inv_sqrt)

    # Mass-weight the Hessian
    hess = hessian_au.copy()
    hess *= scale
    hess = 0.5 * (hess + hess.T)

    # Mass-weight each dH/dp slice
    mw_dH_dp = np.empty((n3, n3, n_params))
    for j in range(n_params):
        mw_dH_dp[:, :, j] = dH_dp_au[:, :, j] * scale

    # Eigendecompose the mass-weighted Hessian
    eigenvalues, eigenvectors = np.linalg.eigh(hess)

    # Eigenvalue sensitivity: dλ_k/dp_j = v_k^T @ (mw_dH/dp_j) @ v_k
    d_eig_dp = np.einsum("ik,ijp,jk->kp", eigenvectors, mw_dH_dp, eigenvectors)

    # Chain through eigenvalue → frequency conversion
    bohr_to_m = co.BOHR_TO_ANG * 1e-10
    factor = co.HARTREE_TO_J / (co.AMU_TO_KG * bohr_to_m**2)
    vals_si = eigenvalues * factor
    denom = 2.0 * np.pi * co.SPEED_OF_LIGHT_MS * 100.0

    # freq = sign(λ) * sqrt(|λ|*factor) / denom
    # d(freq)/d(λ) = factor / (2 * sqrt(|λ|*factor) * denom)
    #              = sqrt(factor) / (2 * sqrt(|λ|) * denom)  [regularised]
    abs_vals_si = np.maximum(np.abs(vals_si), epsilon)
    d_freq_d_eig = factor / (2.0 * np.sqrt(abs_vals_si) * denom)

    # d(freq_k)/dp_j = d(freq_k)/d(λ_k) * d(λ_k)/dp_j
    d_freq_dp = d_freq_d_eig[:, np.newaxis] * d_eig_dp

    # Compute frequencies (same as hessian_to_frequencies)
    freqs_arr = np.sign(vals_si) * np.sqrt(np.abs(vals_si)) / denom

    if sort:
        order = np.argsort(freqs_arr)
        freqs_arr = freqs_arr[order]
        d_freq_dp = d_freq_dp[order, :]

    return freqs_arr.tolist(), d_freq_dp


# ---- JAX-compatible frequency sensitivity ----


def _jax_frequency_param_jacobian(
    hessian_au: np.ndarray,
    dH_dp_au: np.ndarray,
    masses_3n: np.ndarray,
    *,
    epsilon: float = 1e-20,
) -> tuple:
    """JIT-compatible frequency sensitivity via closed-form eigenvalue derivatives.

    Pure JAX equivalent of :func:`frequency_param_jacobian`.  Operates on
    JAX arrays and is fully compatible with ``jax.jit``, ``jax.grad``,
    and ``jax.vmap``.

    Unlike the NumPy version, this function takes pre-resolved masses
    (one per Cartesian DOF, length ``3N``) instead of element symbols,
    because string lookups are not JAX-traceable.

    Args:
        hessian_au: ``(3N, 3N)`` Hessian in Hartree/Bohr².
        dH_dp_au: ``(3N, 3N, n_params)`` Hessian parameter Jacobian
            in Hartree/Bohr².
        masses_3n: ``(3N,)`` atomic masses repeated per Cartesian DOF
            (e.g. ``[m_C, m_C, m_C, m_H, m_H, m_H, ...]``).
        epsilon: Regularisation floor for near-zero eigenvalues.

    Returns:
        ``(frequencies, d_freq_d_params)`` — frequencies is ``(3N,)``
        sorted ascending in cm⁻¹; Jacobian is ``(3N, n_params)``.

    """
    from q2mm.backends.mm._jax_common import ensure_jax

    ensure_jax(engine_name="jax_frequency_sensitivity")

    from q2mm.backends.mm._jax_common import jnp

    # Mass-weighting scale: 1/sqrt(m_i * m_j)
    inv_sqrt = 1.0 / jnp.sqrt(masses_3n)
    scale = jnp.outer(inv_sqrt, inv_sqrt)

    # Mass-weight the Hessian
    hess = hessian_au * scale
    hess = 0.5 * (hess + hess.T)

    # Mass-weight each dH/dp slice (broadcasting over last axis)
    mw_dH_dp = dH_dp_au * scale[:, :, None]

    # Eigendecompose the mass-weighted Hessian
    eigenvalues, eigenvectors = jnp.linalg.eigh(hess)

    # Eigenvalue sensitivity: dλ_k/dp_j = v_k^T @ (mw_dH/dp_j) @ v_k
    d_eig_dp = jnp.einsum("ik,ijp,jk->kp", eigenvectors, mw_dH_dp, eigenvectors)

    # Chain through eigenvalue → frequency conversion
    bohr_to_m = co.BOHR_TO_ANG * 1e-10
    factor = co.HARTREE_TO_J / (co.AMU_TO_KG * bohr_to_m**2)
    denom = 2.0 * jnp.pi * co.SPEED_OF_LIGHT_MS * 100.0

    vals_si = eigenvalues * factor

    # d(freq)/d(λ) with regularisation for near-zero eigenvalues
    abs_vals_si = jnp.maximum(jnp.abs(vals_si), epsilon)
    d_freq_d_eig = factor / (2.0 * jnp.sqrt(abs_vals_si) * denom)

    # d(freq_k)/dp_j = d(freq_k)/d(λ_k) * d(λ_k)/dp_j
    d_freq_dp = d_freq_d_eig[:, None] * d_eig_dp

    # Compute frequencies
    freqs_arr = jnp.sign(vals_si) * jnp.sqrt(jnp.abs(vals_si)) / denom

    # Sort ascending (argsort is fine here — used only for output ordering,
    # not differentiated through)
    order = jnp.argsort(freqs_arr)
    freqs_arr = freqs_arr[order]
    d_freq_dp = d_freq_dp[order, :]

    return freqs_arr, d_freq_dp


def _jax_frequencies_from_hessian(
    hessian_au: np.ndarray,
    masses_3n: np.ndarray,
) -> np.ndarray:
    """JIT-compatible frequencies from a mass-weighted Hessian eigendecomposition.

    Returns only frequencies — no parameter Jacobian.  Use this helper
    inside a ``jax.jit`` / ``jax.grad`` graph when you only need
    ``freqs`` and the outer autodiff will supply parameter gradients
    via reverse-mode AD.  Avoids the expensive ``jax.jacrev`` of the
    Hessian needed by :func:`_jax_frequency_param_jacobian`.

    Args:
        hessian_au: ``(3N, 3N)`` Hessian in Hartree/Bohr².
        masses_3n: ``(3N,)`` masses repeated per Cartesian DOF.

    Returns:
        ``(3N,)`` frequencies in cm⁻¹, sorted ascending.

    """
    from q2mm.backends.mm._jax_common import ensure_jax

    ensure_jax(engine_name="jax_frequencies")

    from q2mm.backends.mm._jax_common import jnp

    inv_sqrt = 1.0 / jnp.sqrt(masses_3n)
    scale = jnp.outer(inv_sqrt, inv_sqrt)

    hess = hessian_au * scale
    hess = 0.5 * (hess + hess.T)

    eigenvalues = jnp.linalg.eigvalsh(hess)

    bohr_to_m = co.BOHR_TO_ANG * 1e-10
    factor = co.HARTREE_TO_J / (co.AMU_TO_KG * bohr_to_m**2)
    denom = 2.0 * jnp.pi * co.SPEED_OF_LIGHT_MS * 100.0

    vals_si = eigenvalues * factor
    freqs_arr = jnp.sign(vals_si) * jnp.sqrt(jnp.abs(vals_si)) / denom
    return jnp.sort(freqs_arr)


def symbols_to_masses_3n(symbols: list[str] | Sequence[str]) -> list[float]:
    """Convert element symbols to a flat ``3N`` mass array.

    Each atom's mass is repeated 3 times (one per Cartesian DOF).
    The returned list can be converted to a JAX array for use with
    :func:`_jax_frequency_param_jacobian`.

    Args:
        symbols: Element symbols (length *N*).

    Returns:
        list[float]: Masses of length ``3N``.

    """
    resolved = _resolve_symbols(symbols)
    return [co.MASSES[s] for s in resolved for _ in range(3)]


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
    replace_with: float = 1.0,
    zer_out_neg: bool = False,
    units: str = co.GAUSSIAN,
    strict: bool = True,
) -> np.ndarray:
    """Replace the most negative eigenvalue to invert TS curvature.

    From Limé & Norrby (J. Comput. Chem. 2015, 36, 244–250,
    DOI:10.1002/jcc.23797): the reaction coordinate eigenvalue is forced to a
    large positive value so that the TS is treated as an energy minimum by
    the MM force field.

    The default replacement is 1.0 Hartree/Bohr² (atomic units).  When
    *units* is ``co.KJMOLA``, the replacement is converted via
    :func:`~q2mm.models.units.hessian_au_to_kjmola2` (≈ 9376) to
    kJ/mol/Å².  This operates on **Cartesian** Hessian eigenvalues —
    not mass-weighted ones.

    Args:
        eigenvalues (np.ndarray): Eigenvalues from Cartesian Hessian decomposition.
        replace_with (float): Replacement value in Hartree/Bohr². Defaults to 1.0.
        zer_out_neg (bool): If True, zero out remaining negative eigenvalues after
            replacing the most negative. Defaults to False.
        units (str): Unit system of the eigenvalues.  If ``co.GAUSSIAN``
            (default), *replace_with* is used as-is (Hartree/Bohr²).
            If ``co.KJMOLA``, *replace_with* is converted via
            :func:`~q2mm.models.units.hessian_au_to_kjmola2`.
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
    replaced_eigenvalues[index_to_replace] = hessian_au_to_kjmola2(replace_with) if units == co.KJMOLA else replace_with

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
        :func:`mass_weight_hessian`).

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


def invert_ts_curvature_jax(
    hessian_matrix: np.ndarray,
    replace_with: float = 1.0,
) -> np.ndarray:
    """JIT-compatible transition-state curvature inversion.

    JAX sibling of :func:`invert_ts_curvature`: replaces the most negative
    eigenvalue of a Cartesian Hessian (in Hartree/Bohr²) with a large
    positive value so Seminario projection yields positive force constants,
    and zeros out any other negative eigenvalues.  All operations are
    traceable by ``jax.jit`` / ``jax.grad`` so the inversion can live inside
    the analytical loss pipeline.

    ``jnp.linalg.eigh`` returns eigenvalues in ascending order, so the most
    negative eigenvalue is ``evals[0]``.  We use ``jnp.where`` rather than
    Python branching to keep the control flow traceable: positive-definite
    Hessians pass through unchanged (within machine precision) because only
    negative eigenvalues are rewritten.

    Based on Limé & Norrby (J. Comput. Chem. 2015, 36, 244–250,
    DOI:10.1002/jcc.23797).

    Args:
        hessian_matrix: ``(3N, 3N)`` Cartesian Hessian in Hartree/Bohr².
        replace_with: Replacement value for the reaction-coordinate
            eigenvalue in Hartree/Bohr².  Defaults to ``1.0``.

    Returns:
        ``(3N, 3N)`` modified Hessian with TS curvature inverted.

    """
    from q2mm.backends.mm._jax_common import ensure_jax

    ensure_jax(engine_name="invert_ts_curvature_jax")

    from q2mm.backends.mm._jax_common import jnp

    hess = 0.5 * (hessian_matrix + hessian_matrix.T)
    evals, evecs = jnp.linalg.eigh(hess)

    new_smallest = jnp.where(evals[0] < 0, replace_with, evals[0])
    rest = jnp.where(evals[1:] < 0, 0.0, evals[1:])
    new_evals = jnp.concatenate([new_smallest[jnp.newaxis], rest])

    return (evecs * new_evals) @ evecs.T


def invert_ts_curvature(
    hessian_matrix: np.ndarray,
) -> np.ndarray:
    """Invert the curvature of a transition-state Hessian.

    Decomposes the Hessian, replaces the negative reaction-coordinate
    eigenvalue with a large positive value, and reconstructs.  This
    converts a saddle-point Hessian into one that Seminario projection
    can safely use to produce positive force constants.

    Based on Limé & Norrby (J. Comput. Chem. 2015, 36, 244–250).

    Args:
        hessian_matrix: Hessian matrix to process.

    Returns:
        Modified Hessian with inverted TS curvature.

    """
    eigenvalues, eigenvectors = decompose(hessian_matrix)
    modified_evals = replace_neg_eigenvalue(eigenvalues, zer_out_neg=True, strict=False)
    return reform_hessian(modified_evals, eigenvectors)
