"""Eigenmatrix evaluator — computes MM eigenmatrix projection and residuals.

Projects the MM Hessian onto QM eigenvectors to produce an eigenmatrix,
then compares diagonal (eigenvalue) and off-diagonal elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from q2mm.backends.contracts import (
    Capability,
    HessianJacobianRequest,
    HessianRequest,
    PreparedBackend,
    UnsupportedCapabilityError,
)
from q2mm.models.observations import Observation


@dataclass
class EigenmatrixResult:
    """Container for computed MM eigenmatrix.

    Attributes:
        eigenmatrix: The eigenmatrix from projecting the MM Hessian
            onto QM eigenvectors.

    """

    eigenmatrix: np.ndarray


class EigenmatrixEvaluator:
    """Evaluates MM Hessian eigenmatrix against QM reference eigenmatrix.

    Projects the MM Hessian onto the QM eigenvector basis to produce an
    eigenmatrix, then compares diagonal elements (eigenvalues) and
    off-diagonal elements (mode coupling).

    The QM eigenvectors are computed once and cached, since the QM basis
    is fixed across optimization iterations.
    """

    EIGENMATRIX_KINDS = frozenset({"eig_diagonal", "eig_offdiagonal"})
    HANDLED_KINDS = EIGENMATRIX_KINDS

    def __init__(self) -> None:
        """Initialize with empty eigenvector cache."""
        self._qm_eigenvectors: dict[int, np.ndarray] = {}

    def compute(
        self,
        prepared: PreparedBackend,
        parameters: np.ndarray,
        *,
        mol_idx: int = 0,
    ) -> EigenmatrixResult:
        """Compute MM eigenmatrix by projecting MM Hessian onto QM eigenvectors.

        Args:
            prepared: The prepared per-case backend session (its molecule must
                have a QM Hessian).
            parameters: Full parameter vector.
            mol_idx: Molecule index for eigenvector caching.

        Returns:
            EigenmatrixResult with the computed eigenmatrix.

        Raises:
            ValueError: If the molecule has no QM Hessian.

        """
        from q2mm.models.hessian import mass_weighted_eigenmatrix, mass_weighted_normal_modes

        mol = prepared.molecule
        mm_hess = np.asarray(prepared.hessian(HessianRequest(parameters=parameters)).hessian)

        if mol_idx not in self._qm_eigenvectors:
            if mol.hessian is None:
                raise ValueError(
                    f"Molecule {mol_idx} ({mol.name}) has no QM Hessian. "
                    "Eigenmatrix training requires a QM Hessian for the "
                    "eigenvector basis."
                )
            _, qm_evecs = mass_weighted_normal_modes(mol.hessian, mol.symbols)
            self._qm_eigenvectors[mol_idx] = qm_evecs

        qm_evecs = self._qm_eigenvectors[mol_idx]
        eigenmatrix = mass_weighted_eigenmatrix(mm_hess, qm_evecs, mol.symbols)

        return EigenmatrixResult(eigenmatrix=eigenmatrix)

    def residuals(
        self,
        computed: EigenmatrixResult,
        references: list[Observation],
    ) -> list[float]:
        """Compute weighted residuals for eigenmatrix references.

        Args:
            computed: Output from :meth:`compute`.
            references: Reference eigenmatrix values (eig_diagonal and/or
                eig_offdiagonal).

        Returns:
            List of ``w * (ref - calc)`` residuals.

        """
        result: list[float] = []
        for ref in references:
            calc_value = self._extract(computed, ref)
            diff = ref.value - calc_value
            result.append(ref.weight * diff)
        return result

    @staticmethod
    def _extract(computed: EigenmatrixResult, ref: Observation) -> float:
        """Extract a calculated eigenmatrix element.

        Args:
            computed: Eigenmatrix result.
            ref: Reference value to match.

        Returns:
            The calculated eigenmatrix element.

        """
        if ref.kind == "eig_diagonal":
            n = computed.eigenmatrix.shape[0]
            if ref.data_idx < 0 or ref.data_idx >= n:
                raise IndexError(
                    f"Eigenmatrix data_idx={ref.data_idx} out of range (matrix has {n} modes). Label: {ref.label!r}"
                )
            return float(computed.eigenmatrix[ref.data_idx, ref.data_idx])
        elif ref.kind == "eig_offdiagonal":
            row, col = ref.atom_indices[:2]
            return float(computed.eigenmatrix[row, col])
        raise ValueError(f"EigenmatrixEvaluator cannot handle kind: {ref.kind}")

    def supports_analytical_gradient(self, prepared: PreparedBackend) -> bool:
        """Check if the backend declares Hessian parameter Jacobians.

        Args:
            prepared: The prepared backend session to check.

        Returns:
            ``True`` if the backend declares ``HESSIAN_PARAMETER_JACOBIAN``.

        """
        return prepared.info.supports(Capability.HESSIAN_PARAMETER_JACOBIAN)

    def gradient(
        self,
        prepared: PreparedBackend,
        parameters: np.ndarray,
        references: list[Observation],
        n_params: int,
        *,
        mol_idx: int = 0,
    ) -> np.ndarray:
        """Compute analytical gradient of the eigenmatrix score contribution.

        Uses the identity ``d(Q^T H Q)/dp = Q^T · (dH/dp) · Q`` where
        Q is the (constant) QM eigenvector matrix.

        Args:
            prepared: The prepared backend session (must support Hessian
                parameter Jacobians; its molecule must have a QM Hessian).
            parameters: Full parameter vector.
            references: Reference eigenmatrix values for this molecule.
            n_params: Length of the gradient vector.
            mol_idx: Molecule index for QM eigenvector caching.

        Returns:
            Gradient vector of shape ``(n_params,)``.

        """
        from q2mm.models.hessian import mass_weight_scale_3n, mass_weighted_normal_modes

        if not prepared.info.supports(Capability.HESSIAN_PARAMETER_JACOBIAN):
            raise UnsupportedCapabilityError(prepared.info.name, Capability.HESSIAN_PARAMETER_JACOBIAN)

        mol = prepared.molecule
        result = prepared.hessian_parameter_jacobian(HessianJacobianRequest(parameters=parameters))
        hess = np.asarray(result.hessian)
        dH_dp = np.asarray(result.jacobian)

        # Get cached QM normal modes (or compute and cache)
        if mol_idx not in self._qm_eigenvectors:
            if mol.hessian is None:
                raise ValueError(
                    f"Molecule {mol_idx} ({mol.name}) has no QM Hessian. "
                    "Eigenmatrix training requires a QM Hessian for the "
                    "eigenvector basis."
                )
            _, qm_evecs = mass_weighted_normal_modes(mol.hessian, mol.symbols)
            self._qm_eigenvectors[mol_idx] = qm_evecs

        qm_evecs = self._qm_eigenvectors[mol_idx]

        # Mass-weighting is linear (H_mw = S ⊙ H), so it carries through to
        # the parameter Jacobian: d(Q^T H_mw Q)/dp = Q^T (S ⊙ dH/dp) Q.
        scale = mass_weight_scale_3n(mol.symbols)
        hess_mw = hess * scale
        dH_dp_mw = dH_dp * scale[:, :, None]

        # d(eigenmatrix)/dp_j = Q^T @ (S ⊙ dH_dp)[:,:,j] @ Q
        d_eigmat_dp = np.einsum("ir,ijp,jc->rcp", qm_evecs, dH_dp_mw, qm_evecs)

        eigmat = qm_evecs.T @ hess_mw @ qm_evecs
        n = eigmat.shape[0]

        grad = np.zeros(n_params)
        for ref in references:
            if ref.kind == "eig_diagonal":
                idx = ref.data_idx
                if idx is None or idx < 0 or idx >= n:
                    raise IndexError(
                        f"Eigenmatrix data_idx={idx!r} out of range (matrix has {n} modes). Label: {ref.label!r}"
                    )
                calc_value = float(eigmat[idx, idx])
                diff = ref.value - calc_value
                grad += -2.0 * ref.weight**2 * diff * d_eigmat_dp[idx, idx, :]
            elif ref.kind == "eig_offdiagonal":
                if ref.atom_indices is None or len(ref.atom_indices) < 2:
                    raise ValueError(
                        f"Off-diagonal eigenmatrix reference requires at least two atom_indices. Label: {ref.label!r}"
                    )
                row, col = ref.atom_indices[:2]
                if row < 0 or col < 0 or row >= n or col >= n:
                    raise IndexError(
                        f"Off-diagonal eigenmatrix indices ({row}, {col}) out of "
                        f"range for {n}×{n} matrix. Label: {ref.label!r}"
                    )
                calc_value = float(eigmat[row, col])
                diff = ref.value - calc_value
                grad += -2.0 * ref.weight**2 * diff * d_eigmat_dp[row, col, :]
            else:
                raise ValueError(f"EigenmatrixEvaluator cannot handle kind: {ref.kind}")
        return grad

    @staticmethod
    def extract_value(calc: dict[str, Any], ref: Observation) -> float:
        """Extract a calculated eigenmatrix value from a results dict.

        Backward-compatible bridge for ObjectiveFunction._extract_value.
        Delegates to :meth:`_extract` via a temporary :class:`EigenmatrixResult`.

        Args:
            calc: Results dict from ``_evaluate_molecule``.
            ref: The reference value to match.

        Returns:
            The calculated eigenmatrix element.

        """
        return EigenmatrixEvaluator._extract(EigenmatrixResult(eigenmatrix=calc["eigenmatrix"]), ref)

    def reset(self) -> None:
        """Clear cached QM eigenvectors."""
        self._qm_eigenvectors.clear()
