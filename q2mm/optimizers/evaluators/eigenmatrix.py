"""Eigenmatrix evaluator — computes MM eigenmatrix projection and residuals.

Projects the MM Hessian onto QM eigenvectors to produce an eigenmatrix,
then compares diagonal (eigenvalue) and off-diagonal elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.optimizers.objective import ReferenceValue


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
        engine: MMEngine,
        mol: Q2MMMolecule,
        ff: ForceField,
        *,
        structure: Any | None = None,
        mol_idx: int = 0,
    ) -> EigenmatrixResult:
        """Compute MM eigenmatrix by projecting MM Hessian onto QM eigenvectors.

        Args:
            engine: The MM backend.
            mol: The molecule being evaluated (must have a QM Hessian).
            ff: The current force field.
            structure: Optional pre-built engine context/handle.
            mol_idx: Molecule index for eigenvector caching.

        Returns:
            EigenmatrixResult with the computed eigenmatrix.

        Raises:
            ValueError: If the molecule has no QM Hessian.

        """
        from q2mm.models.hessian import decompose, transform_to_eigenmatrix

        target = structure if structure is not None else mol
        mm_hess = engine.hessian(target, ff)

        if mol_idx not in self._qm_eigenvectors:
            if mol.hessian is None:
                raise ValueError(
                    f"Molecule {mol_idx} ({mol.name}) has no QM Hessian. "
                    "Eigenmatrix training requires a QM Hessian for the "
                    "eigenvector basis."
                )
            _, qm_evecs = decompose(mol.hessian)
            self._qm_eigenvectors[mol_idx] = qm_evecs

        qm_evecs = self._qm_eigenvectors[mol_idx]
        eigenmatrix = transform_to_eigenmatrix(mm_hess, qm_evecs)

        return EigenmatrixResult(eigenmatrix=eigenmatrix)

    def residuals(
        self,
        computed: EigenmatrixResult,
        references: list[ReferenceValue],
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
    def _extract(computed: EigenmatrixResult, ref: ReferenceValue) -> float:
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

    def supports_analytical_gradient(self, engine: MMEngine) -> bool:
        """Eigenmatrix gradients require eigenvector derivatives.

        Args:
            engine: The MM backend to check.

        Returns:
            Always ``False`` — not yet implemented.

        """
        return False

    def gradient(
        self,
        engine: MMEngine,
        mol: Q2MMMolecule,
        ff: ForceField,
        references: list[ReferenceValue],
        n_params: int,
        *,
        structure: Any | None = None,
    ) -> np.ndarray | None:
        """Not yet implemented — eigenmatrix analytical gradients.

        Differentiating through eigendecomposition (eigenvector
        derivatives) is planned for a future release.

        Returns:
            ``None`` — analytical gradients are not yet supported.

        """
        return None

    @staticmethod
    def extract_value(calc: dict[str, Any], ref: ReferenceValue) -> float:
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
