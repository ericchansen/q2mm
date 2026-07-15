"""Raw Hessian element evaluator — computes MM Hessian and extracts matrix elements.

Unlike the eigenmatrix evaluator, this works directly with the raw Cartesian
Hessian matrix without projecting onto QM eigenvectors.
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
class HessianResult:
    """Container for computed MM Hessian.

    Attributes:
        hessian: The raw Cartesian Hessian in Hartree/Bohr².

    """

    hessian: np.ndarray


class HessianElementEvaluator:
    """Evaluates raw MM Hessian elements against QM reference.

    Computes the full MM Hessian and extracts individual matrix elements
    at specified (row, col) positions for comparison with QM values.
    """

    HANDLED_KINDS = frozenset({"hessian_element"})

    def compute(
        self,
        prepared: PreparedBackend,
        parameters: np.ndarray,
    ) -> HessianResult:
        """Compute the raw MM Hessian.

        Args:
            prepared: The prepared per-case backend session.
            parameters: Full parameter vector.

        Returns:
            HessianResult with the computed Hessian in Hartree/Bohr².

        """
        hess = np.asarray(prepared.hessian(HessianRequest(parameters=parameters)).hessian)
        return HessianResult(hessian=hess)

    def residuals(
        self,
        computed: HessianResult,
        references: list[Observation],
    ) -> list[float]:
        """Compute weighted residuals for Hessian element references.

        Args:
            computed: Output from :meth:`compute`.
            references: Reference Hessian element values.

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
    def _extract(computed: HessianResult, ref: Observation) -> float:
        """Extract a raw Hessian element at (row, col).

        Args:
            computed: Hessian result.
            ref: Reference value with ``atom_indices=(row, col)``.

        Returns:
            The calculated Hessian element.

        """
        if ref.atom_indices is None or len(ref.atom_indices) < 2:
            raise ValueError(
                f"hessian_element requires atom_indices=(row, col), got {ref.atom_indices}. Label: {ref.label!r}"
            )
        row, col = ref.atom_indices[:2]
        n = computed.hessian.shape[0]
        if row < 0 or row >= n or col < 0 or col >= n:
            raise IndexError(f"Hessian indices ({row}, {col}) out of range for {n}×{n} matrix. Label: {ref.label!r}")
        return float(computed.hessian[row, col])

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
        """Compute analytical gradient of the Hessian element score contribution.

        Extracts ``dH[row,col]/dp`` directly from the Hessian parameter
        Jacobian tensor.

        Args:
            prepared: The prepared backend session (must support Hessian
                parameter Jacobians).
            parameters: Full parameter vector.
            references: Reference Hessian element values for this molecule.
            n_params: Length of the gradient vector.
            mol_idx: Molecule index (unused).

        Returns:
            Gradient vector of shape ``(n_params,)``.

        """
        if not prepared.info.supports(Capability.HESSIAN_PARAMETER_JACOBIAN):
            raise UnsupportedCapabilityError(prepared.info.name, Capability.HESSIAN_PARAMETER_JACOBIAN)

        result = prepared.hessian_parameter_jacobian(HessianJacobianRequest(parameters=parameters))
        hess = np.asarray(result.hessian)
        dH_dp = np.asarray(result.jacobian)

        n = hess.shape[0]
        grad = np.zeros(n_params)
        for ref in references:
            if ref.atom_indices is None or len(ref.atom_indices) < 2:
                raise ValueError(
                    f"hessian_element requires atom_indices=(row, col), got {ref.atom_indices}. Label: {ref.label!r}"
                )
            row, col = ref.atom_indices[:2]
            if row < 0 or row >= n or col < 0 or col >= n:
                raise IndexError(
                    f"Hessian indices ({row}, {col}) out of range for {n}×{n} matrix. Label: {ref.label!r}"
                )
            calc_value = float(hess[row, col])
            diff = ref.value - calc_value
            grad += -2.0 * ref.weight**2 * diff * dH_dp[row, col, :]
        return grad

    @staticmethod
    def extract_value(calc: dict[str, Any], ref: Observation) -> float:
        """Extract a calculated Hessian element from a results dict.

        Backward-compatible bridge for ObjectiveFunction._extract_value.
        Delegates to :meth:`_extract` via a temporary :class:`HessianResult`.

        Args:
            calc: Results dict from ``_evaluate_molecule``.
            ref: The reference value to match.

        Returns:
            The calculated Hessian element.

        """
        return HessianElementEvaluator._extract(HessianResult(hessian=calc["raw_hessian"]), ref)

    def reset(self) -> None:
        """No cached state to clear."""
