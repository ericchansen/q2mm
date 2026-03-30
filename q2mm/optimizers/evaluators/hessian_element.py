"""Raw Hessian element evaluator — computes MM Hessian and extracts matrix elements.

Unlike the eigenmatrix evaluator, this works directly with the raw Cartesian
Hessian matrix without projecting onto QM eigenvectors.
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
        engine: MMEngine,
        mol: Q2MMMolecule,
        ff: ForceField,
        *,
        structure: Any | None = None,
    ) -> HessianResult:
        """Compute the raw MM Hessian.

        Args:
            engine: The MM backend.
            mol: The molecule being evaluated.
            ff: The current force field.
            structure: Optional pre-built engine context/handle.

        Returns:
            HessianResult with the computed Hessian in Hartree/Bohr².

        """
        target = structure if structure is not None else mol
        hess = engine.hessian(target, ff)
        return HessianResult(hessian=hess)

    def residuals(
        self,
        computed: HessianResult,
        references: list[ReferenceValue],
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
    def _extract(computed: HessianResult, ref: ReferenceValue) -> float:
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

    def supports_analytical_gradient(self, engine: MMEngine) -> bool:
        """Raw Hessian element gradients are not yet implemented.

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
        """Not yet implemented — raw Hessian element analytical gradients.

        Returns:
            ``None`` — analytical gradients are not yet supported.

        """
        return None

    @staticmethod
    def extract_value(calc: dict[str, Any], ref: ReferenceValue) -> float:
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
