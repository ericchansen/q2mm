"""Frequency evaluator — computes MM vibrational frequencies and residuals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.optimizers.objective import ReferenceValue


@dataclass
class FrequencyResult:
    """Container for computed MM vibrational frequencies.

    Attributes:
        frequencies: List of frequencies in cm⁻¹, ordered by mode index.

    """

    frequencies: list[float] = field(default_factory=list)


class FrequencyEvaluator:
    """Evaluates MM vibrational frequencies against QM reference frequencies.

    Calls ``engine.frequencies()`` once per molecule and matches by
    positional mode index (``data_idx``).
    """

    HANDLED_KINDS = frozenset({"frequency"})

    def compute(
        self,
        engine: MMEngine,
        mol: Q2MMMolecule,
        ff: ForceField,
        *,
        structure: Any | None = None,
        on_error: str = "raise",
    ) -> FrequencyResult:
        """Compute MM vibrational frequencies.

        Args:
            engine: The MM backend.
            mol: The molecule being evaluated.
            ff: The current force field.
            structure: Optional pre-built engine context/handle.
            on_error: Error handling for eigendecomposition failures.
                ``"raise"`` (default) propagates exceptions.
                ``"penalty"`` returns large penalty frequencies so the
                optimizer can retreat from pathological parameter regions.

        Returns:
            FrequencyResult with computed frequencies.

        """
        target = structure if structure is not None else mol
        freqs = engine.frequencies(target, ff, on_error=on_error)
        return FrequencyResult(frequencies=list(freqs))

    def residuals(
        self,
        computed: FrequencyResult,
        references: list[ReferenceValue],
    ) -> list[float]:
        """Compute weighted residuals for frequency references.

        Args:
            computed: Output from :meth:`compute`.
            references: Reference frequency values (``kind="frequency"``).

        Returns:
            List of ``w * (ref - calc)`` residuals.

        Raises:
            IndexError: If a reference ``data_idx`` is out of range.

        """
        result: list[float] = []
        for ref in references:
            if ref.data_idx < 0 or ref.data_idx >= len(computed.frequencies):
                raise IndexError(
                    f"Frequency data_idx={ref.data_idx} out of range "
                    f"(molecule has {len(computed.frequencies)} modes). "
                    f"Label: {ref.label!r}"
                )
            calc_value = computed.frequencies[ref.data_idx]
            diff = ref.value - calc_value
            result.append(ref.weight * diff)
        return result

    def supports_analytical_gradient(self, engine: MMEngine) -> bool:
        """Check if the engine supports Hessian parameter Jacobians.

        Args:
            engine: The MM backend to check.

        Returns:
            ``True`` if the engine supports ``hessian_and_param_jacobian()``.

        """
        return engine.supports_analytical_hessian_gradients()

    def gradient(
        self,
        engine: MMEngine,
        mol: Q2MMMolecule,
        ff: ForceField,
        references: list[ReferenceValue],
        n_params: int,
        *,
        structure: Any | None = None,
        mol_idx: int = 0,
    ) -> np.ndarray:
        """Compute analytical gradient of the frequency score contribution.

        Uses eigenvalue sensitivity to differentiate frequencies w.r.t.
        force field parameters without differentiating through
        the eigendecomposition backward pass.

        Args:
            engine: The MM backend (must support Hessian parameter Jacobians).
            mol: The molecule being evaluated.
            ff: The current force field.
            references: Reference frequency values for this molecule.
            n_params: Length of the gradient vector.
            structure: Optional pre-built engine context/handle.
            mol_idx: Molecule index (unused).

        Returns:
            Gradient vector of shape ``(n_params,)``.

        """
        from q2mm.models.hessian import frequency_param_jacobian

        if not engine.supports_analytical_hessian_gradients():
            raise TypeError(
                f"{engine.name} does not support hessian_and_param_jacobian(). "
                "Cannot compute analytical frequency gradient."
            )

        target = structure if structure is not None else mol
        hess, dH_dp = engine.hessian_and_param_jacobian(target, ff)
        freqs, d_freq_dp = frequency_param_jacobian(hess, dH_dp, mol.symbols)

        grad = np.zeros(n_params)
        for ref in references:
            if ref.data_idx < 0 or ref.data_idx >= len(freqs):
                raise IndexError(
                    f"Frequency data_idx={ref.data_idx} out of range "
                    f"(molecule has {len(freqs)} modes). "
                    f"Label: {ref.label!r}"
                )
            calc_value = freqs[ref.data_idx]
            diff = ref.value - calc_value
            grad += -2.0 * ref.weight**2 * diff * d_freq_dp[ref.data_idx, :]
        return grad

    @staticmethod
    def extract_value(calc: dict[str, Any], ref: ReferenceValue) -> float:
        """Extract a calculated frequency from a results dict.

        Args:
            calc: Results dict from ``_evaluate_molecule``.
            ref: The reference value to match.

        Returns:
            The calculated frequency for the given mode index.

        Raises:
            IndexError: If ``data_idx`` is out of range.

        """
        freqs = calc["frequencies"]
        if ref.data_idx < 0 or ref.data_idx >= len(freqs):
            raise IndexError(
                f"Frequency data_idx={ref.data_idx} out of range "
                f"(molecule has {len(freqs)} modes). Label: {ref.label!r}"
            )
        return freqs[ref.data_idx]
