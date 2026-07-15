"""Frequency evaluator — computes MM vibrational frequencies and residuals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from q2mm.backends.contracts import (
    Capability,
    FrequencyRequest,
    HessianJacobianRequest,
    PreparedBackend,
    UnsupportedCapabilityError,
)
from q2mm.models.observations import Observation


@dataclass
class FrequencyResult:
    """Container for computed MM vibrational frequencies.

    Attributes:
        frequencies: List of frequencies in cm⁻¹, ordered by mode index.

    """

    frequencies: list[float] = field(default_factory=list)


class FrequencyEvaluator:
    """Evaluates MM vibrational frequencies against QM reference frequencies.

    Calls ``prepared.frequencies()`` once per molecule and matches by
    positional mode index (``data_idx``).
    """

    HANDLED_KINDS = frozenset({"frequency"})

    def compute(
        self,
        prepared: PreparedBackend,
        parameters: np.ndarray,
        *,
        on_error: str = "raise",
    ) -> FrequencyResult:
        """Compute MM vibrational frequencies.

        Args:
            prepared: The prepared per-case backend session.
            parameters: Full parameter vector.
            on_error: Error handling for eigendecomposition failures.
                ``"raise"`` (default) propagates exceptions.
                ``"penalty"`` returns large penalty frequencies so the
                optimizer can retreat from pathological parameter regions.

        Returns:
            FrequencyResult with computed frequencies.

        """
        result = prepared.frequencies(FrequencyRequest(parameters=parameters, on_error=on_error))
        return FrequencyResult(frequencies=[float(f) for f in result.frequencies])

    def residuals(
        self,
        computed: FrequencyResult,
        references: list[Observation],
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
        """Compute analytical gradient of the frequency score contribution.

        Uses eigenvalue sensitivity to differentiate frequencies w.r.t.
        force field parameters without differentiating through
        the eigendecomposition backward pass.

        Args:
            prepared: The prepared backend session (must support Hessian
                parameter Jacobians).
            parameters: Full parameter vector.
            references: Reference frequency values for this molecule.
            n_params: Length of the gradient vector.
            mol_idx: Molecule index (unused).

        Returns:
            Gradient vector of shape ``(n_params,)``.

        """
        from q2mm.models.hessian import frequency_param_jacobian

        if not prepared.info.supports(Capability.HESSIAN_PARAMETER_JACOBIAN):
            raise UnsupportedCapabilityError(prepared.info.name, Capability.HESSIAN_PARAMETER_JACOBIAN)

        result = prepared.hessian_parameter_jacobian(HessianJacobianRequest(parameters=parameters))
        hess = np.asarray(result.hessian)
        dH_dp = np.asarray(result.jacobian)
        symbols = prepared.molecule.symbols
        freqs, d_freq_dp = frequency_param_jacobian(hess, dH_dp, symbols)

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
    def extract_value(calc: dict[str, Any], ref: Observation) -> float:
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
