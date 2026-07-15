"""Energy evaluator — computes MM single-point energy and residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from q2mm.backends.contracts import (
    Capability,
    EnergyRequest,
    ParameterGradientRequest,
    PreparedBackend,
    UnsupportedCapabilityError,
)
from q2mm.models.observations import Observation


@dataclass
class EnergyResult:
    """Container for a computed MM energy value.

    Attributes:
        energy: The MM single-point energy (kcal/mol).

    """

    energy: float


class EnergyEvaluator:
    """Evaluates MM single-point energies against QM reference energies.

    This is the simplest evaluator: it calls ``prepared.energy()`` once
    per molecule and compares against reference energy values.
    """

    HANDLED_KINDS = frozenset({"energy"})

    def compute(
        self,
        prepared: PreparedBackend,
        parameters: np.ndarray,
    ) -> EnergyResult:
        """Compute the MM single-point energy.

        Args:
            prepared: The prepared per-case backend session.
            parameters: Full parameter vector.

        Returns:
            EnergyResult with the computed energy.

        """
        result = prepared.energy(EnergyRequest(parameters=parameters))
        return EnergyResult(energy=float(result.energy))

    def residuals(
        self,
        computed: EnergyResult,
        references: list[Observation],
    ) -> list[float]:
        """Compute weighted residuals for energy references.

        Args:
            computed: Output from :meth:`compute`.
            references: Reference energy values (all with ``kind="energy"``).

        Returns:
            List of ``w * (ref - calc)`` residuals.

        """
        result: list[float] = []
        for ref in references:
            diff = ref.value - computed.energy
            result.append(ref.weight * diff)
        return result

    def supports_analytical_gradient(self, prepared: PreparedBackend) -> bool:
        """Energy gradients are available when the backend declares them.

        Args:
            prepared: The prepared backend session to check.

        Returns:
            ``True`` if the backend declares ``PARAMETER_GRADIENT``.

        """
        return prepared.info.supports(Capability.PARAMETER_GRADIENT)

    def gradient(
        self,
        prepared: PreparedBackend,
        parameters: np.ndarray,
        references: list[Observation],
        n_params: int,
        *,
        mol_idx: int = 0,
    ) -> np.ndarray:
        """Compute analytical gradient of the energy score contribution.

        Uses the backend's parameter-gradient result to obtain ``dE/dp`` and
        chains through the weighted-residual score:

        ``d(score)/d(p) = -2 * sum_i [w_i^2 * (ref_i - E_calc) * dE/dp]``

        Args:
            prepared: The prepared backend session (must support gradients).
            parameters: Full parameter vector.
            references: Reference energy values for this molecule.
            n_params: Length of the gradient vector.
            mol_idx: Molecule index (unused).

        Returns:
            Gradient vector of shape ``(n_params,)``.

        Raises:
            UnsupportedCapabilityError: If the backend does not declare
                analytical parameter gradients.

        """
        if not prepared.info.supports(Capability.PARAMETER_GRADIENT):
            raise UnsupportedCapabilityError(prepared.info.name, Capability.PARAMETER_GRADIENT)

        result = prepared.parameter_gradient(ParameterGradientRequest(parameters=parameters))
        calc_energy = float(result.energy)
        de_dp = np.asarray(result.gradient)

        if len(de_dp) != n_params:
            raise ValueError(f"parameter_gradient returned {len(de_dp)} derivatives but expected {n_params}")

        grad = np.zeros(n_params)
        for ref in references:
            diff = ref.value - calc_energy
            grad += -2.0 * ref.weight**2 * diff * de_dp
        return grad

    @staticmethod
    def extract_value(calc: dict[str, Any], ref: Observation) -> float:
        """Extract calculated energy from a results dict.

        Args:
            calc: Results dict from ``_evaluate_molecule``.
            ref: The reference value to match.

        Returns:
            The calculated energy.

        """
        return calc["energy"]
