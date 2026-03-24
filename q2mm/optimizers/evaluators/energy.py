"""Energy evaluator — computes MM single-point energy and residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.optimizers.objective import ReferenceValue


@dataclass
class EnergyResult:
    """Container for a computed MM energy value.

    Attributes:
        energy: The MM single-point energy (kcal/mol or engine units).

    """

    energy: float


class EnergyEvaluator:
    """Evaluates MM single-point energies against QM reference energies.

    This is the simplest evaluator: it calls ``engine.energy()`` once
    per molecule and compares against reference energy values.
    """

    def compute(
        self,
        engine: MMEngine,
        mol: Q2MMMolecule,
        ff: ForceField,
        *,
        structure: Any | None = None,
    ) -> EnergyResult:
        """Compute the MM single-point energy.

        Args:
            engine: The MM backend.
            mol: The molecule being evaluated.
            ff: The current force field.
            structure: Optional pre-built engine context/handle.

        Returns:
            EnergyResult with the computed energy.

        """
        target = structure if structure is not None else mol
        energy = engine.energy(target, ff)
        return EnergyResult(energy=energy)

    def residuals(
        self,
        computed: EnergyResult,
        references: list[ReferenceValue],
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

    def supports_analytical_gradient(self, engine: MMEngine) -> bool:
        """Energy gradients are available when the engine provides them.

        Args:
            engine: The MM backend to check.

        Returns:
            ``True`` if *engine* supports ``energy_and_param_grad()``.

        """
        return engine.supports_analytical_gradients()

    def gradient(
        self,
        engine: MMEngine,
        mol: Q2MMMolecule,
        ff: ForceField,
        references: list[ReferenceValue],
        n_params: int,
        *,
        structure: Any | None = None,
    ) -> np.ndarray:
        """Compute analytical gradient of the energy score contribution.

        Uses the engine's ``energy_and_param_grad()`` to obtain
        ``dE/dp`` and chains through the weighted-residual score:

        ``d(score)/d(p) = -2 * sum_i [w_i^2 * (ref_i - E_calc) * dE/dp]``

        Args:
            engine: The MM backend (must support analytical gradients).
            mol: The molecule being evaluated.
            ff: The current force field.
            references: Reference energy values for this molecule.
            n_params: Length of the gradient vector.
            structure: Optional pre-built engine context/handle.

        Returns:
            Gradient vector of shape ``(n_params,)``.

        Raises:
            TypeError: If the engine does not support analytical gradients.

        """
        if not engine.supports_analytical_gradients():
            raise TypeError(
                f"{engine.name} does not support energy_and_param_grad(). Cannot compute analytical energy gradient."
            )

        target = structure if structure is not None else mol
        calc_energy, de_dp = engine.energy_and_param_grad(target, ff)

        if len(de_dp) != n_params:
            raise ValueError(f"energy_and_param_grad returned {len(de_dp)} derivatives but expected {n_params}")

        grad = np.zeros(n_params)
        for ref in references:
            diff = ref.value - calc_energy
            grad += -2.0 * ref.weight**2 * diff * de_dp
        return grad

    @staticmethod
    def extract_value(calc: dict[str, Any], ref: ReferenceValue) -> float:
        """Extract calculated energy from a results dict.

        Args:
            calc: Results dict from ``_evaluate_molecule``.
            ref: The reference value to match.

        Returns:
            The calculated energy.

        """
        return calc["energy"]
