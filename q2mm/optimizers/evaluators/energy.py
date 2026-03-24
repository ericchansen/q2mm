"""Energy evaluator — computes MM single-point energy and residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
