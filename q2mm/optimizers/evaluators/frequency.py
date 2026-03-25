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
    ) -> FrequencyResult:
        """Compute MM vibrational frequencies.

        Args:
            engine: The MM backend.
            mol: The molecule being evaluated.
            ff: The current force field.
            structure: Optional pre-built engine context/handle.

        Returns:
            FrequencyResult with computed frequencies.

        """
        target = structure if structure is not None else mol
        freqs = engine.frequencies(target, ff)
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
        """Frequency gradients require differentiating through the Hessian eigendecomposition.

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
        """Not yet implemented — frequency analytical gradients.

        Differentiating through Hessian → eigendecomposition → frequencies
        is planned for a future release.

        Returns:
            ``None`` — analytical gradients are not yet supported.

        """
        return None

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
