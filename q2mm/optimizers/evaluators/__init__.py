"""Per-data-type evaluators for force field optimization.

Each evaluator encapsulates the logic for one category of observable
(energy, frequency, geometry, eigenmatrix).  The :class:`Evaluator`
protocol defines the interface that :class:`~q2mm.optimizers.objective.ObjectiveFunction`
delegates to.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for per-data-type evaluators.

    Each evaluator knows how to:

    1. **compute** — run the MM engine and produce calculated values for
       its data type (energy, frequencies, geometry, eigenmatrix).
    2. **residuals** — compare computed values against reference data and
       return weighted residuals.
    3. **gradient** *(optional)* — compute analytical gradient of this
       evaluator's score contribution w.r.t. force field parameters.
    """

    def compute(
        self,
        engine: MMEngine,
        mol: Q2MMMolecule,
        ff: ForceField,
        *,
        structure: Any | None = None,
    ) -> Any:
        """Run the MM engine and return computed observables.

        Args:
            engine: The MM backend.
            mol: The molecule being evaluated.
            ff: The current force field.
            structure: Optional pre-built engine context/handle for
                backends that support runtime parameter updates.

        Returns:
            Computed data (type depends on evaluator).

        """
        ...

    def residuals(self, computed: Any, reference: Any) -> list[float]:
        """Compare computed values to reference and return weighted residuals.

        Args:
            computed: Output from :meth:`compute`.
            reference: Reference data entries for this evaluator's kind.

        Returns:
            List of ``w_i * (ref_i - calc_i)`` residuals.

        """
        ...

    def supports_analytical_gradient(self, engine: MMEngine) -> bool:
        """Whether this evaluator can compute analytical gradients on *engine*.

        Args:
            engine: The MM backend to check.

        Returns:
            ``True`` if :meth:`gradient` is implemented for this engine.

        """
        ...

    def gradient(
        self,
        engine: MMEngine,
        mol: Q2MMMolecule,
        ff: ForceField,
        references: Any,
        n_params: int,
        *,
        structure: Any | None = None,
    ) -> np.ndarray | None:
        """Compute analytical gradient of this evaluator's score contribution.

        The score contribution is ``sum_i (w_i * (ref_i - calc_i))^2``,
        so the gradient is:

        ``d(score)/d(p) = -2 * sum_i [w_i^2 * (ref_i - calc_i) * d(calc_i)/d(p)]``

        Args:
            engine: The MM backend.
            mol: The molecule being evaluated.
            ff: The current force field.
            references: Reference data entries for this evaluator's kind.
            n_params: Number of force field parameters (length of
                the gradient vector).
            structure: Optional pre-built engine context/handle.

        Returns:
            Gradient vector of shape ``(n_params,)``, or ``None`` if
            analytical gradients are not yet supported for this evaluator.

        """
        ...
