"""Per-data-type evaluators for force field optimization.

Each evaluator encapsulates the logic for one category of observable
(energy, frequency, geometry, eigenmatrix, raw Hessian element).  The
:class:`Evaluator` protocol defines the interface that
:class:`~q2mm.optimizers.objective.ObjectiveFunction` delegates to.

Evaluators operate on a :class:`~q2mm.backends.contracts.PreparedBackend`
session (which owns the molecule, base force field, parameter layout, and
reusable native state) plus a **full parameter vector**.  They never receive a
:class:`~q2mm.models.forcefield.ForceField` or a native handle across the
boundary — the prepared session validates the vector and returns typed results.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from q2mm.backends.contracts import PreparedBackend
from q2mm.models.observations import Observation


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for per-data-type evaluators.

    Each evaluator knows how to:

    1. **compute** — run the prepared backend for a parameter vector and
       produce calculated values for its data type.
    2. **residuals** — compare computed values against reference data and
       return weighted residuals.
    3. **gradient** *(optional)* — compute analytical gradient of this
       evaluator's score contribution w.r.t. force field parameters.
    """

    def compute(
        self,
        prepared: PreparedBackend,
        parameters: np.ndarray,
    ) -> object:
        """Run the prepared backend and return computed observables.

        Args:
            prepared: The prepared per-case backend session.
            parameters: Full parameter vector (length ``len(layout)``).

        Returns:
            Computed data (type depends on evaluator).

        """
        ...

    def residuals(self, computed: object, references: list[Observation]) -> list[float]:
        """Compare computed values to reference and return weighted residuals.

        Args:
            computed: Output from :meth:`compute`.
            references: Reference data entries for this evaluator's kind.

        Returns:
            List of ``w_i * (ref_i - calc_i)`` residuals.

        """
        ...

    def supports_analytical_gradient(self, prepared: PreparedBackend) -> bool:
        """Whether this evaluator can compute analytical gradients.

        Args:
            prepared: The prepared backend session to check.

        Returns:
            ``True`` if :meth:`gradient` is implemented for this backend.

        """
        ...

    def gradient(
        self,
        prepared: PreparedBackend,
        parameters: np.ndarray,
        references: list[Observation],
        n_params: int,
        *,
        mol_idx: int = 0,
    ) -> np.ndarray | None:
        """Compute analytical gradient of this evaluator's score contribution.

        The score contribution is ``sum_i (w_i * (ref_i - calc_i))^2``, so the
        gradient is
        ``-2 * sum_i [w_i^2 * (ref_i - calc_i) * d(calc_i)/d(p)]``.

        Args:
            prepared: The prepared backend session.
            parameters: Full parameter vector.
            references: Reference data entries for this evaluator's kind.
            n_params: Length of the gradient vector (``len(layout)``).
            mol_idx: Molecule index for per-molecule caching.

        Returns:
            Gradient vector of shape ``(n_params,)``, or ``None`` if analytical
            gradients are not supported for this evaluator.

        """
        ...
