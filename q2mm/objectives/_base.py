"""Shared executor base for the objective package (private).

:class:`BaseObjectiveExecutor` implements the evaluation bookkeeping,
residual/regularization/metric plumbing, and the
:class:`~q2mm.objectives.protocols.ObjectiveEvaluator` surface once.
Concrete executors (:class:`~q2mm.objectives.python.PythonObjectiveExecutor`,
:class:`~q2mm.objectives.jax.JaxObjectiveExecutor`) supply only the
backend-specific pieces: per-observation calculated values, the gradient
mode, and (optionally) an overridden ``_total`` / ``value_and_gradient``.

All parameter vectors crossing the public surface are full-length
(``len(plan.layout)``); active/full projection is the optimizer's job via
:class:`~q2mm.models.parameters.ActiveParameterSpace`.  :meth:`reset` clears
only the evaluation counters and history — it never discards prepared
backend sessions, so a case is prepared exactly once per executor lifetime.
"""

from __future__ import annotations

import numpy as np

from q2mm.objectives.metrics import (
    raw_residual,
    regularization_gradient,
    regularization_residuals,
    regularization_value,
    weighted_residual,
)
from q2mm.objectives.plan import KIND_TO_CATEGORY, ObjectivePlan
from q2mm.objectives.protocols import Evaluation, GradientMode, ObjectiveGradientError


class BaseObjectiveExecutor:
    """Common evaluation machinery for concrete objective executors."""

    def __init__(self, plan: ObjectivePlan, *, fd_step: float = 1e-4) -> None:
        if not np.isfinite(fd_step) or fd_step <= 0.0:
            raise ValueError(f"fd_step must be positive and finite, got {fd_step!r}.")
        self._plan = plan
        self._fd_step = float(fd_step)
        self._n_eval = 0
        self._n_gradient_eval = 0
        self._history: list[float] = []

    # -- plan / identity ---------------------------------------------------

    @property
    def plan(self) -> ObjectivePlan:
        return self._plan

    @property
    def gradient_mode(self) -> GradientMode:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def finite_difference_step(self) -> float | None:
        """The executor's central FD step, or ``None`` when not FD mode."""
        return self._fd_step if self.gradient_mode is GradientMode.FINITE_DIFFERENCE else None

    # -- counters ----------------------------------------------------------

    @property
    def n_evaluations(self) -> int:
        return self._n_eval

    @property
    def n_gradient_evaluations(self) -> int:
        """Extra evaluations spent inside finite-difference gradients."""
        return self._n_gradient_eval

    @property
    def history(self) -> tuple[float, ...]:
        return tuple(self._history)

    def reset(self) -> None:
        """Clear the evaluation counters and history, retaining prepared state."""
        self._n_eval = 0
        self._n_gradient_eval = 0
        self._history.clear()

    def _record(self, score: float) -> None:
        self._n_eval += 1
        self._history.append(float(score))

    def record_evaluation(self, score: float) -> None:
        """Record an already-computed *score* as one evaluation (no backend work)."""
        self._record(float(score))

    # -- validation --------------------------------------------------------

    def _as_full(self, full_vector: np.ndarray) -> np.ndarray:
        arr = np.array(full_vector, dtype=float, copy=True)
        n = self._plan.n_params
        if arr.shape != (n,):
            raise ValueError(f"full_vector must have shape ({n},), got {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("full_vector must be finite.")
        return arr

    # -- abstract computation hooks ---------------------------------------

    def _calculated(self, full_vector: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        """Per-observation calculated values, aligned with observations order."""
        raise NotImplementedError

    def _data_gradient(self, full_vector: np.ndarray) -> np.ndarray:
        """Analytical gradient of the data term (no regularization).

        Subclasses that declare :attr:`GradientMode.ANALYTICAL` must
        implement this.  The default raises so a mis-declared executor
        fails loudly rather than silently.
        """
        raise ObjectiveGradientError(f"{type(self).__name__} does not implement analytical gradients.")

    # -- residual helpers --------------------------------------------------

    def _raw_residuals(self, calc: np.ndarray) -> np.ndarray:
        obs = self._plan.observations.values
        return np.array(
            [raw_residual(o.kind, float(o.value), float(c)) for o, c in zip(obs, calc, strict=True)],
            dtype=float,
        )

    def _weighted_residuals(self, calc: np.ndarray) -> np.ndarray:
        obs = self._plan.observations.values
        return np.array(
            [
                weighted_residual(o.kind, float(o.value), float(c), float(o.weight))
                for o, c in zip(obs, calc, strict=True)
            ],
            dtype=float,
        )

    def _total(self, full_vector: np.ndarray) -> float:
        calc = self._calculated(full_vector)
        weighted = self._weighted_residuals(calc)
        data = float(np.sum(weighted**2))
        return data + regularization_value(full_vector, self._plan.reference_params, self._plan.regularization)

    # -- public surface ----------------------------------------------------

    def value(self, full_vector: np.ndarray) -> float:
        full = self._as_full(full_vector)
        score = self._total(full)
        self._record(score)
        return score

    def sample(self, full_vector: np.ndarray) -> float:
        """Evaluate the objective without recording an evaluation."""
        return self._total(self._as_full(full_vector))

    def residuals(self, full_vector: np.ndarray) -> np.ndarray:
        full = self._as_full(full_vector)
        return self._weighted_residuals(self._calculated(full))

    def least_squares_residuals(self, full_vector: np.ndarray) -> np.ndarray:
        full = self._as_full(full_vector)
        data = self._weighted_residuals(self._calculated(full))
        reg = regularization_residuals(full, self._plan.reference_params, self._plan.regularization)
        if reg.size == 0:
            return data
        return np.concatenate([data, reg])

    def evaluate(self, full_vector: np.ndarray) -> Evaluation:
        full = self._as_full(full_vector)
        calc = self._calculated(full)
        raw = self._raw_residuals(calc)
        weighted = self._weighted_residuals(calc)
        data = float(np.sum(weighted**2))
        reg = regularization_value(full, self._plan.reference_params, self._plan.regularization)
        category_scores: dict[str, float] = {}
        for obs, w in zip(self._plan.observations.values, weighted, strict=True):
            category = KIND_TO_CATEGORY[obs.kind]
            category_scores[category] = category_scores.get(category, 0.0) + float(w) ** 2
        return Evaluation(
            total=data + reg,
            data_value=data,
            regularization=reg,
            calculated=calc,
            raw_residuals=raw,
            weighted_residuals=weighted,
            category_scores=category_scores,
        )

    def value_and_gradient(self, full_vector: np.ndarray) -> tuple[float, np.ndarray]:
        full = self._as_full(full_vector)
        mode = self.gradient_mode
        if mode is GradientMode.NONE:
            raise ObjectiveGradientError(
                f"{type(self).__name__} declares gradient_mode=none; no analytical/FD gradient is available. "
                "Use SciPy's internal finite differences (do not request evaluator gradients)."
            )
        if mode is GradientMode.ANALYTICAL:
            value = self._total(full)
            grad = self._data_gradient(full) + regularization_gradient(
                full, self._plan.reference_params, self._plan.regularization
            )
        else:  # FINITE_DIFFERENCE
            value = self._total(full)
            grad = self._finite_difference_gradient(full)
        self._record(value)
        return value, np.asarray(grad, dtype=float)

    def gradient(self, full_vector: np.ndarray) -> np.ndarray:
        return self.value_and_gradient(full_vector)[1]

    def _finite_difference_gradient(self, full_vector: np.ndarray) -> np.ndarray:
        """Central finite-difference gradient of ``_total`` (includes reg).

        Sub-evaluations are tracked in :attr:`n_gradient_evaluations` so the
        finite-difference cost is visible without polluting the main
        evaluation count or history.
        """
        step = self._fd_step
        n = len(full_vector)
        grad = np.zeros(n, dtype=float)
        for j in range(n):
            plus = full_vector.copy()
            plus[j] += step
            minus = full_vector.copy()
            minus[j] -= step
            f_plus = self._total(plus)
            f_minus = self._total(minus)
            self._n_gradient_eval += 2
            grad[j] = (f_plus - f_minus) / (2.0 * step)
        return grad
