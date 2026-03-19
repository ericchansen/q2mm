"""Scipy-based force field optimizers for Q2MM.

Provides a clean, composable optimization framework built on
:mod:`scipy.optimize` and the Q2MM clean model layer.
"""

from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

__all__ = [
    "ObjectiveFunction",
    "ReferenceData",
]

# Lazy import: scipy is an optional dependency
try:
    from q2mm.optimizers.scipy_opt import ScipyOptimizer, OptimizationResult  # noqa: F401

    __all__ += ["ScipyOptimizer", "OptimizationResult"]
except ImportError:
    pass
