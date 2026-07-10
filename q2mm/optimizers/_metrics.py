"""Shared numeric helpers for optimizer result types."""

from __future__ import annotations


def fractional_improvement(initial: float, final: float) -> float:
    """Fractional improvement of a score (0 = no change, 1 = perfect).

    Args:
        initial: Objective value before optimization (``initial_score``).
        final: Objective value after optimization (``final_score``).

    Returns:
        ``(initial - final) / initial``, or ``0.0`` when ``initial`` is zero.

    """
    if initial == 0:
        return 0.0
    return (initial - final) / initial
