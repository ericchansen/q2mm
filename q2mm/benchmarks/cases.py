"""Benchmark-case metadata, separate from the scientific optimization core.

:class:`BenchmarkCase` wraps an immutable
:class:`~q2mm.models.problem.OptimizationProblem` with the dataset/
publication/reporting metadata a benchmark runner needs (QM frequencies
for reporting, pre-computed normal modes for PES-distortion analysis,
which functional forms to benchmark by default, human-readable
description) — none of which belongs in the scientific optimization
model itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from q2mm.models.problem import OptimizationProblem
from q2mm.models.results import deep_freeze

__all__ = ["BenchmarkCase"]


@dataclass(frozen=True, eq=False)
class BenchmarkCase:
    """One loaded benchmark system, ready for optimization and reporting.

    Deeply immutable: :attr:`metadata` and :attr:`normal_modes` are
    ``MappingProxyType`` views over dicts private to this instance (never
    the caller's own dict), every ``numpy.ndarray`` reachable from
    :attr:`qm_freqs_per_mol` or :attr:`normal_modes` is a read-only
    defensive copy, and every sequence field is a tuple.

    Attributes:
        key: Registry key (e.g. ``"ch3f"``, ``"rh-enamide"``).
        name: Human-readable system name.
        problem: The immutable :class:`~q2mm.models.problem.OptimizationProblem`
            (training cases, starting force field, parameter layout,
            active space, and observations).
        qm_freqs_per_mol: QM real (non-imaginary/non-rigid) frequencies
            per training case, in case order — for reporting only; not
            part of ``problem.observations``.
        metadata: Extra info (level of theory, publication, DOI,
            starting-point audit, etc.).
        normal_modes: Pre-computed normal-mode eigendecomposition for PES
            distortion analysis, or ``None`` when not available.
        default_forms: Functional forms to benchmark by default.
        description: One-line human-readable description.

    """

    key: str
    name: str
    problem: OptimizationProblem
    qm_freqs_per_mol: tuple[np.ndarray, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    normal_modes: Mapping[str, np.ndarray] | None = None
    default_forms: tuple[str, ...] = ("mm3",)
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "qm_freqs_per_mol", tuple(deep_freeze(arr) for arr in self.qm_freqs_per_mol))
        object.__setattr__(self, "default_forms", tuple(self.default_forms))
        object.__setattr__(self, "metadata", deep_freeze(self.metadata))
        object.__setattr__(
            self, "normal_modes", deep_freeze(self.normal_modes) if self.normal_modes is not None else None
        )
