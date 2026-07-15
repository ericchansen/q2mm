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
from types import MappingProxyType
from typing import Any

import numpy as np

from q2mm.models.problem import OptimizationProblem

__all__ = ["BenchmarkCase"]


def _freeze(value: Any) -> Any:
    """Recursively convert *value* into an immutable equivalent.

    - ``numpy.ndarray`` -> a defensive, read-only copy (breaks aliasing
      with the caller's array *and* rejects in-place mutation via
      ``arr[:] = ...`` on the stored copy itself).
    - ``Mapping`` (e.g. ``dict``) -> a ``MappingProxyType`` wrapping a
      *new* dict built from recursively-frozen values (never the
      caller's own dict object, so later mutation of the caller's dict
      cannot leak through).
    - ``list``/``tuple`` -> a tuple of recursively-frozen elements.
    - Anything else (``str``, ``int``, ``float``, ``bool``, ``None``,
      enums, other already-immutable value objects) is returned as-is.
    """
    if isinstance(value, np.ndarray):
        frozen = np.array(value, copy=True)
        frozen.setflags(write=False)
        return frozen
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


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
        object.__setattr__(self, "qm_freqs_per_mol", tuple(_freeze(arr) for arr in self.qm_freqs_per_mol))
        object.__setattr__(self, "default_forms", tuple(self.default_forms))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        object.__setattr__(self, "normal_modes", _freeze(self.normal_modes) if self.normal_modes is not None else None)
