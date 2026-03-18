"""Abstract base classes for QM and MM engine backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class QMEngine(ABC):
    """Abstract base class for quantum mechanics engines.

    All QM backends (Psi4, Gaussian, ORCA, etc.) must implement this interface.
    """

    @abstractmethod
    def energy(self, structure, method: str = "b3lyp", basis: str = "def2-svp") -> float:
        """Calculate single-point energy in Hartrees."""
        ...

    @abstractmethod
    def hessian(self, structure, method: str = "b3lyp", basis: str = "def2-svp") -> np.ndarray:
        """Calculate Hessian matrix (second derivatives of energy)."""
        ...

    @abstractmethod
    def optimize(self, structure, method: str = "b3lyp", basis: str = "def2-svp"):
        """Optimize geometry. Returns optimized structure."""
        ...

    @abstractmethod
    def frequencies(self, structure, method: str = "b3lyp", basis: str = "def2-svp") -> list[float]:
        """Calculate vibrational frequencies in cm^-1."""
        ...

    def is_available(self) -> bool:
        """Check if this engine is installed and accessible."""
        return False

    def supports_runtime_params(self) -> bool:
        """Whether parameters can be updated without rebuilding engine state."""
        return False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...


class MMEngine(ABC):
    """Abstract base class for molecular mechanics engines.

    All MM backends (Tinker, OpenMM, MacroModel, etc.) must implement this interface.
    """

    @abstractmethod
    def energy(self, structure, forcefield) -> float:
        """Calculate MM energy."""
        ...

    @abstractmethod
    def minimize(self, structure, forcefield):
        """Energy-minimize structure. Returns minimized structure."""
        ...

    @abstractmethod
    def hessian(self, structure, forcefield) -> np.ndarray:
        """Calculate MM Hessian matrix."""
        ...

    def is_available(self) -> bool:
        """Check if this engine is installed and accessible."""
        return False

    def supports_runtime_params(self) -> bool:
        """Whether parameters can be updated without rebuilding engine state."""
        return False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...
