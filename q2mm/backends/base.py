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
        """Calculate Hessian matrix (second derivatives of energy).

        Returns
        -------
        np.ndarray
            Shape ``(3N, 3N)`` Hessian in **Hartree/Bohr²** (atomic units).
        """
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
        """Calculate MM energy in kcal/mol."""
        ...

    @abstractmethod
    def minimize(self, structure, forcefield):
        """Energy-minimize structure. Returns minimized structure."""
        ...

    @abstractmethod
    def hessian(self, structure, forcefield) -> np.ndarray:
        """Calculate MM Hessian matrix.

        Returns
        -------
        np.ndarray
            Shape ``(3N, 3N)`` Hessian in **Hartree/Bohr²** (atomic units).
            Implementors must convert from engine-native units before
            returning (e.g. OpenMM kJ/mol/nm² → Hartree/Bohr²).
        """
        ...

    @abstractmethod
    def frequencies(self, structure, forcefield) -> list[float]:
        """Calculate vibrational frequencies in cm^-1."""
        ...

    def is_available(self) -> bool:
        """Check if this engine is installed and accessible."""
        return False

    def supports_runtime_params(self) -> bool:
        """Whether parameters can be updated without rebuilding engine state."""
        return False

    def supports_analytical_gradients(self) -> bool:
        """Whether this engine provides analytical parameter gradients.

        Engines returning ``True`` must implement ``energy_and_param_grad()``.
        """
        return False

    def energy_and_param_grad(self, structure, forcefield) -> tuple[float, np.ndarray]:
        """Compute energy and analytical gradient w.r.t. MM parameters.

        Must be implemented by engines for which
        :meth:`supports_analytical_gradients` returns ``True``.

        Parameters
        ----------
        structure
            Molecular structure or engine-specific context.
        forcefield
            Force field or parameter set used for the calculation.

        Returns
        -------
        tuple[float, np.ndarray]
            ``(energy, grad)`` where ``energy`` is the MM energy in kcal/mol
            and ``grad`` is a 1-D array of ``dE/dp`` derivatives.
        """
        raise NotImplementedError(
            f"{self.name} does not implement energy_and_param_grad(). "
            "Override this method when supports_analytical_gradients() returns True."
        )

    def create_context(self, structure, forcefield):
        """Create a reusable engine context/handle for a molecule.

        Only needed when :meth:`supports_runtime_params` returns ``True``.
        The returned handle can be passed as ``structure`` to other methods,
        allowing the engine to update parameters in-place rather than
        rebuilding the simulation state each evaluation.
        """
        raise NotImplementedError(
            f"{self.name} does not support reusable contexts. "
            "Override create_context() if supports_runtime_params() returns True."
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...
