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

    All MM backends (Tinker, OpenMM, MacroModel, etc.) must implement
    this interface.

    **Unit contracts** (canonical units for ForceField parameters):

    - ``bond_k``: kcal/(mol·Å²) — energy convention ``E = k·(r − r₀)²``
    - ``angle_k``: kcal/(mol·rad²)
    - ``torsion_k``, ``vdw_epsilon``: kcal/mol
    - ``bond_eq``, ``vdw_radius``: Å
    - ``angle_eq``: degrees

    Engines convert from canonical units to engine-native at the boundary
    (e.g. kcal → kJ for OpenMM, Å → nm).  Output contracts:

    - ``energy()`` returns kcal/mol
    - ``hessian()`` returns Hartree/Bohr² (atomic units)
    - ``frequencies()`` returns cm⁻¹
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

    def supported_functional_forms(self) -> frozenset[str]:
        """Functional forms this engine can evaluate.

        Returns a frozenset of :class:`~q2mm.models.forcefield.FunctionalForm`
        values (as strings).  Override in subclasses; the default is all
        forms (no restriction).

        Examples::

            # OpenMM supports both harmonic and MM3:
            return frozenset({"harmonic", "mm3"})

            # A future AMOEBA-only engine:
            return frozenset({"amoeba"})
        """
        from q2mm.models.forcefield import FunctionalForm

        return frozenset(f.value for f in FunctionalForm)

    def _validate_forcefield(self, forcefield) -> None:
        """Raise ``ValueError`` if the force field's functional form is unsupported.

        Called by engines at the start of ``create_context`` / ``energy`` /
        etc.  Does nothing when ``forcefield.functional_form`` is ``None``
        (legacy / unset).
        """
        ff_form = getattr(forcefield, "functional_form", None)
        if ff_form is None:
            return
        form_value = ff_form.value if hasattr(ff_form, "value") else str(ff_form)
        supported = self.supported_functional_forms()
        if form_value not in supported:
            raise ValueError(
                f"{self.name} does not support functional form {ff_form!r}. Supported: {sorted(supported)}"
            )
