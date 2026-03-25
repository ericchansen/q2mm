"""Abstract base classes for QM and MM engine backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Q2MMMolecule


def coerce_molecule(structure: Any, *, engine_name: str = "MMEngine") -> Q2MMMolecule:
    """Coerce *structure* to a :class:`Q2MMMolecule`.

    Handles common input patterns across MM backends:

    - :class:`Q2MMMolecule` — returned as-is.
    - Engine handle with a ``.molecule`` attribute — returns the molecule.
    - ``str`` / ``Path`` — loaded via :meth:`Q2MMMolecule.from_xyz`.

    Args:
        structure: Input to coerce.
        engine_name: Name shown in ``TypeError`` messages.

    Returns:
        Q2MMMolecule: The coerced molecule.

    Raises:
        TypeError: If *structure* cannot be coerced.

    """
    from q2mm.models.molecule import Q2MMMolecule

    if isinstance(structure, Q2MMMolecule):
        return structure
    # Duck-typed handle support (JaxHandle, JaxMDHandle, etc.)
    mol = getattr(structure, "molecule", None)
    if mol is not None and isinstance(mol, Q2MMMolecule):
        return mol
    if isinstance(structure, (str, Path)):
        return Q2MMMolecule.from_xyz(structure)
    raise TypeError(
        f"{engine_name} expects a Q2MMMolecule, compatible handle, or XYZ path; got {type(structure).__name__}."
    )


class QMEngine(ABC):
    """Abstract base class for quantum mechanics engines.

    All QM backends (Psi4, Gaussian, ORCA, etc.) must implement this interface.
    """

    @abstractmethod
    def energy(self, structure: Q2MMMolecule, method: str = "b3lyp", basis: str = "def2-svp") -> float:
        """Calculate single-point energy in Hartrees.

        Args:
            structure: Molecular structure.
            method: QM method or functional (e.g. ``"b3lyp"``, ``"mp2"``).
            basis: Basis set name (e.g. ``"def2-svp"``, ``"6-31+G(d)"``).

        Returns:
            Electronic energy in Hartrees.

        """
        ...

    @abstractmethod
    def hessian(self, structure: Q2MMMolecule, method: str = "b3lyp", basis: str = "def2-svp") -> np.ndarray:
        """Calculate Hessian matrix (second derivatives of energy).

        Args:
            structure: Molecular structure.
            method: QM method or functional.
            basis: Basis set name.

        Returns:
            Shape ``(3N, 3N)`` Hessian in **Hartree/Bohr²** (atomic units).

        """
        ...

    @abstractmethod
    def optimize(self, structure: Q2MMMolecule, method: str = "b3lyp", basis: str = "def2-svp") -> tuple:
        """Optimize geometry.

        Args:
            structure: Molecular structure.
            method: QM method or functional.
            basis: Basis set name.

        Returns:
            Optimized structure as ``(energy, atoms, coordinates)``.

        """
        ...

    @abstractmethod
    def frequencies(self, structure: Q2MMMolecule, method: str = "b3lyp", basis: str = "def2-svp") -> list[float]:
        """Calculate vibrational frequencies in cm⁻¹.

        Args:
            structure: Molecular structure.
            method: QM method or functional.
            basis: Basis set name.

        Returns:
            Vibrational frequencies in cm⁻¹.

        """
        ...

    def is_available(self) -> bool:
        """Check if this engine is installed and accessible.

        Returns:
            bool: ``True`` if the engine binary or library can be located.

        """
        return False

    def supports_runtime_params(self) -> bool:
        """Whether parameters can be updated without rebuilding engine state.

        Returns:
            bool: ``True`` if the engine supports in-place parameter updates.

        """
        return False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name.

        Returns:
            str: Engine display name (e.g. ``"Psi4 (b3lyp/def2-svp)"``).

        """
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
    def energy(self, structure: Q2MMMolecule, forcefield: ForceField) -> float:
        """Calculate MM energy in kcal/mol.

        Args:
            structure: Molecular structure.  Concrete engines may widen
                this to accept engine-specific handles (e.g.
                ``OpenMMHandle``) per the Liskov Substitution Principle.
            forcefield: Force field parameters.

        Returns:
            Potential energy in kcal/mol.

        """
        ...

    @abstractmethod
    def minimize(self, structure: Q2MMMolecule, forcefield: ForceField) -> tuple:
        """Energy-minimize structure.

        Args:
            structure: Molecular structure.
            forcefield: Force field parameters.

        Returns:
            ``(energy, atoms, coordinates)`` tuple.

        """
        ...

    @abstractmethod
    def hessian(self, structure: Q2MMMolecule, forcefield: ForceField) -> np.ndarray:
        """Calculate MM Hessian matrix.

        Args:
            structure: Molecular structure.
            forcefield: Force field parameters.

        Returns:
            Shape ``(3N, 3N)`` Hessian in **Hartree/Bohr²** (atomic units).

        """
        ...

    def frequencies(self, structure: Q2MMMolecule, forcefield: ForceField) -> list[float]:
        """Calculate vibrational frequencies in cm⁻¹.

        Default implementation: compute Hessian via :meth:`hessian` then
        convert to frequencies via :func:`~q2mm.models.hessian.hessian_to_frequencies`.
        Engines with specialised needs (e.g. path-based inputs) may override.

        Args:
            structure: Molecular structure (or engine-specific handle).
            forcefield: Force field parameters.

        Returns:
            Vibrational frequencies in cm⁻¹.

        """
        from q2mm.models.hessian import hessian_to_frequencies

        hess_au = self.hessian(structure, forcefield)
        mol = coerce_molecule(structure, engine_name=self.__class__.__name__)
        return hessian_to_frequencies(hess_au, list(mol.symbols))

    def is_available(self) -> bool:
        """Check if this engine is installed and accessible.

        Returns:
            bool: ``True`` if the engine binary or library can be located.

        """
        return False

    def supports_runtime_params(self) -> bool:
        """Whether parameters can be updated without rebuilding engine state.

        Returns:
            bool: ``True`` if the engine supports in-place parameter updates.

        """
        return False

    def supports_analytical_gradients(self) -> bool:
        """Whether this engine provides analytical parameter gradients.

        Engines returning ``True`` must implement ``energy_and_param_grad()``.

        Returns:
            bool: ``True`` if the engine provides analytical parameter gradients.

        """
        return False

    def energy_and_param_grad(self, structure: Q2MMMolecule, forcefield: ForceField) -> tuple[float, np.ndarray]:
        """Compute energy and analytical gradient w.r.t. MM parameters.

        Must be implemented by engines for which
        :meth:`supports_analytical_gradients` returns ``True``.

        Args:
            structure: Molecular structure or engine-specific context.
            forcefield: Force field parameters.

        Returns:
            ``(energy, grad)`` where ``energy`` is the MM energy in
            kcal/mol and ``grad`` is a 1-D array of ``dE/dp`` derivatives.

        Raises:
            NotImplementedError: If the engine does not support analytical
                gradients.

        """
        raise NotImplementedError(
            f"{self.name} does not implement energy_and_param_grad(). "
            "Override this method when supports_analytical_gradients() returns True."
        )

    def create_context(self, structure: Q2MMMolecule, forcefield: ForceField) -> object:
        """Create a reusable engine context/handle for a molecule.

        Only needed when :meth:`supports_runtime_params` returns ``True``.
        The returned handle can be passed as ``structure`` to other methods,
        allowing the engine to update parameters in-place rather than
        rebuilding the simulation state each evaluation.

        Args:
            structure: Molecular structure.
            forcefield: Force field parameters.

        Returns:
            An engine-specific handle object for reuse across evaluations.

        Raises:
            NotImplementedError: If the engine does not support reusable
                contexts.

        """
        raise NotImplementedError(
            f"{self.name} does not support reusable contexts. "
            "Override create_context() if supports_runtime_params() returns True."
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name.

        Returns:
            str: Engine display name (e.g. ``"OpenMM"``, ``"Tinker"``).

        """
        ...

    def supported_functional_forms(self) -> frozenset[str]:
        """Functional forms this engine can evaluate.

        Override in subclasses; the default is all forms (no restriction).

        Returns:
            frozenset[str]: Set of
                :class:`~q2mm.models.forcefield.FunctionalForm` values
                (as strings).

        Example:
            .. code-block:: python

                # OpenMM supports both harmonic and MM3:
                return frozenset({"harmonic", "mm3"})

                # A future AMOEBA-only engine:
                return frozenset({"amoeba"})

        """
        from q2mm.models.forcefield import FunctionalForm

        return frozenset(f.value for f in FunctionalForm)

    def _validate_forcefield(self, forcefield: ForceField) -> None:
        """Raise ``ValueError`` if the force field's functional form is unsupported.

        Called by engines at the start of ``create_context`` / ``energy`` /
        etc.  Does nothing when ``forcefield.functional_form`` is ``None``
        (legacy / unset).

        Args:
            forcefield: Force field whose ``functional_form`` attribute is
                checked against :meth:`supported_functional_forms`.

        Raises:
            ValueError: If the force field's functional form is not in the
                set returned by :meth:`supported_functional_forms`.

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
