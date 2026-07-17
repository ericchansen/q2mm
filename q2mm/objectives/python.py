"""Python-dispatch objective executor over the typed backend contract.

:class:`PythonObjectiveExecutor` evaluates the objective by dispatching
typed :class:`~q2mm.backends.contracts.PreparedBackend` requests — one
prepared session per stable case ID, Python dispatch, shared residual /
regularization / metric semantics from :mod:`q2mm.objectives.metrics`.

It works with any backend that implements the prepared-session contract
(OpenMM, Tinker, JAX, ...).  Its declared :attr:`gradient_mode` is chosen
by the caller:

- :attr:`~q2mm.objectives.protocols.GradientMode.NONE` (default): only
  scalar values are produced; a gradient-based optimizer must supply its
  own gradients (e.g. SciPy's internal finite differences).
- :attr:`~q2mm.objectives.protocols.GradientMode.ANALYTICAL`: exact
  gradients via the backend's parameter-gradient / Hessian-parameter
  Jacobian.  Requested categories that have no analytical path (geometry)
  or that the backend does not declare raise
  :class:`~q2mm.objectives.protocols.ObjectiveGradientError` — there is no
  silent fallback.
- :attr:`~q2mm.objectives.protocols.GradientMode.FINITE_DIFFERENCE`:
  explicit central finite-difference gradients (see the base class).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from q2mm.backends.contracts import (
    Backend,
    Capability,
    EnergyRequest,
    EnergyUnit,
    FrequencyRequest,
    HessianJacobianRequest,
    HessianRequest,
    MinimizationRequest,
    ParameterGradientRequest,
    PreparationRequest,
    PreparedBackend,
    UnsupportedCapabilityError,
)
from q2mm.models.forcefield import ForceField
from q2mm.models.observations import (
    Observation,
    ParameterTetherObservation,
    RelativeEnergyObservation,
    ThermodynamicQuantity,
    energy_conversion_from_kcal_per_mol,
)
from q2mm.objectives._base import BaseObjectiveExecutor
from q2mm.objectives._observables import extract_calc_value, geometry_computed
from q2mm.objectives.plan import KIND_TO_CATEGORY, ObjectivePlan
from q2mm.objectives.protocols import (
    GradientMode,
    ObjectiveGradientError,
    UnsupportedObservationError,
)

__all__ = ["PythonObjectiveExecutor"]

_GEOMETRY_KINDS = frozenset({"bond_length", "bond_angle", "torsion_angle"})
_EIGENMATRIX_KINDS = frozenset({"eig_diagonal", "eig_offdiagonal"})
_UNSUPPORTED_KINDS = frozenset(
    {
        "atomic_partial_charge",
        "direct_electrostatic_potential",
        "scan_energy",
    }
)


class PythonObjectiveExecutor(BaseObjectiveExecutor):
    """Python-dispatch objective executor over the typed backend contract.

    Args:
        plan: The immutable objective plan.
        backend: MM backend implementing the prepared-session contract.
        base_force_field: Base force field supplying the topology/structure
            for prepared sessions.  Parameter *values* are always taken
            from the evaluated vector via ``layout.replace``; only the
            structure of *base_force_field* is used.
        gradient_mode: Declared gradient capability (see module docstring).

    Raises:
        ObjectiveGradientError: If ``gradient_mode`` is ``ANALYTICAL`` but a
            required category has no analytical path or the backend does
            not declare the needed capability.

    """

    #: Minimizer iteration cap for geometry references.  Set high so the
    #: relaxed geometry converges to the same MM minimum the JAX executor's
    #: implicit-diff relaxation reaches, keeping calculated geometry
    #: observables in tight cross-executor parity.
    geometry_minimize_iterations: int = 2000

    def __init__(
        self,
        plan: ObjectivePlan,
        backend: Backend,
        base_force_field: ForceField,
        *,
        gradient_mode: GradientMode = GradientMode.NONE,
        fd_step: float = 1e-4,
    ) -> None:
        super().__init__(plan, fd_step=fd_step)
        self._backend = backend
        self._base_ff = base_force_field
        self._gradient_mode = gradient_mode
        #: Error handling for eigendecomposition in frequency evaluation.
        #: ``"raise"`` (default) propagates; ``"penalty"`` returns large
        #: penalty frequencies so the optimizer retreats.
        self.on_error: str = "raise"
        self._prepared: dict[str, PreparedBackend] = {}
        self._qm_eigenvectors: dict[str, np.ndarray] = {}

        self._validate_observation_support()
        if gradient_mode is GradientMode.ANALYTICAL:
            self._validate_analytical_support()

    @property
    def backend(self) -> Backend:
        """The MM backend this executor evaluates against."""
        return self._backend

    @property
    def base_force_field(self) -> ForceField:
        """The base force field supplying prepared-session structure."""
        return self._base_ff

    @property
    def gradient_mode(self) -> GradientMode:
        """The declared gradient capability for this executor."""
        return self._gradient_mode

    # -- capability validation --------------------------------------------

    def _validate_observation_support(self) -> None:
        unsupported = {str(observation.kind) for observation in self._plan.observations.values} & _UNSUPPORTED_KINDS
        if unsupported:
            raise UnsupportedObservationError(
                type(self).__name__,
                unsupported,
                "the MM backend contract exposes no calculated atomic-charge, direct-ESP, or constrained-scan result",
            )
        if any(
            isinstance(observation, RelativeEnergyObservation)
            and observation.quantity is ThermodynamicQuantity.ENTHALPY
            for observation in self._plan.observations.values
        ):
            raise UnsupportedObservationError(
                type(self).__name__,
                {"relative_enthalpy"},
                "the MM backend contract exposes potential energy, not thermochemical enthalpy",
            )

    def _validate_analytical_support(self) -> None:
        categories = self._plan.categories
        info = self._backend.info
        if "geometry" in categories:
            raise ObjectiveGradientError(
                "Analytical gradients are not available for geometry references "
                "(bond_length/bond_angle/torsion_angle) through the Python executor. "
                "Use a JaxObjectiveExecutor for differentiable geometry, or request finite differences."
            )
        if categories & {"energy", "relative_energy"} and not info.supports(Capability.PARAMETER_GRADIENT):
            raise ObjectiveGradientError(
                f"Backend {info.name!r} does not declare PARAMETER_GRADIENT; "
                "analytical energy gradients are unavailable."
            )
        needs_jacobian = categories & {"frequency", "eigenmatrix", "hessian"}
        if needs_jacobian and not info.supports(Capability.HESSIAN_PARAMETER_JACOBIAN):
            raise ObjectiveGradientError(
                f"Backend {info.name!r} does not declare HESSIAN_PARAMETER_JACOBIAN; "
                f"analytical gradients for {sorted(needs_jacobian)} are unavailable."
            )

    # -- prepared sessions -------------------------------------------------

    def _prepared_for(self, case_id: str) -> PreparedBackend:
        session = self._prepared.get(case_id)
        if session is None:
            idx = self._plan.case_index(case_id)
            session = self._backend.prepare(
                PreparationRequest(
                    case_id=case_id,
                    molecule=self._plan.molecules[idx],
                    force_field=self._base_ff,
                )
            )
            self._prepared[case_id] = session
        return session

    # -- per-case computed observables ------------------------------------

    def _compute_case(self, case_id: str, full: np.ndarray, needed: set[str]) -> dict:
        prepared = self._prepared_for(case_id)
        mol = prepared.molecule
        computed: dict = {}

        if "energy" in needed:
            computed["energy"] = float(prepared.energy(EnergyRequest(parameters=full)).energy)

        if needed & _GEOMETRY_KINDS:
            coords = np.asarray(
                prepared.minimize(
                    MinimizationRequest(parameters=full, max_iterations=self.geometry_minimize_iterations)
                ).coordinates
            )
            computed.update(geometry_computed(mol, coords, needed))

        # Hessian-derived observables (frequency / eigenmatrix / hessian
        # element) share ONE MM Hessian per case per vector.  When a
        # frequency reference coexists with an eigenmatrix/Hessian-element
        # reference, derive the frequencies from that same Hessian instead
        # of triggering a second Hessian evaluation.
        hess_based = needed & (_EIGENMATRIX_KINDS | {"hessian_element"})
        if hess_based:
            mm_hess = np.asarray(prepared.hessian(HessianRequest(parameters=full)).hessian)
            if "frequency" in needed:
                from q2mm.models.hessian import hessian_to_frequencies

                computed["frequencies"] = np.asarray(
                    hessian_to_frequencies(mm_hess, list(mol.symbols), on_error=self.on_error)  # type: ignore[arg-type]
                )
            if "hessian_element" in needed:
                computed["raw_hessian"] = mm_hess
            if needed & _EIGENMATRIX_KINDS:
                from q2mm.models.hessian import mass_weighted_eigenmatrix, mass_weighted_normal_modes

                if case_id not in self._qm_eigenvectors:
                    if mol.hessian is None:
                        raise ValueError(
                            f"Case {case_id!r} ({mol.name}) has no QM Hessian; eigenmatrix training requires one."
                        )
                    _, qm_evecs = mass_weighted_normal_modes(mol.hessian, mol.symbols)
                    self._qm_eigenvectors[case_id] = qm_evecs
                computed["eigenmatrix"] = mass_weighted_eigenmatrix(
                    mm_hess, self._qm_eigenvectors[case_id], mol.symbols
                )
        elif "frequency" in needed:
            fr = prepared.frequencies(FrequencyRequest(parameters=full, on_error=self.on_error))
            computed["frequencies"] = np.asarray(fr.frequencies)

        return computed

    def _calculated(self, full_vector: np.ndarray) -> np.ndarray:
        observations = self._plan.observations.values
        by_case: dict[str, list[int]] = defaultdict(list)
        relative_indices: list[int] = []
        tether_indices: list[int] = []
        for gi, obs in enumerate(observations):
            if isinstance(obs, RelativeEnergyObservation):
                relative_indices.append(gi)
            elif isinstance(obs, ParameterTetherObservation):
                tether_indices.append(gi)
            else:
                by_case[obs.case_id].append(gi)

        calc = np.empty(len(observations), dtype=float)
        energy_cache: dict[str, float] = {}

        def case_energy(case_id: str) -> float:
            if case_id not in energy_cache:
                result = self._prepared_for(case_id).energy(EnergyRequest(parameters=full_vector))
                if result.unit is not EnergyUnit.KCAL_PER_MOL:
                    raise ValueError("MM relative-energy evaluation requires canonical kcal/mol backend results.")
                energy_cache[case_id] = float(result.energy)
            return energy_cache[case_id]

        for gi in relative_indices:
            observation = observations[gi]
            assert isinstance(observation, RelativeEnergyObservation)
            delta_kcal = case_energy(observation.case_id) - case_energy(observation.reference_case_id)
            calc[gi] = delta_kcal * energy_conversion_from_kcal_per_mol(observation.unit)
        for gi in tether_indices:
            observation = observations[gi]
            assert isinstance(observation, ParameterTetherObservation)
            calc[gi] = float(full_vector[self._plan.layout.index_of(observation.parameter_id)])
        for case_id, indices in by_case.items():
            needed: set[str] = {str(observations[gi].kind) for gi in indices}
            computed = self._compute_case(case_id, full_vector, needed)
            for gi in indices:
                observation = observations[gi]
                assert isinstance(observation, Observation)
                calc[gi] = extract_calc_value(computed, observation)
        return calc

    # -- analytical gradient ----------------------------------------------

    def _data_gradient(self, full_vector: np.ndarray) -> np.ndarray:
        observations = self._plan.observations.values
        n_params = self._plan.n_params
        total = np.zeros(n_params, dtype=float)

        gradient_cache: dict[str, tuple[float, np.ndarray]] = {}

        def case_energy_gradient(case_id: str) -> tuple[float, np.ndarray]:
            if case_id not in gradient_cache:
                prepared = self._prepared_for(case_id)
                if not prepared.info.supports(Capability.PARAMETER_GRADIENT):
                    raise UnsupportedCapabilityError(prepared.info.name, Capability.PARAMETER_GRADIENT)
                result = prepared.parameter_gradient(ParameterGradientRequest(parameters=full_vector))
                gradient_cache[case_id] = (float(result.energy), np.asarray(result.gradient))
            return gradient_cache[case_id]

        by_case_category: dict[str, dict[str, list[Observation]]] = defaultdict(lambda: defaultdict(list))
        for obs in observations:
            if isinstance(obs, RelativeEnergyObservation):
                case_energy, case_gradient = case_energy_gradient(obs.case_id)
                reference_energy, reference_gradient = case_energy_gradient(obs.reference_case_id)
                factor = energy_conversion_from_kcal_per_mol(obs.unit)
                calculated = factor * (case_energy - reference_energy)
                derivative = factor * (case_gradient - reference_gradient)
                total += -2.0 * obs.weight**2 * (obs.value - calculated) * derivative
            elif isinstance(obs, ParameterTetherObservation):
                index = self._plan.layout.index_of(obs.parameter_id)
                total[index] += -2.0 * obs.weight**2 * (obs.value - full_vector[index])
            else:
                assert isinstance(obs, Observation)
                by_case_category[obs.case_id][KIND_TO_CATEGORY[obs.kind]].append(obs)

        for case_id, cats in by_case_category.items():
            prepared = self._prepared_for(case_id)
            jac_cache: dict[str, np.ndarray] = {}
            for category, refs in cats.items():
                if category == "energy":
                    total += self._energy_gradient(prepared, full_vector, refs, n_params)
                elif category == "geometry":
                    raise ObjectiveGradientError("Analytical gradients are not available for geometry references.")
                else:  # frequency / eigenmatrix / hessian
                    total += self._hessian_based_gradient(prepared, full_vector, category, refs, n_params, jac_cache)
        return total

    @staticmethod
    def _energy_gradient(
        prepared: PreparedBackend, full: np.ndarray, refs: list[Observation], n_params: int
    ) -> np.ndarray:
        if not prepared.info.supports(Capability.PARAMETER_GRADIENT):
            raise UnsupportedCapabilityError(prepared.info.name, Capability.PARAMETER_GRADIENT)
        result = prepared.parameter_gradient(ParameterGradientRequest(parameters=full))
        calc_energy = float(result.energy)
        de_dp = np.asarray(result.gradient)
        grad = np.zeros(n_params, dtype=float)
        for ref in refs:
            diff = ref.value - calc_energy
            grad += -2.0 * ref.weight**2 * diff * de_dp
        return grad

    def _hessian_based_gradient(
        self,
        prepared: PreparedBackend,
        full: np.ndarray,
        category: str,
        refs: list[Observation],
        n_params: int,
        jac_cache: dict[str, np.ndarray],
    ) -> np.ndarray:
        if not prepared.info.supports(Capability.HESSIAN_PARAMETER_JACOBIAN):
            raise UnsupportedCapabilityError(prepared.info.name, Capability.HESSIAN_PARAMETER_JACOBIAN)
        if "hessian" not in jac_cache:
            result = prepared.hessian_parameter_jacobian(HessianJacobianRequest(parameters=full))
            jac_cache["hessian"] = np.asarray(result.hessian)
            jac_cache["jacobian"] = np.asarray(result.jacobian)
        hess = jac_cache["hessian"]
        dH_dp = jac_cache["jacobian"]
        grad = np.zeros(n_params, dtype=float)

        if category == "frequency":
            from q2mm.models.hessian import frequency_param_jacobian

            freqs, d_freq_dp = frequency_param_jacobian(hess, dH_dp, prepared.molecule.symbols)
            for ref in refs:
                if ref.data_idx < 0 or ref.data_idx >= len(freqs):
                    raise IndexError(f"Frequency data_idx={ref.data_idx} out of range. Label: {ref.label!r}")
                diff = ref.value - freqs[ref.data_idx]
                grad += -2.0 * ref.weight**2 * diff * d_freq_dp[ref.data_idx, :]
            return grad

        if category == "hessian":
            n = hess.shape[0]
            for ref in refs:
                if ref.atom_indices is None or len(ref.atom_indices) < 2:
                    raise ValueError(f"hessian_element requires atom_indices=(row, col). Label: {ref.label!r}")
                row, col = ref.atom_indices[:2]
                if row < 0 or row >= n or col < 0 or col >= n:
                    raise IndexError(f"Hessian indices ({row}, {col}) out of range. Label: {ref.label!r}")
                diff = ref.value - float(hess[row, col])
                grad += -2.0 * ref.weight**2 * diff * dH_dp[row, col, :]
            return grad

        # eigenmatrix
        from q2mm.models.hessian import mass_weight_scale_3n, mass_weighted_normal_modes

        mol = prepared.molecule
        case_id = prepared.case_id
        if case_id not in self._qm_eigenvectors:
            if mol.hessian is None:
                raise ValueError(f"Case {case_id!r} ({mol.name}) has no QM Hessian.")
            _, qm_evecs = mass_weighted_normal_modes(mol.hessian, mol.symbols)
            self._qm_eigenvectors[case_id] = qm_evecs
        qm_evecs = self._qm_eigenvectors[case_id]
        scale = mass_weight_scale_3n(mol.symbols)
        hess_mw = hess * scale
        dH_dp_mw = dH_dp * scale[:, :, None]
        d_eigmat_dp = np.einsum("ir,ijp,jc->rcp", qm_evecs, dH_dp_mw, qm_evecs)
        eigmat = qm_evecs.T @ hess_mw @ qm_evecs
        n = eigmat.shape[0]
        for ref in refs:
            if ref.kind == "eig_diagonal":
                idx = ref.data_idx
                if idx < 0 or idx >= n:
                    raise IndexError(f"Eigenmatrix data_idx={idx} out of range. Label: {ref.label!r}")
                diff = ref.value - float(eigmat[idx, idx])
                grad += -2.0 * ref.weight**2 * diff * d_eigmat_dp[idx, idx, :]
            else:
                if ref.atom_indices is None or len(ref.atom_indices) < 2:
                    raise ValueError(f"eig_offdiagonal requires atom_indices=(row, col). Label: {ref.label!r}")
                row, col = ref.atom_indices[:2]
                if row < 0 or row >= n or col < 0 or col >= n:
                    raise IndexError(f"Off-diagonal indices ({row}, {col}) out of range. Label: {ref.label!r}")
                diff = ref.value - float(eigmat[row, col])
                grad += -2.0 * ref.weight**2 * diff * d_eigmat_dp[row, col, :]
        return grad
