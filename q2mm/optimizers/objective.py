"""Objective function for force field optimization.

Wraps the ForceField ↔ MM-backend ↔ reference-data loop into a single
callable that :func:`scipy.optimize.minimize` can drive.

Scoring approach
----------------
This module uses **raw weighted residuals**:

.. math:: r_i = w_i (x_{ref,i} - x_{calc,i})

The objective value is ``sum(r_i**2)``.  This is the standard form expected
by ``scipy.optimize.least_squares`` and gradient-based minimizers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.backends.contracts import (
    Backend,
    BatchedEnergyRequest,
    Capability,
    PreparationRequest,
    PreparedBackend,
)
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Molecule
from q2mm.models.observations import Observation, ObservationSet
from q2mm.models.parameters import ParameterLayout

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from q2mm.optimizers.spec import ObjectiveSpec


# ---- Per-data-type evaluators (lazy-imported in ObjectiveFunction.__init__) ----


# ---- Objective function ----


class ObjectiveFunction:
    """Objective function for scipy-based force field optimization.

    Evaluates the weighted sum-of-squares between MM-calculated and
    reference data for one or more molecules.

    Args:
        forcefield (ForceField): The force field whose parameters are
            being optimized.
        backend (Backend): The MM backend (OpenMM, Tinker, etc.).
        molecules (list[Molecule]): Training set molecules.
        reference (ObservationSet): QM/experimental reference observations.
        layout (ParameterLayout | None): The full-vector layout for
            *forcefield* — required whenever *forcefield* is given, since
            :class:`~q2mm.models.forcefield.ForceField` no longer exposes
            vector extraction/replacement itself.

    """

    def __init__(
        self,
        forcefield: ForceField | None,
        backend: Backend,
        molecules: list[Molecule],
        reference: ObservationSet,
        *,
        case_ids: Sequence[str] | None = None,
        layout: ParameterLayout | None = None,
        regularization: float = 0.0,
        reference_params: np.ndarray | None = None,
    ) -> None:
        """Initialize the objective function.

        Args:
            forcefield (ForceField | None): The force field whose
                parameters are being optimized.
            backend (Backend): The MM backend (OpenMM, Tinker, JAX, ...).
                One prepared session is built per training case.
            molecules (list[Molecule]): Training set molecules.
            reference (ObservationSet): QM/experimental reference
                observations.
            case_ids (Sequence[str] | None): Stable case ID for each
                entry of *molecules*, in the same order.  When ``None``
                (default), positions are auto-labelled ``"0"``, ``"1"``, ...
            layout (ParameterLayout | None): Layout for *forcefield*'s
                scalar parameters.  Required whenever *forcefield* is
                not ``None``.
            regularization: L2 penalty strength (λ).
            reference_params: Parameter vector treated as the L2 "anchor."

        Raises:
            ValueError: If *forcefield* is given without *layout*, if
                *reference_params* has the wrong shape/length, if
                *case_ids* has a different length than *molecules* or
                contains duplicates, or if *regularization* is negative
                or requires a forcefield/reference_params that were not
                supplied.

        """
        if forcefield is not None and layout is None:
            raise ValueError("layout is required whenever forcefield is provided.")
        self.forcefield = forcefield
        self.layout = layout
        self.backend = backend
        self.molecules = molecules
        self.reference = reference

        resolved_case_ids = [str(i) for i in range(len(molecules))] if case_ids is None else list(case_ids)
        if len(resolved_case_ids) != len(molecules):
            raise ValueError(
                f"case_ids length ({len(resolved_case_ids)}) must match molecules length ({len(molecules)})."
            )
        if len(set(resolved_case_ids)) != len(resolved_case_ids):
            raise ValueError(f"case_ids must be unique, got {resolved_case_ids}.")
        self.case_ids: tuple[str, ...] = tuple(resolved_case_ids)
        self._case_id_to_index: dict[str, int] = {cid: i for i, cid in enumerate(self.case_ids)}
        self.n_eval = 0
        self.fd_step = 1e-4
        self.history: list[float] = []
        self.regularization = float(regularization)
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative")
        if reference_params is not None:
            self._reference_params = np.asarray(reference_params, dtype=float)
            if self._reference_params.ndim != 1:
                raise ValueError("reference_params must be a 1-D vector")
            if layout is not None and len(self._reference_params) != len(layout):
                raise ValueError(
                    f"reference_params length ({len(self._reference_params)}) does not match layout ({len(layout)})"
                )
        elif forcefield is not None:
            assert layout is not None  # guaranteed by the check above
            self._reference_params = layout.vector(forcefield)
        else:
            if self.regularization > 0:
                raise ValueError("regularization > 0 requires a forcefield or explicit reference_params")
            self._reference_params = np.array([], dtype=float)
        #: Error handling for eigendecomposition in frequency evaluation.
        #: ``"raise"`` (default) propagates exceptions; ``"penalty"``
        #: returns large penalty frequencies so the optimizer retreats.
        self.on_error: str = "raise"
        # One prepared backend session per case ID (built lazily, reused).
        # The prepared session owns the molecule, base force field, layout,
        # and any reusable native state.
        self._prepared: dict[int, PreparedBackend] = {}
        # Per-data-type evaluator instances (created once, reused).
        # Lazy imports to break circular dependency (evaluators import
        # Observation from this module).
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator
        from q2mm.optimizers.evaluators.hessian_element import HessianElementEvaluator

        self._evaluators = [
            EnergyEvaluator(),
            FrequencyEvaluator(),
            GeometryEvaluator(),
            EigenmatrixEvaluator(),
            HessianElementEvaluator(),
        ]
        # Build kind → evaluator lookup from HANDLED_KINDS
        self._kind_to_evaluator: dict[str, Any] = {}
        for ev in self._evaluators:
            for kind in ev.HANDLED_KINDS:
                self._kind_to_evaluator[kind] = ev

    def _mol_idx(self, ref: Observation) -> int:
        """Resolve *ref*'s stable ``case_id`` to a position in ``self.molecules``.

        Raises:
            KeyError: If ``ref.case_id`` is not one of this objective's
                ``case_ids``.

        """
        try:
            return self._case_id_to_index[ref.case_id]
        except KeyError:
            raise KeyError(
                f"Observation {ref.label!r} (kind={ref.kind!r}) references case_id={ref.case_id!r}, "
                f"which is not among this objective's case_ids: {self.case_ids}."
            ) from None

    def __call__(self, param_vector: np.ndarray) -> float:
        """Evaluate objective for a given parameter vector.

        This is the function signature that :func:`scipy.optimize.minimize`
        expects: ``f(x) -> float``.

        Args:
            param_vector (np.ndarray): Flat parameter vector.

        Returns:
            float: Sum-of-squared weighted residuals.

        """
        residuals = self._compute_residuals(param_vector)
        score = float(np.sum(residuals**2))

        if self.regularization > 0:
            diff = param_vector - self._reference_params
            score += self.regularization * float(np.dot(diff, diff))

        self.n_eval += 1
        self.history.append(score)
        return score

    def residuals(self, param_vector: np.ndarray) -> np.ndarray:
        """Compute weighted residual vector (for least-squares methods).

        When L2 regularization is active, ``sqrt(λ) * (params - ref)`` is
        appended to the residual vector so that
        ``sum(residuals²) == data_loss + λ * ||params - ref||²``.

        Args:
            param_vector (np.ndarray): Flat parameter vector.

        Returns:
            np.ndarray: Weighted residuals for each reference observation,
            with optional L2 regularization terms appended.

        """
        r = self._compute_residuals(param_vector)

        if self.regularization > 0:
            diff = param_vector - self._reference_params
            l2_residuals = np.sqrt(self.regularization) * diff
            r = np.concatenate([r, l2_residuals])

        self.n_eval += 1
        self.history.append(float(np.sum(r**2)))
        return r

    def is_energy_only(self) -> bool:
        """Return ``True`` if every reference is an energy value.

        When true, :meth:`batched_scores` can use the backend's vectorised
        ``batched_energy`` path (e.g. ``jax.vmap``) for GPU-parallel
        sensitivity analysis.
        """
        return all(ref.kind == "energy" for ref in self.reference.values)

    def batched_scores(self, param_matrix: np.ndarray) -> np.ndarray:
        """Evaluate objective for multiple parameter vectors in one call.

        When the backend declares ``BATCHED_ENERGY`` and all
        references are energy-only, energy evaluations are vectorised
        (e.g. via ``jax.vmap``) for GPU-parallel evaluation.

        Otherwise falls back to sequential ``__call__`` per vector.

        Args:
            param_matrix: Shape ``(batch, n_params)`` parameter vectors.

        Returns:
            np.ndarray: Shape ``(batch,)`` objective scores.

        Raises:
            ValueError: If *param_matrix* has wrong number of columns.

        """
        param_matrix = np.atleast_2d(np.asarray(param_matrix, dtype=float))
        if param_matrix.ndim != 2 or param_matrix.shape[1] != len(self.layout):
            raise ValueError(f"param_matrix must have shape (batch, {len(self.layout)}), got {param_matrix.shape}")

        use_batched = self._prepared_for(0).info.supports(Capability.BATCHED_ENERGY) and self.is_energy_only()
        if not use_batched:
            return np.array([self(pvec) for pvec in param_matrix])

        return self._batched_energy_scores(param_matrix)

    def _batched_energy_scores(self, param_matrix: np.ndarray) -> np.ndarray:
        """Compute scores for a batch of param vectors (energy-only fast path).

        Uses each prepared session's ``batched_energy`` to evaluate all
        parameter vectors in a single vectorised call per molecule.
        """
        batch_size = len(param_matrix)
        # Group energy references by molecule index
        mol_refs: dict[int, list] = {}
        for ref in self.reference.values:
            mol_refs.setdefault(self._mol_idx(ref), []).append(ref)

        # For each molecule, batch-evaluate energies
        mol_energies: dict[int, np.ndarray] = {}
        for mol_idx in mol_refs:
            prepared = self._prepared_for(mol_idx)
            result = prepared.batched_energy(BatchedEnergyRequest(parameter_matrix=param_matrix))
            mol_energies[mol_idx] = np.asarray(result.energies)

        # Compute residuals and scores
        scores = np.zeros(batch_size)
        for ref in self.reference.values:
            energies = mol_energies[self._mol_idx(ref)]
            residuals = ref.weight * (ref.value - energies)
            scores += residuals**2

        n = len(param_matrix)
        self.n_eval += n

        # Add L2 regularization to batched scores
        if self.regularization > 0:
            diff = param_matrix - self._reference_params[np.newaxis, :]
            scores += self.regularization * np.sum(diff**2, axis=1)

        for s in scores:
            self.history.append(float(s))
        return scores

    def _can_batch_hessians(self) -> bool:
        """Check whether batched Hessian evaluation can be used.

        Batching is possible when the backend declares the
        :attr:`~q2mm.backends.contracts.Capability.BATCHED_HESSIAN` capability,
        there are at least two molecules, and some references require
        Hessian-derived data (frequencies or eigenmatrix).
        """
        from q2mm.backends.contracts import Capability

        if not self.backend.info.supports(Capability.BATCHED_HESSIAN):
            return False
        if len(self.molecules) < 2:
            return False

        hessian_kinds = {"frequency", "eig_diagonal", "eig_offdiagonal"}
        return any(ref.kind in hessian_kinds for ref in self.reference.values)

    def _precompute_batched_hessians(
        self,
        param_vector: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Pre-compute Hessians for topology-compatible prepared sessions.

        Groups the objective's per-case prepared sessions into typed Hessian
        batches through the backend-neutral
        :func:`~q2mm.backends.contracts.prepare_hessian_batches` helper (which
        checks :attr:`~q2mm.backends.contracts.Capability.BATCHED_HESSIAN` and
        the batch-preparer protocol), then evaluates each batch with a typed
        :class:`~q2mm.backends.contracts.BatchedHessianRequest` carrying the full
        parameter vector.  The typed
        :class:`~q2mm.backends.contracts.BatchedHessianResult` is mapped back to
        molecule indices via each batched case's stable case ID.

        Returns a mapping from molecule index to its ``(3N, 3N)`` Hessian in
        Hartree/Bohr^2.  Only molecules that need Hessian-derived data are
        included.  A genuine batched-evaluation failure raises a typed
        :class:`~q2mm.backends.contracts.EvaluationError` — there is no silent
        sequential fallback.
        """
        from q2mm.backends.contracts import BatchedHessianRequest, prepare_hessian_batches

        hessian_kinds = {"frequency", "eig_diagonal", "eig_offdiagonal"}
        mol_indices_needing_hess: set[int] = set()
        for ref in self.reference.values:
            if ref.kind in hessian_kinds:
                mol_indices_needing_hess.add(self._mol_idx(ref))

        if not mol_indices_needing_hess:
            return {}

        idx_list = sorted(mol_indices_needing_hess)

        # Reuse the compiled per-case prepared sessions already owned by the
        # objective — batches share one topology executable per group, but every
        # session keeps its own coordinates/native state.  Cases are mapped back
        # to molecule indices by their stable case IDs.  Batching goes entirely
        # through the backend-neutral contracts helper — no concrete backend
        # batching module is imported here.
        sessions = [self._prepared_for(i) for i in idx_list]
        request = BatchedHessianRequest(parameters=param_vector)

        hess_map: dict[int, np.ndarray] = {}
        for batch in prepare_hessian_batches(self.backend, sessions):
            result = batch.hessians(request)
            for case_id, hess in zip(result.case_ids, result.hessians):
                hess_map[self._case_id_to_index[case_id]] = np.asarray(hess)

        return hess_map

    def _compute_residuals(self, param_vector: np.ndarray) -> np.ndarray:
        """Compute weighted residuals for all reference observations.

        When the backend declares batched-Hessian support and multiple molecules
        share the same topology, Hessians are computed in a single ``jax.vmap``
        call per topology group (batched path).  Otherwise, falls back to
        per-molecule sequential evaluation.

        Args:
            param_vector: Full parameter vector (length ``len(layout)``).

        Returns:
            np.ndarray: Array of ``w_i * (ref_i - calc_i)`` residuals.

        """
        # Batched Hessian pre-computation (capability-gated).  A genuine
        # batched-evaluation failure propagates as a typed error rather than
        # silently falling back to sequential evaluation.
        precomputed_hessians: dict[int, np.ndarray] = {}
        if self._can_batch_hessians():
            precomputed_hessians = self._precompute_batched_hessians(param_vector)

        calc_cache: dict[int, dict] = {}

        residuals = []
        for ref in self.reference.values:
            mol_idx = self._mol_idx(ref)
            if mol_idx not in calc_cache:
                calc_cache[mol_idx] = self._evaluate_molecule(
                    mol_idx,
                    param_vector,
                    precomputed_hessian=precomputed_hessians.get(mol_idx),
                )

            calc = calc_cache[mol_idx]
            calc_value = self._extract_value(calc, ref)
            diff = ref.value - calc_value
            # Torsion angles wrap around 360°
            if ref.kind == "torsion_angle":
                diff = (diff + 180.0) % 360.0 - 180.0
            residual = ref.weight * diff
            residuals.append(residual)

        return np.array(residuals)

    def gradient(self, param_vector: np.ndarray) -> np.ndarray:
        """Compute analytical gradient of the score w.r.t. parameters.

        Delegates to each evaluator's ``gradient()`` method where
        analytical gradients are available.  For evaluators that do not
        yet support analytical gradients, falls back to finite-difference
        approximation of that evaluator's score contribution.

        The score is ``sum_i (w_i * (ref_i - calc_i))**2``, so each
        evaluator computes:

        ``d(score)/d(p) = -2 * sum_i [w_i^2 * (ref_i - calc_i) * d(calc_i)/d(p)]``

        Note:
            This method does **not** increment ``n_eval`` or append to
            ``history``.  SciPy's ``minimize`` calls ``fun(x)`` and ``jac(x)``
            separately, so tracking state here would double-count evaluations.
            Evaluation counting is handled exclusively in ``__call__``.

        Args:
            param_vector (np.ndarray): Flat parameter vector (same as
                :meth:`__call__`).

        Returns:
            np.ndarray: Gradient of the score with respect to each parameter.

        Note:
            Evaluators that support analytical gradients (e.g. energy via
            ``parameter_gradient``) are used directly.  Evaluators that
            do not support them are handled transparently via central
            finite-difference fallback — no error is raised.

        Note:
            ``n_eval`` and ``history`` track only objective-function
            evaluations made through ``__call__``.  The finite-difference
            gradient evaluations performed here are internal to the
            gradient computation and are intentionally excluded from
            those counters.

        """
        n_params = len(param_vector)
        total_grad = np.zeros(n_params)

        # Group references by molecule and evaluator kind
        refs_by_mol: dict[int, dict[str, list[Observation]]] = {}
        for ref in self.reference.values:
            mol_refs = refs_by_mol.setdefault(self._mol_idx(ref), {})
            # Map kinds to evaluator categories
            category = self._kind_to_category(ref.kind)
            mol_refs.setdefault(category, []).append(ref)

        # Process each molecule's evaluator contributions
        for mol_idx, category_refs in refs_by_mol.items():
            prepared = self._prepared_for(mol_idx)

            for category, refs in category_refs.items():
                evaluator = self._get_evaluator(category)
                if evaluator.supports_analytical_gradient(prepared):
                    grad = evaluator.gradient(
                        prepared,
                        param_vector,
                        refs,
                        n_params,
                        mol_idx=mol_idx,
                    )
                    total_grad += grad
                else:
                    # Finite-difference fallback for this evaluator's contribution
                    grad = self._finite_difference_gradient(
                        param_vector,
                        mol_idx,
                        category,
                        refs,
                    )
                    total_grad += grad

        # L2 regularization gradient: d/dp [λ * ||p - p_ref||²] = 2λ(p - p_ref)
        if self.regularization > 0:
            diff = param_vector - self._reference_params
            total_grad += 2.0 * self.regularization * diff

        # Warn about zero-gradient slots which may indicate incomplete
        # analytical gradient support (e.g. missing improper torsions).
        n_zero = int(np.sum(total_grad == 0))
        if n_zero > 0:
            logger.debug(
                "gradient: %d/%d parameter slots have zero gradient",
                n_zero,
                len(total_grad),
            )

        return total_grad

    def per_evaluator_gradient_support(self) -> dict[str, bool]:
        """Return per-category analytical gradient support for the current backend.

        Queries each evaluator's :meth:`supports_analytical_gradient` for
        the categories that have at least one reference value.  The result
        reflects the **actual** gradient dispatch path used by
        :meth:`gradient` — no hardcoded assumptions.

        Returns:
            Mapping from evaluator category (``"energy"``, ``"frequency"``,
            etc.) to ``True`` if analytical gradients are available, ``False``
            if finite-difference fallback will be used.

        """
        categories: dict[str, bool] = {}
        for ref in self.reference.values:
            cat = self._kind_to_category(ref.kind)
            if cat not in categories:
                evaluator = self._get_evaluator(cat)
                prepared = self._prepared_for(self._mol_idx(ref))
                categories[cat] = evaluator.supports_analytical_gradient(prepared)
        return dict(sorted(categories.items()))

    def to_jax_spec(
        self,
    ) -> ObjectiveSpec:
        """Build a JAX-compatible objective specification.

        Encodes reference data, regularization settings, and parameter
        bounds into the :class:`~q2mm.optimizers.spec.ObjectiveSpec`
        format consumed by :class:`~q2mm.optimizers.jaxloss.JaxLoss`.

        Geometry references (bond_length, bond_angle, torsion_angle)
        are included; the JIT loss handles them via implicit
        differentiation through an inner ``jaxopt.LBFGS`` geometry
        minimization.

        Returns:
            ObjectiveSpec ready for JIT compilation.

        Raises:
            ValueError: If the objective has no references.

        """
        from q2mm.optimizers.spec import ObjectiveSpec, _build_molecule_spec

        # Group references by molecule index
        refs_by_mol: dict[int, list[Observation]] = {}
        for ref in self.reference.values:
            refs_by_mol.setdefault(self._mol_idx(ref), []).append(ref)

        # Build per-molecule specs
        mol_specs = []
        categories: set[str] = set()
        for mol_idx in sorted(refs_by_mol):
            mol = self.molecules[mol_idx]
            refs = refs_by_mol[mol_idx]
            spec = _build_molecule_spec(
                mol_idx=mol_idx,
                symbols=tuple(mol.symbols),
                refs=refs,
                topology=mol,
            )
            mol_specs.append(spec)
            # Track which categories are present
            if spec.has_energy:
                categories.add("energy")
            if spec.has_frequency:
                categories.add("frequency")
            if spec.has_hessian:
                categories.add("hessian")
            if spec.has_eigenmatrix:
                categories.add("eigenmatrix")
            if spec.has_geometry:
                categories.add("geometry")

        if not categories:
            raise ValueError(
                "No references found. ObjectiveFunction.to_jax_spec() requires at least one reference value."
            )

        # Parameter bounds from the layout
        bounds = self.layout.bounds
        lower = np.array(bounds[:, 0], dtype=float)
        upper = np.array(bounds[:, 1], dtype=float)

        return ObjectiveSpec(
            molecules=tuple(mol_specs),
            n_params=len(self.layout),
            regularization=self.regularization,
            reference_params=self._reference_params.copy(),
            lower_bounds=lower,
            upper_bounds=upper,
            supported_categories=frozenset(categories),
        )

    def jax_sessions(self, spec: ObjectiveSpec) -> dict[int, PreparedBackend]:
        """Return the objective's per-case prepared sessions for a JAX spec.

        Reuses the objective's own :meth:`_prepared_for` cache so each stable
        case is prepared exactly once.  The returned map (``mol_idx ->``
        prepared session) is passed to :class:`~q2mm.optimizers.jaxloss.JaxLoss`
        so the Objective + JaxLoss path never prepares a case twice.
        """
        return {ms.mol_idx: self._prepared_for(ms.mol_idx) for ms in spec.molecules}

    def _finite_difference_gradient(
        self,
        param_vector: np.ndarray,
        mol_idx: int,
        category: str,
        refs: list[Observation],
        step: float | None = None,
    ) -> np.ndarray:
        """Compute finite-difference gradient for one evaluator's contribution.

        Uses central differences: ``(f(x+h) - f(x-h)) / (2h)`` for each
        parameter, where ``f`` is the sum-of-squared weighted residuals
        from *refs* only.

        .. warning::

            For ``frequency`` and ``eigenmatrix`` categories, the FD
            perturbation evaluates at the *original* (unperturbed)
            geometry rather than re-optimizing at each perturbed
            parameter set.  This is an approximation — the true
            derivative includes an implicit geometry-relaxation term.
            For small parameter perturbations this is usually
            acceptable, but the resulting gradient may be inaccurate
            when the potential energy surface is highly anharmonic.

        Args:
            param_vector: Current parameter vector.
            mol_idx: Molecule index.
            category: Evaluator category (``"energy"``, ``"frequency"``,
                ``"geometry"``, or ``"eigenmatrix"``).
            refs: Reference values for this evaluator and molecule.
            step: Finite-difference step size. Defaults to
                :attr:`fd_step` (configurable, initially ``1e-4``).

        Returns:
            Gradient vector of shape ``(n_params,)``.

        """
        if step is None:
            step = self.fd_step
        n_params = len(param_vector)
        grad = np.zeros(n_params)

        for j in range(n_params):
            params_plus = param_vector.copy()
            params_plus[j] += step
            score_plus = self._partial_score(params_plus, mol_idx, category, refs)

            params_minus = param_vector.copy()
            params_minus[j] -= step
            score_minus = self._partial_score(params_minus, mol_idx, category, refs)

            grad[j] = (score_plus - score_minus) / (2.0 * step)

        return grad

    def _partial_score(
        self,
        param_vector: np.ndarray,
        mol_idx: int,
        category: str,
        refs: list[Observation],
    ) -> float:
        """Evaluate score contribution from a subset of references.

        .. warning::

            For ``frequency`` and ``eigenmatrix`` categories this
            evaluates at the *unperturbed* geometry.  Strictly, the
            Hessian (and therefore frequencies / eigenmatrix) should
            be computed at the minimum-energy geometry for the
            perturbed parameters.  See the note on
            :meth:`_finite_difference_gradient`.

        Args:
            param_vector: Parameter vector to evaluate.
            mol_idx: Molecule index.
            category: Evaluator category.
            refs: Reference values to score.

        Returns:
            Sum-of-squared weighted residuals for the given references.

        """
        prepared = self._prepared_for(mol_idx)
        evaluator = self._get_evaluator(category)

        if category == "geometry":
            needed_kinds = frozenset(r.kind for r in refs)
            computed = evaluator.compute(prepared, param_vector, needed_kinds=needed_kinds)
        elif category == "eigenmatrix":
            # NOTE: Evaluates the Hessian at the *original* (unperturbed)
            # geometry, which is an approximation.  A more rigorous approach
            # would re-optimize the geometry at the perturbed parameters first
            # (see issue #149).
            computed = evaluator.compute(prepared, param_vector, mol_idx=mol_idx)
        elif category == "frequency":
            # NOTE: Same approximation as eigenmatrix — Hessian evaluated at
            # original geometry rather than re-optimized (see issue #149).
            computed = evaluator.compute(prepared, param_vector, on_error=self.on_error)
        else:
            computed = evaluator.compute(prepared, param_vector)

        residuals = evaluator.residuals(computed, refs)
        return float(np.sum(np.array(residuals) ** 2))

    @staticmethod
    def _kind_to_category(kind: str) -> str:
        """Map a reference value kind to its evaluator category.

        Args:
            kind: Reference value kind string.

        Returns:
            Evaluator category: ``"energy"``, ``"frequency"``,
            ``"geometry"``, ``"eigenmatrix"``, or ``"hessian"``.

        Raises:
            ValueError: If the kind is unknown.

        """
        _KIND_CATEGORIES = {
            "energy": "energy",
            "frequency": "frequency",
            "bond_length": "geometry",
            "bond_angle": "geometry",
            "torsion_angle": "geometry",
            "eig_diagonal": "eigenmatrix",
            "eig_offdiagonal": "eigenmatrix",
            "hessian_element": "hessian",
        }
        if kind not in _KIND_CATEGORIES:
            raise ValueError(f"Unknown reference kind: {kind}")
        return _KIND_CATEGORIES[kind]

    def _get_evaluator(self, category: str) -> Any:
        """Get the evaluator instance for a category.

        Args:
            category: Evaluator category string.

        Returns:
            The evaluator instance.

        Raises:
            ValueError: If the category is unknown.

        """
        # Find the first evaluator whose HANDLED_KINDS intersects with
        # the kinds in this category.
        for ev in self._evaluators:
            # Check if this evaluator handles any kind in the category
            for kind in ev.HANDLED_KINDS:
                if self._kind_to_category(kind) == category:
                    return ev
        raise ValueError(f"Unknown evaluator category: {category}")

    def _prepared_for(self, mol_idx: int) -> PreparedBackend:
        """Return the prepared backend session for a molecule, building once.

        Exactly one prepared session is built per stable case ID.  The session
        owns the molecule, base force field, parameter layout, and any reusable
        native state, and is reused across every evaluation of that case.

        Args:
            mol_idx (int): Index into the molecules list.

        Returns:
            PreparedBackend: The cached per-case session.

        """
        if mol_idx not in self._prepared:
            self._prepared[mol_idx] = self.backend.prepare(
                PreparationRequest(
                    case_id=self.case_ids[mol_idx],
                    molecule=self.molecules[mol_idx],
                    force_field=self.forcefield,
                )
            )
        return self._prepared[mol_idx]

    def _evaluate_molecule(
        self,
        mol_idx: int,
        param_vector: np.ndarray,
        *,
        precomputed_hessian: np.ndarray | None = None,
    ) -> dict:
        """Run MM calculations for a single molecule.

        Delegates to per-data-type evaluators where available, populating
        the results dict with the same keys for backward compatibility.

        When *precomputed_hessian* is provided (from batched vmap
        evaluation), it is used directly for frequency and eigenmatrix
        calculations, avoiding a redundant Hessian evaluation.

        Args:
            mol_idx (int): Index into the molecules list.
            param_vector: Full parameter vector (length ``len(layout)``).
            precomputed_hessian: Optional ``(3N, 3N)`` Hessian in
                Hartree/Bohr² from batched evaluation.

        Returns:
            dict: Calculated results keyed by data type.

        """
        mol = self.molecules[mol_idx]
        prepared = self._prepared_for(mol_idx)
        result: dict = {}

        # Determine what data types are needed for this molecule
        needed = {ref.kind for ref in self.reference.values if self._mol_idx(ref) == mol_idx}

        energy_ev = self._kind_to_evaluator.get("energy")
        freq_ev = self._kind_to_evaluator.get("frequency")
        geom_ev = self._kind_to_evaluator.get("bond_length")
        eigm_ev = self._kind_to_evaluator.get("eig_diagonal")

        if "energy" in needed and energy_ev is not None:
            er = energy_ev.compute(prepared, param_vector)
            result["energy"] = er.energy

        if "frequency" in needed and freq_ev is not None:
            if precomputed_hessian is not None:
                from q2mm.models.hessian import hessian_to_frequencies

                result["frequencies"] = hessian_to_frequencies(
                    precomputed_hessian,
                    list(mol.symbols),
                    on_error=self.on_error,
                )
            else:
                fr = freq_ev.compute(prepared, param_vector, on_error=self.on_error)
                result["frequencies"] = fr.frequencies

        if geom_ev is not None and needed & geom_ev.HANDLED_KINDS:
            geo_needed = frozenset(needed & geom_ev.HANDLED_KINDS)
            gr = geom_ev.compute(prepared, param_vector, needed_kinds=geo_needed)
            if "bond_length" in geo_needed:
                result["bond_lengths"] = gr.bond_lengths
                result["bond_lengths_by_atoms"] = gr.bond_lengths_by_atoms
            if "bond_angle" in geo_needed:
                result["bond_angles"] = gr.bond_angles
                result["bond_angles_by_atoms"] = gr.bond_angles_by_atoms
            if "torsion_angle" in geo_needed:
                result["torsion_coords"] = gr.torsion_coords

        if eigm_ev is not None and needed & eigm_ev.HANDLED_KINDS:
            if precomputed_hessian is not None:
                from q2mm.models.hessian import mass_weighted_eigenmatrix, mass_weighted_normal_modes

                # Use precomputed Hessian to build eigenmatrix directly, in the
                # same mass-weighted normal-mode basis as reference generation
                # and EigenmatrixEvaluator / JaxLoss.
                eigm_evaluator = eigm_ev
                if mol_idx not in eigm_evaluator._qm_eigenvectors:
                    if mol.hessian is None:
                        raise ValueError(
                            f"Molecule {mol_idx} ({mol.name}) has no QM Hessian. "
                            "Eigenmatrix training requires a QM Hessian for the "
                            "eigenvector basis."
                        )
                    _, qm_evecs = mass_weighted_normal_modes(mol.hessian, mol.symbols)
                    eigm_evaluator._qm_eigenvectors[mol_idx] = qm_evecs

                qm_evecs = eigm_evaluator._qm_eigenvectors[mol_idx]
                result["eigenmatrix"] = mass_weighted_eigenmatrix(
                    precomputed_hessian,
                    qm_evecs,
                    mol.symbols,
                )
            else:
                emr = eigm_ev.compute(prepared, param_vector, mol_idx=mol_idx)
                result["eigenmatrix"] = emr.eigenmatrix

        hess_ev = self._kind_to_evaluator.get("hessian_element")
        if "hessian_element" in needed and hess_ev is not None:
            hr = hess_ev.compute(prepared, param_vector)
            result["raw_hessian"] = hr.hessian

        return result

    @staticmethod
    def _extract_value(calc: dict, ref: Observation) -> float:
        """Extract a calculated value matching a reference observation.

        Delegates to per-data-type evaluators for extraction logic.
        Uses each evaluator's ``HANDLED_KINDS`` to find the right handler.

        Args:
            calc (dict): Calculated results from :meth:`_evaluate_molecule`.
            ref (Observation): Reference observation to match.

        Returns:
            float: The calculated value corresponding to the reference.

        Raises:
            IndexError: If ``data_idx`` is out of range.
            KeyError: If atom-identity match fails.
            ValueError: If ``ref.kind`` is unknown or torsion is missing
                atom indices.

        """
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator
        from q2mm.optimizers.evaluators.hessian_element import HessianElementEvaluator

        _EVALUATOR_CLASSES = [
            EnergyEvaluator,
            FrequencyEvaluator,
            GeometryEvaluator,
            EigenmatrixEvaluator,
            HessianElementEvaluator,
        ]
        for cls in _EVALUATOR_CLASSES:
            if ref.kind in cls.HANDLED_KINDS:
                return cls.extract_value(calc, ref)
        raise ValueError(f"Unknown reference kind: {ref.kind}")

    def reset(self) -> None:
        """Reset evaluation counter, history, and cached prepared sessions."""
        self.n_eval = 0
        self.history.clear()
        self._prepared.clear()
        for ev in self._evaluators:
            if hasattr(ev, "reset"):
                ev.reset()
