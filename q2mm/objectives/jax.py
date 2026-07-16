"""JAX-native objective executor with per-case JIT and Python aggregation.

:class:`JaxObjectiveExecutor` compiles each training case's loss
contribution into its **own** small ``jax.jit(jax.value_and_grad(...))``
program and aggregates the per-case value and gradient by dispatching them
from Python and summing.  No single XLA program ever contains all
molecules — this preserves the per-case JIT split that prevents
compilation OOM on multi-molecule systems, and is never merged with the
subprocess/native Python backend path.

Geometry references are handled via implicit differentiation through an
inner ``jaxopt.LBFGS`` geometry minimization, giving exact analytical
parameter gradients of the relaxed observables.

The Phase-3 :class:`~q2mm.backends.mm.jax_engine.PreparedJax` sessions are
prepared once per case and reused for both the compiled loss (via each
session's private ``_energy_kernel``) and the observable extraction used by
:meth:`evaluate`; the executor never prepares a case twice.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.backends.contracts import PreparationRequest
from q2mm.models.forcefield import ForceField
from q2mm.objectives._base import BaseObjectiveExecutor
from q2mm.objectives._observables import extract_calc_value, geometry_computed
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import GradientMode

if TYPE_CHECKING:
    from q2mm.backends.mm.jax_engine import JaxBackend, PreparedJax

logger = logging.getLogger(__name__)

__all__ = ["JaxObjectiveExecutor"]

# Inner geometry-minimizer tolerance + iteration cap (see issue #243).
_GEOM_INNER_TOL = 1e-8
_GEOM_INNER_MAXITER = 500

_GEOMETRY_KINDS = frozenset({"bond_length", "bond_angle", "torsion_angle"})
_EIG_KINDS = frozenset({"eig_diagonal", "eig_offdiagonal"})


# ---------------------------------------------------------------------------
# Private JAX kernels (executor-specific — never merged with the Python path)
# ---------------------------------------------------------------------------


def _relax_coords(energy_fn, params, coords0):  # noqa: ANN001, ANN202
    """Return the implicit-diff relaxed geometry for a molecule."""
    import jaxopt

    def energy_of_coords(coords, p):  # noqa: ANN001, ANN202
        return energy_fn(p, coords)

    solver = jaxopt.LBFGS(
        fun=energy_of_coords,
        tol=_GEOM_INNER_TOL,
        maxiter=_GEOM_INNER_MAXITER,
        implicit_diff=True,
    )
    sol = solver.run(coords0, params)
    return sol.params


def _bond_lengths(coords, atoms):  # noqa: ANN001, ANN202
    import jax.numpy as jnp

    d = coords[atoms[:, 0]] - coords[atoms[:, 1]]
    return jnp.sqrt(jnp.sum(d * d, axis=-1))


def _bond_angles_deg(coords, atoms):  # noqa: ANN001, ANN202
    import jax.numpy as jnp

    v1 = coords[atoms[:, 0]] - coords[atoms[:, 1]]
    v2 = coords[atoms[:, 2]] - coords[atoms[:, 1]]
    n1 = jnp.linalg.norm(v1, axis=-1)
    n2 = jnp.linalg.norm(v2, axis=-1)
    denom = jnp.maximum(n1 * n2, 1e-12)
    cos = jnp.sum(v1 * v2, axis=-1) / denom
    cos = jnp.clip(cos, -1.0 + 1e-12, 1.0 - 1e-12)
    return jnp.arccos(cos) * (180.0 / jnp.pi)


def _torsion_angles_deg(coords, atoms):  # noqa: ANN001, ANN202
    import jax.numpy as jnp

    p0 = coords[atoms[:, 0]]
    p1 = coords[atoms[:, 1]]
    p2 = coords[atoms[:, 2]]
    p3 = coords[atoms[:, 3]]
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    b2_norm = jnp.linalg.norm(b2, axis=-1, keepdims=True)
    b2_hat = b2 / jnp.maximum(b2_norm, 1e-12)
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    m = jnp.cross(n1, b2_hat)
    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(m * n2, axis=-1)
    return jnp.arctan2(y, x) * (180.0 / jnp.pi)


def _resolve_geom_atoms(ref: object, attr: str, arity: int, topology: object) -> tuple[int, ...]:
    """Resolve a geometry reference to an explicit atom-index tuple."""
    if ref.atom_indices is not None and len(ref.atom_indices) >= arity:  # type: ignore[attr-defined]
        return tuple(int(i) for i in ref.atom_indices[:arity])  # type: ignore[attr-defined]
    records = getattr(topology, attr, None)
    if records is None:
        raise ValueError(f"{ref.kind} reference {ref.label!r} requires molecule.{attr}.")  # type: ignore[attr-defined]
    if ref.data_idx < 0 or ref.data_idx >= len(records):  # type: ignore[attr-defined]
        raise ValueError(f"{ref.kind} reference {ref.label!r} has data_idx out of range.")  # type: ignore[attr-defined]
    record = records[ref.data_idx]  # type: ignore[attr-defined]
    index_attrs = ("atom_i", "atom_j", "atom_k", "atom_l")[:arity]
    return tuple(int(getattr(record, a)) for a in index_attrs)


def _build_mol_arrays(mol: object, refs: list) -> dict:
    """Build per-molecule reference arrays grouped by category."""
    symbols = tuple(mol.symbols)  # type: ignore[attr-defined]
    n3 = 3 * len(symbols)
    data: dict[str, list] = {
        "energy_refs": [],
        "energy_weights": [],
        "freq_indices": [],
        "freq_refs": [],
        "freq_weights": [],
        "hess_indices": [],
        "hess_refs": [],
        "hess_weights": [],
        "ediag_indices": [],
        "ediag_refs": [],
        "ediag_weights": [],
        "eoff_indices": [],
        "eoff_refs": [],
        "eoff_weights": [],
        "bond_atoms": [],
        "bond_refs": [],
        "bond_weights": [],
        "angle_atoms": [],
        "angle_refs": [],
        "angle_weights": [],
        "torsion_atoms": [],
        "torsion_refs": [],
        "torsion_weights": [],
    }
    for ref in refs:
        kind = ref.kind
        if kind == "energy":
            data["energy_refs"].append(ref.value)
            data["energy_weights"].append(ref.weight)
        elif kind == "frequency":
            data["freq_indices"].append(ref.data_idx)
            data["freq_refs"].append(ref.value)
            data["freq_weights"].append(ref.weight)
        elif kind == "hessian_element":
            row, col = ref.atom_indices[:2]
            data["hess_indices"].append(row * n3 + col)
            data["hess_refs"].append(ref.value)
            data["hess_weights"].append(ref.weight)
        elif kind == "eig_diagonal":
            data["ediag_indices"].append(ref.data_idx)
            data["ediag_refs"].append(ref.value)
            data["ediag_weights"].append(ref.weight)
        elif kind == "eig_offdiagonal":
            row, col = ref.atom_indices[:2]
            data["eoff_indices"].append(row * n3 + col)
            data["eoff_refs"].append(ref.value)
            data["eoff_weights"].append(ref.weight)
        elif kind == "bond_length":
            data["bond_atoms"].append(_resolve_geom_atoms(ref, "bonds", 2, mol))
            data["bond_refs"].append(ref.value)
            data["bond_weights"].append(ref.weight)
        elif kind == "bond_angle":
            data["angle_atoms"].append(_resolve_geom_atoms(ref, "angles", 3, mol))
            data["angle_refs"].append(ref.value)
            data["angle_weights"].append(ref.weight)
        elif kind == "torsion_angle":
            data["torsion_atoms"].append(_resolve_geom_atoms(ref, "torsions", 4, mol))
            data["torsion_refs"].append(ref.value)
            data["torsion_weights"].append(ref.weight)
        else:
            raise ValueError(f"Unknown reference kind: {kind!r}")
    return data


class JaxObjectiveExecutor(BaseObjectiveExecutor):
    """JAX-native executor: per-case JIT loss/gradient + Python aggregation.

    Args:
        plan: The immutable objective plan.
        backend: A ``JaxBackend`` instance.
        base_force_field: Base force field supplying topology/structure for
            prepared sessions.

    Raises:
        TypeError: If *backend* is not a ``JaxBackend``.

    """

    def __init__(self, plan: ObjectivePlan, backend: object, base_force_field: ForceField) -> None:
        super().__init__(plan)
        from q2mm.backends.mm._jax_common import ensure_jax
        from q2mm.backends.mm.jax_engine import JaxBackend

        if not isinstance(backend, JaxBackend):
            raise TypeError(f"JaxObjectiveExecutor requires a JaxBackend, got {type(backend).__name__}.")
        ensure_jax(engine_name="JaxObjectiveExecutor")
        self._backend: JaxBackend = backend
        self._base_ff = base_force_field
        #: Frequency eigendecomposition error handling for ``evaluate``.
        self.on_error: str = "raise"
        self._sessions: dict[str, PreparedJax] = {}
        self._compiled_value_fns: list[Callable] = []
        self._compiled_vag_fns: list[Callable] = []
        self._compiled_reg_value_fn: Callable | None = None
        self._compiled_reg_vag_fn: Callable | None = None
        self._case_meta: dict[str, dict[str, Any]] = {}
        self._hess_fns: dict[str, Callable] = {}
        self._build()

    @property
    def backend(self) -> JaxBackend:
        """The JAX backend this executor evaluates against."""
        return self._backend

    @property
    def base_force_field(self) -> ForceField:
        """The base force field supplying prepared-session structure."""
        return self._base_ff

    @property
    def gradient_mode(self) -> GradientMode:
        """Always :attr:`GradientMode.ANALYTICAL` (per-case JIT autodiff)."""
        return GradientMode.ANALYTICAL

    def _prepared_for(self, case_id: str) -> PreparedJax:
        session = self._sessions.get(case_id)
        if session is None:
            idx = self._plan.case_index(case_id)
            session = self._backend.prepare(  # type: ignore[assignment]
                PreparationRequest(
                    case_id=case_id,
                    molecule=self._plan.molecules[idx],
                    force_field=self._base_ff,
                )
            )
            self._sessions[case_id] = session
        return session

    def _build(self) -> None:
        import jax
        import jax.numpy as jnp
        from q2mm.models.hessian import (
            _jax_frequencies_from_hessian,
            mass_weight_scale_3n,
            mass_weighted_normal_modes,
            symbols_to_masses_3n,
        )
        from q2mm.models.units import KCALMOLA2_TO_HESSIAN_AU

        plan = self._plan
        hess_au_scale = float(KCALMOLA2_TO_HESSIAN_AU)

        # Group observations by case (case order).
        refs_by_case: dict[str, list] = {cid: [] for cid in plan.case_ids}
        for obs in plan.observations.values:
            refs_by_case[obs.case_id].append(obs)

        entries: list[dict] = []
        for case_id in plan.case_ids:
            refs = refs_by_case[case_id]
            if not refs:
                continue
            idx = plan.case_index(case_id)
            mol = plan.molecules[idx]
            session = self._prepared_for(case_id)
            arrays = _build_mol_arrays(mol, refs)
            symbols = tuple(mol.symbols)
            coords = jnp.array(mol.geometry, dtype=jnp.float64)

            has_energy = len(arrays["energy_refs"]) > 0
            has_freq = len(arrays["freq_refs"]) > 0
            has_hess = len(arrays["hess_refs"]) > 0
            has_ediag = len(arrays["ediag_refs"]) > 0
            has_eoff = len(arrays["eoff_refs"]) > 0
            has_eig = has_ediag or has_eoff
            has_bond = len(arrays["bond_refs"]) > 0
            has_angle = len(arrays["angle_refs"]) > 0
            has_tors = len(arrays["torsion_refs"]) > 0
            has_geom = has_bond or has_angle or has_tors
            needs_hessian = has_freq or has_hess or has_eig

            entry: dict = {
                "case_id": case_id,
                "symbols": symbols,
                "session": session,
                "coords": coords,
                "has_energy": has_energy,
                "has_freq": has_freq,
                "has_hess": has_hess,
                "has_ediag": has_ediag,
                "has_eoff": has_eoff,
                "has_eig": has_eig,
                "has_bond": has_bond,
                "has_angle": has_angle,
                "has_tors": has_tors,
                "has_geom": has_geom,
                "needs_hessian": needs_hessian,
            }

            meta: dict = {"symbols": symbols, "session": session, "scale": hess_au_scale}
            if needs_hessian:
                entry["masses_3n"] = jnp.array(symbols_to_masses_3n(symbols), dtype=jnp.float64)
            if has_eig:
                if mol.hessian is None:
                    raise ValueError(f"Case {case_id!r} ({mol.name}) has no QM Hessian; eigenmatrix requires one.")
                _, qm_evecs = mass_weighted_normal_modes(mol.hessian, symbols)
                entry["qm_evecs"] = jnp.array(qm_evecs, dtype=jnp.float64)
                entry["mw_scale"] = jnp.array(mass_weight_scale_3n(symbols), dtype=jnp.float64)
                meta["qm_evecs"] = np.asarray(qm_evecs)

            if has_energy:
                entry["energy_refs"] = jnp.array(arrays["energy_refs"], dtype=jnp.float64)
                entry["energy_weights"] = jnp.array(arrays["energy_weights"], dtype=jnp.float64)
            if has_freq:
                entry["freq_indices"] = jnp.array(arrays["freq_indices"], dtype=jnp.int32)
                entry["freq_refs"] = jnp.array(arrays["freq_refs"], dtype=jnp.float64)
                entry["freq_weights"] = jnp.array(arrays["freq_weights"], dtype=jnp.float64)
            if has_hess:
                entry["hess_indices"] = jnp.array(arrays["hess_indices"], dtype=jnp.int32)
                entry["hess_refs"] = jnp.array(arrays["hess_refs"], dtype=jnp.float64)
                entry["hess_weights"] = jnp.array(arrays["hess_weights"], dtype=jnp.float64)
            if has_ediag:
                entry["ediag_indices"] = jnp.array(arrays["ediag_indices"], dtype=jnp.int32)
                entry["ediag_refs"] = jnp.array(arrays["ediag_refs"], dtype=jnp.float64)
                entry["ediag_weights"] = jnp.array(arrays["ediag_weights"], dtype=jnp.float64)
            if has_eoff:
                entry["eoff_indices"] = jnp.array(arrays["eoff_indices"], dtype=jnp.int32)
                entry["eoff_refs"] = jnp.array(arrays["eoff_refs"], dtype=jnp.float64)
                entry["eoff_weights"] = jnp.array(arrays["eoff_weights"], dtype=jnp.float64)
            if has_bond:
                entry["bond_atoms"] = jnp.array(arrays["bond_atoms"], dtype=jnp.int32)
                entry["bond_refs"] = jnp.array(arrays["bond_refs"], dtype=jnp.float64)
                entry["bond_weights"] = jnp.array(arrays["bond_weights"], dtype=jnp.float64)
            if has_angle:
                entry["angle_atoms"] = jnp.array(arrays["angle_atoms"], dtype=jnp.int32)
                entry["angle_refs"] = jnp.array(arrays["angle_refs"], dtype=jnp.float64)
                entry["angle_weights"] = jnp.array(arrays["angle_weights"], dtype=jnp.float64)
            if has_tors:
                entry["torsion_atoms"] = jnp.array(arrays["torsion_atoms"], dtype=jnp.int32)
                entry["torsion_refs"] = jnp.array(arrays["torsion_refs"], dtype=jnp.float64)
                entry["torsion_weights"] = jnp.array(arrays["torsion_weights"], dtype=jnp.float64)

            entries.append(entry)
            self._case_meta[case_id] = meta

        # Per-session Hessian functions (one per topology group).
        hess_fn_cache: dict[int, Callable] = {}

        def _make_hess_fn(efn):  # noqa: ANN001, ANN202
            def _energy_of_flat(fc, p):  # noqa: ANN001, ANN202
                return efn(p, fc.reshape(-1, 3))

            return jax.hessian(_energy_of_flat, argnums=0)

        for entry in entries:
            session = entry["session"]
            h_id = id(session)
            if entry["needs_hessian"] and h_id not in hess_fn_cache:
                hess_fn_cache[h_id] = _make_hess_fn(session._energy_kernel())
            if entry["needs_hessian"]:
                self._hess_fns[entry["case_id"]] = hess_fn_cache[h_id]

        def _make_nongeom_loss(entry_data: dict, mol_hess_fn: Callable | None, scale: float) -> Callable:
            coords = entry_data["coords"]
            flat_coords = coords.reshape(-1)
            energy_fn = entry_data["session"]._energy_kernel()

            def _loss(params):  # noqa: ANN001, ANN202
                total = jnp.float64(0.0)
                if entry_data["has_energy"]:
                    energy = energy_fn(params, coords)
                    residuals = entry_data["energy_weights"] * (entry_data["energy_refs"] - energy)
                    total = total + jnp.sum(residuals**2)
                if entry_data["needs_hessian"]:
                    assert mol_hess_fn is not None
                    hess_au = mol_hess_fn(flat_coords, params) * scale
                    if entry_data["has_freq"]:
                        freqs = _jax_frequencies_from_hessian(hess_au, entry_data["masses_3n"])
                        calc = freqs[entry_data["freq_indices"]]
                        residuals = entry_data["freq_weights"] * (entry_data["freq_refs"] - calc)
                        total = total + jnp.sum(residuals**2)
                    if entry_data["has_hess"]:
                        n3 = hess_au.shape[0]
                        idx = entry_data["hess_indices"]
                        calc = hess_au[idx // n3, idx % n3]
                        residuals = entry_data["hess_weights"] * (entry_data["hess_refs"] - calc)
                        total = total + jnp.sum(residuals**2)
                    if entry_data["has_eig"]:
                        hess_mw = hess_au * entry_data["mw_scale"]
                        eigmat = entry_data["qm_evecs"].T @ hess_mw @ entry_data["qm_evecs"]
                        if entry_data["has_ediag"]:
                            idx = entry_data["ediag_indices"]
                            calc = eigmat[idx, idx]
                            residuals = entry_data["ediag_weights"] * (entry_data["ediag_refs"] - calc)
                            total = total + jnp.sum(residuals**2)
                        if entry_data["has_eoff"]:
                            idx = entry_data["eoff_indices"]
                            n3e = eigmat.shape[0]
                            calc = eigmat[idx // n3e, idx % n3e]
                            residuals = entry_data["eoff_weights"] * (entry_data["eoff_refs"] - calc)
                            total = total + jnp.sum(residuals**2)
                return total

            return _loss

        def _make_geom_loss(entry_data: dict) -> Callable:
            coords = entry_data["coords"]
            energy_fn = entry_data["session"]._energy_kernel()

            def _loss(params):  # noqa: ANN001, ANN202
                total = jnp.float64(0.0)
                relaxed = _relax_coords(energy_fn, params, coords)
                if entry_data["has_bond"]:
                    calc = _bond_lengths(relaxed, entry_data["bond_atoms"])
                    residuals = entry_data["bond_weights"] * (entry_data["bond_refs"] - calc)
                    total = total + jnp.sum(residuals**2)
                if entry_data["has_angle"]:
                    calc = _bond_angles_deg(relaxed, entry_data["angle_atoms"])
                    residuals = entry_data["angle_weights"] * (entry_data["angle_refs"] - calc)
                    total = total + jnp.sum(residuals**2)
                if entry_data["has_tors"]:
                    calc = _torsion_angles_deg(relaxed, entry_data["torsion_atoms"])
                    diff = entry_data["torsion_refs"] - calc
                    diff = (diff + 180.0) % 360.0 - 180.0
                    residuals = entry_data["torsion_weights"] * diff
                    total = total + jnp.sum(residuals**2)
                return total

            return _loss

        loss_fns: list[Callable] = []
        for entry in entries:
            if entry["has_energy"] or entry["needs_hessian"]:
                mol_hess_fn = self._hess_fns.get(entry["case_id"])
                loss_fns.append(_make_nongeom_loss(entry, mol_hess_fn, hess_au_scale))
            if entry["has_geom"]:
                loss_fns.append(_make_geom_loss(entry))

        self._compiled_value_fns = [jax.jit(fn) for fn in loss_fns]
        self._compiled_vag_fns = [jax.jit(jax.value_and_grad(fn)) for fn in loss_fns]

        reg = plan.regularization
        if reg > 0:
            ref_params = jnp.array(plan.reference_params, dtype=jnp.float64)

            def _reg_fn(params):  # noqa: ANN001, ANN202
                diff = params - ref_params
                return reg * jnp.dot(diff, diff)

            self._compiled_reg_value_fn = jax.jit(_reg_fn)
            self._compiled_reg_vag_fn = jax.jit(jax.value_and_grad(_reg_fn))

    # -- value / gradient via per-case JIT + Python aggregation -----------

    def _total(self, full_vector: np.ndarray) -> float:
        import jax.numpy as jnp

        p = jnp.array(full_vector, dtype=jnp.float64)
        total = jnp.float64(0.0)
        for fn in self._compiled_value_fns:
            total = total + fn(p)
        if self._compiled_reg_value_fn is not None:
            total = total + self._compiled_reg_value_fn(p)
        return float(total)

    def value_and_grad_jax(self, full_vector: object):  # noqa: ANN201
        """Aggregate value and gradient as JAX-native types (per-case dispatch)."""
        import jax.numpy as jnp

        p = jnp.array(full_vector, dtype=jnp.float64)
        total_loss = jnp.float64(0.0)
        total_grad = jnp.zeros_like(p)
        for fn in self._compiled_vag_fns:
            loss_i, grad_i = fn(p)
            total_loss = total_loss + loss_i
            total_grad = total_grad + grad_i
        if self._compiled_reg_vag_fn is not None:
            reg_loss, reg_grad = self._compiled_reg_vag_fn(p)
            total_loss = total_loss + reg_loss
            total_grad = total_grad + reg_grad
        return total_loss, total_grad

    def loss_and_grad(self, full_vector: np.ndarray) -> tuple[float, np.ndarray]:
        """Host-typed value+gradient with a finite penalty on NaN/Inf."""
        loss_jax, grad_jax = self.value_and_grad_jax(full_vector)
        loss = float(loss_jax)
        grad = np.asarray(grad_jax, dtype=float)
        if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
            logger.warning("JaxObjectiveExecutor returned non-finite values; substituting penalty")
            return 1e30, np.zeros_like(grad)
        return loss, grad

    def value_and_gradient(self, full_vector: np.ndarray) -> tuple[float, np.ndarray]:
        """Total value and full-length gradient via the per-case JIT path."""
        full = self._as_full(full_vector)
        value, grad = self.loss_and_grad(full)
        self._record(value)
        return value, grad

    # -- observable extraction for evaluate() -----------------------------

    def _calculated(self, full_vector: np.ndarray) -> np.ndarray:
        from collections import defaultdict

        import jax.numpy as jnp
        from q2mm.models.hessian import hessian_to_frequencies, mass_weighted_eigenmatrix

        observations = self._plan.observations.values
        by_case: dict[str, list[int]] = defaultdict(list)
        for gi, obs in enumerate(observations):
            by_case[obs.case_id].append(gi)

        calc = np.empty(len(observations), dtype=float)
        p = jnp.array(full_vector, dtype=jnp.float64)

        for case_id, indices in by_case.items():
            needed: set[str] = {str(observations[gi].kind) for gi in indices}
            meta = self._case_meta[case_id]
            session = meta["session"]
            idx = self._plan.case_index(case_id)
            mol = self._plan.molecules[idx]
            symbols = meta["symbols"]
            energy_fn = session._energy_kernel()
            coords0 = jnp.array(mol.geometry, dtype=jnp.float64)
            computed: dict = {}

            if "energy" in needed:
                computed["energy"] = float(energy_fn(p, coords0))

            if needed & (_EIG_KINDS | {"frequency", "hessian_element"}):
                hess_fn = self._hess_fns[case_id]
                hess_au = np.asarray(hess_fn(coords0.reshape(-1), p)) * meta["scale"]
                if "hessian_element" in needed:
                    computed["raw_hessian"] = hess_au
                if "frequency" in needed:
                    computed["frequencies"] = np.asarray(
                        hessian_to_frequencies(hess_au, list(symbols), on_error=self.on_error)  # type: ignore[arg-type]
                    )
                if needed & _EIG_KINDS:
                    computed["eigenmatrix"] = mass_weighted_eigenmatrix(hess_au, meta["qm_evecs"], symbols)

            if needed & _GEOMETRY_KINDS:
                relaxed = np.asarray(_relax_coords(energy_fn, p, coords0))
                computed.update(geometry_computed(mol, relaxed, needed))

            for gi in indices:
                calc[gi] = extract_calc_value(computed, observations[gi])
        return calc
