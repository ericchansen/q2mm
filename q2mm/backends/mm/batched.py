"""Batched, topology-compatible Hessian evaluation using JAX vmap.

Groups topology-compatible prepared JAX sessions into typed batch objects and
uses ``jax.vmap`` to compute Hessians for multiple geometries in a single
vectorized call.  Every session keeps its own molecule/coordinates/native
state; a batch shares only one compiled coordinate-Hessian kernel (the
representative session's).  The batch's evaluation surface is a typed
:class:`~q2mm.backends.contracts.BatchedHessianRequest` carrying one full
parameter vector; no ForceField crosses the boundary.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover
    _HAS_JAX = False

from q2mm.backends.contracts import (
    BatchedHessianRequest,
    BatchedHessianResult,
    EvaluationError,
    HessianUnit,
    readonly_array,
)
from q2mm.models.units import KCALMOLA2_TO_HESSIAN_AU

if TYPE_CHECKING:
    from q2mm.backends.mm.jax_engine import PreparedJax, _JaxState


# ---------------------------------------------------------------------------
# Topology signature
# ---------------------------------------------------------------------------


def _topology_signature(state: _JaxState) -> str:
    """Create a hashable signature for a native state's topology using SHA-256.

    Two states with the same signature are guaranteed to share identical
    connectivity and parameter mapping, so their energy functions are
    interchangeable up to coordinate differences.
    """
    h = hashlib.sha256()
    n_atoms = state.molecule.geometry.shape[0] if state.molecule is not None else 0
    h.update(f"n_atoms={n_atoms}".encode())
    h.update(f"n_bt={state.n_bond_types}".encode())
    h.update(f"n_at={state.n_angle_types}".encode())
    h.update(f"n_tt={state.n_torsion_types}".encode())
    h.update(f"n_vt={state.n_vdw_types}".encode())
    h.update(f"form={state.functional_form}".encode())
    for name, arr in [
        ("bonds", state.bond_indices),
        ("angles", state.angle_indices),
        ("torsions", state.torsion_indices),
        ("vdw", state.vdw_pair_indices),
    ]:
        h.update(f"{name}={sorted(map(tuple, arr))}".encode() if len(arr) > 0 else f"{name}=[]".encode())
    for name, arr in [
        ("bmap", state.bond_param_map),
        ("amap", state.angle_param_map),
        ("tmap", state.torsion_param_map),
    ]:
        h.update(f"{name}={list(arr)}".encode() if len(arr) > 0 else f"{name}=[]".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Typed batch object
# ---------------------------------------------------------------------------


class PreparedJaxBatch:
    """A typed batch of topology-compatible :class:`PreparedJax` sessions.

    All sessions share the same atom count, connectivity, and parameter mapping
    (only their coordinates differ).  The batch shares one compiled
    coordinate-Hessian kernel (the representative session's private native
    state), while every session retains its own molecule/coordinates/native
    state.  The only evaluation surface is :meth:`hessians`, which takes a typed
    :class:`BatchedHessianRequest` carrying one full parameter vector and
    returns a typed :class:`BatchedHessianResult`.
    """

    def __init__(self, sessions: list[PreparedJax]) -> None:
        from q2mm.backends.mm.jax_engine import PreparedJax

        if not sessions:
            raise EvaluationError("PreparedJaxBatch requires at least one prepared session.")
        for s in sessions:
            if not isinstance(s, PreparedJax):
                raise EvaluationError(f"PreparedJaxBatch requires PreparedJax sessions; got {type(s).__name__}.")

        rep = sessions[0]
        rep_state = rep._state
        rep_len = len(rep.layout)
        rep_form = rep_state.functional_form
        rep_natoms = len(rep.molecule.symbols)
        rep_sig = _topology_signature(rep_state)
        if rep.info.provenance is None:  # pragma: no cover - JAX always has provenance
            raise EvaluationError("PreparedJaxBatch representative session has no provenance.")
        rep_prov = rep.info.provenance

        case_ids: list[str] = []
        for s in sessions:
            cid = s.case_id
            if not isinstance(cid, str) or not cid:
                raise EvaluationError("PreparedJaxBatch: every session must have a non-empty case_id.")
            case_ids.append(cid)
            prov = s.info.provenance
            if prov is None or prov.backend != rep_prov.backend or prov.role is not rep_prov.role:
                raise EvaluationError("PreparedJaxBatch: all sessions must share provenance backend/role.")
            if len(s.layout) != rep_len:
                raise EvaluationError(f"PreparedJaxBatch: incompatible layout length {len(s.layout)} != {rep_len}.")
            if s._state.functional_form != rep_form:
                raise EvaluationError(
                    f"PreparedJaxBatch: incompatible functional form {s._state.functional_form!r} != {rep_form!r}."
                )
            if len(s.molecule.symbols) != rep_natoms:
                raise EvaluationError(
                    f"PreparedJaxBatch: incompatible atom count {len(s.molecule.symbols)} != {rep_natoms}."
                )
            if _topology_signature(s._state) != rep_sig:
                raise EvaluationError("PreparedJaxBatch: all sessions must share the same topology signature.")
        if len(set(case_ids)) != len(case_ids):
            raise EvaluationError(f"PreparedJaxBatch: case IDs must be unique; got {case_ids}.")

        self._sessions = list(sessions)
        # The representative native state owns the shared compiled kernel.
        self._state: _JaxState = rep_state
        self._case_ids = tuple(case_ids)
        # Defensive, read-only copies of each session's geometry.
        self._geometries = [readonly_array(s.molecule.geometry) for s in sessions]
        self._n_params = rep_len
        self._provenance = rep_prov

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Stable case IDs of the batched sessions, in row order."""
        return self._case_ids

    def hessians(self, request: BatchedHessianRequest) -> BatchedHessianResult:
        """Compute per-case Cartesian Hessians for one full parameter vector.

        Args:
            request: Typed request carrying a full parameter vector applied to
                every case in the batch.

        Returns:
            BatchedHessianResult: ``(n_cases, 3N, 3N)`` Hessians in
            Hartree/Bohr^2, one per case in :attr:`case_ids` order.

        Raises:
            EvaluationError: If the parameter vector length is wrong or the
                batched evaluation fails.

        """
        if not isinstance(request, BatchedHessianRequest):
            raise EvaluationError("PreparedJaxBatch.hessians expects a BatchedHessianRequest.")
        params_np = np.asarray(request.parameters, dtype=np.float64)
        if params_np.ndim != 1 or params_np.shape[0] != self._n_params:
            raise EvaluationError(
                f"PreparedJaxBatch: parameter vector has {params_np.shape} entries, "
                f"expected ({self._n_params},) (len(layout))."
            )
        try:
            hess_list = self._batched_hessians(params_np)
        except Exception as exc:  # noqa: BLE001 - normalize to typed error
            raise EvaluationError(f"JAX batched-Hessian evaluation failed: {exc}") from exc
        stacked = np.stack(hess_list, axis=0)
        return BatchedHessianResult(
            case_ids=self._case_ids,
            hessians=readonly_array(stacked),
            unit=HessianUnit.HARTREE_PER_BOHR2,
            provenance=self._provenance,
        )

    def _batched_hessians(self, params_np: np.ndarray) -> list[np.ndarray]:
        """Vectorized coordinate-Hessian evaluation over the batch geometries."""
        state = self._state
        params = jnp.array(params_np, dtype=jnp.float64)

        if state._coord_hess_fn is None:

            def _energy_of_flat_coords(flat_coords: jnp.ndarray, params_: jnp.ndarray) -> jnp.ndarray:
                return state._energy_fn(params_, flat_coords.reshape(-1, 3))

            state._coord_hess_fn = jax.jit(jax.hessian(_energy_of_flat_coords, argnums=0))

        if len(self._geometries) == 1:
            flat = jnp.array(self._geometries[0], dtype=jnp.float64).flatten()
            hess = state._coord_hess_fn(flat, params)
            return [np.asarray(hess) * KCALMOLA2_TO_HESSIAN_AU]

        if state._batched_coord_hess_fn is None:
            state._batched_coord_hess_fn = jax.jit(jax.vmap(state._coord_hess_fn, in_axes=(0, None)))

        batch_coords = jnp.stack([jnp.array(g, dtype=jnp.float64).flatten() for g in self._geometries])
        batch_hess = state._batched_coord_hess_fn(batch_coords, params)
        return [np.asarray(batch_hess[i]) * KCALMOLA2_TO_HESSIAN_AU for i in range(len(self._geometries))]


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def group_by_topology(sessions: list[PreparedJax]) -> list[PreparedJaxBatch]:
    """Group prepared sessions into typed batches by topology compatibility.

    Two sessions are compatible if they share bond/angle/torsion connectivity,
    vdW pair list, and parameter mappings -- in practice, multiple
    conformations (GS, TS) of the same molecule.  Each returned
    :class:`PreparedJaxBatch` shares only the compiled kernel of its
    representative session; every session keeps its own coordinates and native
    state.

    Args:
        sessions: Prepared JAX sessions to group.

    Returns:
        List of :class:`PreparedJaxBatch`, one per distinct topology.

    """
    groups: dict[str, list[PreparedJax]] = {}
    order: list[str] = []
    for session in sessions:
        sig = _topology_signature(session._state)
        if sig not in groups:
            groups[sig] = []
            order.append(sig)
        groups[sig].append(session)
    return [PreparedJaxBatch(groups[sig]) for sig in order]
