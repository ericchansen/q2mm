"""JIT-compiled loss function for JAX-native force field optimization.

Compiles an :class:`~q2mm.optimizers.spec.ObjectiveSpec` into a single
``jax.jit``-compiled ``params → loss`` function that runs entirely inside
JAX's XLA backend.  This eliminates Python-loop overhead and enables
end-to-end gradient computation via ``jax.grad``.

The compiled loss supports **energy**, **frequency**, **hessian-element**,
**eigenmatrix**, and **geometry** (bond_length / bond_angle /
torsion_angle) reference types.  Geometry references are handled via
implicit differentiation: each loss call runs an inner
``jaxopt.LBFGS(fun=handle._energy_fn, implicit_diff=True)`` geometry
minimization at the current parameters, computes bond/angle/torsion
observables from the relaxed coordinates, and accumulates weighted
residuals.  The implicit-function theorem gives the exact gradient of
the relaxed observables with respect to the force-field parameters
without autodiff-through-iteration.  See
issue #243 (https://github.com/ericchansen/q2mm/issues/243) for the design decision.

Usage::

    from q2mm.optimizers.jaxloss import JaxLoss

    spec = objective_function.to_jax_spec()
    jax_loss = JaxLoss(spec, engine, molecules, forcefield)

    loss = jax_loss(params)
    loss, grad = jax_loss.loss_and_grad(params)

"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from q2mm.backends.mm.jax_engine import JaxEngine, JaxHandle
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Q2MMMolecule
    from q2mm.optimizers.spec import ObjectiveSpec


# Tolerance + iteration cap for the inner geometry minimizer used when a
# molecule has geometry references.  The outer gradient does NOT depend on
# inner_tol for well-conditioned convex problems (see issue #243),
# but a tight-enough tol is needed so that ``∇_x E(x*) ≈ 0`` at which the
# implicit-function theorem applies.
_GEOM_INNER_TOL = 1e-8
_GEOM_INNER_MAXITER = 500
# Penalty added per geometry reference when the inner solver does not
# converge.  Set to 0.0 because:
# 1. The actual geometry residuals (bonds/angles) naturally penalize poor
#    convergence — wrong structures yield large residuals.
# 2. A large binary penalty (the original 1e4) caused 40× score inflation
#    for nearly-converged geometries where jaxopt's L-BFGS didn't reach
#    the strict 1e-8 gradient-norm tolerance, making the optimizer report
#    false divergence.
_GEOM_NONCONV_PENALTY = 0.0
# Harmonic restraint (kcal/(mol·Å²) per Cartesian coordinate) that keeps the
# inner geometry relaxation near the initial QM structure.  For well-behaved
# systems the minimum is very close to coords0, so a moderate restraint adds
# negligible energy (<0.1 kcal/mol total).  For pathological TS systems with
# negative force constants (up to −3753 kcal/(mol·rad²) in Pd benchmarks),
# the restraint prevents the LBFGS solver from diverging to NaN.  The
# restraint also improves conditioning of the implicit-differentiation
# Hessian by adding k·I to ∂²E/∂x².
_GEOM_RESTRAINT_K = 100.0


def _relax_coords(energy_fn, params, coords0):  # noqa: ANN001, ANN202
    """Return the relaxed geometry and convergence flag for a molecule.

    Uses ``jaxopt.LBFGS(fun=energy_of_coords, implicit_diff=True)`` so the
    outer ``jax.grad`` sees the exact parameter-gradient of ``x*`` via the
    implicit function theorem, avoiding autodiff-through-iteration.

    A harmonic restraint to the initial geometry prevents divergence for
    transition-state systems where negative force constants produce an
    unbounded PES.

    When the inner solver does not converge (gradient norm > tol after
    maxiter), the caller should add a penalty to the loss.

    Args:
        energy_fn: ``(params, coords) -> scalar`` energy function
            (typically ``handle._energy_fn``).  Coords shape ``(N, 3)``.
        params: Current parameter vector (JAX array).
        coords0: Initial coordinates, shape ``(N, 3)``.

    Returns:
        tuple: ``(relaxed_coords, converged)`` where ``converged`` is a
        scalar boolean (JAX array).

    """
    import jaxopt

    from q2mm.backends.mm._jax_common import jnp

    def energy_of_coords(coords, p):  # noqa: ANN001, ANN202
        mm_energy = energy_fn(p, coords)
        # Harmonic restraint to initial geometry — prevents divergence for
        # TS systems with negative force constants while barely affecting
        # well-behaved systems where coords ≈ coords0.
        disp = coords - coords0
        restraint = 0.5 * _GEOM_RESTRAINT_K * jnp.sum(disp * disp)
        return mm_energy + restraint

    solver = jaxopt.LBFGS(
        fun=energy_of_coords,
        tol=_GEOM_INNER_TOL,
        maxiter=_GEOM_INNER_MAXITER,
        implicit_diff=True,
    )
    sol = solver.run(coords0, params)
    converged = sol.state.error <= _GEOM_INNER_TOL
    return sol.params, converged


def _bond_lengths(coords, atoms):  # noqa: ANN001, ANN202
    """Compute bond lengths (Å) for pairs of atoms.

    Args:
        coords: ``(N, 3)`` Cartesian coordinates.
        atoms: ``(M, 2)`` integer array of atom-index pairs.

    Returns:
        ``(M,)`` array of bond lengths.

    """
    from q2mm.backends.mm._jax_common import jnp

    d = coords[atoms[:, 0]] - coords[atoms[:, 1]]
    return jnp.sqrt(jnp.sum(d * d, axis=-1))


def _bond_angles_deg(coords, atoms):  # noqa: ANN001, ANN202
    """Compute bond angles in degrees for atom triples.

    The middle atom is the vertex.  Norms are floored by a small
    epsilon so that degenerate (zero-length arm) geometries encountered
    during relaxation produce finite values instead of NaNs, and the
    cos is clipped to ``[-1+ε, 1-ε]`` to avoid NaN gradients at
    collinear geometries (see issue #243).

    Args:
        coords: ``(N, 3)`` Cartesian coordinates.
        atoms: ``(M, 3)`` integer array of atom-index triples
            (outer, vertex, outer).

    Returns:
        ``(M,)`` array of angles in degrees.

    """
    from q2mm.backends.mm._jax_common import jnp

    v1 = coords[atoms[:, 0]] - coords[atoms[:, 1]]
    v2 = coords[atoms[:, 2]] - coords[atoms[:, 1]]
    n1 = jnp.linalg.norm(v1, axis=-1)
    n2 = jnp.linalg.norm(v2, axis=-1)
    denom = jnp.maximum(n1 * n2, 1e-12)
    cos = jnp.sum(v1 * v2, axis=-1) / denom
    cos = jnp.clip(cos, -1.0 + 1e-12, 1.0 - 1e-12)
    return jnp.arccos(cos) * (180.0 / jnp.pi)


def _torsion_angles_deg(coords, atoms):  # noqa: ANN001, ANN202
    """Compute torsion (dihedral) angles in degrees for atom quadruples.

    Uses the numerically stable ``atan2`` formulation so the result is
    smooth across the ±180° wrap.  Norms are floored by a small
    epsilon so degenerate geometries (collinear triplets, zero-length
    ``b2``) produce finite values instead of NaN/inf, matching the
    NumPy reference behavior in :func:`q2mm.geometry.dihedral_angle`.

    Args:
        coords: ``(N, 3)`` Cartesian coordinates.
        atoms: ``(M, 4)`` integer array of atom-index quadruples.

    Returns:
        ``(M,)`` array of dihedrals in degrees, in ``[-180, 180]``.

    """
    from q2mm.backends.mm._jax_common import jnp

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


class JaxLoss:
    """Per-molecule JIT-compiled loss for JAX-native optimization.

    Each molecule's loss contribution (Hessian-derived and geometry) is
    compiled into its own small XLA program.  The aggregate loss and
    gradient are computed by dispatching these per-molecule programs from
    Python and summing — no single XLA program contains all molecules,
    which prevents compilation OOM on multi-molecule systems.

    The primary API is :meth:`loss_and_grad` (returns Python float +
    NumPy array) and :meth:`value_and_grad_jax` (returns JAX types,
    for use with jaxopt ``value_and_grad=True``).

    A JAX-traceable :attr:`_loss_fn` is also available for callers that
    need to JIT or vmap the loss (e.g. JaxMultiStartOptimizer).  This
    function loops over the per-molecule compiled functions; when traced,
    the loop is unrolled.  It is safe for single-molecule systems but
    will OOM on large multi-molecule systems if JIT'd externally.

    Args:
        spec: Compiled objective specification.
        engine: JaxEngine instance (must be a JaxEngine).
        molecules: Training set molecules (same order as spec).
        forcefield: Base force field (for topology/handle creation).

    Raises:
        TypeError: If engine is not a JaxEngine.
        ValueError: If spec has no supported categories.

    """

    def __init__(
        self,
        spec: ObjectiveSpec,
        engine: JaxEngine,
        molecules: list[Q2MMMolecule],
        forcefield: ForceField,
    ) -> None:
        from q2mm.backends.mm._jax_common import ensure_jax
        from q2mm.backends.mm.jax_engine import JaxEngine

        if not isinstance(engine, JaxEngine):
            raise TypeError(f"JaxLoss requires a JaxEngine, got {type(engine).__name__}")

        ensure_jax(engine_name="JaxLoss")

        self._spec = spec
        self._engine = engine
        self._molecules = molecules
        self._forcefield = forcefield

        # Pre-build handles and compile per-molecule loss fragments
        self._handles: dict[int, JaxHandle] = {}
        self._loss_fn: Callable | None = None
        self._compiled_nongeom_vag_fns: list = []
        self._compiled_geom_vag_fns: list = []
        self._compiled_reg_vag_fn: Callable | None = None

        self._build()

    def _build(self) -> None:
        """Pre-build JaxHandles and compile per-molecule loss functions.

        Each molecule's Hessian-derived loss (eigenmatrix, frequency,
        hessian-element, energy) and geometry loss are JIT-compiled
        independently.  This prevents XLA compilation OOM on multi-molecule
        systems where a monolithic graph would exceed GPU memory.

        The aggregate loss and gradient are computed by dispatching the
        per-molecule compiled functions from Python and summing — no
        outer JIT wraps the full set.
        """
        from q2mm.backends.mm._jax_common import jax, jnp
        from q2mm.models.hessian import (
            _jax_frequencies_from_hessian,
            invert_ts_curvature_jax,
            symbols_to_masses_3n,
        )
        from q2mm.models.units import KCALMOLA2_TO_HESSIAN_AU

        spec = self._spec
        engine = self._engine
        forcefield = self._forcefield

        # Pre-build handles for all molecules in the spec
        for mol_spec in spec.molecules:
            mol = self._molecules[mol_spec.mol_idx]
            handle = engine._get_handle(mol, forcefield)
            self._handles[mol_spec.mol_idx] = handle

        # Pre-compute static data for each molecule
        mol_data = []
        for mol_spec in spec.molecules:
            handle = self._handles[mol_spec.mol_idx]
            mol = self._molecules[mol_spec.mol_idx]
            coords = jnp.array(mol.geometry, dtype=jnp.float64)

            entry: dict = {
                "mol_spec": mol_spec,
                "handle": handle,
                "coords": coords,
            }

            if mol_spec.has_frequency or mol_spec.has_hessian or mol_spec.has_eigenmatrix:
                entry["masses_3n"] = jnp.array(symbols_to_masses_3n(mol_spec.symbols), dtype=jnp.float64)

            if mol_spec.has_eigenmatrix:
                if mol.hessian is None:
                    raise ValueError(
                        f"Molecule {mol_spec.mol_idx} ({mol.name}) has no QM Hessian. "
                        "Eigenmatrix training requires a QM Hessian."
                    )
                from q2mm.models.hessian import decompose

                _, qm_evecs = decompose(mol.hessian)
                entry["qm_evecs"] = jnp.array(qm_evecs, dtype=jnp.float64)

            # Convert reference arrays to JAX
            if mol_spec.has_energy:
                entry["energy_refs"] = jnp.array(mol_spec.energy_refs)
                entry["energy_weights"] = jnp.array(mol_spec.energy_weights)
            if mol_spec.has_frequency:
                entry["freq_indices"] = jnp.array(mol_spec.freq_indices, dtype=jnp.int32)
                entry["freq_refs"] = jnp.array(mol_spec.freq_refs)
                entry["freq_weights"] = jnp.array(mol_spec.freq_weights)
            if mol_spec.has_hessian:
                entry["hess_indices"] = jnp.array(mol_spec.hess_indices, dtype=jnp.int32)
                entry["hess_refs"] = jnp.array(mol_spec.hess_refs)
                entry["hess_weights"] = jnp.array(mol_spec.hess_weights)
            if mol_spec.has_eigenmatrix:
                if len(mol_spec.eig_diag_refs) > 0:
                    entry["ediag_indices"] = jnp.array(mol_spec.eig_diag_indices, dtype=jnp.int32)
                    entry["ediag_refs"] = jnp.array(mol_spec.eig_diag_refs)
                    entry["ediag_weights"] = jnp.array(mol_spec.eig_diag_weights)
                if len(mol_spec.eig_offdiag_refs) > 0:
                    entry["eoff_indices"] = jnp.array(mol_spec.eig_offdiag_indices, dtype=jnp.int32)
                    entry["eoff_refs"] = jnp.array(mol_spec.eig_offdiag_refs)
                    entry["eoff_weights"] = jnp.array(mol_spec.eig_offdiag_weights)
            if mol_spec.has_bond_length:
                entry["bond_atoms"] = jnp.array(mol_spec.bond_atoms, dtype=jnp.int32)
                entry["bond_refs"] = jnp.array(mol_spec.bond_refs)
                entry["bond_weights"] = jnp.array(mol_spec.bond_weights)
            if mol_spec.has_bond_angle:
                entry["angle_atoms"] = jnp.array(mol_spec.angle_atoms, dtype=jnp.int32)
                entry["angle_refs"] = jnp.array(mol_spec.angle_refs)
                entry["angle_weights"] = jnp.array(mol_spec.angle_weights)
            if mol_spec.has_torsion:
                entry["torsion_atoms"] = jnp.array(mol_spec.torsion_atoms, dtype=jnp.int32)
                entry["torsion_refs"] = jnp.array(mol_spec.torsion_refs)
                entry["torsion_weights"] = jnp.array(mol_spec.torsion_weights)
            if mol_spec.has_geometry:
                entry["n_geom_refs"] = len(mol_spec.bond_refs) + len(mol_spec.angle_refs) + len(mol_spec.torsion_refs)

            mol_data.append(entry)

        # ---- Per-molecule Hessian functions ----
        #
        # Build a Hessian function per topology group.  Each per-molecule
        # loss is JIT-compiled independently, preventing XLA compilation
        # OOM on multi-molecule systems.
        hess_fn_cache: dict[int, object] = {}  # id(handle) → hess_fn

        hess_au_scale = float(KCALMOLA2_TO_HESSIAN_AU)

        def _make_hess_fn(efn):  # noqa: ANN001, ANN202
            """Closure to capture the correct energy_fn per topology."""

            def _energy_of_flat(fc, p):  # noqa: ANN001, ANN202
                return efn(p, fc.reshape(-1, 3))

            return jax.hessian(_energy_of_flat, argnums=0)

        for entry in mol_data:
            handle = entry["handle"]
            h_id = id(handle)
            if h_id not in hess_fn_cache and entry["mol_spec"].needs_hessian_computation:
                hess_fn_cache[h_id] = _make_hess_fn(handle._energy_fn)

        # ---- Per-molecule non-geometry loss factory ----

        def _make_mol_nongeom_loss(entry_data: dict, mol_hess_fn: object | None, scale: float) -> Callable:
            """Return a ``params → scalar`` loss for one molecule's non-geometry refs."""
            ms = entry_data["mol_spec"]
            coords = entry_data["coords"]
            flat_coords = coords.reshape(-1)
            handle = entry_data["handle"]
            energy_fn = handle._energy_fn

            def _mol_loss(params: np.ndarray) -> np.ndarray:
                total = jnp.float64(0.0)

                if ms.has_energy:
                    energy = energy_fn(params, coords)
                    residuals = entry_data["energy_weights"] * (entry_data["energy_refs"] - energy)
                    total = total + jnp.sum(residuals**2)

                if ms.needs_hessian_computation:
                    hess_au = mol_hess_fn(flat_coords, params) * scale

                    if ms.invert_ts_curvature:
                        hess_au = invert_ts_curvature_jax(hess_au)

                    if ms.has_frequency:
                        freqs = _jax_frequencies_from_hessian(hess_au, entry_data["masses_3n"])
                        calc_freqs = freqs[entry_data["freq_indices"]]
                        residuals = entry_data["freq_weights"] * (entry_data["freq_refs"] - calc_freqs)
                        total = total + jnp.sum(residuals**2)

                    if ms.has_hessian:
                        n3 = hess_au.shape[0]
                        indices = entry_data["hess_indices"]
                        rows = indices // n3
                        cols = indices % n3
                        calc_hess = hess_au[rows, cols]
                        residuals = entry_data["hess_weights"] * (entry_data["hess_refs"] - calc_hess)
                        total = total + jnp.sum(residuals**2)

                    if ms.has_eigenmatrix:
                        qm_evecs = entry_data["qm_evecs"]
                        eigmat = qm_evecs.T @ hess_au @ qm_evecs

                        if "ediag_indices" in entry_data:
                            idx = entry_data["ediag_indices"]
                            calc_diag = eigmat[idx, idx]
                            residuals = entry_data["ediag_weights"] * (entry_data["ediag_refs"] - calc_diag)
                            total = total + jnp.sum(residuals**2)

                        if "eoff_indices" in entry_data:
                            idx = entry_data["eoff_indices"]
                            n3e = eigmat.shape[0]
                            rows = idx // n3e
                            cols = idx % n3e
                            calc_off = eigmat[rows, cols]
                            residuals = entry_data["eoff_weights"] * (entry_data["eoff_refs"] - calc_off)
                            total = total + jnp.sum(residuals**2)

                return total

            return _mol_loss

        # Build per-molecule non-geometry loss functions
        nongeom_loss_fns: list[Callable] = []
        for entry in mol_data:
            ms = entry["mol_spec"]
            if not ms.needs_hessian_computation and not ms.has_energy:
                continue
            h_id = id(entry["handle"])
            mol_hess_fn = hess_fn_cache.get(h_id)
            nongeom_loss_fns.append(_make_mol_nongeom_loss(entry, mol_hess_fn, hess_au_scale))

        n_nongeom = len(nongeom_loss_fns)
        logger.debug("JaxLoss: %d per-molecule non-geometry loss functions", n_nongeom)

        # ---- Per-molecule geometry loss ----

        def _geometry_loss_from_entry(entry: dict, params: np.ndarray) -> np.ndarray:
            total = jnp.float64(0.0)
            handle = entry["handle"]
            coords = entry["coords"]
            mol_spec = entry["mol_spec"]
            energy_fn = handle._energy_fn

            relaxed, geom_converged = _relax_coords(energy_fn, params, coords)
            total = total + jnp.where(
                geom_converged,
                0.0,
                _GEOM_NONCONV_PENALTY * entry["n_geom_refs"],
            )

            if mol_spec.has_bond_length:
                calc = _bond_lengths(relaxed, entry["bond_atoms"])
                residuals = entry["bond_weights"] * (entry["bond_refs"] - calc)
                total = total + jnp.sum(residuals**2)

            if mol_spec.has_bond_angle:
                calc = _bond_angles_deg(relaxed, entry["angle_atoms"])
                residuals = entry["angle_weights"] * (entry["angle_refs"] - calc)
                total = total + jnp.sum(residuals**2)

            if mol_spec.has_torsion:
                calc = _torsion_angles_deg(relaxed, entry["torsion_atoms"])
                diff = entry["torsion_refs"] - calc
                diff = (diff + 180.0) % 360.0 - 180.0
                residuals = entry["torsion_weights"] * diff
                total = total + jnp.sum(residuals**2)

            return total

        geom_loss_fns: list[Callable] = []
        for entry in mol_data:
            if not entry["mol_spec"].has_geometry:
                continue

            def _make_geom_loss_fn(entry_data: dict) -> Callable[[np.ndarray], np.ndarray]:
                def _geom_loss_fn(params: np.ndarray) -> np.ndarray:
                    return _geometry_loss_from_entry(entry_data, params)

                return _geom_loss_fn

            geom_loss_fns.append(_make_geom_loss_fn(entry))

        n_geom = len(geom_loss_fns)
        logger.debug("JaxLoss: %d per-molecule geometry loss functions", n_geom)

        # ---- Regularization ----

        reg = spec.regularization
        ref_params = jnp.array(spec.reference_params, dtype=jnp.float64)

        def _reg_fn(params: np.ndarray) -> np.ndarray:
            diff = params - ref_params
            return reg * jnp.dot(diff, diff)

        # ---- JIT-compile per-molecule functions ----

        self._compiled_nongeom_vag_fns = [jax.jit(jax.value_and_grad(fn)) for fn in nongeom_loss_fns]
        self._compiled_geom_vag_fns = [jax.jit(jax.value_and_grad(fn)) for fn in geom_loss_fns]
        if reg > 0:
            self._compiled_reg_vag_fn = jax.jit(jax.value_and_grad(_reg_fn))

        # ---- Aggregate _loss_fn (JAX-traceable) ----
        #
        # This function loops over per-molecule JIT'd functions.  When
        # called from Python, each inner call dispatches to its pre-compiled
        # XLA executable.  When JIT'd or vmapped (e.g. by JaxMultiStart),
        # the loop is unrolled — safe for single-molecule systems but will
        # OOM on large multi-molecule systems.  Standard optimizers should
        # use loss_and_grad() (Python dispatch) instead.
        compiled_nongeom_fns = tuple(jax.jit(fn) for fn in nongeom_loss_fns)
        compiled_geom_fns = tuple(jax.jit(fn) for fn in geom_loss_fns)
        compiled_reg = jax.jit(_reg_fn) if reg > 0 else None

        def _loss_fn(params: np.ndarray) -> np.ndarray:
            total = jnp.float64(0.0)
            for fn in compiled_nongeom_fns:
                total = total + fn(params)
            for fn in compiled_geom_fns:
                total = total + fn(params)
            if compiled_reg is not None:
                total = total + compiled_reg(params)
            return total

        self._loss_fn = _loss_fn

    def __call__(self, params: np.ndarray) -> float:
        """Evaluate the loss via Python dispatch over per-molecule functions.

        Each per-molecule JIT'd function is dispatched independently,
        avoiding the XLA compilation OOM that occurs when all molecules
        are compiled into one graph.

        Args:
            params: Flat parameter vector (NumPy or JAX array).

        Returns:
            Scalar loss value.

        """
        loss, _ = self.value_and_grad_jax(params)
        return float(loss)

    def value_and_grad_jax(self, params: np.ndarray) -> tuple:
        """Evaluate loss and gradient, returning JAX-native types.

        This is the core dispatcher.  Each per-molecule compiled
        ``value_and_grad`` function is called independently from Python,
        and the results are accumulated as JAX arrays.

        Use this method when the caller needs JAX array outputs
        (e.g. jaxopt with ``value_and_grad=True``).

        Args:
            params: Flat parameter vector (NumPy or JAX array).

        Returns:
            ``(loss_jax, gradient_jax)`` — JAX scalar and JAX array.

        """
        from q2mm.backends.mm._jax_common import jnp

        p = jnp.array(params, dtype=jnp.float64)
        total_loss = jnp.float64(0.0)
        total_grad = jnp.zeros_like(p)

        for fn in self._compiled_nongeom_vag_fns:
            loss_i, grad_i = fn(p)
            total_loss = total_loss + loss_i
            total_grad = total_grad + grad_i

        for fn in self._compiled_geom_vag_fns:
            loss_i, grad_i = fn(p)
            total_loss = total_loss + loss_i
            total_grad = total_grad + grad_i

        if self._compiled_reg_vag_fn is not None:
            reg_loss, reg_grad = self._compiled_reg_vag_fn(p)
            total_loss = total_loss + reg_loss
            total_grad = total_grad + reg_grad

        return total_loss, total_grad

    def loss_and_grad(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate loss and gradient, returning host types.

        Convenience wrapper around :meth:`value_and_grad_jax` that
        converts the loss to a Python float and the gradient to a
        NumPy array.

        When the loss or gradient contains NaN/Inf (e.g. from
        out-of-range parameters), returns a large finite penalty
        (``1e30``) and a zero gradient so that line-search optimizers
        like L-BFGS-B can recover gracefully.

        Args:
            params: Flat parameter vector.

        Returns:
            ``(loss, gradient)`` — loss is a Python float, gradient is
            a NumPy array with the same shape as *params*.

        """
        loss_jax, grad_jax = self.value_and_grad_jax(params)
        loss = float(loss_jax)
        grad = np.asarray(grad_jax)
        if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
            logger.warning("JaxLoss returned non-finite values; substituting penalty")
            return 1e30, np.zeros_like(grad)
        return loss, grad

    @property
    def spec(self) -> ObjectiveSpec:
        """The compiled objective specification."""
        return self._spec
