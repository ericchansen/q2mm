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
``docs/how-it-works/geometry-refs-spike.md`` for the design decision.

Usage::

    from q2mm.optimizers.jaxloss import JaxLoss

    spec = objective_function.to_jax_spec()
    jax_loss = JaxLoss(spec, engine, molecules, forcefield)

    loss = jax_loss(params)
    loss, grad = jax_loss.loss_and_grad(params)

"""

from __future__ import annotations

import logging
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
# inner_tol for well-conditioned convex problems (see geometry-refs-spike.md),
# but a tight-enough tol is needed so that ``∇_x E(x*) ≈ 0`` at which the
# implicit-function theorem applies.
_GEOM_INNER_TOL = 1e-8
_GEOM_INNER_MAXITER = 200


def _relax_coords(energy_fn, params, coords0):  # noqa: ANN001, ANN202
    """Return the relaxed geometry ``x*(params)`` for a single molecule.

    Uses ``jaxopt.LBFGS(fun=energy_of_coords, implicit_diff=True)`` so the
    outer ``jax.grad`` sees the exact parameter-gradient of ``x*`` via the
    implicit function theorem, avoiding autodiff-through-iteration.

    Args:
        energy_fn: ``(params, coords) -> scalar`` energy function
            (typically ``handle._energy_fn``).  Coords shape ``(N, 3)``.
        params: Current parameter vector (JAX array).
        coords0: Initial coordinates, shape ``(N, 3)``.

    Returns:
        Relaxed coordinates, shape ``(N, 3)``.

    """
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
    collinear geometries (see geometry-refs-spike.md).

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
    """JIT-compiled loss function for JAX-native optimization.

    Compiles a pure-JAX ``params → loss`` function from an
    :class:`~q2mm.optimizers.spec.ObjectiveSpec`.  The compiled function
    is fully compatible with ``jax.jit``, ``jax.grad``, and
    ``jax.value_and_grad``.

    The loss function is the sum of squared weighted residuals across
    all supported evaluator categories (energy, frequency,
    hessian-element, eigenmatrix), plus an optional L2 regularization
    term.

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
        self._compiled_loss_fn = None
        self._compiled_loss_and_grad_fn = None

        self._build()

    def _build(self) -> None:
        """Pre-build JaxHandles and compile the loss function."""
        from q2mm.backends.mm._jax_common import jax, jnp
        from q2mm.models.hessian import (
            _jax_frequencies_from_hessian,
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

            mol_data.append(entry)

        # Regularization arrays
        reg = spec.regularization
        ref_params = jnp.array(spec.reference_params, dtype=jnp.float64)

        hess_au_scale = float(KCALMOLA2_TO_HESSIAN_AU)

        def _loss_fn(params: np.ndarray) -> np.ndarray:
            """Pure JAX loss function: params → scalar loss."""
            total = jnp.float64(0.0)

            for entry in mol_data:
                handle = entry["handle"]
                coords = entry["coords"]
                mol_spec = entry["mol_spec"]
                energy_fn = handle._energy_fn

                # Energy contribution
                if mol_spec.has_energy:
                    energy = energy_fn(params, coords)
                    residuals = entry["energy_weights"] * (entry["energy_refs"] - energy)
                    total = total + jnp.sum(residuals**2)

                # Geometry contributions — relax coords at current params,
                # then compute bond/angle/torsion observables.  Implicit
                # differentiation through the inner jaxopt.LBFGS gives the
                # exact gradient w.r.t. params; see geometry-refs-spike.md.
                if mol_spec.has_geometry:
                    relaxed = _relax_coords(energy_fn, params, coords)

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
                        # Wrap torsion residuals into [-180, 180) before squaring.
                        diff = entry["torsion_refs"] - calc
                        diff = (diff + 180.0) % 360.0 - 180.0
                        residuals = entry["torsion_weights"] * diff
                        total = total + jnp.sum(residuals**2)

                # Hessian-dependent contributions
                if mol_spec.needs_hessian_computation:
                    flat_coords = coords.flatten()

                    def _energy_of_flat(fc: np.ndarray, p: np.ndarray) -> np.ndarray:
                        return energy_fn(p, fc.reshape(-1, 3))

                    hess_fn = jax.hessian(_energy_of_flat, argnums=0)
                    hess_kcal = hess_fn(flat_coords, params)
                    hess_au = hess_kcal * hess_au_scale

                    if mol_spec.invert_ts_curvature:
                        from q2mm.models.hessian import invert_ts_curvature_jax

                        hess_au = invert_ts_curvature_jax(hess_au)

                    # Frequency contribution
                    if mol_spec.has_frequency:
                        freqs = _jax_frequencies_from_hessian(hess_au, entry["masses_3n"])
                        calc_freqs = freqs[entry["freq_indices"]]
                        residuals = entry["freq_weights"] * (entry["freq_refs"] - calc_freqs)
                        total = total + jnp.sum(residuals**2)

                    # Hessian element contribution
                    if mol_spec.has_hessian:
                        n3 = hess_au.shape[0]
                        indices = entry["hess_indices"]
                        rows = indices // n3
                        cols = indices % n3
                        calc_hess = hess_au[rows, cols]
                        residuals = entry["hess_weights"] * (entry["hess_refs"] - calc_hess)
                        total = total + jnp.sum(residuals**2)

                    # Eigenmatrix contribution
                    if mol_spec.has_eigenmatrix:
                        qm_evecs = entry["qm_evecs"]
                        eigmat = qm_evecs.T @ hess_au @ qm_evecs

                        if "ediag_indices" in entry:
                            idx = entry["ediag_indices"]
                            calc_diag = eigmat[idx, idx]
                            residuals = entry["ediag_weights"] * (entry["ediag_refs"] - calc_diag)
                            total = total + jnp.sum(residuals**2)

                        if "eoff_indices" in entry:
                            idx = entry["eoff_indices"]
                            n3 = eigmat.shape[0]
                            rows = idx // n3
                            cols = idx % n3
                            calc_off = eigmat[rows, cols]
                            residuals = entry["eoff_weights"] * (entry["eoff_refs"] - calc_off)
                            total = total + jnp.sum(residuals**2)

            # L2 regularization
            if reg > 0:
                diff = params - ref_params
                total = total + reg * jnp.dot(diff, diff)

            return total

        self._loss_fn = _loss_fn
        self._compiled_loss_fn = jax.jit(_loss_fn)
        self._compiled_loss_and_grad_fn = jax.jit(jax.value_and_grad(_loss_fn))

    def __call__(self, params: np.ndarray) -> float:
        """Evaluate the JIT-compiled loss function.

        Args:
            params: Flat parameter vector (NumPy or JAX array).

        Returns:
            Scalar loss value.

        """
        from q2mm.backends.mm._jax_common import jnp

        p = jnp.array(params, dtype=jnp.float64)
        return float(self._compiled_loss_fn(p))

    def loss_and_grad(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate loss and gradient in a single JIT-compiled call.

        Args:
            params: Flat parameter vector.

        Returns:
            ``(loss, gradient)`` — loss is a scalar, gradient has the
            same shape as *params*.

        """
        from q2mm.backends.mm._jax_common import jnp

        p = jnp.array(params, dtype=jnp.float64)
        loss, grad = self._compiled_loss_and_grad_fn(p)
        return float(loss), np.asarray(grad)

    @property
    def spec(self) -> ObjectiveSpec:
        """The compiled objective specification."""
        return self._spec
