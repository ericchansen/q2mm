"""JIT-compiled loss function for JAX-native force field optimization.

Compiles an :class:`~q2mm.optimizers.spec.ObjectiveSpec` into a single
``jax.jit``-compiled ``params → loss`` function that runs entirely inside
JAX's XLA backend.  This eliminates Python-loop overhead and enables
end-to-end gradient computation via ``jax.grad``.

The compiled loss supports **energy**, **frequency**, **hessian-element**,
and **eigenmatrix** reference types.  Geometry references (bond_length,
bond_angle, torsion_angle) are excluded — they require differentiable
energy minimization via implicit differentiation, which is planned for
a later phase.

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
            _jax_frequency_param_jacobian,
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

                # Hessian-dependent contributions
                if mol_spec.needs_hessian_computation:
                    flat_coords = coords.flatten()

                    def _energy_of_flat(fc: np.ndarray, p: np.ndarray) -> np.ndarray:
                        return energy_fn(p, fc.reshape(-1, 3))

                    hess_fn = jax.hessian(_energy_of_flat, argnums=0)
                    hess_kcal = hess_fn(flat_coords, params)
                    hess_au = hess_kcal * hess_au_scale

                    # Frequency contribution
                    if mol_spec.has_frequency:
                        dh_dp_kcal = jax.jacrev(hess_fn, argnums=1)(flat_coords, params)
                        dh_dp_au = dh_dp_kcal * hess_au_scale

                        freqs, d_freq_dp = _jax_frequency_param_jacobian(hess_au, dh_dp_au, entry["masses_3n"])
                        calc_freqs = freqs[entry["freq_indices"]]
                        residuals = entry["freq_weights"] * (entry["freq_refs"] - calc_freqs)
                        total = total + jnp.sum(residuals**2)

                    # Hessian element contribution
                    if mol_spec.has_hessian:
                        if not mol_spec.has_frequency:
                            dh_dp_kcal = jax.jacrev(hess_fn, argnums=1)(flat_coords, params)
                            dh_dp_au = dh_dp_kcal * hess_au_scale

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
