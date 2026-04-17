"""Tests for the JAX-native TS-curvature inversion helper.

See :func:`q2mm.models.hessian.invert_ts_curvature_jax`.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from q2mm.models.hessian import (  # noqa: E402
    invert_ts_curvature,
    invert_ts_curvature_jax,
)


def _ts_like_hessian(seed: int = 0) -> np.ndarray:
    """Synthesize a symmetric matrix with one negative eigenvalue."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((6, 6))
    sym = 0.5 * (a + a.T)
    _, evecs = np.linalg.eigh(sym)
    evals = np.array([-0.3, 0.1, 0.5, 0.9, 1.4, 2.0])
    return (evecs * evals) @ evecs.T


def _multi_neg_hessian() -> np.ndarray:
    """Symmetric matrix with two negative eigenvalues."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal((5, 5))
    sym = 0.5 * (a + a.T)
    _, evecs = np.linalg.eigh(sym)
    evals = np.array([-0.4, -0.1, 0.3, 0.7, 1.2])
    return (evecs * evals) @ evecs.T


@pytest.mark.jax
class TestInvertTsCurvatureJax:
    def test_parity_single_negative(self) -> None:
        """JAX version matches NumPy on a single-negative-eigenvalue Hessian."""
        hess = _ts_like_hessian()
        np_out = invert_ts_curvature(hess)
        jax_out = np.asarray(invert_ts_curvature_jax(jnp.asarray(hess)))
        np.testing.assert_allclose(jax_out, np_out, atol=1e-5)

    def test_parity_multi_negative(self) -> None:
        """Higher-order saddle: zero the extras, replace the most negative."""
        hess = _multi_neg_hessian()
        np_out = invert_ts_curvature(hess)
        jax_out = np.asarray(invert_ts_curvature_jax(jnp.asarray(hess)))
        np.testing.assert_allclose(jax_out, np_out, atol=1e-5)

    def test_positive_definite_passthrough(self) -> None:
        """No negative eigenvalues → output equals input (within eps)."""
        rng = np.random.default_rng(2)
        a = rng.standard_normal((6, 6))
        spd = a @ a.T + np.eye(6)
        out = np.asarray(invert_ts_curvature_jax(jnp.asarray(spd)))
        np.testing.assert_allclose(out, spd, atol=1e-5)

    def test_jit_compatible(self) -> None:
        """Must compile and run under ``jax.jit``."""
        hess = _ts_like_hessian()
        fn = jax.jit(invert_ts_curvature_jax)
        out = np.asarray(fn(jnp.asarray(hess)))
        expected = invert_ts_curvature(hess)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_differentiable_through_residual(self) -> None:
        """Gradients flow through the inversion (simulates loss-graph use)."""
        hess = _ts_like_hessian()

        def loss(h: jnp.ndarray) -> jnp.ndarray:
            h_inv = invert_ts_curvature_jax(h)
            return jnp.sum(jnp.linalg.eigvalsh(h_inv) ** 2)

        grad = jax.grad(loss)(jnp.asarray(hess))
        assert np.isfinite(np.asarray(grad)).all()
        assert np.asarray(grad).shape == hess.shape

    def test_smallest_replaced_is_positive(self) -> None:
        """After inversion, lowest eigenvalue must be ≥ 0 (convex)."""
        hess = _ts_like_hessian()
        out = np.asarray(invert_ts_curvature_jax(jnp.asarray(hess)))
        evals = np.linalg.eigvalsh(out)
        assert evals.min() >= -1e-5

    def test_custom_replace_value(self) -> None:
        """replace_with kwarg is honored."""
        hess = _ts_like_hessian()
        out = np.asarray(invert_ts_curvature_jax(jnp.asarray(hess), replace_with=2.5))
        evals = np.sort(np.linalg.eigvalsh(out))
        assert np.any(np.isclose(evals, 2.5, atol=1e-5))
