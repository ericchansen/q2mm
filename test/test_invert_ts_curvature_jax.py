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

    def test_parity_asymmetric_input(self) -> None:
        """F8: NumPy version symmetrizes its input like the JAX twin.

        Regression: the NumPy ``invert_ts_curvature`` fed the raw matrix to
        ``eigh`` (which reads a single triangle), so an asymmetric input
        silently diverged from ``invert_ts_curvature_jax`` (which symmetrizes
        first).  Both must now agree, and both must equal the result of
        explicitly symmetrizing before inversion.
        """
        hess = _ts_like_hessian()
        asym = hess.copy()
        asym[0, 1] += 0.2
        asym[1, 0] -= 0.2

        np_out = invert_ts_curvature(asym)
        jax_out = np.asarray(invert_ts_curvature_jax(jnp.asarray(asym)))
        np.testing.assert_allclose(np_out, jax_out, atol=1e-5)

        sym = 0.5 * (asym + asym.T)
        np.testing.assert_allclose(np_out, invert_ts_curvature(sym), atol=1e-5)


@pytest.mark.jax
class TestJaxSupportModule:
    """Direct coverage for :mod:`q2mm._jax_support`.

    ``q2mm.models.hessian``'s three JAX-native helpers (this file's
    ``invert_ts_curvature_jax``, plus the frequency/frequency-sensitivity
    helpers) and ``q2mm.backends.mm._jax_common.ensure_jax`` all delegate
    to this one dependency-free, foundational module rather than either
    depending on the other (see
    ``test_architecture_doc.py::test_models_package_never_imports_outer_layers``).
    """

    def test_has_jax_is_true_here(self) -> None:
        """Confirm this test file itself requires jax (see ``pytest.importorskip`` above)."""
        from q2mm._jax_support import has_jax

        assert has_jax() is True

    def test_has_jax_is_side_effect_free(self) -> None:
        """Calling ``has_jax()`` must never trigger the lazy JAX import."""
        import q2mm._jax_support as jax_support

        was_initialized = jax_support._initialized
        jax_support.has_jax()
        # Merely calling has_jax() must not change initialization state,
        # regardless of whether some earlier test already initialized it.
        assert jax_support._initialized == was_initialized

    def test_load_jax_populates_cache_and_returns_jax_numpy(self) -> None:
        from q2mm._jax_support import load_jax

        jax_mod, jnp = load_jax(caller_name="test_load_jax_populates_cache_and_returns_jax_numpy")
        assert jax_mod is not None
        assert jnp is not None
        # A trivial numerical sanity check that the cached module is a real,
        # working jax.numpy, not a stub.
        assert float(jnp.asarray([1.0, 2.0]).sum()) == pytest.approx(3.0)

    def test_load_jax_is_idempotent(self) -> None:
        from q2mm._jax_support import load_jax

        jax_first, jnp_first = load_jax(caller_name="first_call")
        jax_second, jnp_second = load_jax(caller_name="second_call")
        assert jax_second is jax_first
        assert jnp_second is jnp_first

    def test_load_jax_enables_float64(self) -> None:
        """Hessian eigenvalue analysis needs float64 (see module docstring).

        By the time this test runs, some test in the session has always
        already imported ``jax`` at least once (this very module does so
        at collection time via ``pytest.importorskip("jax")`` above), so
        ``jax.config.jax_enable_x64`` — a process-wide, one-way switch —
        is expected to already be ``True`` by the normal env-var-honoring
        contract of :func:`q2mm._jax_support.load_jax`.
        """
        from q2mm._jax_support import load_jax

        _, jnp = load_jax(caller_name="test_load_jax_enables_float64")
        arr = jnp.asarray([1.0])
        assert str(arr.dtype) == "float64"

    def test_jax_common_and_hessian_share_the_same_cached_jax_module(self) -> None:
        """Both ``_jax_common.ensure_jax`` and a direct ``load_jax`` call resolve identically.

        One canonical import, not two independent ones: they must
        resolve to the exact same cached ``jax``/``jnp`` module objects.
        """
        from q2mm._jax_support import load_jax
        from q2mm.backends.mm import _jax_common

        _jax_common.ensure_jax(engine_name="test_shared_cache")
        jax_direct, jnp_direct = load_jax(caller_name="test_shared_cache")
        assert _jax_common.jax is jax_direct
        assert _jax_common.jnp is jnp_direct

    def test_module_is_dependency_free(self) -> None:
        """``q2mm/_jax_support.py`` must import nothing else from ``q2mm``.

        It is the one canonical, foundational JAX-import-guard helper
        shared by both ``q2mm.models.hessian`` (which cannot depend on
        ``q2mm.backends``) and ``q2mm.backends.mm._jax_common``, so it
        must sit genuinely below both — not merely attached to one of
        them — the same way ``q2mm.constants``/``q2mm.geometry`` do.
        """
        import ast
        from pathlib import Path

        import q2mm._jax_support as jax_support

        path = Path(jax_support.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        q2mm_imports = sorted(mod for mod in modules if mod == "q2mm" or mod.startswith("q2mm."))
        assert not q2mm_imports, (
            f"q2mm/_jax_support.py must stay dependency-free from the rest of q2mm, but imports: {q2mm_imports}"
        )
