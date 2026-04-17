"""Geometry-refs spike: compare implicit-diff approaches for inner minimization.

Runs both Option A (:class:`jaxopt.LBFGS` with ``implicit_diff=True``) and
Option B (hand-rolled :func:`jax.custom_vjp` via the implicit function theorem)
on a triatomic toy system whose relaxed geometry has a closed form. Emits
three tables (accuracy, timing, ill-conditioning) that back the decision
documented in ``docs/how-it-works/geometry-refs-spike.md``.

Run with::

    JAX_PLATFORMS=cpu .venv/bin/python scripts/spike_geom_implicit_diff.py

Test system: 1D linear triatomic A-B-C with atom A fixed at the origin::

    E(x_B, x_C; k1, r1, k2, r2) =
        0.5 * k1 * (x_B - r1)^2
      + 0.5 * k2 * ((x_C - x_B) - r2)^2
      - F * x_C                                   (external force on atom C)

Closed-form relaxed geometry (from stationarity)::

    b_12* = x_B*              = r1 + F / k1
    b_23* = x_C* - x_B*       = r2 + F / k2

Observables: bond lengths ``b_12`` and ``b_23``.
Loss: ``L(p) = (b_12* - b1_ref)^2 + (b_23* - b2_ref)^2``.

Because the closed form is known, every gradient number in the spike
tables can be checked against an exact analytical derivative.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

# Default to CPU for reproducible spike numbers; override with JAX_PLATFORMS=cuda.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize as jax_minimize
from jaxopt import LBFGS

jax.config.update("jax_enable_x64", True)

FORCE = 0.1  # external force on atom C, in arbitrary units


# -----------------------------------------------------------------------------
# Toy system
# -----------------------------------------------------------------------------


def energy(x: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Harmonic two-bond potential plus external force on atom C."""
    k1, r1, k2, r2 = p
    x_B, x_C = x[0], x[1]
    return 0.5 * k1 * (x_B - r1) ** 2 + 0.5 * k2 * ((x_C - x_B) - r2) ** 2 - FORCE * x_C


def observables(x: jnp.ndarray) -> jnp.ndarray:
    """Return ``(b_12, b_23)`` bond lengths from atomic coordinates."""
    return jnp.array([x[0], x[1] - x[0]])


def closed_form_geometry(p: jnp.ndarray) -> jnp.ndarray:
    """Closed-form relaxed geometry ``(x_B*, x_C*)`` derived from stationarity."""
    k1, r1, k2, r2 = p
    x_B = r1 + FORCE / k1
    x_C = x_B + r2 + FORCE / k2
    return jnp.array([x_B, x_C])


def closed_form_grad(p: jnp.ndarray, refs: jnp.ndarray) -> jnp.ndarray:
    """Closed-form loss gradient ``∂L/∂p`` for the bond-length objective."""
    k1, r1, k2, r2 = p
    b12 = r1 + FORCE / k1
    b23 = r2 + FORCE / k2
    d12 = 2.0 * (b12 - refs[0])
    d23 = 2.0 * (b23 - refs[1])
    return jnp.array(
        [
            d12 * (-FORCE / k1**2),  # ∂L/∂k1
            d12,  # ∂L/∂r1
            d23 * (-FORCE / k2**2),  # ∂L/∂k2
            d23,  # ∂L/∂r2
        ]
    )


# -----------------------------------------------------------------------------
# Option A: jaxopt.LBFGS with implicit_diff=True
# -----------------------------------------------------------------------------


def make_option_a(tol: float = 1e-10, maxiter: int = 200) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a relaxed-geometry function using jaxopt's built-in implicit diff."""
    solver = LBFGS(fun=energy, tol=tol, maxiter=maxiter, implicit_diff=True)

    def relax(p: jnp.ndarray) -> jnp.ndarray:
        x0 = jnp.array([1.0, 2.0])
        return solver.run(x0, p).params

    return relax


# -----------------------------------------------------------------------------
# Option B: hand-rolled custom_vjp via the implicit function theorem
# -----------------------------------------------------------------------------


def make_option_b(tol: float = 1e-10, maxiter: int = 200) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a relaxed-geometry function with a hand-rolled ``custom_vjp``."""

    @jax.custom_vjp
    def relax(p: jnp.ndarray) -> jnp.ndarray:
        x0 = jnp.array([1.0, 2.0])
        res = jax_minimize(
            lambda x: energy(x, p),
            x0,
            method="BFGS",
            tol=tol,
            options={"maxiter": maxiter},
        )
        return res.x

    def relax_fwd(p: jnp.ndarray) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        x = relax(p)
        return x, (x, p)

    def relax_bwd(saved: tuple[jnp.ndarray, jnp.ndarray], g: jnp.ndarray) -> tuple[jnp.ndarray]:
        x, p = saved
        # H = ∂²E/∂x² at x*   (n_x × n_x)
        h_mat = jax.hessian(energy, argnums=0)(x, p)
        # M = ∂²E/∂x∂p at x*  (n_x × n_p)
        m_mat = jax.jacfwd(jax.grad(energy, argnums=0), argnums=1)(x, p)
        # Implicit function theorem: dx*/dp = -H⁻¹ M.
        # VJP: (dL/dp) = g · (dx*/dp) = -g · H⁻¹ · M.
        # Solve Hᵀ λ = g, then dL/dp = -λᵀ M.
        lam = jnp.linalg.solve(h_mat.T, g)
        return (-(lam @ m_mat),)

    relax.defvjp(relax_fwd, relax_bwd)
    return relax


# -----------------------------------------------------------------------------
# Pipeline: loss(p) = ||obs(relax(p)) - refs||²
# -----------------------------------------------------------------------------


def make_loss(
    relax_fn: Callable[[jnp.ndarray], jnp.ndarray], refs: jnp.ndarray
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a scalar loss ``L(p) = Σ (obs(x*(p)) - refs)²`` around a relaxer."""

    def loss(p: jnp.ndarray) -> jnp.ndarray:
        x_star = relax_fn(p)
        obs = observables(x_star)
        return jnp.sum((obs - refs) ** 2)

    return loss


# -----------------------------------------------------------------------------
# Benchmark harness
# -----------------------------------------------------------------------------


def _time_grad(
    grad_fn: Callable[[jnp.ndarray], jnp.ndarray],
    p: jnp.ndarray,
    n_warmup: int = 3,
    n_iter: int = 20,
) -> tuple[jnp.ndarray, float]:
    """Time a JIT-compiled gradient function. Returns (last_grad, ms_per_call)."""
    for _ in range(n_warmup):
        jax.block_until_ready(grad_fn(p))
    t0 = time.perf_counter()
    g = grad_fn(p)
    for _ in range(n_iter - 1):
        g = grad_fn(p)
    jax.block_until_ready(g)
    elapsed = (time.perf_counter() - t0) / n_iter
    return g, elapsed * 1e3


def run_accuracy_table() -> None:
    """Print Table 1: gradient accuracy vs inner solver tolerance."""
    p = jnp.array([2.0, 1.5, 3.0, 1.2])
    refs = jnp.array([1.6, 1.3])
    g_exact = closed_form_grad(p, refs)

    print("=" * 78)
    print("TABLE 1: Gradient accuracy vs inner solver tolerance")
    print("=" * 78)
    print(f"Reference (closed form): {np.asarray(g_exact)}")
    print()
    header = f"{'tol':>10} | {'method':>10} | {'max |err|':>12} | {'rel err':>12}"
    print(header)
    print("-" * len(header))

    for tol in [1e-3, 1e-6, 1e-9, 1e-12]:
        loss_a = jax.jit(make_loss(make_option_a(tol=tol), refs))
        loss_b = jax.jit(make_loss(make_option_b(tol=tol), refs))
        g_a = jax.grad(loss_a)(p)
        g_b = jax.grad(loss_b)(p)
        for name, g in [("A: jaxopt", g_a), ("B: custom", g_b)]:
            err = float(jnp.max(jnp.abs(g - g_exact)))
            rel = float(jnp.max(jnp.abs((g - g_exact) / g_exact)))
            print(f"{tol:>10.0e} | {name:>10} | {err:>12.3e} | {rel:>12.3e}")

    print()
    print("Finite-difference baseline (full-pipeline FD, end-to-end):")
    loss_ref = jax.jit(make_loss(make_option_a(tol=1e-12), refs))
    for eps in [1e-3, 1e-5, 1e-7]:
        g_fd = np.zeros(4)
        for i in range(4):
            dp = np.zeros(4)
            dp[i] = eps
            g_fd[i] = (loss_ref(p + dp) - loss_ref(p - dp)) / (2 * eps)
        err = float(np.max(np.abs(g_fd - np.asarray(g_exact))))
        rel = float(np.max(np.abs((g_fd - np.asarray(g_exact)) / np.asarray(g_exact))))
        print(f"{eps:>10.0e} | {'FD':>10} | {err:>12.3e} | {rel:>12.3e}")


def run_timing_table() -> None:
    """Print Table 2: wall-time per gradient evaluation."""
    p = jnp.array([2.0, 1.5, 3.0, 1.2])
    refs = jnp.array([1.6, 1.3])

    print()
    print("=" * 78)
    print("TABLE 2: Wall-time per gradient evaluation (JIT'd, 20 iterations)")
    print("=" * 78)
    header = f"{'method':>12} | {'tol':>8} | {'ms/grad':>10}"
    print(header)
    print("-" * len(header))

    for tol in [1e-6, 1e-10]:
        grad_a = jax.jit(jax.grad(make_loss(make_option_a(tol=tol), refs)))
        grad_b = jax.jit(jax.grad(make_loss(make_option_b(tol=tol), refs)))
        _, ms_a = _time_grad(grad_a, p)
        _, ms_b = _time_grad(grad_b, p)
        print(f"{'A: jaxopt':>12} | {tol:>8.0e} | {ms_a:>10.3f}")
        print(f"{'B: custom':>12} | {tol:>8.0e} | {ms_b:>10.3f}")


def run_conditioning_table() -> None:
    """Print Table 3: Hessian ill-conditioning sensitivity (shrink ``k_1`` → 0)."""
    refs = jnp.array([1.6, 1.3])

    print()
    print("=" * 78)
    print("TABLE 3: Hessian ill-conditioning (shrink k_1 -> 0)")
    print("=" * 78)
    header = f"{'k1':>10} | {'cond(H)':>12} | {'A err':>12} | {'B err':>12}"
    print(header)
    print("-" * len(header))

    for k1 in [2.0, 0.1, 1e-3, 1e-5]:
        p = jnp.array([k1, 1.5, 3.0, 1.2])
        g_exact = closed_form_grad(p, refs)
        x_star = closed_form_geometry(p)
        h_mat = jax.hessian(energy, argnums=0)(x_star, p)
        cond = float(jnp.linalg.cond(h_mat))
        g_a = jax.grad(jax.jit(make_loss(make_option_a(tol=1e-10), refs)))(p)
        g_b = jax.grad(jax.jit(make_loss(make_option_b(tol=1e-10), refs)))(p)
        err_a = float(jnp.max(jnp.abs(g_a - g_exact)))
        err_b = float(jnp.max(jnp.abs(g_b - g_exact)))
        print(f"{k1:>10.1e} | {cond:>12.3e} | {err_a:>12.3e} | {err_b:>12.3e}")


def main() -> None:
    """Entry point: run all three spike tables."""
    print(f"JAX devices: {jax.devices()}")
    print()
    run_accuracy_table()
    run_timing_table()
    run_conditioning_table()


if __name__ == "__main__":
    main()
