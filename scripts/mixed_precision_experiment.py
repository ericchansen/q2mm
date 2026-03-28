#!/usr/bin/env python3
"""Mixed-precision experiment: float32 Hessian + float64 eigendecomp.

Tests whether computing the Hessian in float32 (GPU-friendly) and then
casting to float64 for eigendecomposition produces acceptable frequencies.

Usage (inside ci-jax container):
    python scripts/mixed_precision_experiment.py
"""

from __future__ import annotations

import os
import time

import numpy as np


def run() -> None:
    """Compare three precision paths for Hessian → frequencies."""
    os.environ.pop("JAX_ENABLE_X64", None)

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from q2mm.backends.mm.jax_engine import JaxEngine, _compile_energy_fn
    from q2mm.models.units import KCALMOLA2_TO_HESSIAN_AU
    from q2mm.models.hessian import hessian_to_frequencies
    from q2mm.diagnostics.systems import SYSTEMS

    engine = JaxEngine()
    threshold = 0.1  # cm⁻¹

    for sys_name in ["ch3f", "rh-enamide"]:
        system = SYSTEMS[sys_name]
        data = system.loader(engine)
        ff = data.forcefield

        for mol_i, mol in enumerate(data.molecules):
            n = len(mol.symbols)
            label = f"{sys_name} mol{mol_i} ({n} atoms, {3 * n} DOF)"
            print(f"\n{'=' * 70}")
            print(f"  {label}")
            print(f"{'=' * 70}")

            handle = engine.create_context(mol, ff)

            # ── 1. BASELINE: Full float64 ──────────────────────────────
            jax.config.update("jax_enable_x64", True)
            handle._energy_fn = _compile_energy_fn(handle, ff)
            handle._coord_hess_fn = None
            params64, coords64 = engine._params_and_coords(handle, ff)

            def _e64(fc: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
                return handle._energy_fn(p, fc.reshape(-1, 3))

            hess_fn64 = jax.jit(jax.hessian(_e64, argnums=0))

            t0 = time.perf_counter()
            hess_kcal_f64 = np.asarray(hess_fn64(coords64.flatten(), params64))
            t_f64 = time.perf_counter() - t0

            hess_au_f64 = hess_kcal_f64 * float(KCALMOLA2_TO_HESSIAN_AU)
            freqs_f64 = np.array(hessian_to_frequencies(hess_au_f64, list(mol.symbols)))

            # ── 2. Float32 Hessian ─────────────────────────────────────
            jax.config.update("jax_enable_x64", False)
            handle._energy_fn = _compile_energy_fn(handle, ff)

            p32 = jnp.array(np.asarray(params64), dtype=jnp.float32)
            c32 = jnp.array(np.asarray(coords64), dtype=jnp.float32)

            def _e32(fc: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
                return handle._energy_fn(p, fc.reshape(-1, 3))

            hess_fn32 = jax.jit(jax.hessian(_e32, argnums=0))

            t0 = time.perf_counter()
            hess_kcal_f32 = np.asarray(hess_fn32(c32.flatten(), p32))
            t_f32 = time.perf_counter() - t0

            # ── 3A. MIXED: f32 Hessian → cast f64 → f64 eigendecomp ──
            hess_au_mixed = hess_kcal_f32.astype(np.float64) * float(KCALMOLA2_TO_HESSIAN_AU)
            freqs_mixed = np.array(hessian_to_frequencies(hess_au_mixed, list(mol.symbols)))

            # ── 3B. FULL-F32: f32 Hessian → f64 eigendecomp (no cast before unit conv)
            # Same as mixed — numpy eigendecomp is always f64. The only
            # difference is the INPUT Hessian precision.

            # ── Compare ────────────────────────────────────────────────
            real = np.abs(freqs_f64) > 50.0
            diff_mixed = np.abs(freqs_f64 - freqs_mixed)

            # Hessian element-level comparison
            hess_diff = np.abs(hess_kcal_f64 - hess_kcal_f32.astype(np.float64))
            hess_max = np.max(np.abs(hess_kcal_f64))

            print(f"\n  Hessian computation time:  f64={t_f64:.3f}s  f32={t_f32:.3f}s  ({t_f64 / t_f32:.2f}x)")
            print(
                f"  Hessian element diff:     max={np.max(hess_diff):.2e}  "
                f"mean={np.mean(hess_diff):.2e}  "
                f"rel_max={np.max(hess_diff) / hess_max:.2e}"
            )

            max_d_real = float(np.max(diff_mixed[real])) if real.any() else 0.0
            mean_d_real = float(np.mean(diff_mixed[real])) if real.any() else 0.0
            rmsd_real = float(np.sqrt(np.mean(diff_mixed[real] ** 2))) if real.any() else 0.0
            verdict = "✅ PASS" if max_d_real < threshold else "❌ FAIL"

            print("\n  Mixed precision (f32 Hessian → f64 eigendecomp):")
            print(f"    Max Δ (real modes):  {max_d_real:.6f} cm⁻¹  {verdict}")
            print(f"    Mean Δ (real modes): {mean_d_real:.6f} cm⁻¹")
            print(f"    RMSD (real modes):   {rmsd_real:.6f} cm⁻¹")

            # Worst 5
            if real.any():
                ridx = np.where(real)[0]
                worst = ridx[np.argsort(diff_mixed[real])[-5:][::-1]]
                print("\n    Worst 5 deviations:")
                for w in worst:
                    print(
                        f"      freq[{w:3d}]: f64={freqs_f64[w]:10.4f}  "
                        f"mixed={freqs_mixed[w]:10.4f}  "
                        f"Δ={diff_mixed[w]:.6f} cm⁻¹"
                    )

            # Near-zero mode comparison (these are the hardest)
            near_zero = np.abs(freqs_f64) <= 50.0
            if near_zero.any():
                max_nz = float(np.max(diff_mixed[near_zero]))
                print(f"\n    Near-zero modes (|freq| ≤ 50): max Δ = {max_nz:.4f} cm⁻¹ (informational)")

            # Only process the first molecule for rh-enamide
            if sys_name == "rh-enamide":
                break

    # Re-enable x64 at end
    jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":
    run()
