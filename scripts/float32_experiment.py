#!/usr/bin/env python3
"""Float32 viability experiment for JaxEngine harmonic force fields.

Issue #178: Test whether float32 is viable for JaxEngine's harmonic-only
force fields.  If it is, consumer GPUs unlock 64x more throughput.

Usage (inside ci-jax container):
    # Float64 baseline
    python scripts/float32_experiment.py run64

    # Float32 experiment (JAX_ENABLE_X64=0 must be set BEFORE this script)
    JAX_ENABLE_X64=0 python scripts/float32_experiment.py run32

    # Compare results
    python scripts/float32_experiment.py compare /tmp/f64.json /tmp/f32.json
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np


def run_experiment(output_path: str) -> None:
    """Run frequency computation with current JAX precision config."""
    # Import engine first — this triggers _jax_common.py which may enable x64
    from q2mm.backends.mm.jax_engine import JaxEngine
    from q2mm.diagnostics.systems import SYSTEMS

    import jax
    import jax.numpy as jnp

    precision = "float64" if jax.config.jax_enable_x64 else "float32"
    print(f"\n{'=' * 60}")
    print(f"Running with {precision} (x64_enabled={jax.config.jax_enable_x64})")
    # Verify actual dtype
    test_arr = jnp.array([1.0])
    print(f"jnp.array([1.0]).dtype = {test_arr.dtype}")
    print(f"{'=' * 60}")

    engine = JaxEngine()
    results: dict[str, dict] = {"precision": precision}

    # --- CH3F ---
    ch3f = SYSTEMS["ch3f"]
    ch3f_data = ch3f.loader(engine)

    for i, mol in enumerate(ch3f_data.molecules):
        t0 = time.perf_counter()
        freqs = engine.frequencies(mol, ch3f_data.forcefield)
        elapsed = time.perf_counter() - t0
        results[f"ch3f_mol{i}"] = {
            "frequencies": np.asarray(freqs).tolist(),
            "elapsed_s": elapsed,
        }
        print(f"  CH3F mol{i}: {len(freqs)} freqs, {elapsed:.4f}s")

    # --- Rh-enamide ---
    rh = SYSTEMS["rh-enamide"]
    rh_data = rh.loader(engine)

    for i, mol in enumerate(rh_data.molecules):
        t0 = time.perf_counter()
        freqs = engine.frequencies(mol, rh_data.forcefield)
        elapsed = time.perf_counter() - t0
        results[f"rh_mol{i}"] = {
            "frequencies": np.asarray(freqs).tolist(),
            "elapsed_s": elapsed,
        }
        print(f"  Rh-enamide mol{i}: {len(freqs)} freqs, {elapsed:.4f}s")

    # --- Throughput: repeated energy evals (post-JIT) ---
    # NOTE: engine.energy() returns a Python float which forces a
    # device→host sync on GPU.  For pure GPU throughput measurement,
    # call the jitted _energy_fn directly with block_until_ready().
    # Here we measure the realistic engine-level throughput including
    # the sync, since that's what the optimizer actually calls.
    import jax

    mol = ch3f_data.molecules[0]
    ff = ch3f_data.forcefield
    handle = engine._get_handle(mol, ff)
    params, coords = engine._params_and_coords(handle, ff)

    # Warm up JIT
    for _ in range(3):
        handle._energy_fn(params, coords).block_until_ready()

    # Pure-JAX throughput (no device→host sync per call)
    n_iters = 500
    t0 = time.perf_counter()
    for _ in range(n_iters):
        handle._energy_fn(params, coords).block_until_ready()
    elapsed = time.perf_counter() - t0
    results["throughput_ch3f"] = {
        "n_iters": n_iters,
        "elapsed_s": elapsed,
        "evals_per_sec": n_iters / elapsed,
    }
    print(f"  CH3F throughput: {n_iters / elapsed:.0f} evals/s ({elapsed:.3f}s)")

    # Rh-enamide throughput (larger molecule)
    mol_rh = rh_data.molecules[0]
    ff_rh = rh_data.forcefield
    handle_rh = engine._get_handle(mol_rh, ff_rh)
    params_rh, coords_rh = engine._params_and_coords(handle_rh, ff_rh)

    for _ in range(3):
        handle_rh._energy_fn(params_rh, coords_rh).block_until_ready()

    n_rh = 100
    t0 = time.perf_counter()
    for _ in range(n_rh):
        handle_rh._energy_fn(params_rh, coords_rh).block_until_ready()
    elapsed = time.perf_counter() - t0
    results["throughput_rh"] = {
        "n_iters": n_rh,
        "elapsed_s": elapsed,
        "evals_per_sec": n_rh / elapsed,
    }
    print(f"  Rh-enamide throughput: {n_rh / elapsed:.0f} evals/s ({elapsed:.3f}s)")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


def compare_results(f64_file: str, f32_file: str) -> None:
    """Compare float64 vs float32 frequency results."""
    with open(f64_file) as f:
        f64 = json.load(f)
    with open(f32_file) as f:
        f32 = json.load(f)

    print("\n" + "=" * 70)
    print("FLOAT32 vs FLOAT64 COMPARISON")
    print("=" * 70)

    all_diffs: list[float] = []

    # Validate that both files have comparable system keys
    meta_keys = {"precision", "throughput_ch3f", "throughput_rh"}
    f64_systems = sorted(k for k in f64 if k not in meta_keys)
    f32_systems = sorted(k for k in f32 if k not in meta_keys)
    if f64_systems != f32_systems:
        missing_in_f32 = set(f64_systems) - set(f32_systems)
        missing_in_f64 = set(f32_systems) - set(f64_systems)
        if missing_in_f32:
            print(f"  WARNING: keys in f64 but not f32: {missing_in_f32}")
        if missing_in_f64:
            print(f"  WARNING: keys in f32 but not f64: {missing_in_f64}")

    for key in f64_systems:
        if key not in f32:
            print(f"  {key}: MISSING in f32 file, skipping")
            continue
        freqs_64 = np.array(f64[key]["frequencies"])
        freqs_32 = np.array(f32[key]["frequencies"])

        if len(freqs_64) != len(freqs_32):
            print(f"  {key}: LENGTH MISMATCH ({len(freqs_64)} vs {len(freqs_32)})")
            continue

        abs_diff = np.abs(freqs_64 - freqs_32)
        # Use abs(freq) > 1.0 to avoid division by near-zero imaginary freqs
        safe_denom = np.where(np.abs(freqs_64) > 1.0, np.abs(freqs_64), 1.0)
        rel_diff = abs_diff / safe_denom
        all_diffs.extend(abs_diff.tolist())

        rmsd = np.sqrt(np.mean(abs_diff**2))

        print(f"\n  {key}:")
        print(f"    Frequencies:   {len(freqs_64)}")
        print(f"    Max abs diff:  {np.max(abs_diff):.6f} cm-1")
        print(f"    Mean abs diff: {np.mean(abs_diff):.6f} cm-1")
        print(f"    Max rel diff:  {np.max(rel_diff) * 100:.6f}%")
        print(f"    RMSD:          {rmsd:.6f} cm-1")

        # Show worst 3
        worst_idx = np.argsort(abs_diff)[-3:][::-1]
        for idx in worst_idx:
            print(f"      freq[{idx}]: f64={freqs_64[idx]:.4f}, f32={freqs_32[idx]:.4f}, diff={abs_diff[idx]:.6f} cm-1")

    all_diffs_arr = np.array(all_diffs)
    if len(all_diffs_arr) == 0:
        print("\n  No comparable frequencies found. Check input files.")
        return

    print("\n  OVERALL:")
    print(f"    Total frequencies compared: {len(all_diffs_arr)}")
    print(f"    Max abs diff:  {np.max(all_diffs_arr):.6f} cm-1")
    print(f"    Mean abs diff: {np.mean(all_diffs_arr):.6f} cm-1")
    print(f"    RMSD:          {np.sqrt(np.mean(all_diffs_arr**2)):.6f} cm-1")

    # Throughput comparison
    if "throughput_ch3f" in f64 and "throughput_ch3f" in f32:
        tp64_ch3f = f64["throughput_ch3f"]["evals_per_sec"]
        tp32_ch3f = f32["throughput_ch3f"]["evals_per_sec"]
        print("\n  THROUGHPUT (CPU):")
        print(f"    CH3F     float64: {tp64_ch3f:.0f} evals/s")
        print(f"    CH3F     float32: {tp32_ch3f:.0f} evals/s  ({tp32_ch3f / tp64_ch3f:.2f}x)")

    if "throughput_rh" in f64 and "throughput_rh" in f32:
        tp64_rh = f64["throughput_rh"]["evals_per_sec"]
        tp32_rh = f32["throughput_rh"]["evals_per_sec"]
        print(f"    Rh-enam  float64: {tp64_rh:.0f} evals/s")
        print(f"    Rh-enam  float32: {tp32_rh:.0f} evals/s  ({tp32_rh / tp64_rh:.2f}x)")

    # Verdict
    threshold = 0.1  # cm-1
    max_diff = np.max(all_diffs_arr)
    print(f"\n  {'=' * 50}")
    if max_diff < threshold:
        print("  VERDICT: Float32 IS VIABLE")
        print(f"    Max diff {max_diff:.6f} < {threshold} cm-1 threshold")
        print("    Safe for harmonic-only force fields.")
    else:
        print("  VERDICT: Float32 NOT VIABLE")
        print(f"    Max diff {max_diff:.6f} >= {threshold} cm-1 threshold")
        print("    Float64 required for acceptable frequency accuracy.")
    print(f"  {'=' * 50}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/float32_experiment.py run64 [output.json]")
        print("  JAX_ENABLE_X64=0 python scripts/float32_experiment.py run32 [output.json]")
        print("  python scripts/float32_experiment.py compare f64.json f32.json")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "run64":
        out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/float64_results.json"
        run_experiment(out)
    elif cmd == "run32":
        out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/float32_results.json"
        run_experiment(out)
    elif cmd == "compare":
        if len(sys.argv) < 4:
            print("Usage: ... compare f64.json f32.json")
            sys.exit(1)
        compare_results(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
