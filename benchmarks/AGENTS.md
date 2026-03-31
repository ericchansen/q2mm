# Agent Instructions: Benchmarks

## Running Benchmarks

1. **NEVER use `--no-save`.** Benchmark runs can take hours (JAX-MD CPU
   Rh-enamide takes ~6.5 h). Output force fields and result JSON are not
   reproducible bit-for-bit — once lost, the entire run must be repeated.

2. **Save outputs to version control.** After a benchmark completes:
   - Force fields → `benchmarks/<system>/forcefields/`
   - Result JSON → `benchmarks/<system>/results/`
   - Commit them on the working branch before moving on.

3. **Run benchmarks sequentially.** GPU vs CPU comparisons must run one at
   a time on an otherwise idle system. Parallel runs contaminate timing
   data because they compete for CPU, memory, and GPU resources.

4. **Use `JAX_PLATFORMS=cpu`** for CPU baselines — this ensures JAX doesn't
   silently use the GPU.

## Example

```bash
# GPU run — saves to benchmarks/rh-enamide/
q2mm-benchmark --system rh-enamide --backend jax-md --optimizer L-BFGS-B

# CPU baseline — run AFTER GPU finishes
JAX_PLATFORMS=cpu q2mm-benchmark --system rh-enamide --backend jax-md --optimizer L-BFGS-B

# Commit results immediately
git add benchmarks/rh-enamide/
git commit -m "bench: rh-enamide JAX-MD GPU and CPU L-BFGS-B results"
```

## Updating Documentation

When new benchmark data is collected, update **both**:
- `benchmarks/GPU_BENCHMARKS.md` — repo-level summary
- `docs/benchmarks/gpu.md` — published docs page

Do not leave stale numbers in the docs.
