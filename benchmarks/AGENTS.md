# Agent Instructions: Benchmarks

Read this entire file before running any benchmark.

---

## 1. Pre-flight Checklist (MANDATORY before any benchmark)

Before running ANY benchmark, verify:

1. **Platform check:**
   ```bash
   python -c "import openmm; [print(openmm.Platform.getPlatform(i).getName()) for i in range(openmm.Platform.getNumPlatforms())]"
   ```
   Must show `CUDA` for GPU benchmarks.

2. **JAX device check:**
   ```bash
   python -c "import jax; print(jax.devices())"
   ```
   Must show `CudaDevice` for GPU benchmarks.

3. **GPU utilization baseline:**
   ```bash
   nvidia-smi
   ```
   Note current GPU usage before starting.

4. **If OpenCL appears instead of CUDA:** STOP. Do not proceed. Use WSL2
   instead. OpenCL gives ~14% GPU utilization and wastes hours.

---

## 2. Saving Rules

- **NEVER use `--no-save`** — results and force fields must always be saved.
  Runs can take hours; output is not reproducible bit-for-bit.
- **Output convention:** `benchmark_results/<system>_<date>/` for timestamped runs.
- **File naming:** `{system}_{engine}_{ff}_{device}_{optimizer}` with underscore
  delimiters.
- **Force CPU baseline:** Set `JAX_PLATFORMS=cpu` environment variable.
- **Run benchmarks sequentially** on an idle system for consistent timing.
  Parallel runs contaminate timing data.
- **Update both** `benchmarks/` archive AND `docs/benchmarks/` summary pages
  after collecting new data. Do not leave stale numbers in the docs.

---

## 3. Expected Runtimes (RTX 5090, Rh-enamide system)

| Backend          | Optimizer   | Device | Approx Time | Notes                                    |
| ---------------- | ----------- | ------ | ----------- | ---------------------------------------- |
| JAX              | L-BFGS-B    | GPU    | ~6 min      | 2.08x vs CPU                             |
| JAX              | L-BFGS-B    | CPU    | ~9 min      |                                          |
| JAX              | Nelder-Mead | CPU    | ~9 min      | Hits max iterations                      |
| JAX-MD (OPLSAA)  | L-BFGS-B    | GPU    | ~100 min    | 5.6x vs CPU                              |
| JAX-MD (OPLSAA)  | L-BFGS-B    | CPU    | ~6.6 hrs    |                                          |
| OpenMM           | L-BFGS-B    | CUDA   | TBD         | Not yet benchmarked with CUDA            |
| OpenMM           | Nelder-Mead | CUDA   | TBD         | Not yet benchmarked with CUDA            |
| OpenMM           | L-BFGS-B    | OpenCL | ~5+ hours   | DO NOT USE — OpenCL is broken on this system |
| Tinker           | L-BFGS-B    | CPU    | TBD         | Subprocess-based, no GPU                 |

---

## 4. When to Abort

- **OpenCL fallback:** If benchmark output shows `OpenMM (OpenCL)` instead
  of `OpenMM (CUDA)`, kill immediately.
- **Zero GPU utilization:** Check `nvidia-smi` during the first few minutes.
  If GPU util is <5%, something is wrong.
- **LAPACK eigenvalue failure:** Powell optimizer can trigger "Eigenvalues did
  not converge" on Rh-enamide. This is caught gracefully and recorded as
  FAILED — don't abort the whole matrix for this.
- **No progress for 30+ minutes:** Check if the process is actually computing
  (CPU/GPU activity) or hung.

---

## 5. Platform Troubleshooting

| Symptom                                 | Cause                        | Fix                                          |
| --------------------------------------- | ---------------------------- | -------------------------------------------- |
| `OpenMM (OpenCL)` in output             | CUDA plugin missing          | Install `OpenMM-CUDA-12` or use WSL2         |
| `OpenMM (CPU)` in output                | No GPU platform available    | Install CUDA plugin, check `nvidia-smi`      |
| JAX shows `CpuDevice` only              | JAX CUDA not installed       | Use WSL2 with `jax[cuda12]`                  |
| Very slow OpenMM evals (~mins each)     | Using CPU instead of CUDA    | Check platform, switch to CUDA               |
| `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`    | Old CUDA plugin vs new GPU   | Install `OpenMM-CUDA-12` (has NVRTC JIT)     |

---

## 6. Benchmark Commands Reference

```bash
# Full matrix (all backends x all optimizers)
q2mm-benchmark --system rh-enamide --output benchmark_results/rh_enamide_YYYY-MM-DD

# Single backend
q2mm-benchmark --system rh-enamide --backend jax --optimizer L-BFGS-B \
  --output benchmark_results/rh_enamide_YYYY-MM-DD

# CPU baseline for comparison
JAX_PLATFORMS=cpu q2mm-benchmark --system rh-enamide --backend jax \
  --optimizer L-BFGS-B --output benchmark_results/rh_enamide_YYYY-MM-DD

# CH3F small molecule (fast, good for testing)
q2mm-benchmark --output benchmark_results/ch3f_YYYY-MM-DD
```
