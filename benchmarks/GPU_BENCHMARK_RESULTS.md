# GPU Benchmark Results

RTX 5090 (32 GB VRAM, Blackwell sm_120) vs AMD Ryzen 7 7800X3D (JAX backend, float64)

## Per-Evaluation Throughput

| System | Atoms | Modes | Params | GPU ms/eval | CPU ms/eval | GPU speedup |
|--------|-------|-------|--------|-------------|-------------|-------------|
| CH3F | 5 | 9 | 5 | 46.3 | 15.0 | 0.32× |
| rh-enamide | 36–62 | 1,273 | 94 | 110.4 | 386.7 | **3.50×** |

GPU overhead dominates for tiny molecules. At ~36+ atoms the GPU wins on
per-evaluation throughput, and the advantage grows with system size.

## GRAD→SIMP Cycling

### CH3F (5 atoms)

| Device | Cycles | Evals | Time | Final score |
|--------|--------|-------|------|-------------|
| GPU | 10 | 372 | 17.2 s | 0.000766 |
| CPU | 10 | 362 | 5.4 s | 0.000766 |

Identical final scores. CPU faster for small molecules.

### rh-enamide (9 structures, 36–62 atoms)

| Device | Cycles | Evals | Time | Final score |
|--------|--------|-------|------|-------------|
| GPU | 3 | 15,530 | 1,722 s | 34.30 |
| CPU | 4 | 1,842 | 716 s | 32.78 |

GPU converges in fewer cycles but the optimizer takes more evaluations per
cycle (different numerical paths on GPU vs CPU). Per-eval throughput is 3.5×
faster on GPU, but total wall-clock time is longer due to the extra evaluations.

## Key Observations

1. **GPU per-eval speedup scales with molecule size** — confirmed by
   CH3F (0.32×) → rh-enamide (3.50×) trend.
2. **JIT compilation overhead** is significant (~7 min for rh-enamide on first
   run) but amortised across evaluations.
3. **Optimizer convergence paths differ** between GPU and CPU, even with float64
   enabled. This affects total evaluation count and wall-clock comparisons.
4. **GPU VRAM usage** reached ~30 GB / 32 GB for rh-enamide, suggesting
   molecules with >100 atoms may need memory-management strategies.

## Hardware & Software

- GPU: NVIDIA RTX 5090 (32 GB, Blackwell sm_120, CUDA 12.8)
- Container: `q2mm-gpu:latest` (nvidia/cuda:12.8.0 + micromamba + JAX 0.5.x)
- JAX: float64 enabled via `jax.config.update("jax_enable_x64", True)`
- OpenMM CUDA: unsupported (PTX error 222 on sm_120)
- Force field: auto-generated harmonic (JAX doesn't support MM3 yet)
