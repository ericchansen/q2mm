# GPU Acceleration Benchmarks

Hardware and software environment, methodology, and results for
GPU-accelerated Q2MM parameter optimization on NVIDIA RTX 5090
(Blackwell architecture).

## Hardware

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA RTX 5090 (Blackwell, sm_120) |
| **VRAM** | 32 GB GDDR7 |
| **CPU** | AMD (multi-core, used for CPU baseline) |
| **Driver** | NVIDIA 591.74 |
| **CUDA runtime** | 13.1 (driver-reported; JAX uses `jax[cuda12]` wheels targeting CUDA 12.x) |
| **OS** | Ubuntu (WSL2) |

## Software

| Package | Version |
|---------|---------|
| JAX | 0.9.2 |
| jax[cuda12] | 0.9.2 |
| jax-md | 0.2.8 |
| Python | 3.12 |

## Methodology

- All benchmarks run **sequentially** (never in parallel) to avoid
  resource contention that would invalidate timing
- CPU baselines use `JAX_PLATFORMS=cpu` to force CPU-only execution
- Per-evaluation time is the fair comparison metric — different devices
  may take different optimization paths (different eval counts)
- L-BFGS-B optimizer used for all runs (gradient-based; benefits from
  GPU-accelerated gradient and objective function evaluations, which
  include Hessian/frequency computations internally)
- Benchmarks use the `q2mm-benchmark` CLI with results saved to
  `benchmarks/<system>/` (per `CONTRIBUTING.md` policy)

## Results

### Rh-enamide (9 molecules, 36–62 atoms each, 182 parameters)

| Backend | Device | s/eval | Evals | Wall Time | GPU Speedup |
|---------|--------|-------:|------:|----------:|:-----------:|
| JAX (harmonic) | GPU | 12.60 | 31 | 390.5 s | **2.08×** |
| JAX (harmonic) | CPU | 26.17 | 21 | 549.6 s | — |
| JAX-MD (OPLSAA) | GPU | 13.44 | 447 | 6,008.9 s | **5.61×** |
| JAX-MD (OPLSAA) | CPU | 75.38 | 316 | 23,819.4 s | — |

### CH₃F (1 molecule, 5 atoms, 8 parameters)

| Backend | Device | s/eval | Evals | Wall Time | GPU Speedup |
|---------|--------|-------:|------:|----------:|:-----------:|
| JAX (harmonic) | GPU | 0.054 | 132 | 7.1 s | 0.20× |
| JAX (harmonic) | CPU | 0.011 | 95 | 1.0 s | — |

### OpenMM

OpenMM's CUDA platform works on RTX 5090 (Blackwell / sm_120) via the
`OpenMM-CUDA-12` pip package, which provides CUDA plugin binaries that
JIT-compile kernels at runtime using NVRTC.

```bash
pip install OpenMM-CUDA-12
```

OpenMM benchmarks were run on CPU only (existing results in `rh-enamide/results/`).
GPU benchmarks with `OpenMM-CUDA-12` are planned (see issue #194).

## Key Findings

1. **JAX-MD OPLSAA achieves 5.61× per-evaluation GPU speedup** on the
   Rh-enamide system. This is the most computationally intensive backend
   (full OPLSAA force field with LJ, Coulomb, and torsion terms), and
   benefits most from GPU parallelism across the 9-molecule batch.

2. **JAX harmonic achieves 2.08× per-evaluation GPU speedup** on
   Rh-enamide. The simpler energy expression (bonds + angles only)
   leaves less work for GPU parallelism, but the batched Hessian
   computation via `jax.vmap` still benefits.

3. **Small molecules (CH₃F) are faster on CPU.** With only 5 atoms and
   8 parameters, GPU kernel launch overhead dominates. CPU completes
   each evaluation in 11 ms vs 54 ms on GPU. This is expected — GPU
   acceleration only pays off above a minimum system size.

4. **GPU speedup scales with computational complexity.** The progression
   from 0.20× (CH₃F, trivial) → 2.08× (Rh harmonic, moderate) →
   5.61× (Rh OPLSAA, heavy) shows that larger, more complex systems
   will see even greater GPU acceleration.

5. **Eval counts differ between CPU and GPU** due to floating-point
   differences (float64 on both, but different reduction order).
   Per-evaluation time is the apples-to-apples metric.

## Reproducing

```bash
# Activate the virtual environment with jax[cuda12] installed
source .venv/bin/activate

# Rh-enamide GPU
q2mm-benchmark --system rh-enamide --backend jax --optimizer L-BFGS-B --output benchmarks/rh-enamide

# Rh-enamide CPU
JAX_PLATFORMS=cpu q2mm-benchmark --system rh-enamide --backend jax --optimizer L-BFGS-B --output benchmarks/rh-enamide

# Rh-enamide JAX-MD GPU (warning: ~100 min)
q2mm-benchmark --system rh-enamide --backend jax-md --optimizer L-BFGS-B --output benchmarks/rh-enamide

# Rh-enamide JAX-MD CPU (warning: ~6.6 hours)
JAX_PLATFORMS=cpu q2mm-benchmark --system rh-enamide --backend jax-md --optimizer L-BFGS-B --output benchmarks/rh-enamide

# CH3F GPU vs CPU
q2mm-benchmark --backend jax --optimizer L-BFGS-B --output benchmarks/ch3f
JAX_PLATFORMS=cpu q2mm-benchmark --backend jax --optimizer L-BFGS-B --output benchmarks/ch3f
```

## Date

Benchmarks run: 2026-03-30 (sequential, single-machine)
