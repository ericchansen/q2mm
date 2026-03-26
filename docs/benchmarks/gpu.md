# GPU Acceleration

GPU vs CPU benchmarks using the JAX backend on an NVIDIA RTX 5090
(32 GB VRAM, Blackwell sm_120) with an AMD Ryzen 7 7800X3D CPU.
All runs use float64 precision.

---

## Results

### Per-evaluation throughput

| System | Atoms | Modes | Params | GPU (ms/eval) | CPU (ms/eval) | Speedup |
|--------|------:|------:|-------:|--------------:|--------------:|--------:|
| CH₃F | 5 | 9 | 8 | 2.20 | 0.40 | 0.18× |
| rh-enamide | 36–62 | 1,273 | 94 | 36.2 | 21.9 | 0.60× |

Per-eval times measured as the mean of 50 (CH₃F) or 10 (rh-enamide)
isolated calls after JIT warmup.

### GRAD→SIMP cycling

Full [`OptimizationLoop`][q2mm.optimizers.cycling.OptimizationLoop]
with L-BFGS-B (GRAD, `maxiter=200`) and Nelder-Mead (SIMP,
`maxiter=200`, `max_params=5`, `convergence=0.01`).

**CH₃F** (1 molecule, 8 parameters, synthetic reference):

| Device | Cycles | Evals | Wall time | Final score |
|--------|-------:|------:|----------:|------------:|
| GPU | 2 | 664 | 2.3 s | 1.9 × 10⁻⁵ |
| CPU | 2 | 682 | 0.8 s | 1.9 × 10⁻⁵ |

**rh-enamide** (9 molecules, 94 parameters, QM frequency reference):

| Device | Cycles | Evals | Wall time | Final score |
|--------|-------:|------:|----------:|------------:|
| GPU | 3 | 30,637 | 1,117 s | 34.56 |
| CPU | 4 | 30,936 | 686 s | 32.78 |

Score is the weighted sum of squared frequency deviations (QM − MM)
across all modes.  Lower is better; initial score is 2,161 for both
runs.

!!! warning "Force field caveat"
    rh-enamide uses an auto-generated **harmonic** force field (94
    params) because JaxEngine does not support MM3.  These scores are
    not comparable to the 182-parameter MM3 results on the
    [rh-enamide page](rh-enamide.md).

### Takeaway

**CPU is faster than GPU for these workloads.**  The rh-enamide system
(the larger of the two) shows a 1.6× CPU advantage.  GPU acceleration
for frequency-based fitting likely requires larger molecules or training
sets to amortize kernel-launch and data-transfer overhead.  Systems with
hundreds of atoms per structure or energy-only objectives (which can use
`jax.vmap` batching) are better candidates for GPU speedup.

---

## JIT Compilation

JAX compiles functions on the first call.  Subsequent calls reuse the
compiled kernel.

| System | JIT warmup |
|--------|----------:|
| CH₃F | < 1 s |
| rh-enamide (9 molecules) | ~6 s (GPU), ~3 s (CPU) |

---

## Memory

| System | Peak VRAM | Available |
|--------|----------:|----------:|
| CH₃F (1 molecule) | ~2 GB | 32 GB |
| rh-enamide (9 molecules) | ~30 GB | 32 GB |

---

## Compatibility

| Component | Status |
|-----------|--------|
| JAX CUDA (Blackwell / sm_120) | ✅ Works |
| OpenMM CUDA (Blackwell / sm_120) | ❌ PTX error 222 |
| JAX force fields | Harmonic only (no MM3) |

---

## Reproducing

Each benchmark was run **alone** on an otherwise idle system.

```bash
# GPU — rh-enamide
docker run --rm --gpus all -v "$PWD:/work" -w /work q2mm-gpu:latest bash -c \
  "pip install -e . --no-deps -q && \
   python scripts/run_cycling_benchmark.py \
     --molecule rh-enamide --engine jax \
     --max-cycles 10 --max-params 5 --convergence 0.01 \
     --output benchmarks/rh-enamide/results-cycling-gpu"

# CPU — rh-enamide (same container, JAX forced to CPU)
docker run --rm -v "$PWD:/work" -w /work q2mm-gpu:latest bash -c \
  "pip install -e . --no-deps -q && \
   JAX_PLATFORMS=cpu python scripts/run_cycling_benchmark.py \
     --molecule rh-enamide --engine jax \
     --max-cycles 10 --max-params 5 --convergence 0.01 \
     --output benchmarks/rh-enamide/results-cycling-cpu"
```

Raw JSON results: `benchmarks/{molecule}/results-cycling-{gpu,cpu}/`.

## Hardware & Software

| Component | Version |
|-----------|---------|
| GPU | NVIDIA RTX 5090 (32 GB, Blackwell sm_120) |
| CPU | AMD Ryzen 7 7800X3D (8 cores, 16 threads) |
| CUDA | 12.8 |
| Container | `q2mm-gpu:latest` (nvidia/cuda:12.8.0 + micromamba) |
| JAX | 0.5.x with CUDA 12 |
| Precision | float64 |
