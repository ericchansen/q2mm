# GPU Acceleration

GPU benchmarks using the JAX backend.  Results below were collected on
an NVIDIA RTX 5090 (32 GB VRAM, Blackwell sm_120) with an AMD Ryzen 7
7800X3D CPU.  All runs use float64 precision.

---

## Reproducing

```bash
# GPU run (requires NVIDIA container toolkit)
docker run --rm --gpus all -v "$PWD:/work" -w /work q2mm-gpu:latest bash -c \
  "pip install -e . --no-deps -q && \
   python scripts/run_cycling_benchmark.py \
     --molecule rh-enamide --engine jax \
     --max-cycles 10 --max-params 5 \
     --convergence 0.01 \
     --output benchmarks/rh-enamide/results-cycling-gpu"

# CPU run (same container, JAX forced to CPU)
docker run --rm -v "$PWD:/work" -w /work q2mm-gpu:latest bash -c \
  "pip install -e . --no-deps -q && \
   JAX_PLATFORMS=cpu python scripts/run_cycling_benchmark.py \
     --molecule rh-enamide --engine jax \
     --max-cycles 10 --max-params 5 \
     --convergence 0.01 \
     --output benchmarks/rh-enamide/results-cycling-cpu"
```

Raw JSON results are archived in
`benchmarks/{molecule}/results-cycling-{gpu,cpu}/`.

---

## Results

*Benchmarks are being re-collected.  This section will be updated with
accurate numbers.*

---

## JIT Compilation Overhead

JAX compiles energy/gradient functions on the first evaluation per
molecule.  Subsequent evaluations reuse the compiled kernel.

| System | Approximate first-run overhead |
|--------|-------------------------------:|
| CH₃F (5 atoms) | ~3 s |
| rh-enamide (9 structures, 36–62 atoms) | ~7 min |

For short runs the JIT cost dominates.  Multi-cycle optimizations
amortize it.  Future work: AOT compilation could cache kernels to disk.

---

## Memory Usage

| System | Peak VRAM | Available |
|--------|----------:|----------:|
| CH₃F (1 molecule) | ~2 GB | 32 GB |
| rh-enamide (9 molecules) | ~30 GB | 32 GB |

Larger training sets may require molecule batching, gradient
checkpointing, or multi-GPU distribution.

---

## Compatibility

| Component | Status |
|-----------|--------|
| JAX CUDA (Blackwell / sm_120) | ✅ Works |
| OpenMM CUDA (Blackwell / sm_120) | ❌ PTX error 222 — not yet supported |
| JAX force fields | Harmonic only (no MM3) |

---

## Hardware & Software

| Component | Version |
|-----------|---------|
| GPU | NVIDIA RTX 5090 (32 GB, Blackwell sm_120) |
| CPU | AMD Ryzen 7 7800X3D (8 cores, 16 threads) |
| CUDA | 12.8 |
| Container | `q2mm-gpu:latest` (nvidia/cuda:12.8.0 + micromamba) |
| JAX | 0.5.x with CUDA 12 support |
| Precision | float64 (`jax.config.update("jax_enable_x64", True)`) |
