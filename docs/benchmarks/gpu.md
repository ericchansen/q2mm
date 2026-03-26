# GPU Acceleration

GPU vs CPU benchmarks using the JAX backend on an NVIDIA RTX 5090 (32 GB
VRAM, Blackwell sm_120).  All runs use float64 precision.

!!! info "Data"
    **Results:**
    [CH₃F GPU](https://github.com/ericchansen/q2mm/tree/master/benchmarks/ch3f/results-cycling-gpu) ·
    [CH₃F CPU](https://github.com/ericchansen/q2mm/tree/master/benchmarks/ch3f/results-cycling-cpu) ·
    [rh-enamide GPU](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/results-cycling-gpu) ·
    [rh-enamide CPU](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/results-cycling-cpu)

    **Script:** [`scripts/run_cycling_benchmark.py`](https://github.com/ericchansen/q2mm/blob/master/scripts/run_cycling_benchmark.py)

---

## Per-Evaluation Throughput

The GPU advantage scales with molecule size.  Tiny molecules are
dominated by kernel launch / data transfer overhead; larger systems
amortize that cost across more arithmetic.

| System | Atoms | Modes | Params | GPU (ms/eval) | CPU (ms/eval) | GPU speedup |
|--------|------:|------:|-------:|--------------:|--------------:|:-----------:|
| CH₃F | 5 | 9 | 5 | 46.3 | 15.0 | 0.32× |
| rh-enamide | 36–62 | 1,273 | 94 | 110.4 | 386.7 | **3.50×** |

!!! tip "Rule of thumb"
    GPU acceleration pays off at roughly **30+ atoms per molecule**.  Below
    that threshold, CPU is faster because kernel launch overhead exceeds the
    arithmetic savings.

---

## GRAD→SIMP Cycling

Full optimization loop results using the
[`OptimizationLoop`](../api.md) with L-BFGS-B (GRAD) and Nelder-Mead
(SIMP) stages.

### CH₃F (5 atoms, 5 parameters)

| Device | Cycles | Evals | Time | Final score | Improvement |
|--------|-------:|------:|-----:|------------:|:-----------:|
| GPU | 10 | 372 | 17.2 s | 0.000766 | 99.65% |
| CPU | 10 | 362 | 5.4 s | 0.000766 | 99.65% |

Identical final scores and eval counts — the optimizer follows the same
path.  CPU is **3.2× faster** in wall-clock time because the per-eval
overhead dominates for a 5-atom molecule.

### rh-enamide (9 structures, 36–62 atoms, 94 parameters)

!!! warning "Apples-to-oranges caveat"
    This benchmark uses an **auto-generated harmonic force field** (94
    parameters) because JAX does not support MM3.  The scores here are
    *not* comparable to the 182-parameter MM3 results on the
    [rh-enamide page](rh-enamide.md).  The purpose is to compare GPU vs
    CPU on the *same* problem, not to compare force field quality.

| Device | Cycles | Evals | ms/eval | Opt time | Final score |
|--------|-------:|------:|--------:|---------:|------------:|
| GPU | 3 | 15,530 | **110** | 1,714 s¹ | 34.30 |
| CPU | 4 | 1,842 | 387 | 712 s | 32.78 |

¹ *Includes ~7 min of JIT compilation on first evaluation.  Excluding
JIT, the GPU optimization time is ~1,300 s.*

**Score** is the weighted sum of squared frequency deviations (QM − MM)
across all 1,273 vibrational modes.  Lower is better; initial score is
2,161.  Both runs start from identical Seminario-estimated parameters.

The GPU is **3.5× faster per evaluation** (110 ms vs 387 ms), but the
optimizer takes **8.4× more evaluations** on GPU.  The evaluation count
difference is an open question — GPU and CPU produce slightly different
floating-point results (different BLAS/LAPACK libraries, different FMA
behavior), which causes the L-BFGS-B line search to make different step
decisions.  Whether this is a systematic issue or run-dependent noise
requires repeated trials to determine.

!!! note "Single run — no confidence intervals"
    These are single-run results.  The eval count difference may vary
    across runs depending on optimizer sensitivity to floating-point
    rounding.  Repeated trials with different random seeds are needed
    for robust conclusions about total wall-clock time.

---

## JIT Compilation Overhead

JAX's just-in-time compilation adds a one-time cost on the first
evaluation.  Subsequent evaluations reuse the compiled kernel.

| System | First-run overhead | Amortized over |
|--------|-------------------:|:--------------:|
| CH₃F | ~3 s | negligible for any run |
| rh-enamide | ~7 min | significant for short runs |

For rh-enamide, JIT compilation accounts for a large fraction of a
single-cycle run.  Multi-cycle runs amortize this cost.  Future work:
AOT compilation (``jax.jit(...).lower().compile()``) could cache compiled
kernels to disk.

---

## Memory Usage

GPU VRAM consumption scales with molecule size and the number of
structures evaluated simultaneously.

| System | Peak VRAM | Available | Utilization |
|--------|----------:|----------:|:-----------:|
| CH₃F (1 molecule) | ~2 GB | 32 GB | 6% |
| rh-enamide (9 molecules) | ~30 GB | 32 GB | 92% |

!!! warning "VRAM limits"
    The rh-enamide system nearly saturates a 32 GB GPU.  Larger training
    sets (more molecules or >100 atoms) may require:

    - Batching molecules across multiple kernel launches
    - Gradient checkpointing to trade compute for memory
    - Multi-GPU distribution

---

## When to Use GPU

| Scenario | Recommendation |
|----------|:---------------|
| Small molecules (< 20 atoms) | **Use CPU** — GPU overhead exceeds benefit |
| Medium molecules (20–60 atoms) | **GPU helps** — 2–4× per-eval speedup |
| Large molecules (> 60 atoms) | **GPU recommended** — speedup grows with size |
| Many short runs (< 5 cycles) | **CPU preferred** — JIT compilation dominates |
| Long optimization runs | **GPU preferred** — JIT cost amortized |
| Limited VRAM (< 16 GB) | **CPU preferred** — large systems may OOM |

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
| Force field | Auto-generated harmonic (JAX does not support MM3) |

!!! note "OpenMM CUDA"
    OpenMM's CUDA platform does not yet support Blackwell (sm_120) GPUs.
    The conda-forge PTX binaries produce `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`
    (error 222).  GPU benchmarks use the JAX backend exclusively.

---

*Results generated by `scripts/run_cycling_benchmark.py`.
Raw JSON archived in `benchmarks/ch3f/results-cycling-{gpu,cpu}/` and
`benchmarks/rh-enamide/results-cycling-{gpu,cpu}/`.*
