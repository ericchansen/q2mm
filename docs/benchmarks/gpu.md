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
(the larger of the two) shows a 1.6× CPU advantage.  This is expected
behaviour — see [Why CPU Wins](#why-cpu-wins) below.

---

## Why CPU Wins

Three factors combine to make GPUs slower for q2mm's current workloads.

### 1. Float64 on a consumer GPU

q2mm requires float64 (double precision) for numerically stable
Hessians and eigenvalue decomposition.  Consumer GPUs artificially
limit double-precision throughput:

| Card | FP32 (TFLOPS) | FP64 (TFLOPS) | FP64 : FP32 |
|------|-------------:|-------------:|:-----------:|
| **RTX 5090** | 104.8 | 1.6 | **1 : 64** |
| RTX 4090 | 82.6 | 1.3 | 1 : 64 |
| A100 (datacenter) | 19.5 | 9.7 | 1 : 2 |
| H100 (datacenter) | 67 | 34 | 1 : 2 |

The RTX 5090 delivers only **1.6 TFLOPS** in float64 — comparable to
the Ryzen 7 7800X3D CPU, which has no float64 penalty.  This single
factor accounts for most of the GPU's disadvantage.

Sources: [TechPowerUp RTX 5090 specs][tp5090], [NVIDIA A100 datasheet][a100]

[tp5090]: https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216
[a100]: https://www.nvidia.com/en-us/data-center/a100/

### 2. Small matrices, many kernel launches

Each GPU operation incurs fixed kernel-launch overhead.  JAX's own
[benchmarking guide][jaxbench] shows that **for 10×10 matrices, GPU is
10× slower than CPU**; the crossover happens around 1000×1000.

q2mm's Hessian sizes fall well below the crossover:

| System | Atoms | Hessian shape |
|--------|------:|:--------------|
| CH₃F | 5 | 15 × 15 |
| rh-enamide (largest) | 62 | 186 × 186 |

Each molecule is evaluated in its own kernel launch — 9 separate
launches per eval for rh-enamide, with no batching across molecules.

[jaxbench]: https://docs.jax.dev/en/latest/benchmarking.html

### 3. Sequential molecule evaluation

The objective function loops over molecules in Python:

```python
for ref in self.reference.values:
    if mol_idx not in calc_cache:
        calc_cache[mol_idx] = self._evaluate_molecule(mol_idx, ff)
```

Each `_evaluate_molecule` → `engine.hessian()` → one JAX kernel launch.
There is no `vmap` over molecules for the frequency objective.  This
pattern cannot saturate a GPU.

---

## Comparison with JAX-ReaxFF

[JAX-ReaxFF][jaxreaxff] reports **10–100× GPU speedup** for reactive
force field parameter fitting.  Their architecture differs from q2mm in
every dimension that matters for GPU performance:

| Aspect | JAX-ReaxFF | q2mm (current) |
|--------|-----------|----------------|
| Molecule batching | `vmap` over all geometries in one call | Per-molecule Python loop |
| Derivatives needed | 1st order (gradients) | 2nd order (Hessians) |
| Optimization method | Gradient-based (L-BFGS) | Scipy minimize + Simplex cycling |
| Precision | float32 sufficient | float64 required |
| System sizes | 100s of atoms | 5–62 atoms |

Their speedup is primarily **algorithmic** (gradient-based optimization
needs ~1000× fewer evaluations than genetic algorithms) rather than
purely hardware-driven.

[jaxreaxff]: https://github.com/cagrikymk/JAX-ReaxFF

Source: [JAX-ReaxFF paper (ChemRxiv)][jaxreaxff-paper],
[JAX-ReaxFF poster (Zenodo)][jaxreaxff-poster]

[jaxreaxff-paper]: https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2021-b342n
[jaxreaxff-poster]: https://zenodo.org/records/6863899

---

## When GPU Would Help

GPU acceleration would become beneficial under different conditions:

- **Datacenter GPUs** (A100 / H100) — 6–20× more FP64 throughput
  than consumer cards
- **Large molecules** (> 500 atoms) — Hessians become 1500×1500+,
  enough to saturate GPU cores
- **Energy-only objectives** — No Hessians; batched energy via
  `jax.vmap` already works (see
  [`JaxEngine.batched_energy`][q2mm.backends.mm.jax_engine.JaxEngine.batched_energy])
- **Many molecules batched** — `vmap` over 100+ structures in one
  kernel launch (requires padding to uniform atom count)
- **End-to-end differentiable loss** — A single JIT-compiled function
  from parameters → loss (like JAX-ReaxFF) would eliminate Python loop
  overhead and enable GPU kernel fusion

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

## Further Reading

- [JAX GPU Performance Tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html) — official guide on maximizing GPU throughput
- [JAX Benchmarking Guide](https://docs.jax.dev/en/latest/benchmarking.html) — `block_until_ready`, CPU vs GPU crossover points
- [Efficient Hessians in JAX](https://stackoverflow.com/questions/70572362/compute-efficiently-hessian-matrices-in-jax) — `jacfwd(jacrev(...))` patterns
- [JAX GPU slower than CPU (issue #18816)](https://github.com/jax-ml/jax/issues/18816) — community reports of same phenomenon
- [JAX-ReaxFF paper](https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2021-b342n) — gradient-based FF optimization with GPU speedup
- [DMFF](https://github.com/deepmodeling/DMFF) — differentiable molecular force field platform
