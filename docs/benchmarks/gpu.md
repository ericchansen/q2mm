# GPU Acceleration

GPU vs CPU benchmarks using the JAX backend on an NVIDIA RTX 5090
(32 GB VRAM, Blackwell sm_120) with an AMD Ryzen 7 7800X3D CPU.
All runs use float64 precision.

**The headline result is counter-intuitive: CPU is faster than GPU for
every workload tested.**  This page explains why, what would need to
change, and the planned path to making GPU acceleration viable.

---

## Results

### Per-evaluation throughput

| System | Atoms | Modes | Params | GPU (ms/eval) | CPU (ms/eval) | Speedup |
|--------|------:|------:|-------:|--------------:|--------------:|--------:|
| CH₃F | 5 | 9 | 8 | 2.20 | 0.40 | 0.18× |
| rh-enamide | 36–62 | 1,273 | 94 | 36.2 | 21.9 | 0.60× |

Per-eval times measured as the mean of 50 (CH₃F) or 10 (rh-enamide)
isolated calls after JIT warmup.

Each "eval" computes the full objective: for every molecule in the
training set, run the energy function, compute the Hessian
(`jax.hessian`), diagonalise it to get vibrational frequencies, and sum
the weighted squared deviations from QM reference values.

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

Both devices converge to similar scores and use nearly identical eval
counts (~30 k).  The 1.6× wall-time difference comes entirely from
the per-eval throughput gap, not from algorithmic differences.

!!! warning "Force field caveat"
    rh-enamide uses an auto-generated **harmonic** force field (94
    params) because JaxEngine does not support MM3.  These scores are
    not comparable to the 182-parameter MM3 results on the
    [rh-enamide page](rh-enamide.md).

### Takeaway

**CPU is faster than GPU for these workloads.**  The rh-enamide system
(the larger of the two) shows a 1.6× CPU advantage, and CH₃F is
5.5× faster on CPU.  The smaller the molecule, the worse the GPU
performs — exactly what you would expect given the factors described
below.

---

## Why CPU Wins

Three factors combine to make GPUs slower for q2mm's current workloads.
Understanding each one is important because they point to different
solutions.

### 1. Float64 on a consumer GPU

q2mm currently forces float64 (double precision) globally for
numerically stable Hessians and eigenvalue decomposition.  Consumer
NVIDIA GPUs artificially limit double-precision throughput to push
scientific users toward datacenter cards:

| Card | FP32 (TFLOPS) | FP64 (TFLOPS) | FP64 : FP32 |
|------|-------------:|-------------:|:-----------:|
| **RTX 5090** | 104.8 | 1.6 | **1 : 64** |
| RTX 4090 | 82.6 | 1.3 | 1 : 64 |
| A100 (datacenter) | 19.5 | 9.7 | 1 : 2 |
| H100 (datacenter) | 67 | 34 | 1 : 2 |

The RTX 5090 delivers only **1.6 TFLOPS** in float64 — roughly
comparable to the Ryzen 7 7800X3D CPU, which executes float64 at full
speed with no penalty relative to float32.  This single factor accounts
for most of the GPU's disadvantage.

However, **float64 may not be strictly necessary** for the current
harmonic-only JaxEngine.  For harmonic terms like `E = k(r − r₀)²`,
the Hessian is exactly `2k` — there is no catastrophic cancellation in
the autodiff tape that would corrupt float32 results.  Numerical
analysis shows that resolving 1 cm⁻¹ at a 1000 cm⁻¹ mode requires
only ~8 % relative precision in the Hessian element, while float32
provides 1.2 × 10⁻⁷ relative precision — five orders of magnitude more
than needed.  Switching to float32 would unlock the full 104.8 TFLOPS
on the RTX 5090, a potential **64× throughput increase**.  Float64
would become necessary when VdW terms (with 1/r¹² − 1/r⁶ near-
cancellation), Morse potentials, or very soft modes (~10 cm⁻¹) are
involved.  See [Next Steps](#next-steps) for the plan to validate this.

Sources: [TechPowerUp RTX 5090 specs][tp5090], [NVIDIA A100 datasheet][a100]

[tp5090]: https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216
[a100]: https://www.nvidia.com/en-us/data-center/a100/

### 2. Small matrices, many kernel launches

Every GPU operation incurs fixed kernel-launch overhead — the time to
dispatch work to the GPU, regardless of how much work there is.
JAX's own [benchmarking guide][jaxbench] demonstrates that **for
10×10 matrices, GPU is 10× slower than CPU**; the GPU only pulls
ahead around 1000×1000.

q2mm's Hessian sizes fall well below the crossover:

| System | Atoms | Hessian shape |
|--------|------:|:--------------|
| CH₃F | 5 | 15 × 15 |
| rh-enamide (largest) | 62 | 186 × 186 |

At these sizes the GPU spends more time launching kernels than doing
actual arithmetic.  Worse, each molecule gets its own independent
kernel launch — for rh-enamide that is 9 separate Hessian computations
per eval, each too small to keep the GPU busy.

[jaxbench]: https://docs.jax.dev/en/latest/benchmarking.html

### 3. Sequential molecule evaluation

The objective function loops over molecules in Python, calling the JAX
engine once per molecule:

```python
for ref in self.reference.values:
    if mol_idx not in calc_cache:
        calc_cache[mol_idx] = self._evaluate_molecule(mol_idx, ff)
```

Each `_evaluate_molecule` → `engine.hessian()` → one JAX kernel launch.
There is no `vmap` over molecules for the frequency objective — each
of those 9 small Hessian kernels is dispatched and completed
sequentially.  This is the worst-case pattern for GPU utilisation:
many tiny kernels with Python overhead between each one, giving the
GPU no opportunity to amortise its fixed costs.

The energy-only path already supports batching via
[`JaxEngine.batched_energy`][q2mm.backends.mm.jax_engine.JaxEngine.batched_energy],
which uses `jax.vmap` to evaluate many parameter vectors in a single
kernel.  Extending this pattern to the Hessian/frequency pipeline is
a key part of the path forward.

---

## Comparison with JAX-ReaxFF

[JAX-ReaxFF][jaxreaxff] reports **10–100× GPU speedup** for reactive
force field parameter fitting.  Their design choices are instructive
because they show what GPU-friendly force field fitting looks like:

| Aspect | JAX-ReaxFF | q2mm (current) |
|--------|-----------|----------------|
| Molecule batching | `vmap` over all geometries in one call | Per-molecule Python loop |
| Derivatives needed | 1st order (gradients only) | 2nd order (Hessians for frequencies) |
| Loss function | Single JIT-compiled params → loss | Python loop calling engine per molecule |
| Precision | float32 sufficient | float64 (but [may not be needed](#1-float64-on-a-consumer-gpu)) |
| System sizes | 100s of atoms | 5–62 atoms |

Note that q2mm already uses gradient-based L-BFGS for the GRAD step in
cycling — the optimisation method itself is not the gap.  The critical
difference is **how the loss function is structured**.  JAX-ReaxFF
compiles a single function from parameters to scalar loss via
`jax.jit`, letting XLA fuse all per-molecule evaluations, gradient
computations, and the loss reduction into one optimised GPU kernel.
q2mm instead calls back into Python between each molecule, which
prevents XLA from seeing the full computation graph.

Their speedup compared to prior ReaxFF tools is also partly
**algorithmic**: replacing genetic algorithms with gradient-based
optimisation reduces the number of evaluations by ~1000×.  q2mm
already uses gradient-based methods, so that particular advantage
does not apply here.

[jaxreaxff]: https://github.com/cagrikymk/JAX-ReaxFF

Source: [JAX-ReaxFF paper (ChemRxiv)][jaxreaxff-paper],
[JAX-ReaxFF poster (Zenodo)][jaxreaxff-poster]

[jaxreaxff-paper]: https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2021-b342n
[jaxreaxff-poster]: https://zenodo.org/records/6863899

---

## Next Steps

The path to making GPU acceleration viable for q2mm involves three
independent improvements, each of which also benefits CPU performance.
Tracked in [issue #176](https://github.com/ericchansen/q2mm/issues/176).

### Float32 viability test

Run existing benchmarks with `jax_enable_x64=False` and compare
frequency accuracy against the float64 baseline.  If frequencies
agree to < 0.1 cm⁻¹ for harmonic force fields, float32 can be used
by default on consumer GPUs, unlocking 64× more throughput.  Float64
would remain available (and recommended) for force fields with VdW,
Morse, or other terms prone to cancellation in autodiff.

### Batching Hessians across molecules with `vmap`

Pad molecules to a uniform atom count and use `jax.vmap(hessian_fn)`
to compute all Hessians in a single kernel launch, replacing the
per-molecule Python loop.  This directly addresses
[factors 2 and 3](#2-small-matrices-many-kernel-launches) above.

### End-to-end JIT-compiled loss function

Restructure the frequency objective as a single `jax.jit`-compiled
function: `params → energy → hessian → eigenvalues → frequencies →
residuals → loss`.  This lets XLA fuse the entire computation graph
into optimised kernels, eliminates Python loop overhead, and enables
`jax.grad(loss)` for analytical parameter gradients through the full
pipeline — including through the eigenvalue decomposition.

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
