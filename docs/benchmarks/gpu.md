# GPU Acceleration

GPU vs CPU benchmarks using JAX-based backends on an NVIDIA RTX 5090
(32 GB VRAM, Blackwell sm_120).  All runs use float64 precision.

**GPU acceleration delivers real speedups for medium-to-large molecular
systems:** JAX-MD OPLSAA achieves **5.6× per-evaluation speedup** and
JAX harmonic achieves **2.1×** on the 9-molecule Rh-enamide system.
Small molecules (CH₃F, 5 atoms) remain faster on CPU due to kernel
launch overhead.

This page summarizes dedicated GPU-vs-CPU comparison runs.  For the latest
full backend × form × optimizer CH₃F matrix, see
[Small Molecules](small-molecules.md).

---

## Results

### Full optimization (L-BFGS-B)

End-to-end L-BFGS-B optimization using `q2mm-benchmark`.  All
benchmarks run **sequentially** on an otherwise idle system to ensure
valid timing.  Per-evaluation time is the fair comparison metric — eval
counts may differ between CPU and GPU due to float64 reduction-order
differences.

**Rh-enamide** (9 molecules, 36–62 atoms each, 182 parameters):

| Backend | Device | s/eval | Evals | Wall Time | GPU Speedup |
|---------|--------|-------:|------:|----------:|:-----------:|
| JAX-MD (OPLSAA) | GPU | 13.44 | 447 | 6,009 s | **5.61×** |
| JAX-MD (OPLSAA) | CPU | 75.38 | 316 | 23,819 s | — |
| JAX (harmonic) | GPU | 12.60 | 31 | 391 s | **2.08×** |
| JAX (harmonic) | CPU | 26.17 | 21 | 550 s | — |

**CH₃F** (1 molecule, 5 atoms, 8 parameters):

| Backend | Device | s/eval | Evals | Wall Time | GPU Speedup |
|---------|--------|-------:|------:|----------:|:-----------:|
| JAX (harmonic) | GPU | 0.054 | 132 | 7.1 s | 0.20× |
| JAX (harmonic) | CPU | 0.011 | 95 | 1.0 s | — |

These CH₃F numbers come from the dedicated JAX harmonic L-BFGS-B GPU-vs-CPU
comparison, not the later 24-combo full matrix.

### Takeaway

GPU speedup **scales with computational complexity**:

- **0.20×** — CH₃F (5 atoms, trivial): GPU kernel launch overhead dominates
- **2.08×** — Rh-enamide JAX harmonic (bonds + angles): moderate benefit from batched Hessian via `jax.vmap`
- **5.61×** — Rh-enamide JAX-MD OPLSAA (full force field with LJ, Coulomb, torsions): GPU parallelism fully utilised across the 9-molecule batch

The crossover point is somewhere between 5 and 36 atoms.  Systems with
more molecules, more atoms per molecule, or more complex force field
terms benefit most from GPU acceleration.

---

## When CPU is Faster

Three factors can make GPUs slower for small or simple workloads.
Understanding each explains the CH₃F results and predicts when GPU
acceleration will and won't help.

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

However, **float32 is not straightforward** for larger molecules.  Testing
both full-float32 and mixed-precision (float32 Hessian, float64
eigendecomp) showed that frequency errors reach 0.44–0.78 cm⁻¹ in
rh-enamide's softest real vibrational modes — above the strict
0.1 cm⁻¹ threshold.  Small molecules like CH₃F (5 atoms) pass
easily (max error 0.0002 cm⁻¹).  The bottleneck is the Hessian
computation itself: `jax.hessian` in float32 introduces ~10⁻⁶
relative errors in matrix elements, which propagate into eigenvalues
of soft modes.  See [Float32 viability test](#float32-viability-test)
for full methodology and results, including a mixed-precision path
that may be viable for relaxed thresholds or early optimisation cycles.

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
| Optimization | Gradient-based L-BFGS (via JAX) | L-BFGS-B (GRAD) + Nelder-Mead (SIMP) via Scipy |
| Loss function | Single JIT-compiled params → loss | Python loop calling engine per molecule |
| Precision | float32 sufficient | float64 default; mixed precision under investigation ([details](#float32-viability-test)) |
| System sizes | 100s of atoms | 5–62 atoms |

The optimization row deserves careful discussion.  q2mm already uses
gradient-based L-BFGS-B for the GRAD step in its cycling loop, so
the optimizer itself is not the primary gap.  However, there is a
subtle but important difference in **what those gradients flow
through**.  In JAX-ReaxFF, L-BFGS operates on a single JIT-compiled
function that maps parameters all the way to a scalar loss — the
gradients are analytical derivatives of the loss with respect to
every parameter, computed via `jax.grad` through the entire
computation graph in one fused kernel.  In q2mm, L-BFGS-B operates
on a loss function that internally calls back into Python for each
molecule, computes Hessians, converts to frequencies via NumPy
eigenvalue decomposition, and sums residuals — the parameter
gradients for L-BFGS are computed by Scipy via finite differences or
by q2mm's own per-evaluator gradient methods, not by differentiating
through the full pipeline.  This means q2mm's L-BFGS cannot take
advantage of XLA kernel fusion or GPU-native gradient computation,
even though it is technically the same algorithm.

A natural question is whether q2mm could also use gradient-based
L-BFGS through the full frequency pipeline — i.e., differentiate
through the eigenvalue decomposition to get analytical parameter
gradients of the frequency loss.  JAX supports differentiating
through `jnp.linalg.eigh`, so this is feasible in principle.
Doing so would eliminate the need for finite-difference parameter
gradients in the GRAD step and could significantly reduce the number
of evaluations needed per L-BFGS iteration.

The loss function structure is the other critical difference.
JAX-ReaxFF compiles a single function from parameters to scalar
loss via `jax.jit`, letting XLA fuse all per-molecule evaluations,
gradient computations, and the loss reduction into one optimised
GPU kernel.  q2mm instead calls back into Python between each
molecule, which prevents XLA from seeing the full computation
graph and makes it impossible to amortise kernel-launch overhead
across molecules.

Their speedup compared to prior ReaxFF tools is also partly
**algorithmic**: replacing genetic algorithms with gradient-based
optimisation reduces the number of evaluations by ~1000×.  Since
q2mm already uses gradient-based methods for the GRAD step, that
particular improvement does not directly apply.  However, if q2mm
could differentiate through the full frequency pipeline, the number
of evaluations per L-BFGS iteration would drop from O(parameters)
(finite-difference) to O(1) (analytical gradient), which is a
substantial saving for systems like rh-enamide with 94 parameters.

There are two (non-exclusive) paths forward.  One is to integrate
[JAX-ReaxFF][jaxreaxff] as a backend or adopt its architecture for
q2mm's own engines — this would give immediate access to a
battle-tested `vmap`-batched, JIT-compiled, gradient-based pipeline
for reactive force fields.  The other is to restructure q2mm's own
objective function into an end-to-end JIT-compiled
`params → loss` function, which would work with any JAX-based engine
(not just ReaxFF) and preserve q2mm's existing cycling workflow.
Both approaches are worth exploring and are tracked as open
questions in the future directions below.

[jaxreaxff]: https://github.com/cagrikymk/JAX-ReaxFF

Source: [JAX-ReaxFF paper (ChemRxiv)][jaxreaxff-paper],
[JAX-ReaxFF poster (Zenodo)][jaxreaxff-poster]

[jaxreaxff-paper]: https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2021-b342n
[jaxreaxff-poster]: https://zenodo.org/records/6863899

---

## Future Directions

GPU acceleration already delivers meaningful speedups for larger
systems, but several improvements could push the gains further.
Each also benefits CPU performance.  They are listed roughly in order
of effort, but are not strictly sequential — results from earlier
items may change the priority of later ones.

Tracked in [issue #176](https://github.com/ericchansen/q2mm/issues/176)
and linked issues.

### Float32 viability test

Tracked in [#178](https://github.com/ericchansen/q2mm/issues/178).

#### Methodology

Three precision configurations were tested:

1. **Full float64** (baseline) — Hessian computation and eigendecomposition
   both in float64.  This is the current default.
2. **Full float32** — `JAX_ENABLE_X64=0` before import, so `jax.hessian`
   and all JAX operations use float32.  Eigendecomposition via numpy's
   `eigvalsh` still uses float64 internally, but the *input* Hessian is
   float32-precision.
3. **Mixed precision** — Hessian computed in float32 via JAX (GPU-friendly),
   then explicitly cast to float64 before eigendecomposition.  This tests
   whether the Hessian itself retains enough precision in float32 for the
   eigenvalue solver to produce accurate frequencies.

The pipeline under test is:

```
params, coords → jax.hessian(energy_fn) → Hess (kcal/mol/Å²)
    → unit conversion (Hartree/Bohr²) → mass-weighting → eigvalsh → cm⁻¹
```

"Real modes" are frequencies with |ν| > 50 cm⁻¹, excluding the 6
near-zero translation/rotation modes which are numerically unstable
regardless of precision.

Tests run inside the `ci-jax` Docker container on CPU (AMD Ryzen 7
7800X3D).  CPU float32 and float64 have identical throughput, so the
precision comparison is isolated from hardware effects.  On GPU, the
benefit would be the FP32:FP64 throughput ratio (64× on RTX 5090).

Scripts:
[`scripts/float32_experiment.py`](https://github.com/ericchansen/q2mm/blob/master/scripts/float32_experiment.py)
(full float32 vs float64),
[`scripts/mixed_precision_experiment.py`](https://github.com/ericchansen/q2mm/blob/master/scripts/mixed_precision_experiment.py)
(mixed precision).

#### Results: Full float32

| System | Modes | Max Δ (cm⁻¹) | Mean Δ | RMSD |
|--------|------:|-------------:|-------:|-----:|
| CH₃F (all modes) | 15 | 0.224 | 0.053 | 0.099 |
| CH₃F (real modes, >50 cm⁻¹) | 9 | **0.0002** | 0.0001 | 0.0001 |
| rh-enamide (all modes) | 1,404 | 9.127 | 0.174 | 1.007 |
| rh-enamide (real modes, >50 cm⁻¹) | 1,347 | **0.785** | 0.021 | 0.074 |

Full float32 fails the 0.1 cm⁻¹ threshold for rh-enamide by ~8×.
Near-zero modes show errors up to 9.1 cm⁻¹.

#### Results: Mixed precision (float32 Hessian → float64 eigendecomp)

| System | Atoms | DOF | Hess max Δ | Hess rel max | Freq max Δ (real) | Mean Δ | RMSD | Verdict |
|--------|------:|----:|-----------:|-------------:|------------------:|-------:|-----:|---------|
| CH₃F | 5 | 15 | 1.1 × 10⁻⁴ | 8.0 × 10⁻⁸ | **0.0002** | 0.0001 | 0.0001 | ✅ Pass |
| rh-enamide mol0 | 36 | 108 | 18.7 | 1.9 × 10⁻⁶ | **0.443** | 0.022 | 0.061 | ❌ Fail |

"Hess max Δ" is the maximum absolute difference between any element of
the float64 and float32 Hessians (in kcal/mol/Å²).  "Hess rel max" is
relative to the largest Hessian element.

Mixed precision is **better** than full float32 (0.44 vs 0.78 cm⁻¹ max
error) but still exceeds the 0.1 cm⁻¹ threshold by ~4×.

The worst errors concentrate in the **lowest-frequency real modes**
(60–86 cm⁻¹) — soft, floppy motions where small Hessian perturbations
are amplified through the eigenvalue decomposition:

```
freq[ 35]:  f64=  65.0674   mixed=  65.5104   Δ=0.443 cm⁻¹
freq[ 38]:  f64=  85.5928   mixed=  85.7688   Δ=0.176 cm⁻¹
freq[ 37]:  f64=  79.5196   mixed=  79.6864   Δ=0.167 cm⁻¹
freq[ 34]:  f64=  60.6396   mixed=  60.7742   Δ=0.135 cm⁻¹
```

High-frequency modes (>500 cm⁻¹) are much less affected because the
corresponding Hessian eigenvalues are larger relative to the float32
noise floor.

#### Analysis: Where does precision matter?

The error budget breaks down as:

| Stage | Error source | Impact |
|-------|-------------|--------|
| Energy function | float32 evaluates `E = k(r−r₀)²` etc. | Negligible — terms are well-conditioned |
| `jax.hessian` (autodiff) | Forward-over-reverse in float32 | 18.7 kcal/mol/Å² max element error (1.9 × 10⁻⁶ relative) |
| Unit conversion | Linear scaling by `KCALMOLA2_TO_HESSIAN_AU` | No additional error (uniform scaling doesn't change condition number) |
| Mass-weighting | Divides by √(mᵢ · mⱼ); mass range H(1)–Rh(103) | Amplifies existing errors non-uniformly |
| `eigvalsh` | Eigenvalue decomposition | In float64: accurate.  Float32 Hessian errors propagate into eigenvalues of soft modes |

The bottleneck is the **Hessian computation itself**, not the
eigendecomposition.  Even with float64 `eigvalsh`, the float32
Hessian has ~10⁻⁶ relative errors that are large enough to shift
eigenvalues of the softest modes by ~0.4 cm⁻¹.

#### Interpretation

The 0.1 cm⁻¹ threshold is strict but physically motivated: it ensures
that optimised parameters reproduce QM frequencies within typical
experimental resolution.  Whether a relaxed threshold (e.g. 0.5 or
1.0 cm⁻¹) is acceptable depends on the application:

- **Final production fits:** 0.1 cm⁻¹ is appropriate → float64 required
- **Early optimisation cycles / coarse exploration:** 0.5 cm⁻¹ may be
  acceptable → mixed precision could accelerate by 64× on consumer GPUs
- **Small molecules (≤10 atoms):** float32 passes easily → always safe
- **Stiff molecules (no soft modes <100 cm⁻¹):** likely safe even at
  30+ atoms, but not yet tested

**Recommendation:** Keep float64 as default.  The `JAX_ENABLE_X64=0`
environment variable allows opt-in for specific use cases.  A future
adaptive strategy could use float32 for early cycles and switch to
float64 for refinement.

### Batch kernel launches across molecules with `vmap`

Pad molecules to a uniform atom count and use `jax.vmap(hessian_fn)`
to compute all Hessians in a single kernel launch, replacing the
per-molecule Python loop.  This directly addresses
[factors 2 and 3](#2-small-matrices-many-kernel-launches) above.
The energy-only path already supports batching via
[`JaxEngine.batched_energy`][q2mm.backends.mm.jax_engine.JaxEngine.batched_energy],
so the pattern is proven within q2mm; extending it to Hessians is
the next logical step.  For rh-enamide, this would consolidate 9
separate kernel launches into one, giving the GPU a much larger
workload to parallelise.

### `vmap` over molecules in the objective function

A related but distinct idea: instead of (or in addition to) batching
Hessians at the engine level, `vmap` the entire per-molecule
evaluation at the objective function level.  This would require the
objective to be expressible as a pure JAX function — no Python
control flow between molecules — but would let XLA see all molecules
at once for maximum kernel fusion.  This overlaps with the
end-to-end JIT approach below.

### End-to-end JIT-compiled loss function

Restructure the frequency objective as a single `jax.jit`-compiled
function: `params → energy → hessian → eigenvalues → frequencies →
residuals → loss`.  This lets XLA fuse the entire computation graph
into optimised kernels, eliminates Python loop overhead, and enables
`jax.grad(loss)` for analytical parameter gradients through the full
pipeline — including through the eigenvalue decomposition.  If this
works, it would also enable gradient-based L-BFGS through the full
frequency loss (not just finite-difference gradients), reducing the
per-iteration cost from O(parameters) evaluations to O(1).

### JAX-ReaxFF integration

Explore integrating [JAX-ReaxFF][jaxreaxff] as a backend or adopting
its architecture.  JAX-ReaxFF already has a mature, GPU-optimised
pipeline for reactive force fields with `vmap` batching, JIT
compilation, and gradient-based L-BFGS.  Supporting it as an engine
would give q2mm access to reactive force field fitting with proven
GPU speedups.  Alternatively, studying its architecture could inform
how to restructure q2mm's own objective function.

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
| OpenMM CUDA (Blackwell / sm_120) | ✅ Works — `pip install OpenMM-CUDA-12` (Linux / WSL2 / Windows). Uses NVRTC to JIT-compile kernels for sm_120. |
| JAX force fields | Harmonic only (no MM3) |

---

## Reproducing

Each benchmark was run **sequentially** on an otherwise idle system.
CPU baselines use `JAX_PLATFORMS=cpu` to force CPU-only execution.

```bash
# Activate the virtual environment with jax[cuda12] installed
# Linux / WSL2:
source .venv/bin/activate
# Windows (native, JAX CUDA not available — OpenMM CUDA only):
# .venv\Scripts\activate

# Rh-enamide: JAX GPU vs CPU
q2mm-benchmark --system rh-enamide --backend jax --optimizer L-BFGS-B --output benchmarks/rh-enamide
JAX_PLATFORMS=cpu q2mm-benchmark --system rh-enamide --backend jax --optimizer L-BFGS-B --output benchmarks/rh-enamide

# Rh-enamide: JAX-MD GPU vs CPU (warning: GPU ~100 min, CPU ~6.6 hours)
q2mm-benchmark --system rh-enamide --backend jax-md --optimizer L-BFGS-B --output benchmarks/rh-enamide
JAX_PLATFORMS=cpu q2mm-benchmark --system rh-enamide --backend jax-md --optimizer L-BFGS-B --output benchmarks/rh-enamide

# CH₃F: JAX GPU vs CPU (dedicated GPU study)
q2mm-benchmark --system ch3f --backend jax --optimizer L-BFGS-B --output benchmark_results/ch3f
JAX_PLATFORMS=cpu q2mm-benchmark --system ch3f --backend jax --optimizer L-BFGS-B --output benchmark_results/ch3f
```

Raw data: `benchmarks/GPU_BENCHMARKS.md` for the dedicated GPU study.  The
latest full CH₃F matrix artifacts live under `benchmark_results/ch3f/`.

## Hardware & Software

| Component | Version |
|-----------|---------|
| GPU | NVIDIA RTX 5090 (32 GB GDDR7, Blackwell sm_120) |
| CPU | AMD Ryzen 7 7800X3D (8 cores, 16 threads) |
| Driver | NVIDIA 591.74 |
| CUDA runtime | 13.1 (driver-reported; JAX uses `jax[cuda12]` wheels targeting CUDA 12.x) |
| JAX | 0.9.2 |
| jax-md | 0.2.8 |
| Python | 3.12 |
| Precision | float64 |

## Further Reading

- [JAX GPU Performance Tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html) — official guide on maximizing GPU throughput
- [JAX Benchmarking Guide](https://docs.jax.dev/en/latest/benchmarking.html) — `block_until_ready`, CPU vs GPU crossover points
- [Efficient Hessians in JAX](https://stackoverflow.com/questions/70572362/compute-efficiently-hessian-matrices-in-jax) — `jacfwd(jacrev(...))` patterns
- [JAX GPU slower than CPU (issue #18816)](https://github.com/jax-ml/jax/issues/18816) — community reports of same phenomenon
- [JAX-ReaxFF paper](https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2021-b342n) — gradient-based FF optimization with GPU speedup
- [DMFF](https://github.com/deepmodeling/DMFF) — differentiable molecular force field platform
