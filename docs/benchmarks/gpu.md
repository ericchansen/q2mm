# GPU Acceleration

This page answers one question: when does GPU acceleration help q2mm today, and
what kind of speedup is realistic? It focuses on dedicated CPU-vs-GPU
comparison runs rather than on the full benchmark matrices.

## Scope

- Hardware: NVIDIA RTX 5090 (Blackwell), float64 throughout
- Workloads: dedicated JAX and JAX-MD CPU/GPU comparisons on CH₃F and
  Rh-enamide
- Comparison metric: seconds per evaluation (`s/eval`); total evaluation count
  can differ between devices
- Related pages: [Small Molecules](../systems/small-molecules.md) for the full CH₃F matrix
  and [Rh-Enamide](../systems/rh-enamide.md) for the selected overnight large-system run

## CPU-vs-GPU comparisons completed so far

All runs below are full L-BFGS-B optimizations executed sequentially on an idle
machine. Default rows are grouped by system and backend; use the filters and
sortable headers to compare system/backend/device slices directly. `Relative
speed vs CPU` is the device-level comparison to use when deciding whether GPU
is worthwhile.

<div class="benchmark-table-anchor" data-benchmark-table="gpu-comparisons"></div>

| System | Backend | Device | s/eval | Evals | Wall time | Relative speed vs CPU |
|--------|---------|--------|-------:|------:|----------:|----------------------:|
| Rh-enamide | JAX-MD (OPLSAA) | GPU | 13.44 | 447 | 6,009 s | 5.61x |
| Rh-enamide | JAX-MD (OPLSAA) | CPU | 75.38 | 316 | 23,819 s | baseline |
| Rh-enamide | JAX (harmonic) | GPU | 12.60 | 31 | 391 s | 2.08x |
| Rh-enamide | JAX (harmonic) | CPU | 26.17 | 21 | 550 s | baseline |
| CH₃F | JAX (harmonic) | GPU | 0.054 | 132 | 7.1 s | 0.20x |
| CH₃F | JAX (harmonic) | CPU | 0.011 | 95 | 1.0 s | baseline |

## Interpretation

- GPU speedup appears once the workload is large enough and the force field is
  complex enough to keep the device busy. The strongest current example is
  JAX-MD on Rh-enamide at 5.61x faster than CPU on a per-evaluation basis.
- Small systems still favor CPU. CH₃F is faster on CPU because kernel-launch
  overhead dominates the actual arithmetic.
- Compare `s/eval`, not raw evaluation counts. CPU and GPU can take slightly
  different optimization paths because of floating-point reduction-order
  differences.
- The three main reasons GPU can still lose are unchanged: consumer-GPU float64
  throughput is limited, Hessians are still relatively small, and the frequency
  objective evaluates molecules sequentially rather than in one large batched
  kernel.
- The selected overnight Rh-enamide sweep is useful feasibility evidence for
  OpenMM CUDA and large-system screening, but its outcome-by-outcome details
  belong on [Rh-Enamide](../systems/rh-enamide.md) rather than on this device-comparison
  page.
- **Grad-simp cycling now works on JAX GPU** (see [Rh-Enamide](../systems/rh-enamide.md)).
  JAX MM3 grad-simp reached the same 42.7 RMSD as the OpenMM overnight run
  in ~25 minutes on the RTX 5090 — a ~23× optimizer-time improvement. This was
  unblocked by Hessian symmetrisation and bound-aware sensitivity analysis
  that prevent the eigenvalue failures that previously killed every JAX/JAX-MD
  grad-simp attempt.

??? note "Why float64 still matters on RTX 5090-class GPUs"
    q2mm keeps these benchmark runs in float64 because the frequency path is
    still sensitive to Hessian precision, especially for larger systems with
    soft modes.

    | Hardware class | Example | FP64 : FP32 throughput |
    |----------------|---------|-----------------------:|
    | Consumer GPU | [RTX 5090](https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216) | 1 : 64 |
    | Datacenter GPU | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) | 1 : 2 |

    That gap is one reason CH₃F can still be faster on CPU even when a consumer
    GPU is present: q2mm's Hessian/frequency workflow does not get to use the
    much larger FP32 throughput numbers that GeForce-class cards advertise.

    The current float32 story is also still mixed. CH₃F passes comfortably in
    float32, but the larger Rh-enamide tests do not yet make float32 or mixed
    precision a drop-in replacement for the current default workflow. In the
    archived viability study, full float32 produced about 0.78 cm⁻¹ maximum
    error on real Rh-enamide modes, and mixed precision improved that to about
    0.44 cm⁻¹ while still missing the stricter 0.1 cm⁻¹ target. For now, that
    makes float32 an interesting research direction for relaxed thresholds or
    early screening, not the default benchmark setting.

## Multi-System GPU Results

For the full 5-system × 3-optimizer GPU shootout (timing, RMSD,
reliability), see [Optimizer Comparison](optimizer-comparison.md).

## Artifacts and provenance

- Related CH₃F full-matrix artifacts:
  [`q2mm-data/benchmarks/ch3f/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks/ch3f)
- Related Rh-enamide archive:
  [`q2mm-data/benchmarks/`](https://github.com/ericchansen/q2mm-data/tree/main/benchmarks)
  under `rh-enamide/`

## Reproducing

```bash
q2mm-benchmark --system rh-enamide --backend jax --optimizer scipy-lbfgsb --output results/rh-enamide
JAX_PLATFORMS=cpu q2mm-benchmark --system rh-enamide --backend jax --optimizer scipy-lbfgsb --output results/rh-enamide

q2mm-benchmark --system rh-enamide --backend jax-md --optimizer scipy-lbfgsb --output results/rh-enamide
JAX_PLATFORMS=cpu q2mm-benchmark --system rh-enamide --backend jax-md --optimizer scipy-lbfgsb --output results/rh-enamide

q2mm-benchmark --system ch3f --backend jax --optimizer scipy-lbfgsb --output results/ch3f
JAX_PLATFORMS=cpu q2mm-benchmark --system ch3f --backend jax --optimizer scipy-lbfgsb --output results/ch3f
```
