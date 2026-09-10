# OpenMM Backend

The `OpenMMBackend` is Q2MM's most versatile backend, supporting both
**Harmonic** and **MM3** functional forms. It runs in-process via the
[OpenMM](https://openmm.org/) Python API, avoiding subprocess overhead.

---

## Installation

OpenMM is available via conda-forge:

```bash
conda install -c conda-forge openmm
```

Or with pip:

```bash
pip install openmm
```

For **GPU support** (via CUDA), install the CUDA plugin package:

```bash
pip install OpenMM-CUDA-12
```

This provides CUDA plugin binaries that JIT-compile kernels via NVRTC,
supporting all NVIDIA architectures including Blackwell (RTX 5090).
Works on **Linux, WSL2, and native Windows** — requires an NVIDIA GPU
and a compatible driver (≥ 535).

!!! tip "WSL2 recommended for GPU benchmarks"
    For GPU benchmarks and the full CUDA stack (JAX CUDA + JAX-MD +
    OpenMM CUDA), WSL2 is the recommended environment on Windows.
    Native Windows supports OpenMM CUDA but not JAX CUDA or JAX-MD.

!!! warning "Avoid OpenCL for GPU acceleration"
    OpenCL on modern NVIDIA GPUs (e.g. RTX 5090) gives very poor GPU
    utilisation (~14%). **Always prefer CUDA** over OpenCL when an NVIDIA
    GPU is available. The auto-detection order (CUDA > OpenCL > CPU)
    ensures CUDA is selected first when both plugins are installed.

!!! tip "Verify installation"
    ```python
    import openmm
    print(openmm.version.full_version)
    print(openmm.Platform.getNumPlatforms(), "platforms available")
    ```

### Platform detection

OpenMMBackend auto-detects the fastest available compute platform:

| Priority | Platform | Notes |
|----------|----------|-------|
| 1 | CUDA | Requires NVIDIA GPU + CUDA toolkit |
| 2 | OpenCL | AMD/Intel GPUs |
| 3 | CPU | Multi-threaded, available everywhere |
| 4 | Reference | Single-threaded, for debugging only |

Override with the `platform_name` constructor parameter if needed.

---

## Supported energy terms

| Term | Harmonic Mode | MM3 Mode |
|------|:---:|:---:|
| Bonds | ✅ Harmonic | ✅ Cubic/quartic |
| Angles | ✅ Harmonic | ✅ Sextic |
| Torsions | ✅ | ✅ |
| Improper torsions | ❌ | ❌ |
| vdW (LJ 12-6) | ✅ | — |
| vdW (Buckingham exp-6) | — | ✅ |
| Electrostatics | ❌ | ❌ |
| 1-4 scaling | ✅ AMBER (ε/2) | None (MM3) |

### Nonbonded topology and CMAP grids

Nonbonded exclusions are derived from the **bond graph**, even if the molecule
has no angle or torsion records. Pairs separated by one or two bonds are
excluded; harmonic-mode pairs whose shortest path is three bonds retain
half-strength vdW interactions. MM3 keeps its existing unscaled 1-4 interactions.
Atom types declared in `nonbonded_excluded_atom_types` have no nonbonded
interactions, including with other excluded centers.

CMAP supplies a fixed two-dihedral energy correction. The backend converts
`CmapGrid`'s phi-major ordering and -180-degree origin to
[OpenMM's phi-fast ordering and zero-degree origin](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CMAPTorsionForce.html#openmm.openmm.CMAPTorsionForce.addMap),
and converts kcal/mol to kJ/mol once. **Only even grid resolutions are
supported**: an odd resolution would require resampling, so preparation rejects
it rather than choosing an interpolation policy. This conversion retains
OpenMM's native signed-dihedral convention; it does not establish signed-torsion
equivalence with other backends.

---

## Configuration

```python
from q2mm.backends.mm import OpenMMBackend

backend = OpenMMBackend(
    platform_name=None,   # auto-detect (CUDA > OpenCL > CPU > Reference)
    precision=None,       # "single", "mixed", or "double" (GPU only; default: "mixed")
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `platform_name` | `str \| None` | `None` | Force a specific OpenMM platform |
| `precision` | `str \| None` | `None` | GPU precision mode; ignored on CPU |

### Runtime parameter updates

`OpenMMBackend.prepare(PreparationRequest(...))` returns a prepared session
that owns a reusable OpenMM `Context`. Each typed request carries a full
parameter vector, and the session updates the context's force parameters
without rebuilding the system. This makes iterative optimization fast while
keeping all evaluations behind the typed prepared-session contract.

---

## Capabilities

| Prepared-session operation | Supported | Notes |
|--------|:---------:|-------|
| `energy(EnergyRequest)` | ✅ | Returns kcal/mol |
| `minimize(MinimizationRequest)` | ✅ | OpenMM L-BFGS minimizer |
| `hessian(HessianRequest)` | ✅ | **Numerical** (finite-difference) |
| `frequencies(FrequencyRequest)` | ✅ | From numerical Hessian |
| `parameter_gradient(ParameterGradientRequest)` | ✅ | Exact for bond/angle/torsion; vdW via finite differences |
| `Capability.REUSABLE_STATE` | ✅ | Prepared session reuses the OpenMM context |

---

## Serialization

Standalone force-field XML can be written with
`q2mm.io.save_openmm_xml(force_field, path, molecule=...)`. The new backend
surface does not expose a generic `System` XML exporter; prepared sessions are
for typed evaluations, not file I/O.

---

## Limitations

- **Numerical Hessians** — `hessian()` uses finite differences, not analytical
  second derivatives. Accurate but slower than JAX's analytical Hessian.
- **Partial analytical gradients** — `parameter_gradient()` provides exact
  gradients for bond, angle, and torsion parameters via OpenMM global-parameter
  derivatives. vdW parameter gradients are supplemented via central finite
  differences for positive epsilon, using a step no larger than 1% of epsilon
  or 1e-4 kcal/mol. Radii use a 1e-4 Angstrom step, switching to a second-order
  forward difference when a central step would reach zero. Unused rows and
  rows without non-excluded partners have zero derivatives. Zero epsilon in
  same-row pairs has a finite one-sided derivative; zero epsilon mixed with
  another parameter row is not differentiable and is rejected, even when both
  epsilons are zero. Negative interacting parameters and unrepresentable steps
  are rejected rather than perturbed outside the supported domain.
- **No improper torsions** — not yet implemented.
- **No electrostatics** — charge optimization is not supported.

---

## Example

```python
from q2mm.backends.contracts import EnergyRequest, FrequencyRequest, PreparationRequest
from q2mm.backends.mm.openmm import OpenMMBackend
from q2mm.io.amber import load_amber_frcmod
from q2mm.io.xyz import load_xyz

# Load molecule and force field
mol = load_xyz("molecule.xyz")
ff = load_amber_frcmod("params.frcmod")

backend = OpenMMBackend()
session = backend.prepare(PreparationRequest(case_id="example", molecule=mol, force_field=ff))
params = session.layout.vector(ff)

# Single-point energy
e = session.energy(EnergyRequest(parameters=params)).energy
print(f"Energy: {e:.4f} kcal/mol")

# Frequencies
freqs = session.frequencies(FrequencyRequest(parameters=params)).frequencies
print(f"Frequencies: {freqs}")
```

---

## See also

- [Backend comparison table](index.md#backend-overview)
- [Parameter transferability](index.md#parameter-transferability)
- [Benchmarks](../benchmarks/index.md)
- [API Reference: OpenMMBackend](../reference/q2mm/backends/mm/openmm.md)
