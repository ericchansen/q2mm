# Platform Support

Canonical reference for Q2MM platform compatibility across operating
systems and GPU configurations.

---

## Compatibility matrix

| Component | Linux | WSL2 | Windows (native) | macOS |
|-----------|:-----:|:----:|:-----------------:|:-----:|
| Q2MM core | ✅ | ✅ | ✅ | ✅ |
| OpenMM (CPU) | ✅ | ✅ | ✅ | ✅ |
| OpenMM CUDA | ✅ | ✅ | ✅ | ❌ (no NVIDIA GPU) |
| OpenMM OpenCL | ✅ | ✅ | ✅ | ✅ |
| JAX (CPU) | ✅ | ✅ | ✅ | ✅ |
| JAX CUDA (`jax[cuda12]`) | ✅ | ✅ | ❌ | ❌ |
| JAX-MD | ✅ | ✅ | ❌ | ✅ |
| Psi4 | ✅ | ✅ | ❌ | ✅ (conda) |
| Tinker | ✅ | ✅ | ✅ | ✅ |

!!! tip "Linux is the recommended GPU environment"
    Linux provides the full GPU stack: JAX CUDA + JAX-MD + OpenMM CUDA.
    WSL2 counts as Linux for this purpose. Native Windows supports OpenMM
    CUDA but not JAX CUDA or JAX-MD.

---

## GPU setup

For GPU-accelerated workflows, install the CUDA extras you need:

### Linux (including WSL2)

```bash
# Full stack: all backends + OpenMM CUDA + JAX CUDA + JAX-MD
pip install "q2mm[all,openmm-cuda]"
pip install "jax[cuda12]"

# OpenMM CUDA only (no JAX GPU)
pip install "q2mm[openmm,openmm-cuda,optimize]"

# JAX CUDA only (no JAX-MD, no OpenMM)
pip install "q2mm[jax,optimize]"
pip install "jax[cuda12]"

# JAX CUDA + JAX-MD (no OpenMM)
pip install "q2mm[jax,jax-md,optimize]"
pip install "jax[cuda12]"
```

### Windows (native)

```bash
# OpenMM CUDA works on native Windows
pip install "q2mm[openmm,openmm-cuda,optimize]"

# JAX CUDA and JAX-MD are NOT available on native Windows.
# Use Linux or WSL2 for these backends.
```

### macOS

```bash
# CPU-only (no NVIDIA GPUs on macOS)
pip install "q2mm[all]"
```

---

## Verification commands

### NVIDIA driver

```bash
nvidia-smi
# Expected: driver version, GPU name, CUDA version
```

### OpenMM platforms

```python
import openmm
for i in range(openmm.Platform.getNumPlatforms()):
    print(openmm.Platform.getPlatform(i).getName())
# Expected (with CUDA): Reference, CPU, OpenCL, CUDA
```

### JAX devices

```python
import jax
print(jax.devices())
# Expected (with CUDA): [CudaDevice(id=0)]
```

---

## Common issues

### OpenMM CUDA fails with "unsupported GPU architecture"

The pre-built CUDA plugin may not include PTX for very new GPU
architectures. Install `OpenMM-CUDA-12` ≥ 8.5.0, which uses NVRTC
to JIT-compile kernels at runtime — this supports all architectures
including Blackwell (sm_120).

```bash
pip install "OpenMM-CUDA-12>=8.5.0"
```

### OpenCL gives poor GPU utilisation

OpenCL on modern NVIDIA GPUs (e.g. RTX 5090) achieves only ~14% GPU
utilisation. **Always prefer CUDA** over OpenCL when an NVIDIA GPU is
present. If `detect_best_platform()` returns `"OpenCL"`, install the
CUDA plugin:

```bash
pip install OpenMM-CUDA-12
```

### JAX doesn't see the GPU

```bash
# Check that jax[cuda12] is installed (not just jax)
pip install "jax[cuda12]"

# Verify
python -c "import jax; print(jax.devices())"
```

If it still shows CPU only, check that `nvidia-smi` works and that
CUDA libraries are on `LD_LIBRARY_PATH` (Linux/WSL2).

### WSL2: nvidia-smi works but CUDA fails

Ensure you have the **Windows** NVIDIA driver installed (not a Linux
driver inside WSL2). WSL2 uses the Windows driver via GPU
paravirtualisation. See the
[NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

### JAX-MD not available on Windows

JAX-MD does not publish Windows wheels. Use Linux (or WSL2):

```bash
pip install "q2mm[jax-md]"
```

---

## See also

- [Getting Started](getting-started.md) — installation instructions
- [OpenMM Backend](backends/openmm.md) — OpenMM-specific configuration
- [GPU Benchmarks](benchmarks/gpu.md) — GPU vs CPU performance data
