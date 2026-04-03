# JaxEngine

A pure-JAX implementation supporting both harmonic (OPLSAA-style) and MM3
functional forms, including bond, angle, torsion, and vdW energy terms.
Best for small-to-medium molecules where periodic boundaries and neighbor
lists are not needed.  All energy functions are differentiable via
`jax.grad`, enabling analytical gradient computation.

---

## Installation

```bash
pip install jax jaxlib
```

For GPU support, install the CUDA-enabled `jaxlib`:

```bash
pip install jax[cuda12]
```

!!! tip "Verify installation"
    ```python
    import jax
    print(jax.__version__)
    print(jax.default_backend())  # "cpu" or "gpu"
    ```

---

## Configuration

```python
from q2mm.backends.mm import JaxEngine

engine = JaxEngine()
```

JaxEngine has no constructor parameters.  It runs on whichever JAX backend
is active (`cpu` or `gpu`), detected via `jax.default_backend()`.

---

## Supported Energy Terms

| Term | Supported |
|------|:---------:|
| Bonds (harmonic + MM3) | ✅ |
| Angles (harmonic + MM3) | ✅ |
| Torsions (cosine) | ✅ |
| Improper torsions | ❌ |
| vdW (LJ 12-6 + Buckingham exp-6) | ✅ |
| Electrostatics | ❌ |
| 1-4 scaling | ❌ Not implemented |

**Functional forms:** Harmonic and MM3.

---

## Capabilities

| Method | Supported | Notes |
|--------|:---------:|-------|
| `energy()` | ✅ | Pure JAX |
| `minimize()` | ✅ | JAX gradients + SciPy L-BFGS-B |
| `hessian()` | ✅ | **Analytical** via `jax.hessian` |
| `frequencies()` | ✅ | From analytical Hessian |
| `energy_and_param_grad()` | ✅ | **Analytical** via `jax.grad` |
| `batched_energy()` | ✅ | **Vectorized** via `jax.vmap` |
| `supports_runtime_params()` | ✅ | — |
| `supports_analytical_gradients()` | ✅ | — |

---

## GPU Support

JaxEngine runs on whichever device JAX selects.  To use a GPU:

1. Install the CUDA-enabled JAX: `pip install jax[cuda12]`
2. Verify: `python -c "import jax; print(jax.default_backend())"`

The engine name includes the backend string (e.g., `JAX (harmonic, gpu)`
or `JAX (harmonic, cpu)`).

!!! info "Performance"
    In the current benchmark set, JaxEngine is one of the fastest in-process
    backends for harmonic CH₃F optimization and offers analytical gradients
    for energy-based evaluators.  Exact speedups depend on system size,
    objective, and device, so use the
    [benchmark overview](../benchmarks/index.md) and
    [GPU benchmarks](../benchmarks/gpu.md) for workload-specific numbers.

---

## Limitations

- **No 1-4 pair scaling** — non-bonded energies differ from OpenMM/JAX-MD
  for molecules with 1-4 interactions.  See the
  [compatibility notes](index.md#when-becomes).
- **No periodic boundaries** — gas-phase only.

---

## Example

```python
from q2mm.backends.mm import JaxEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule

mol = Q2MMMolecule.from_xyz("molecule.xyz")
ff = ForceField.create_for_molecule(mol)

engine = JaxEngine()
e = engine.energy(mol, ff)
print(f"JAX energy: {e:.4f} kcal/mol")

# Analytical parameter gradients
e, grad = engine.energy_and_param_grad(mol, ff)
print(f"Energy: {e:.4f}, grad shape: {grad.shape}")
```

---

## See Also

- [JaxMDEngine](jax-md.md) — periodic boundaries, neighbor lists, 1-4 scaling
- [Engine comparison table](index.md#engine-overview)
- [GPU benchmarks](../benchmarks/gpu.md)
- [API Reference: JaxEngine](../reference/q2mm/backends/mm/jax_engine.md)
