# JaxEngine

A pure-[JAX](https://jax.readthedocs.io/) implementation supporting both harmonic (OPLSAA-style) and MM3
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

## Supported energy terms

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

## GPU support

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

!!! tip "Optax optimizers"
    JaxEngine pairs naturally with
    [optax adaptive optimizers](../how-it-works/optimization-guide.md#workflow-b-small-rugged)
    (Adam, AdaGrad, SGD) via `OptaxOptimizer`. These use JAX's analytical
    gradients automatically — no finite-difference overhead. On CH₃F MM3,
    Adam achieves 56.3 cm⁻¹ RMSD (10× better than L-BFGS-B).
    See [Small Molecules](../benchmarks/small-molecules.md) for full results.

!!! tip "JaxOpt end-to-end optimization"
    JaxEngine also supports fully JIT-compiled optimization via
    [`JaxOptOptimizer`](../how-it-works/optimization-guide.md#workflow-d-end-to-end-differentiable-jaxopt),
    which runs the entire loss + gradient + L-BFGS step inside JAX with
    no Python callbacks. Supports energy, frequency, hessian element, and
    eigenmatrix objectives. Use `full_method="jaxopt:lbfgs"` in the
    cycling loop for JIT-compiled gradient phases.

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

## See also

- [JaxMDEngine](jax-md.md) — periodic boundaries, neighbor lists, 1-4 scaling
- [Engine comparison table](index.md#engine-overview)
- [GPU benchmarks](../benchmarks/gpu.md)
- [API Reference: JaxEngine](../reference/q2mm/backends/mm/jax_engine.md)
