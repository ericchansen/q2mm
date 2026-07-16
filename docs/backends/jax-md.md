# JAX-MD Backend

Built on the [JAX-MD](https://github.com/jax-md/jax-md) library, this
backend adds **periodic boundary conditions**, **neighbor lists**, and
**configurable 1-4 scaling** on top of [JAX](https://jax.readthedocs.io/)'s differentiable energy
functions.

---

## Installation

```bash
pip install jax jaxlib jax-md
```

For GPU support:

```bash
pip install jax[cuda12] jax-md
```

---

## Supported energy terms

| Term | Supported |
|------|:---------:|
| Bonds (harmonic) | ✅ |
| Angles (harmonic) | ✅ |
| Torsions (cosine) | ✅ |
| Improper torsions | ❌ |
| vdW (LJ 12-6) | ✅ |
| Electrostatics | Infrastructure only (charges zeroed) |
| 1-4 scaling | ✅ Configurable (default: AMBER 0.5) |
| Periodic boundaries | ✅ |
| Neighbor lists | ✅ (jax-md native) |

**Functional forms:** Harmonic only.

---

## Configuration

```python
from q2mm.backends.mm import JaxMdBackend

backend = JaxMdBackend(
    box=(100.0, 100.0, 100.0),   # simulation box dimensions (Å)
    coulomb=None,                 # CoulombHandler; default: CutoffCoulomb(r_cut=12.0)
    nb_options=None,              # NonbondedOptions; default: r_cut=12.0
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `box` | `tuple[float, float, float]` | `(100.0, 100.0, 100.0)` | Periodic box dimensions in Å |
| `coulomb` | `CoulombHandler \| None` | `CutoffCoulomb(r_cut=12.0)` | Electrostatic handler |
| `nb_options` | `NonbondedOptions \| None` | `NonbondedOptions(r_cut=12.0)` | Non-bonded cutoff and options |

---

## Capabilities

| Prepared-session operation | Supported | Notes |
|--------|:---------:|-------|
| `energy(EnergyRequest)` | ✅ | — |
| `minimize(MinimizationRequest)` | ✅ | JAX gradients + SciPy L-BFGS-B |
| `hessian(HessianRequest)` | ✅ | **Analytical** via `jax.hessian` |
| `frequencies(FrequencyRequest)` | ✅ | From analytical Hessian |
| `parameter_gradient(ParameterGradientRequest)` | ✅ | **Analytical** via `jax.grad` |
| `batched_energy(BatchedEnergyRequest)` | ✅ | **Vectorized** via `jax.vmap` |
| `Capability.REUSABLE_STATE` | ✅ | Prepared session reuses compiled JAX functions |

!!! tip "Optax and JaxOpt optimizers"
    JaxMdBackend exposes analytical parameter gradients through
    `parameter_gradient(ParameterGradientRequest)`, making it compatible with
    [Optax](https://optax.readthedocs.io/) and
    [JaxOpt](https://jaxopt.github.io/) optimizers. See the
    [Optimization Guide](../how-it-works/optimization-guide.md) for
    workflow recommendations.

---

## GPU support

JaxMdBackend runs on whichever device JAX selects.  To use a GPU:

1. Install the CUDA-enabled JAX: `pip install jax[cuda12]`
2. Verify: `python -c "import jax; print(jax.default_backend())"`

The backend name includes the JAX device string (e.g., `JAX-MD (OPLSAA, gpu)`).

---

## Limitations

- **Harmonic only** — MM3 functional form is not yet supported.
- **Electrostatics zeroed** — Coulomb energy is computed with zero charges;
  charge optimization is not yet supported.
- **No improper torsions** — topology arrays are empty.
- **64-bit mode forced** — importing this module enables `jax_enable_x64`
  globally, which affects all JAX code in the process.

---

## Example

```python
from q2mm.backends.contracts import EnergyRequest, ParameterGradientRequest, PreparationRequest
from q2mm.backends.mm import JaxMdBackend
from q2mm.io.xyz import load_xyz
from q2mm.models.forcefield import ForceField

mol = load_xyz("molecule.xyz")
ff = ForceField.create_for_molecule(mol)

backend = JaxMdBackend(box=(50.0, 50.0, 50.0))
session = backend.prepare(PreparationRequest(case_id="example", molecule=mol, force_field=ff))
params = session.layout.vector(ff)

e = session.energy(EnergyRequest(parameters=params)).energy
print(f"JAX-MD energy: {e:.4f} kcal/mol")

# Analytical parameter gradients
grad_result = session.parameter_gradient(ParameterGradientRequest(parameters=params))
print(f"Energy: {grad_result.energy:.4f}, grad shape: {grad_result.gradient.shape}")
```

---

## See also

- [JaxBackend](jax-engine.md) — simpler, no periodic boundaries
- [Backend comparison table](index.md#backend-overview)
- [GPU benchmarks](../benchmarks/gpu.md)
- [API Reference: JaxMdBackend](../reference/q2mm/backends/mm/jax_md_engine.md)
